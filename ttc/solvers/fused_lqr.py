import os, sys

import torch
import torch.profiler

import math

import triton
import triton.language as tl

from fla.utils import input_guard, autocast_custom_fwd, autocast_custom_bwd


from ttc.solvers.symplectic_iter import lqr_symplectic_rev_iter, lqr_symplectic_dec_first_step, lqr_symplectic_fwd_iter_diff

import pdb

def fused_lqr_layer_fwd(
    A_diag: torch.Tensor, B: torch.Tensor,
    Q: torch.Tensor, R_inv_diag: torch.Tensor,
    Q_f: torch.Tensor,
    gamma_A: torch.Tensor,
    gamma_B: torch.Tensor,
    gamma_Q: torch.Tensor,
    h: torch.Tensor,
    T: int, num_chunks: int = 1,
    inv_guard: bool = False,
):
    assert all(x.is_cuda for x in [A_diag, B, Q, R_inv_diag, Q_f, gamma_A, gamma_B, gamma_Q, h])

    bs, nh, d = A_diag.shape
    du = B.shape[-1]
    assert T > 0 and T % num_chunks == 0

    chunk_size_T = T // num_chunks

    assert A_diag.shape == (bs, nh, d)
    assert Q.shape == (bs, d, d)
    assert B.shape == (bs, d, du)
    assert R_inv_diag.shape == (bs, nh, du)
    assert Q_f.shape == (bs, nh, d, d)

    assert gamma_A.shape == (bs, nh)
    assert gamma_B.shape == (bs, nh, du)
    assert gamma_Q.shape == (bs, nh, d)

    assert T % chunk_size_T == 0

    # Initialize recurrence from [I, Q_f].
    Y1 = torch.zeros((bs, nh, d, d), device=Q.device, dtype=torch.float32)
    Y1.diagonal(dim1=-2, dim2=-1).fill_(1)
    Y2 = Q_f.to(torch.float32, copy=True, memory_format=torch.contiguous_format)
    Y3_out = torch.zeros(bs, nh, d, du, device=Q.device, dtype=Q.dtype)
    
    for chunk_idx in range(num_chunks):

        # Chunked multiply-only recurrence.
        with torch.profiler.record_function("fwd:lqr_symplectic_rev_iter"):
            grid = lambda meta: (bs, nh, triton.cdiv(d, meta["BY"]),)
            lqr_symplectic_rev_iter[grid](
                A_diag.contiguous(), B.contiguous(),
                Q.contiguous(), R_inv_diag.contiguous(),
                None, # r_ptr

                # Discounted parameters.
                gamma_A.contiguous(),
                gamma_B.contiguous(),
                gamma_Q.contiguous(),

                # Current recurrence state.
                Y1.contiguous(), Y2.contiguous(),

                # Output pointers.
                None, None, Y3_out, None,


                T=chunk_size_T,

                # Triton meta parameters.
                B=bs,
                H=nh,
                D=d,
                DU=du,

                DISCOUNTED=True,
                GAMMA_LOG_SCALE=True,

                IS_AFFINE=False,
                IS_LAST_CHUNK=(chunk_idx == num_chunks - 1),
                IS_INPLACE=True,
                CACHE_FOR_BWD=True,

                NORMALIZE_TYPE=2,
                NORMALIZE_THRESHOLD=1e3,
                INV_GUARD=inv_guard,
                INV_GUARD_EPS=1e-6,
            )

    if torch.isnan(Y1).any() or torch.isnan(Y2).any() or torch.isinf(Y1).any() or torch.isinf(Y2).any() \
        or torch.any(torch.abs(Y1) >= 1e6) or torch.any(torch.abs(Y2) >= 1e6):
        raise RuntimeError("Numerical instability detected during fused LQR forward recurrence.")

    with torch.profiler.record_function("fwd:torch.linalg.lu_solve"):
        LU, pivots = torch.linalg.lu_factor(Y1)
        P = torch.linalg.lu_solve(LU, pivots, Y2)

    
    if torch.isnan(LU).any() or torch.isnan(pivots).any() or torch.isnan(P).any() or torch.isinf(LU).any() or torch.isinf(pivots).any() or torch.isinf(P).any() \
        or torch.any(torch.abs(Y1) >= 1e6) or torch.any(torch.abs(Y2) >= 1e6) or torch.any(torch.abs(P) >= 1e6):
        raise RuntimeError("Numerical instability detected during fused LQR LU solve.")

    with torch.profiler.record_function("fwd:lqr_symplectic_dec_first_step"):

        u = torch.zeros(bs, nh, d, device=Q.device, dtype=Q.dtype)
        lam0 = torch.zeros(bs, nh, d, device=Q.device, dtype=Q.dtype)
        lqr_symplectic_dec_first_step[(bs, nh)](
            A_diag.contiguous(),
            B.contiguous(),
            Q.contiguous(),
            R_inv_diag.contiguous(),
            P.contiguous(),
            h.contiguous(),

            # Discounted parameters.
            gamma_A.contiguous(),
            gamma_B.contiguous(),
            gamma_Q.contiguous(),

            # Output pointers.
            u.contiguous(),
            None,
            lam0.contiguous(),

            # Triton meta parameters.
            B=bs,
            H=nh,
            D=d,
            DU=du,
            DISCOUNTED=True,
            GAMMA_LOG_SCALE=True,
            RETURN_U=True,
            RETURN_LAMBDA=False,
            RETURN_LAMBDA0=True,
        )

    return u, lam0, LU, pivots, Y3_out

def fused_lqr_layer_bwd(
    A_diag: torch.Tensor, B: torch.Tensor,
    Q: torch.Tensor, R_inv_diag: torch.Tensor,
    gamma_A: torch.Tensor, gamma_B: torch.Tensor, gamma_Q: torch.Tensor,
    d_u: torch.Tensor, h: torch.Tensor, lam0: torch.Tensor,
    LU: torch.Tensor, pivots: torch.Tensor, Y3_out: torch.Tensor,
    T: int,
    inv_guard: bool = False,
):
    assert all(x.is_cuda for x in [A_diag, B, Q, R_inv_diag, gamma_A, gamma_B, gamma_Q, d_u, h, lam0, LU, pivots, Y3_out])

    bs, nh, d = A_diag.shape
    du = B.shape[-1]

    assert T > 0

    assert A_diag.shape == (bs, nh, d)
    assert Q.shape == (bs, d, d)
    assert B.shape == (bs, d, du)
    assert R_inv_diag.shape == (bs, nh, du)
    assert Y3_out.shape == (bs, nh, d, du)

    assert gamma_A.shape == (bs, nh)
    assert gamma_B.shape == (bs, nh, du)
    assert gamma_Q.shape == (bs, nh, d)

    assert h.shape == (bs, nh, d)
    assert d_u.shape == (bs, nh, du)
    assert lam0.shape == (bs, nh, d)

    assert LU.shape == (bs, nh, d, d)
    assert pivots.shape == (bs, nh, d)


    Q = Q.to(torch.float32)

    # Backward solve.
    y3 = Y3_out @ d_u[..., None] # [B,H,D,DU] @ [B,H,DU,1] = [B,H,D,1]
    with torch.profiler.record_function("bwd:torch.linalg.lu_solve"):
        d_p = torch.linalg.lu_solve(LU, pivots, y3.to(torch.float32)).squeeze(-1)

    grad_A_diag = torch.zeros_like(A_diag, dtype=torch.float32)
    grad_B = torch.zeros_like(B, dtype=torch.float32)
    grad_Q = torch.zeros_like(Q, dtype=torch.float32)
    grad_R_diag_inv = torch.zeros_like(R_inv_diag, dtype=torch.float32)
    grad_Qf = torch.zeros(bs, nh, d, d, device=Q.device, dtype=torch.float32)
    grad_h0 = torch.zeros_like(h, dtype=h.dtype)

    grad_gamma_A = torch.zeros_like(gamma_A, dtype=torch.float32)
    grad_gamma_B = torch.zeros_like(gamma_B, dtype=torch.float32)
    grad_gamma_Q = torch.zeros_like(gamma_Q, dtype=torch.float32)

    with torch.profiler.record_function("bwd:lqr_symplectic_fwd_iter_diff"):
        lqr_symplectic_fwd_iter_diff[(bs, nh)](
            A_diag.contiguous(),
            B.contiguous(),
            Q.contiguous(),
            R_inv_diag.contiguous(),
            d_p.contiguous(),
            d_u.contiguous(),
            h.contiguous(),
            lam0.contiguous(),

            # Discounted parameters.
            gamma_A.contiguous(),
            gamma_B.contiguous(),
            gamma_Q.contiguous(),

            # Output pointers.
            grad_A_diag.contiguous(),
            grad_B.contiguous(),
            grad_Q.contiguous(),
            grad_R_diag_inv.contiguous(),
            grad_Qf.contiguous(),
            grad_h0.contiguous(),

            # Discounted parameter gradients.
            grad_gamma_A.contiguous(),
            grad_gamma_B.contiguous(),
            grad_gamma_Q.contiguous(),


            T=T,

            # Triton meta parameters.
            B=bs,
            H=nh,
            D=d,
            DU=du,

            DISCOUNTED=True,
            GAMMA_LOG_SCALE=True,
            RETURN_GRAD_R_INV=True,
        )

    if torch.isnan(grad_A_diag).any() or torch.isnan(grad_B).any() or torch.isnan(grad_Q).any() or torch.isnan(grad_R_diag_inv).any() or torch.isnan(grad_Qf).any() or torch.isnan(grad_h0).any() \
        or torch.isinf(grad_A_diag).any() or torch.isinf(grad_B).any() or torch.isinf(grad_Q).any() or torch.isinf(grad_R_diag_inv).any() or torch.isinf(grad_Qf).any() or torch.isinf(grad_h0).any() \
        or torch.any(torch.abs(grad_A_diag) >= 1e6) or torch.any(torch.abs(grad_B) >= 1e6) or torch.any(torch.abs(grad_Q) >= 1e6) or torch.any(torch.abs(grad_R_diag_inv) >= 1e6) or torch.any(torch.abs(grad_Qf) >= 1e6) or torch.any(torch.abs(grad_h0) >= 1e6):
        raise RuntimeError("Numerical instability detected during fused LQR backward pass.")

    # Q_f shares dtype with Q; omit saving Q_f to reduce backward memory.
    return grad_A_diag.to(dtype=A_diag.dtype), grad_B.to(dtype=B.dtype), \
        grad_Q.to(dtype=Q.dtype), grad_R_diag_inv.to(dtype=R_inv_diag.dtype), \
        grad_Qf.to(dtype=Q.dtype), \
        grad_gamma_A.to(dtype=gamma_A.dtype), grad_gamma_B.to(dtype=gamma_B.dtype), grad_gamma_Q.to(dtype=gamma_Q.dtype), \
        grad_h0.to(dtype=h.dtype)

class FusedLQRSolveFunc(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, A_diag: torch.Tensor, B: torch.Tensor,
        Q: torch.Tensor, R_inv_diag: torch.Tensor,
        Q_f: torch.Tensor,
        gamma_A: torch.Tensor,
        gamma_B: torch.Tensor,
        gamma_Q: torch.Tensor,
        h0: torch.Tensor,
        T: int,
        cache_level: str = "LU",
        inv_guard: bool = False,
    ):
        # assert cache_level in ["none", "Lam0", "LU", "LU_lqr"]
        # if cache_level == "none":
        #     ctx.save_for_backward(h0, A_diag, B, Q, R_diag)
        # elif cache_level == "Lam0":
        #     u, lam0 = fused_homo_lqr_layer_fwd_wLam0(A_diag, B, Q, R_diag, h0, T, inv_guard=inv_guard)
        #     ctx.save_for_backward(h0, A_diag, B, Q, R_diag, lam0)
        # elif cache_level == "LU":
        #     u, lam0, LU, pivots, Y3_out = fused_homo_lqr_layer_fwd_wLU(A_diag, B, Q, R_diag, h0, T, inv_guard=inv_guard)
        #     ctx.save_for_backward(h0, A_diag, B, Q, R_diag, lam0, LU, pivots, Y3_out)
        # elif cache_level == "LU_lqr":
        #     u, lam0, LU, pivots, Y3_out = fused_lqr_layer_fwd(A_diag, B, Q, R_diag, Q_f, gamma_A, gamma_B, gamma_Q, h0, T, inv_guard=inv_guard)
        #     ctx.save_for_backward(h0, A_diag, B, Q, R_diag, lam0, LU, pivots, Y3_out)
        # else:
        #     raise ValueError(f"Invalid cache level: {cache_level}")
    
        u, lam0, LU, pivots, Y3_out = fused_lqr_layer_fwd(A_diag, B, Q, R_inv_diag, Q_f, gamma_A, gamma_B, gamma_Q, h0, T, inv_guard=inv_guard)
        ctx.save_for_backward(h0, A_diag, B, Q, R_inv_diag, gamma_A, gamma_B, gamma_Q, lam0, LU, pivots, Y3_out)
        
        ctx.T = T
        ctx.cache_level = cache_level
        ctx.inv_guard = inv_guard
        return u

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, d_u: torch.Tensor):
        cache_level = ctx.cache_level
        T = ctx.T
        inv_guard = ctx.inv_guard
    
        h0, A_diag, B, Q, R_inv_diag, gamma_A, gamma_B, gamma_Q, lam0, LU, pivots, Y3_out = ctx.saved_tensors
        d_A_diag, d_B, d_Q, d_R_diag_inv, d_Qf, d_gamma_A, d_gamma_B, d_gamma_Q, d_h0 = fused_lqr_layer_bwd(A_diag, B, Q, R_inv_diag,
            gamma_A, gamma_B, gamma_Q, d_u, h0, lam0, LU, pivots, Y3_out, T, inv_guard=inv_guard)


        # print(f"grad_A:", T, d_A_diag.min(), d_A_diag.max(), d_A_diag.mean())
        # print(f"d_B:", T, d_B.min(), d_B.max(), d_B.mean())
        # print(f"d_Q:", T, d_Q.min(), d_Q.max(), d_Q.mean())
        # print(f"d_R_diag_inv:", T, d_R_diag_inv.min(), d_R_diag_inv.max(), d_R_diag_inv.mean())
        # print(f"d_Qf:", T, d_Qf.min(), d_Qf.max(), d_Qf.mean())
        # print(f"d_gamma_A:", T, d_gamma_A.min(), d_gamma_A.max(), d_gamma_A.mean())
        # print(f"d_gamma_B:", T, d_gamma_B.min(), d_gamma_B.max(), d_gamma_B.mean())
        # print(f"d_gamma_Q:", T, d_gamma_Q.min(), d_gamma_Q.max(), d_gamma_Q.mean())
        # print(f"d_h0:", T, d_h0.min(), d_h0.max(), d_h0.mean())


        return d_A_diag, d_B, d_Q, d_R_diag_inv, d_Qf, d_gamma_A, d_gamma_B, d_gamma_Q, d_h0, None, None, None


def fused_lqr_solve(A_diag: torch.Tensor, B: torch.Tensor,
        Q: torch.Tensor, R_inv_diag: torch.Tensor,
        Q_f: torch.Tensor,
        gamma_A: torch.Tensor,
        gamma_B: torch.Tensor,
        gamma_Q: torch.Tensor,
        h0: torch.Tensor,
        T: int,
        inv_guard: bool = False,
):
    return FusedLQRSolveFunc.apply(A_diag, B, Q, R_inv_diag, Q_f, gamma_A, gamma_B, gamma_Q, h0, T, inv_guard)
