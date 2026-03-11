import os, sys

import torch
import math

import triton
import triton.language as tl

from ttc.solvers.lin_solver_lqr import solve_traj_actions, solve_traj_states


class KKTLQRSolveFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A_diag: torch.Tensor, B: torch.Tensor,
        Q: torch.Tensor, R_diag: torch.Tensor,
        h_0: torch.Tensor,
    ):
        bs, T, d = A_diag.shape
        du = B.shape[-1]

        assert A_diag.shape == (bs, T, d)
        assert B.shape == (bs, T, d, du)
        assert Q.shape == (bs, T, d, d)
        assert R_diag.shape == (bs, T, du)
        assert h_0.shape == (bs, d)

        u = solve_traj_actions(A_diag, B, Q, R_diag, h_0, r_traj=None, return_traj=False)
        ctx.save_for_backward(h_0, A_diag, B, Q, R_diag)
        ctx.T = T
        return u

    @staticmethod
    def backward(ctx, d_u: torch.Tensor):
        h_0, A_diag, B, Q, R_diag = ctx.saved_tensors
        T = ctx.T

        bs, _, d = A_diag.shape
        du = B.shape[-1]

        r_g = torch.zeros(bs, T, du, device=d_u.device)
        r_g[:, 0, :] = d_u

        u_fwd = solve_traj_actions(A_diag, B, Q, R_diag, h_0, r_traj=None, return_traj=True)
        u_bwd = solve_traj_actions(A_diag, B, Q, R_diag, torch.zeros_like(h_0), r_traj=r_g, return_traj=True)
        cost_fwd, hs_fwd, lam_fwd = solve_traj_states(u_fwd, A_diag, B, Q, R_diag, h_0, r_traj=None, return_states=True, return_costates=True)
        cost_bwd, hs_bwd, lam_bwd, lam0_bwd = solve_traj_states(u_bwd, A_diag, B, Q, R_diag, None, r_traj=r_g, return_states=True, return_costates=True, return_lambda0=True)

        hs_fwd_shift = torch.cat((h_0[:, None, :], hs_fwd[:, :-1, :]), dim=1)
        hs_bwd_shift = torch.cat((torch.zeros_like(h_0[:, None, :]), hs_bwd[:, :-1, :]), dim=1)
        A_grad = hs_fwd_shift * lam_bwd + hs_bwd_shift * lam_fwd

        B_grad = lam_bwd[..., None] * u_fwd[..., None, :] + lam_fwd[..., None] * u_bwd[..., None, :]

        Q_grad = (hs_bwd[..., :, None] * hs_fwd[..., None, :] + hs_fwd[..., :, None] * hs_bwd[..., None, :]) / 2.
        # Q_grad = torch.sum((hs_pytorch_bwd[..., :, None] * hs_pytorch_fwd[..., None, :]), dim=1)
        # Q_grad = torch.sum((hs_pytorch_fwd[..., :, None] * hs_pytorch_bwd[..., None, :]), dim=1)

        R_grad = u_fwd * u_bwd

        h0_grad = lam0_bwd

        return A_grad, B_grad, Q_grad, R_grad, h0_grad

def _generate_parameters(A_diag: torch.Tensor, B: torch.Tensor,
    Q: torch.Tensor, R_inv_diag: torch.Tensor,
    Q_f: torch.Tensor,
    log_gamma_A: torch.Tensor,
    log_gamma_B: torch.Tensor,
    log_gamma_Q: torch.Tensor,
    T: int
):
    bs, h, d = A_diag.shape
    du = B.shape[-1]

    t = (torch.arange(T, device=A_diag.device, dtype=A_diag.dtype) + 1.)[None, None, :] # [1, 1, T]

    decay_A = torch.exp(log_gamma_A[..., None, None] * t[..., None]) # [bs, h, 1, 1] * [bs, h, T, 1] -> [bs, h, T, 1]
    A_diag = A_diag[..., None, :] * decay_A # [bs, h, 1, d] * [bs, h, 1, d] * [bs, h, T, 1] -> [bs, h, T, d]
    A_diag += 1. # add identity matrix
    
    decay_B = torch.exp(log_gamma_B[..., None, None, :] * t[..., None, None]) # [bs, h, 1, 1, du] * [bs, h, T, 1, 1] -> [bs, h, T, 1, du]
    B = B[..., None, None, :, :] * decay_B # [bs, h, 1, d, du] * [bs, h, T, 1, du] -> [bs, h, T, d, du]

    decay_Q = torch.exp(log_gamma_Q[..., None, :] * t[..., :-1, None]) # [bs, h, 1, d] * [bs, h, T, 1] -> [bs, h, T, d]
    Q = decay_Q[..., None, :] * Q[..., None, None, :, :] * decay_Q[..., :, None] # [bs, h, T, 1, d] * [bs, 1, 1, d, d] * [bs, h, T, d, 1] -> [bs, h, T, d, d]
    Q = torch.cat([Q, Q_f[..., None, :, :]], dim=-3)

    R_diag = 1. / R_inv_diag[..., None, :].expand(*R_inv_diag.shape[:-1], T, du)

    return A_diag, B, Q, R_diag

def direct_lqr_solve(A_diag: torch.Tensor, B: torch.Tensor,
    Q: torch.Tensor, R_inv_diag: torch.Tensor,
    Q_f: torch.Tensor,
    log_gamma_A: torch.Tensor,
    log_gamma_B: torch.Tensor,
    log_gamma_Q: torch.Tensor,
    h0: torch.Tensor,
    T: int
):

    bs, h, d = A_diag.shape
    du = B.shape[-1]
    
    A_diag, B, Q, R_diag = _generate_parameters(A_diag, B, Q, R_inv_diag, Q_f, log_gamma_A, log_gamma_B, log_gamma_Q, T)
    
    u = solve_traj_actions(
        A_diag.reshape(-1, T, d),
        B.reshape(-1, T, d, du),
        Q.reshape(-1, T, d, d),
        R_diag.reshape(-1, T, du),
        h0.reshape(-1, d),
        r_traj=None,
        return_traj=False
    )

    return u.reshape(bs, h, d)

def kkt_lqr_solve(A_diag: torch.Tensor, B: torch.Tensor,
    Q: torch.Tensor, R_inv_diag: torch.Tensor,
    Q_f: torch.Tensor,
    log_gamma_A: torch.Tensor,
    log_gamma_B: torch.Tensor,
    log_gamma_Q: torch.Tensor,
    h0: torch.Tensor,
    T: int
):
    bs, h, d = A_diag.shape
    du = B.shape[-1]
    
    A_diag, B, Q, R_diag = _generate_parameters(A_diag, B, Q, R_inv_diag, Q_f, log_gamma_A, log_gamma_B, log_gamma_Q, T)
    u = KKTLQRSolveFunc.apply(
        A_diag.reshape(-1, T, d),
        B.reshape(-1, T, d, du),
        Q.reshape(-1, T, d, d),
        R_diag.reshape(-1, T, du),
        h0.reshape(-1, d),
    )
    return u.reshape(bs, h, d)