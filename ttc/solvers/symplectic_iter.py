import os

import torch
import triton
import triton.language as tl
from triton.runtime.autotuner import Config, autotune

from pathlib import Path

import functools

from fla.ops.utils import chunk_local_cumsum
from fla.utils import check_shared_mem, is_nvidia_hopper

from ttc.solvers.utils import config_pruner

if os.environ.get('TRITON_NO_AUTOTUNE', '0') == '1':
    BY_VALUES = [16]
    NUM_WARPS = [4]
    NUM_STAGES = [2]
else:
    BY_VALUES = [16, 32, 64, 128] if check_shared_mem() else [16, 32, 64]
    NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]
    NUM_STAGES = [2, 3, 4]


@triton.heuristics({
    'DISCOUNTED': lambda args: (args['gamma_A_ptr'] is not None) and (args['gamma_B_ptr'] is not None) and (args['gamma_Q_ptr'] is not None),
    'IS_AFFINE': lambda args: args['r_ptr'] is not None,
    'IS_INPLACE': lambda args: (args['Y1_out_ptr'] is None) or (args['Y2_out_ptr'] is None),
    'CACHE_FOR_BWD': lambda args: args['Y3_out_ptr'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BY': BY}, num_warps=num_warps, num_stages=num_stages)
        for BY in BY_VALUES
        for num_warps in NUM_WARPS
        for num_stages in NUM_STAGES
    ],
    key=['D', 'DU'],
    # Y1/Y2 are often updated in-place; restore them during autotune trials.
    restore_value=['Y1', 'Y2'],
    prune_configs_by={
        'early_config_prune': functools.partial(config_pruner, tile_dim_pairs=[('BY', 'D')]),
    },
)
@triton.jit(do_not_specialize=['T'])
def lqr_symplectic_rev_iter(
    # Input pointers.
    A_diag_ptr,                                 # [B,H,D]
    B_ptr,                                      # [B,D,DU]
    Q_ptr,                                      # [B,D,D]
    R_inv_diag_ptr,                             # [B,H,DU]
    r_ptr,                                      # [B,H,DU]

    gamma_A_ptr,                                # [B,H,1]
    gamma_B_ptr,                                # [B,H,DU]
    gamma_Q_ptr,                                # [B,H,D]

    # Output pointers.
    Y1, Y2,                                     # [B,H,D,D], [B,H,D,D], [B,H,D]
    Y1_out_ptr, Y2_out_ptr,                     # [B,H,D,D], [B,H,D,D]
    Y3_out_ptr, y3_out_ptr,                     # [B,H,D,DU], [B,H,D]

    T: int,                                     # Number of iterations (per chunk).

    # Meta parameters.
    B: tl.constexpr,                            # Batch size.
    H: tl.constexpr,                            # Number of heads.
    D: tl.constexpr,                            # State dimension.
    DU: tl.constexpr,                           # Action dimension.
    BY: tl.constexpr,                           # Rows per program (>=16), should divide D.

    DISCOUNTED: tl.constexpr,                   # If true, use discounted A/B/Q parameters.
    GAMMA_LOG_SCALE: tl.constexpr,              # If true, gamma parameters are in log scale.

    IS_AFFINE: tl.constexpr,                    # If true, include affine term on step 1.
    IS_LAST_CHUNK: tl.constexpr,                # If true, this is the final chunk.
    IS_INPLACE: tl.constexpr,                   # If true, update Y1/Y2 in-place.
    CACHE_FOR_BWD: tl.constexpr,                # If true, cache (-Y2 @ B @ R^-1) for backward.
    
    NORMALIZE_TYPE: tl.constexpr = 0,           # 0: none, odd: row-wise norm, even: global norm.
    NORMALIZE_THRESHOLD: tl.constexpr = 1e3,    # Normalize when Y1/Y2 norm exceeds this threshold.
    INV_GUARD: tl.constexpr = False,            # If true, add diagonal ridge to Y1/Y2.
    INV_GUARD_EPS: tl.constexpr = 1e-6,         # Ridge epsilon.
):
    bi  = tl.program_id(0)  # batch
    hi  = tl.program_id(1)  # head id
    ri = tl.program_id(2)  # row-block id

    if ((bi >= B) or (hi >= H)) or (ri >= tl.cdiv(D, BY)):
        return

    # Flattened batch/head index.
    b = bi * H + hi

    ptr_Y1 = tl.make_block_ptr(
        base=Y1 + b * (D * D),
        shape=(D, D),
        strides=(D, 1),
        offsets=(ri * BY, 0),
        block_shape=(BY, D),
        order=(1, 0),
    )

    ptr_Y2 = tl.make_block_ptr(
        base=Y2 + b * (D * D),
        shape=(D, D),
        strides=(D, 1),
        offsets=(ri * BY, 0),
        block_shape=(BY, D),
        order=(1, 0),
    )

    Y1_rowblk = tl.load(ptr_Y1, boundary_check=(0, 1), padding_option="zero").to(tl.float32)   # (BY,D)
    Y2_rowblk = tl.load(ptr_Y2, boundary_check=(0, 1), padding_option="zero").to(tl.float32)   # (BY,D)
    y3_rowblk = tl.zeros(shape=(BY,), dtype=Y2_rowblk.dtype).to(tl.float32)                # (BY,)

    blk_ptr_A_diag = tl.make_block_ptr(
        base=A_diag_ptr + b * D,
        shape=(D,),
        strides=(1,),
        offsets=(0,),
        block_shape=(D,),
        order=(0,),
    )
    A_diag = tl.load(blk_ptr_A_diag, boundary_check=(0,), padding_option="zero").to(tl.float32)

    blk_ptr_B = tl.make_block_ptr(
        base=B_ptr + bi * (D * DU),
        shape=(D, DU),
        strides=(DU, 1),
        offsets=(0, 0),
        block_shape=(D, DU),
        order=(1, 0),
    )
    B = tl.load(blk_ptr_B, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    blk_ptr_R_inv_diag = tl.make_block_ptr(
        base=R_inv_diag_ptr + b * DU,
        shape=(DU,),
        strides=(1,),
        offsets=(0,),
        block_shape=(DU,),
        order=(0,),
    )
    R_inv_diag = tl.load(blk_ptr_R_inv_diag, boundary_check=(0,), padding_option="zero").to(tl.float32)
    
    blk_ptr_Q = tl.make_block_ptr(
        base=Q_ptr + bi * (D * D),
        shape=(D, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(D, D),
        order=(1, 0),
    )
    Q = tl.load(blk_ptr_Q, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    if DISCOUNTED:
        blk_ptr_gamma_A = tl.make_block_ptr(
            base=gamma_A_ptr + b * 1,
            shape=(1,),
            strides=(1,),
            offsets=(0,),
            block_shape=(1,),
            order=(0,),
        )
        if not GAMMA_LOG_SCALE:
            gamma_A = tl.exp(tl.load(blk_ptr_gamma_A, boundary_check=(0,), padding_option="zero").to(tl.float32))
            log_gamma_A = tl.log(gamma_A)
        else:
            log_gamma_A = tl.load(blk_ptr_gamma_A, boundary_check=(0,), padding_option="zero").to(tl.float32)

        blk_ptr_gamma_B = tl.make_block_ptr(
            base=gamma_B_ptr + b * DU,
            shape=(DU,),
            strides=(1,),
            offsets=(0,),
            block_shape=(DU,),
            order=(0,),
        )
        if not GAMMA_LOG_SCALE:
            gamma_B = tl.load(blk_ptr_gamma_B, boundary_check=(0,), padding_option="zero").to(tl.float32)
            log_gamma_B = tl.log(gamma_B)
        else:
            log_gamma_B = tl.load(blk_ptr_gamma_B, boundary_check=(0,), padding_option="zero").to(tl.float32)

        blk_ptr_gamma_Q = tl.make_block_ptr(
            base=gamma_Q_ptr + b * D,
            shape=(D,),
            strides=(1,),
            offsets=(0,),
            block_shape=(D,),
            order=(0,),
        )
        if not GAMMA_LOG_SCALE:
            gamma_Q = tl.load(blk_ptr_gamma_Q, boundary_check=(0,), padding_option="zero").to(tl.float32)
            log_gamma_Q = tl.log(gamma_Q)
        else:
            log_gamma_Q = tl.load(blk_ptr_gamma_Q, boundary_check=(0,), padding_option="zero").to(tl.float32)
    else:
        log_gamma_A = tl.zeros(shape=(1,), dtype=tl.float32)
        log_gamma_B = tl.zeros(shape=(DU,), dtype=tl.float32)
        log_gamma_Q = tl.zeros(shape=(D,), dtype=tl.float32)


    log_gamma_A_t = log_gamma_A * T
    log_gamma_B_t = log_gamma_B * T
    log_gamma_Q_t = log_gamma_Q * (T - 1)

    if IS_AFFINE:
        blk_ptr_r = tl.make_block_ptr(
            base=r_ptr + b * DU,
            shape=(DU,),
            strides=(1,),
            offsets=(0,),
            block_shape=(DU,),
            order=(0,),
        )
        r = tl.load(blk_ptr_r, boundary_check=(0,), padding_option="zero").to(tl.float32)

    if CACHE_FOR_BWD:
        Y2xBxR_inv_bwd_rowblk = tl.zeros(shape=(BY, DU), dtype=Y2_rowblk.dtype).to(tl.float32)                # (BY, DU)


    for step in range(0, T):

        # NOTE: No support for matrix-vector multiplication in triton.
        # Y2 @ B @ R^-1 @ B^T
        B_t = B * tl.exp(log_gamma_B_t[None, :])
        Y2xB = tl.dot(Y2_rowblk, B_t, input_precision="tf32x3")                 # (BY, D) @ (D,DU) -> (BY,DU)
        Y2xBxR_inv = Y2xB * R_inv_diag[None, :]                                  # (BY, DU) * [DU,] -> (BY, DU)
        Y2xG = tl.dot(Y2xBxR_inv, B_t.trans(1, 0), input_precision="tf32x3")    # (BY, DU) @ (DU, D) -> (BY, D)
        # Compute y3 only on the final step: Y2 @ (-B @ R^-1 @ r).
        if IS_AFFINE:
            if (IS_LAST_CHUNK and (step == T - 1)):
                # Y2 @ (-B @ R^-1 @ r): [BY, BU] @ [BU,] -> [BY,]
                y3_rowblk -= tl.sum(Y2xBxR_inv * r[None, :], axis=1)

        # Cache Y2 @ -B @ R^-1 for backward.
        if CACHE_FOR_BWD:
            if (IS_LAST_CHUNK and (step == T - 1)):
                Y2xBxR_inv_bwd_rowblk = -Y2xBxR_inv

        # Y1 = Y1 + Y2 @ G: [BY, D] + [BY, D] -> [BY, D]
        Y1_rowblk += Y2xG


        A_diag_t = 1. + A_diag * tl.exp(log_gamma_A_t)

        # Y1 = (Y1 + Y2 @ G) @ A^-1: [BY, D] * [1, D] -> [BY, D]
        Y1_rowblk = tl.div_rn(Y1_rowblk, A_diag_t[None, :])

        if not (IS_LAST_CHUNK and (step == T - 1)):
            Q_t = tl.exp(log_gamma_Q_t[:, None]) * Q * tl.exp(log_gamma_Q_t[None, :])
            Y1xQ = tl.dot(Y1_rowblk, Q_t, input_precision="tf32x3")                           # (BY,D) @ (D,D) -> (BY,D)
            Y2_rowblk = tl.fma(Y2_rowblk, A_diag_t[None, :], Y1xQ)
        else:
            Y2_rowblk = Y2_rowblk * A_diag_t[None, :]

        # Advance log-gamma terms.
        log_gamma_A_t -= log_gamma_A
        log_gamma_B_t -= log_gamma_B
        log_gamma_Q_t -= log_gamma_Q

        # Optional normalization for numerical stability.
        if NORMALIZE_TYPE == 0:
            pass

        elif NORMALIZE_TYPE == 1:
            tile_max_1 = tl.sum(tl.abs(Y1_rowblk), axis=1)
            tile_max_2 = tl.sum(tl.abs(Y2_rowblk), axis=1)
            tile_max = tl.maximum(tile_max_1, tile_max_2)
            if NORMALIZE_THRESHOLD > 0:
                tile_max = tl.where(tile_max > NORMALIZE_THRESHOLD, tile_max, 1.0)

            Y1_rowblk = tl.div_rn(Y1_rowblk, tile_max[:, None])
            Y2_rowblk = tl.div_rn(Y2_rowblk, tile_max[:, None])

            if (IS_LAST_CHUNK and (step == T - 1)):
                if IS_AFFINE:
                    y3_rowblk = tl.div_rn(y3_rowblk, tile_max)
                if CACHE_FOR_BWD:
                    Y2xBxR_inv_bwd_rowblk = tl.div_rn(Y2xBxR_inv_bwd_rowblk, tile_max[:, None])

        elif NORMALIZE_TYPE == 2:
            tile_max_1 = tl.max(tl.sum(tl.abs(Y1_rowblk), axis=1))
            tile_max_2 = tl.max(tl.sum(tl.abs(Y2_rowblk), axis=1))
            tile_max = tl.maximum(tile_max_1, tile_max_2)
            if NORMALIZE_THRESHOLD > 0:
                tile_max = tl.where(tile_max > NORMALIZE_THRESHOLD, tile_max, 1.0)

            Y1_rowblk = tl.div_rn(Y1_rowblk, tile_max)
            Y2_rowblk = tl.div_rn(Y2_rowblk, tile_max)

            if (IS_LAST_CHUNK and (step == T - 1)):
                if IS_AFFINE:
                    y3_rowblk = tl.div_rn(y3_rowblk, tile_max)
                if CACHE_FOR_BWD:
                    Y2xBxR_inv_bwd_rowblk = tl.div_rn(Y2xBxR_inv_bwd_rowblk, tile_max)

    # Optional diagonal ridge on Y1.
    if INV_GUARD:
        id_eps = ((ri*BY + tl.arange(0, BY)[:, None]) == tl.arange(0, D)[None, :]) * INV_GUARD_EPS
        Y1_rowblk = Y1_rowblk + id_eps

    # Single store at chunk end.

    if not IS_INPLACE:

        ptr_Y1_out = tl.make_block_ptr(
            base=Y1_out_ptr + b * (D * D),
            shape=(D, D),
            strides=(D, 1),
            offsets=(ri * BY, 0),
            block_shape=(BY, D),
            order=(1, 0),
        )

        ptr_Y2_out = tl.make_block_ptr(
            base=Y2_out_ptr + b * (D * D),
            shape=(D, D),
            strides=(D, 1),
            offsets=(ri * BY, 0),
            block_shape=(BY, D),
            order=(1, 0),
        )

        tl.store(ptr_Y1_out, Y1_rowblk.to(ptr_Y1_out.dtype.element_ty), boundary_check=(0, 1))
        tl.store(ptr_Y2_out, Y2_rowblk.to(ptr_Y2_out.dtype.element_ty), boundary_check=(0, 1))
    else:
        tl.store(ptr_Y1, Y1_rowblk.to(ptr_Y1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(ptr_Y2, Y2_rowblk.to(ptr_Y2.dtype.element_ty), boundary_check=(0, 1))


    if (IS_AFFINE and IS_LAST_CHUNK):
        blk_ptr_y3 = tl.make_block_ptr(
            base=y3_out_ptr + b * D,
            shape=(D,),
            strides=(1,),
            offsets=(ri * BY,),
            block_shape=(BY,),
            order=(0,),
        )
        tl.store(blk_ptr_y3, y3_rowblk.to(blk_ptr_y3.dtype.element_ty), boundary_check=(0,))

    if CACHE_FOR_BWD:
        blk_ptr_Y2xBxR_inv_bwd = tl.make_block_ptr(
            base=Y3_out_ptr + b * (D * DU),
            shape=(D, DU),
            strides=(DU, 1),
            offsets=(ri * BY, 0),
            block_shape=(BY, DU),
            order=(1, 0),
        )
        tl.store(blk_ptr_Y2xBxR_inv_bwd, Y2xBxR_inv_bwd_rowblk.to(blk_ptr_Y2xBxR_inv_bwd.dtype.element_ty), boundary_check=(0, 1))

@triton.heuristics({
    'DISCOUNTED': lambda args: (args['gamma_A_ptr'] is not None) and (args['gamma_B_ptr'] is not None) and (args['gamma_Q_ptr'] is not None),
    'RETURN_U': lambda args: (args['u_ptr'] is not None),
    'RETURN_LAMBDA': lambda args: (args['lambda_ptr'] is not None),
    'RETURN_LAMBDA0': lambda args: (args['lambda0_ptr'] is not None),
})
@triton.autotune(
    configs=[
        triton.Config(kwargs={}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in NUM_STAGES
    ],
    key=['D', 'DU'],
)
@triton.jit(do_not_specialize=['T'])
def lqr_symplectic_dec_first_step(
    # Input pointers.
    A_diag_ptr,                                 # [B,H,D]
    B_ptr,                                      # [B,D,DU]
    Q_ptr,                                      # [B,D,D]
    R_inv_diag_ptr,                             # [B,H,DU]
    P_ptr,                                      # [B,H,D,D]
    h0_ptr,                                     # [B,H,D]

    # Discount parameters.
    gamma_A_ptr,                                # [B,H,1]
    gamma_B_ptr,                                # [B,H,DU]
    gamma_Q_ptr,                                # [B,H,D]

    # Output pointers.
    u_ptr,                                      # [B,DU]
    lambda_ptr,                                 # [B,D]
    lambda0_ptr,                                # [B,D]

    # Meta parameters.
    B: tl.constexpr,                            # Batch size.
    H: tl.constexpr,                            # Number of heads.
    D: tl.constexpr,                            # State dimension.
    DU: tl.constexpr,                           # Action dimension.

    DISCOUNTED: tl.constexpr,                   # If true, use discounted A/B/Q parameters.
    GAMMA_LOG_SCALE: tl.constexpr,              # If true, gamma parameters are in log scale.

    RETURN_U: tl.constexpr,                     # If true, return u.
    RETURN_LAMBDA: tl.constexpr,                # If true, return lambda_1.
    RETURN_LAMBDA0: tl.constexpr,               # If true, return lambda_0.
):
    bi  = tl.program_id(0)  # batch
    hi  = tl.program_id(1)  # head

    b = bi * H + hi

    tl.static_assert((RETURN_U or RETURN_LAMBDA) or RETURN_LAMBDA0)

    if bi >= B or hi >= H:
        return

    blk_ptr_h0 = tl.make_block_ptr(
        base=h0_ptr + b * D,
        shape=(D,),
        strides=(1,),
        offsets=(0,),
        block_shape=(D,),
        order=(0,),
    )

    h = tl.load(blk_ptr_h0, boundary_check=(0,), padding_option="zero").to(tl.float32)   # (D, )
    lam = tl.zeros((D,), dtype=tl.float32)

    blk_ptr_P = tl.make_block_ptr(
        base=(P_ptr + b * (D * D)),
        shape=(D, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(D, D),
        order=(1, 0),
    )
    P = tl.load(blk_ptr_P, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    # NOTE: No support for matrix-vector multiplication in triton.
    lam += tl.sum(P * h[None, :], axis=1)                           # (D,D) @ (D,) -> (D,)

    if RETURN_LAMBDA0:
        blk_ptr_lambda0 = tl.make_block_ptr(
            base=lambda0_ptr + b * D,
            shape=(D,),
            strides=(1,),
            offsets=(0,),
            block_shape=(D,),
            order=(0,),
        )
        tl.store(blk_ptr_lambda0, lam.to(blk_ptr_lambda0.dtype.element_ty), boundary_check=(0,))

    blk_ptr_A_diag = tl.make_block_ptr(
        base=A_diag_ptr + b * D,
        shape=(D,),
        strides=(1,),
        offsets=(0,),
        block_shape=(D,),
        order=(0,),
    )
    A_diag = tl.load(blk_ptr_A_diag, boundary_check=(0,), padding_option="zero").to(tl.float32)
    if DISCOUNTED:
        blk_ptr_gamma_A = tl.make_block_ptr(
            base=gamma_A_ptr + b * 1,
            shape=(1,),
            strides=(1,),
            offsets=(0,),
            block_shape=(1,),
            order=(0,),
        )
        if GAMMA_LOG_SCALE:
            gamma_A = tl.exp(tl.load(blk_ptr_gamma_A, boundary_check=(0,), padding_option="zero").to(tl.float32))
        else:
            gamma_A = tl.load(blk_ptr_gamma_A, boundary_check=(0,), padding_option="zero").to(tl.float32)
        A_diag = 1. + A_diag * gamma_A
    else:
        A_diag = 1. + A_diag

    # lambda_{t+1} <- A_inv^T(lambda_t - Q @ h_t)
    lam = tl.div_rn(lam, A_diag)


    blk_ptr_B = tl.make_block_ptr(
        base=B_ptr + bi * (D * DU),
        shape=(D, DU),
        strides=(DU, 1),
        offsets=(0, 0),
        block_shape=(D, DU),
        order=(1, 0),
    )
    B = tl.load(blk_ptr_B, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    if DISCOUNTED:
        blk_ptr_gamma_B = tl.make_block_ptr(
            base=gamma_B_ptr + b * DU,
            shape=(DU,),
            strides=(1,),
            offsets=(0,),
            block_shape=(DU,),
            order=(0,),
        )
        if GAMMA_LOG_SCALE:
            gamma_B = tl.exp(tl.load(blk_ptr_gamma_B, boundary_check=(0,), padding_option="zero").to(tl.float32))
        else:
            gamma_B = tl.load(blk_ptr_gamma_B, boundary_check=(0,), padding_option="zero").to(tl.float32)
        B = B * gamma_B[None, :]
    # NOTE: B^T @ lambda_t = lambda_t^T @ B. No support for matrix-vector multiplication in triton.
    BTxlam = tl.sum(B * lam[:, None], axis=0)                           # (BU,D) @ (D,) -> (BU,)


    blk_ptr_R_inv_diag = tl.make_block_ptr(
        base=R_inv_diag_ptr + b * DU,
        shape=(DU,),
        strides=(1,),
        offsets=(0,),
        block_shape=(DU,),
        order=(0,),
    )
    R_inv_diag = tl.load(blk_ptr_R_inv_diag, boundary_check=(0,), padding_option="zero").to(tl.float32)
    # u_t <- -R_inv_diag @ B^T @ lambda_t
    u = -BTxlam.reshape((DU,)) * R_inv_diag # [DU, ]

    if RETURN_U:
        blk_ptr_u_out = tl.make_block_ptr(
            base=u_ptr + b * DU,
            shape=(DU,),
            strides=(1,),
            offsets=(0,),
            block_shape=(DU,),
            order=(0,),
        )
        tl.store(blk_ptr_u_out, u.to(blk_ptr_u_out.dtype.element_ty), boundary_check=(0,))

    if RETURN_LAMBDA:
        blk_ptr_lambda_out = tl.make_block_ptr(
            base=lambda_ptr + b * D,
            shape=(D,),
            strides=(1,),
            offsets=(0,),
            block_shape=(D,),
            order=(0,),
        )
        tl.store(blk_ptr_lambda_out, lam.to(blk_ptr_lambda_out.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config(kwargs={}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in NUM_STAGES
    ],
    key=['D', 'DU'],
    # Grad tensors are accumulated in-place; restore between autotune trials.
    restore_value=['grad_A_diag_ptr', 'grad_B_ptr', 'grad_Q_ptr', 'grad_R_diag_ptr', 'grad_h0_ptr'],
)
@triton.jit(do_not_specialize=['T'])
def lqr_symplectic_fwd_iter_diff(
    # Input pointers.
    A_diag_ptr,                                 # [B,H,D]
    B_ptr,                                      # [B,D,DU]
    Q_ptr,                                      # [B,D,D]
    R_inv_diag_ptr,                             # [B,H,DU]
    d_p_ptr,                                    # [B,H,D]
    d_r_ptr,                                    # [B,H,DU]
    h0_ptr,                                     # [B,H,D]
    lambda0_ptr,                                # [B,H,D]

    # Discount parameters.
    gamma_A_ptr,                                # [B,H,1]
    gamma_B_ptr,                                # [B,H,DU]
    gamma_Q_ptr,                                # [B,H,D]

    # Output pointers.
    grad_A_diag_ptr,                            # [B,H,D]
    grad_B_ptr,                                 # [B,D,DU]
    grad_Q_ptr,                                 # [B,D,D]
    grad_R_diag_ptr,                            # [B,H,DU]
    grad_Qf_ptr,                                # [B,H,D,D]
    grad_h0_ptr,                                # [B,H,D]

    grad_gamma_A_ptr,                           # [B,H,1]
    grad_gamma_B_ptr,                           # [B,H,DU]
    grad_gamma_Q_ptr,                           # [B,H,D]

    T,                                          # Number of iterations.

    # Meta parameters.
    B: tl.constexpr,                            # Batch size.
    H: tl.constexpr,                            # Number of heads.
    D: tl.constexpr,                            # State dimension.
    DU: tl.constexpr,                           # Action dimension.

    DISCOUNTED: tl.constexpr,                   # If true, use discounted A/B/Q parameters.
    GAMMA_LOG_SCALE: tl.constexpr,              # If true, gamma parameters are in log scale.
    RETURN_GRAD_R_INV: tl.constexpr,            # If true, return gradient w.r.t. R^{-1}.
):
    bi  = tl.program_id(0)  # batch
    hi  = tl.program_id(1)  # head

    if bi >= B or hi >= H:
        return

    b = bi * H + hi

    # Output pointers.
    blk_ptr_grad_A_diag = grad_A_diag_ptr + b * D + tl.arange(0, D)
    blk_ptr_grad_B = grad_B_ptr + bi * (D * DU) + tl.arange(0, D)[:, None] * DU + tl.arange(0, DU)[None, :]
    blk_ptr_grad_Q = grad_Q_ptr + bi * (D * D) + tl.arange(0, D)[:, None] * D + tl.arange(0, D)[None, :]
    blk_ptr_grad_R_diag = grad_R_diag_ptr + b * DU + tl.arange(0, DU)
    blk_ptr_grad_gamma_A = grad_gamma_A_ptr + b + tl.arange(0, 1)
    blk_ptr_grad_gamma_B = grad_gamma_B_ptr + b * DU + tl.arange(0, DU)
    blk_ptr_grad_gamma_Q = grad_gamma_Q_ptr + b * D + tl.arange(0, D)
    
    blk_ptr_grad_Qf = tl.make_block_ptr(
        base=grad_Qf_ptr + b * D * D,
        shape=(D, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(D, D),
        order=(1, 0),
    )

    blk_ptr_grad_h0 = tl.make_block_ptr(
        base=grad_h0_ptr + b * D,
        shape=(D,),
        strides=(1,),
        offsets=(0,),
        block_shape=(D,),
        order=(0,),
    )

    blk_ptr_A_diag = tl.make_block_ptr(
        base=A_diag_ptr + b * D,
        shape=(D,),
        strides=(1,),
        offsets=(0,),
        block_shape=(D,),
        order=(0,),
    )
    A_diag = tl.load(blk_ptr_A_diag, boundary_check=(0,), padding_option="zero").to(tl.float32)

    blk_ptr_B = tl.make_block_ptr(
        base=B_ptr + bi * (D * DU),
        shape=(D, DU),
        strides=(DU, 1),
        offsets=(0, 0),
        block_shape=(D, DU),
        order=(1, 0),
    )
    B = tl.load(blk_ptr_B, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    blk_ptr_Q = tl.make_block_ptr(
        base=Q_ptr + bi * (D * D),
        shape=(D, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(D, D),
        order=(1, 0),
    )
    Q = tl.load(blk_ptr_Q, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    blk_ptr_R_inv_diag = tl.make_block_ptr(
        base=R_inv_diag_ptr + b * DU,
        shape=(DU,),
        strides=(1,),
        offsets=(0,),
        block_shape=(DU,),
        order=(0,),
    )
    R_inv_diag = tl.load(blk_ptr_R_inv_diag, boundary_check=(0,), padding_option="zero").to(tl.float32)

    if DISCOUNTED:
        blk_ptr_gamma_A = tl.make_block_ptr(
            base=gamma_A_ptr + b * 1,
            shape=(1,),
            strides=(1,),
            offsets=(0,),
            block_shape=(1,),
            order=(0,),
        )
        if not GAMMA_LOG_SCALE:
            gamma_A = tl.load(blk_ptr_gamma_A, boundary_check=(0,), padding_option="zero").to(tl.float32)
            log_gamma_A = tl.log(gamma_A)
        else:
            log_gamma_A = tl.load(blk_ptr_gamma_A, boundary_check=(0,), padding_option="zero").to(tl.float32)

        blk_ptr_gamma_B = tl.make_block_ptr(
            base=gamma_B_ptr + b * DU,
            shape=(DU,),
            strides=(1,),
            offsets=(0,),
            block_shape=(DU,),
            order=(0,),
        )
        if not GAMMA_LOG_SCALE:
            gamma_B = tl.load(blk_ptr_gamma_B, boundary_check=(0,), padding_option="zero").to(tl.float32)
            log_gamma_B = tl.log(gamma_B)
        else:
            log_gamma_B = tl.load(blk_ptr_gamma_B, boundary_check=(0,), padding_option="zero").to(tl.float32)

        blk_ptr_gamma_Q = tl.make_block_ptr(
            base=gamma_Q_ptr + b * D,
            shape=(D,),
            strides=(1,),
            offsets=(0,),
            block_shape=(D,),
            order=(0,),
        )
        if not GAMMA_LOG_SCALE:
            gamma_Q = tl.load(blk_ptr_gamma_Q, boundary_check=(0,), padding_option="zero").to(tl.float32)
            log_gamma_Q = tl.log(gamma_Q)
        else:
            log_gamma_Q = tl.load(blk_ptr_gamma_Q, boundary_check=(0,), padding_option="zero").to(tl.float32)
    else:
        log_gamma_A = tl.zeros(shape=(1,), dtype=tl.float32)
        log_gamma_B = tl.zeros(shape=(DU,), dtype=tl.float32)
        log_gamma_Q = tl.zeros(shape=(D,), dtype=tl.float32)

    log_gamma_A_t = log_gamma_A
    log_gamma_B_t = log_gamma_B
    log_gamma_Q_t = tl.zeros(shape=(D,), dtype=tl.float32)

    blk_ptr_d_r = tl.make_block_ptr(
        base=d_r_ptr + b * DU,
        shape=(DU,),
        strides=(1,),
        offsets=(0,),
        block_shape=(DU,),
        order=(0,),
    )
    d_r = tl.load(blk_ptr_d_r, boundary_check=(0,), padding_option="zero").to(tl.float32) # (DU, )

    blk_ptr_h0 = tl.make_block_ptr(
        base=h0_ptr + b * D,
        shape=(D,),
        strides=(1,),
        offsets=(0,),
        block_shape=(D,),
        order=(0,),
    )
    h = tl.load(blk_ptr_h0, boundary_check=(0,), padding_option="zero").to(tl.float32)   # (D, )
    d_h = tl.zeros((D,), dtype=h.dtype)


    blk_ptr_lambda0 = tl.make_block_ptr(
        base=lambda0_ptr + b * D,
        shape=(D,),
        strides=(1,),
        offsets=(0,),
        block_shape=(D,),
        order=(0,),
    )
    lam = tl.load(blk_ptr_lambda0, boundary_check=(0,), padding_option="zero").to(tl.float32)

    blk_ptr_d_p = tl.make_block_ptr(
        base=d_p_ptr + b * D,
        shape=(D,),
        strides=(1,),
        offsets=(0,),
        block_shape=(D,),
        order=(0,),
    )
    # Y1^-1 @ Y2 @ h + Y1^-1 @ y3 = P @ h + p = p since h = 0
    d_lam = tl.load(blk_ptr_d_p, boundary_check=(0,), padding_option="zero").to(tl.float32)


    # ----- time loop: keep everything on-chip; only stream S panels -----
    for step in range(0, T):
        # compute lam and d_lam
        if step > 0:

            Q_t = tl.exp(log_gamma_Q_t[:, None]) * Q * tl.exp(log_gamma_Q_t[None, :]) # [D,D]

            # NOTE: No support for matrix-vector multiplication in triton.
            Qxh = tl.sum(Q_t * h[None, :], axis=1)                           # (BS,D) @ (D,) -> (BS,)
            d_Qxh = tl.sum(Q_t * d_h[None, :], axis=1)                           # (D,D) @ (D,) -> (D,)

            lam -= Qxh
            d_lam -= d_Qxh

        if step == 0:
            # dh_0 = dlambda_0
            tl.store(blk_ptr_grad_h0, d_lam.to(blk_ptr_grad_h0.dtype.element_ty), boundary_check=(0,))

        A_diag_t = 1. + A_diag * tl.exp(log_gamma_A_t)

        # lambda_{t+1} <- A_inv^T(lambda_t - Q @ h_t)
        lam = tl.div_rn(lam, A_diag_t)
        # d_lambda_{t+1} <- A_inv^T(dlambda_t - Q @ dh_t)
        d_lam = tl.div_rn(d_lam, A_diag_t)

        # dA_t = lambda_t @ dh_t^T + dlambda_t @ h_t^T -> dDiag(A_t) = dh_t .* lambda_t + dlambda_t .* h_t
        dA_t = lam * d_h + d_lam * h
        # A_t = A * gamma_A^t + 1.
        # -> d A = d A_t * gamma_A^t
        # -> d gamma_A = sum(d A_t * A) * t * gamma_A^(t-1)
        # -> d log_gamma_A = sum(d A_t * A) * t * exp(log_gamma_A * t)
        tl.atomic_add(blk_ptr_grad_A_diag, (dA_t * tl.exp(log_gamma_A_t)).to(blk_ptr_grad_A_diag.dtype.element_ty), scope="cta")
        if not GAMMA_LOG_SCALE:
            tl.atomic_add(blk_ptr_grad_gamma_A, (tl.sum(dA_t * A_diag) * (step+1) * tl.exp(log_gamma_A_t - log_gamma_A)).to(blk_ptr_grad_gamma_A.dtype.element_ty), scope="cta")
        else:
            tl.atomic_add(blk_ptr_grad_gamma_A, (tl.sum(dA_t * A_diag) * (step+1) * tl.exp(log_gamma_A_t)).to(blk_ptr_grad_gamma_A.dtype.element_ty), scope="cta")

        B_t = B * tl.exp(log_gamma_B_t[None, :])

        # NOTE: B^T @ lambda_t = lambda_t^T @ B. No support for matrix-vector multiplication in triton.
        BTxlam = tl.sum(B_t * lam[:, None], axis=0)                           # (DU,D) @ (D,) -> (DU,)

        # NOTE: B^T @ lambda_t = lambda_t^T @ B. No support for matrix-vector multiplication in triton.
        d_BTxlam = tl.sum(B_t * d_lam[:, None], axis=0)                           # (DU,D) @ (D,) -> (DU,)

        # u_t <- -R_inv_diag @ B^T @ lambda_t
        u = -BTxlam * R_inv_diag # [DU, ]
        # d_u_t <- -R_inv_diag @ d_B^T @ d_lambda_t 
        d_u = -d_BTxlam * R_inv_diag # [DU, ]

        blk_ptr_d_r = tl.make_block_ptr(
            base=d_r_ptr + b * DU,
            shape=(DU,),
            strides=(1,),
            offsets=(0,),
            block_shape=(DU,),
            order=(0,),
        )
        if (step == 0):
            # d_u_t <- d_u_t - R_inv_diag @ r
            d_u = tl.fma(d_r, -R_inv_diag, d_u) # [DU, ]

        # dB_t = lambda_t @ du_t^T + dlambda_t @ u_t^T
        dB_t = lam[:, None] * d_u[None, :] + d_lam[:, None] * u[None, :]
        # B_t = B * diag(gamma_B)^t
        # -> d B = d B_t * diag(gamma_B)^t
        # -> d gamma_B = sum_i (d B_t)_{ij} * B_{ij} * t * gamma_B^(t-1)
        # -> d log_gamma_B = sum_i (d B_t)_{ij} * B_{ij} * t * exp(log_gamma_B * t)
        tl.atomic_add(blk_ptr_grad_B, (dB_t * tl.exp(log_gamma_B_t[None, :])).to(blk_ptr_grad_B.dtype.element_ty), scope='gpu')
        if not GAMMA_LOG_SCALE:
            tl.atomic_add(blk_ptr_grad_gamma_B, (tl.sum(dB_t * B, axis=0) * (step+1) * tl.exp(log_gamma_B_t - log_gamma_B)).to(blk_ptr_grad_gamma_B.dtype.element_ty), scope='cta')
        else:
            tl.atomic_add(blk_ptr_grad_gamma_B, (tl.sum(dB_t * B, axis=0) * (step+1) * tl.exp(log_gamma_B_t)).to(blk_ptr_grad_gamma_B.dtype.element_ty), scope='cta')

        # NOTE: No support for matrix-vector multiplication in triton.
        Bxu = tl.sum(B_t * u[None, :], axis=1)                           # (D,DU) @ (DU,) -> (D,)
        # NOTE: No support for matrix-vector multiplication in triton.
        d_Bxu = tl.sum(B_t * d_u[None, :], axis=1)                           # (D,DU) @ (DU,) -> (D,)

        # h_{t+1} <- A_t @ h_{t-1} + B_t @ u_t
        h = tl.fma(h, A_diag_t, Bxu)
        # d_h_{t+1} <- A_t @ d_h_{t-1} + B_t @ d_u_t
        d_h = tl.fma(d_h, A_diag_t, d_Bxu)

        # advance log_gamma_A_t, log_gamma_B_t, log_gamma_Q_t here
        # since they are considered as the next step's parameters in the below equations.
        log_gamma_A_t += log_gamma_A
        log_gamma_B_t += log_gamma_B
        log_gamma_Q_t += log_gamma_Q

        # Note that dQ_t denotes the gradient w.r.t. Q_{step+1}. so t = step + 1 in the following equations.
        # dQ_t = (h_t @ dh_t^T + dh_t @ h_t^T) / 2
        dQ_t = (h[:, None] * d_h[None, :] + d_h[:, None] * h[None, :]) / 2.
        if step == T - 1:
            tl.store(blk_ptr_grad_Qf, dQ_t.to(blk_ptr_grad_Qf.dtype.element_ty), boundary_check=(0, 1))
        else:
            # Q_t = diag(gamma_Q)^t * Q * diag(gamma_Q)^t
            # -> d Q = diag(gamma_Q)^t * d Q_t * diag(gamma_Q)^t
            # -> d gamma_Q = (sum_i (d Q_t)_{ij} * Q_{ij} * (gamma_Q_i)^t + sum_j (d Q_t)_{ij} * Q_{ij} * (gamma_Q_j)^t) * (t * gamma_Q^(t-1))
            # -> d log_gamma_Q = (sum_i (d Q_t)_{ij} * Q_{ij} * (gamma_Q_i)^t + sum_j (d Q_t)_{ij} * Q_{ij} * (gamma_Q_j)^t) * (t * exp(log_gamma_Q * t))
            tl.atomic_add(blk_ptr_grad_Q, (tl.exp(log_gamma_Q_t[:, None]) * dQ_t * tl.exp(log_gamma_Q_t[None, :])).to(blk_ptr_grad_Q.dtype.element_ty), scope="gpu")
            if not GAMMA_LOG_SCALE:
                tl.atomic_add(blk_ptr_grad_gamma_Q, (
                    (
                        tl.sum(dQ_t * Q * tl.exp(log_gamma_Q_t[:, None]), axis=0) +
                        tl.sum(dQ_t * Q * tl.exp(log_gamma_Q_t[None, :]), axis=1)
                    ) * (step+1) * tl.exp(log_gamma_Q_t - log_gamma_Q)
                ).to(blk_ptr_grad_gamma_Q.dtype.element_ty), scope="cta")
            else:
                tl.atomic_add(blk_ptr_grad_gamma_Q, (
                    (
                        tl.sum(dQ_t * Q * tl.exp(log_gamma_Q_t[:, None]), axis=0) +
                        tl.sum(dQ_t * Q * tl.exp(log_gamma_Q_t[None, :]), axis=1)
                    ) * (step+1) * tl.exp(log_gamma_Q_t)
                ).to(blk_ptr_grad_gamma_Q.dtype.element_ty), scope="cta")

        # dR_t = u_t @ du_t^T -> dDiag(R_t) = u_t .* du_t^T
        dR_t = u * d_u
        # R = 1 / R_inv -> dR_inv = -1 / R^2 * dR
        if RETURN_GRAD_R_INV:
            tl.atomic_add(blk_ptr_grad_R_diag, tl.div_rn(-dR_t, R_inv_diag * R_inv_diag).to(blk_ptr_grad_R_diag.dtype.element_ty), scope="cta")
        else:
            tl.atomic_add(blk_ptr_grad_R_diag_inv, dR_t.to(blk_ptr_grad_R_diag_inv.dtype.element_ty), scope="cta")
