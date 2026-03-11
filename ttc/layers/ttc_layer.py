import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import triton
import triton.language as tl

from fla.modules.l2norm import l2norm

from ttc.solvers.torch_lqr import direct_lqr_solve, kkt_lqr_solve
from ttc.solvers.fused_lqr import fused_lqr_solve

import pdb

LQR_SOLVER_IMPL_MAP = {
    "riccati": direct_lqr_solve,
    "kkt": kkt_lqr_solve,
    "fused": fused_lqr_solve,
}

class TTCLayer(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, h_dim: int, num_heads: int,
        b_rank: int, q_rank: int, u_dim: int = None,
        bias: bool = False, gamma_bias: bool = False, gamma_bias_init: str = "sqrt_dim",
        rescale: str = "l2_norm", q_reparam: str = "plain", solver_impl: str = "fused",
    ):
        super(TTCLayer, self).__init__()

        if u_dim is None:
            u_dim = h_dim
    
        assert solver_impl in LQR_SOLVER_IMPL_MAP, f"Unknown LQR solver {solver_impl}"
        self.solver = LQR_SOLVER_IMPL_MAP[solver_impl]

        assert rescale in ["none", "l2_norm", "h0_norm"], f"Unknown rescale method {rescale}"
        self.rescale = rescale

        self.q_reparam = q_reparam
        assert self.q_reparam in ["plain", "cholesky"], f"Unknown Q reparameterization {q_reparam}"

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.h_dim = h_dim
        self.u_dim = u_dim
        self.q_rank = q_rank
        self.b_rank = b_rank


        # Per-token projection emits:
        # - headwise blocks: h0, A_diag, R_inv_diag, Qf coefficients, log-gammas
        # - shared coefficients: B and Q bases
        self.in_proj = nn.Linear(in_dim, (h_dim + h_dim + u_dim + h_dim + 1 + u_dim + h_dim) * num_heads + b_rank + q_rank, bias=bias)
        self.out_proj = nn.Linear(u_dim * num_heads, out_dim, bias=bias)

        self.Bs = nn.Parameter(torch.empty(b_rank, h_dim, u_dim))

        self.Qs = nn.Parameter(torch.empty(q_rank, h_dim, h_dim))

        self.ttc_gamma_bias_init = gamma_bias_init
        assert self.ttc_gamma_bias_init in ["sqrt_dim", "zero"], f"Unknown gamma bias initialization {gamma_bias_init}"

        self.bias_gamma = None
        if gamma_bias:
            self.bias_gamma = nn.Parameter(torch.empty(1 + u_dim + h_dim, dtype=torch.float32))

        TTCLayer._init_module(self)

    @staticmethod
    def _init_module(module, std=0.02):

        with torch.no_grad():

            module.Bs.data.normal_(mean=0.0, std=std)

            # Plain mode: initialize Q as Q * Q^T / sqrt(d) to start PSD.
            if module.q_reparam == "plain":
                Qs = torch.empty_like(module.Qs)
                Qs.normal_(mean=0.0, std=std)
                module.Qs.data.copy_(Qs @ Qs.transpose(1, 2) / math.sqrt(module.h_dim))
            # Cholesky mode: keep factorized parameters and form Q in forward.
            elif module.q_reparam == "cholesky": 
                module.Qs.data.normal_(mean=0.0, std=std)
            else:
                raise ValueError(f"Unknown Q reparameterization {module.q_reparam}")

            if module.bias_gamma is not None:
                if module.ttc_gamma_bias_init == "sqrt_dim":
                    module.bias_gamma.data.fill_(math.log(1 / math.sqrt(module.h_dim)))
                elif module.ttc_gamma_bias_init == "zero":
                    module.bias_gamma.data.fill_(0.)
                else:
                    raise ValueError(f"Unknown gamma bias initialization {module.ttc_gamma_bias_init}")

    def forward(self, h0: torch.Tensor, T: int):

        # if horizon is zero, then directly return zeros as if the layer is disabled
        if T == 0:
            return torch.zeros(h0.shape[0], h0.shape[1], self.out_dim, device=h0.device, dtype=h0.dtype)

        bs, seq_len, in_dim = h0.shape
        h0 = h0.view(-1, in_dim) # [bs * seq_len, in_dim]

        # Split projected features into headwise blocks + low-rank mixture coefficients.
        gamma_split = [1, self.u_dim, self.h_dim]
        head_split = [self.h_dim, self.h_dim, self.u_dim, self.h_dim, sum(gamma_split)]
        heads, coeff_B, coeff_Q = torch.split(self.in_proj(h0), [sum(head_split) * self.num_heads, self.b_rank, self.q_rank], dim=-1)
        h0, A_diag, R_inv_diag, coeff_Qf, log_gammas = torch.split(heads.view(-1, self.num_heads, sum(head_split)), head_split, dim=-1)

        A_diag = l2norm(A_diag)

        # Keep Q/Qf coefficients positive to preserve PSD structure.
        coeff_B, coeff_Q, coeff_Qf = l2norm(coeff_B), l2norm(F.softplus(coeff_Q)), l2norm(F.softplus(coeff_Qf))
        B = torch.einsum("...r,rmn->...mn", coeff_B, self.Bs) # [bs * seq_len, h_dim, u_dim, h_dim]
        if self.q_reparam == "plain":
            Q = torch.einsum("...r,rmn->...mn", coeff_Q, self.Qs)  # [bs * seq_len, h_dim, h_dim]
            Q_f = torch.einsum("...r,rmn->...mn", coeff_Qf, self.Qs) # [bs * seq_len, num_heads, h_dim, h_dim]
        elif self.q_reparam == "cholesky":
            Q = torch.einsum("...r,rmn,rkn->...mk", coeff_Q, self.Qs, self.Qs) / math.sqrt(self.h_dim) # [bs * seq_len, h_dim, h_dim]
            Q_f = torch.einsum("...r,rmn,rkn->...mk", coeff_Qf, self.Qs, self.Qs) / math.sqrt(self.h_dim) # [bs * seq_len, num_heads, h_dim, h_dim]
        else:
            raise ValueError(f"Unknown Q reparameterization {self.q_reparam}")

        R_inv_diag = torch.clamp(l2norm(F.softplus(R_inv_diag)), min=1e-6)

        if self.bias_gamma is not None:
            log_gammas = -F.softplus(log_gammas) - F.relu(-self.bias_gamma[None, None, :])
        else:
            log_gammas = -F.softplus(log_gammas)

        log_gamma_A, log_gamma_B, log_gamma_Q = torch.split(log_gammas, gamma_split, dim=-1)
        log_gamma_A = log_gamma_A.squeeze(-1)

        u = self.solver(A_diag, B, Q, R_inv_diag, Q_f, log_gamma_A, log_gamma_B, log_gamma_Q, h0, T)

        if torch.isnan(u).any():
            raise RuntimeError("NaN detected in TTC layer solver output.")

        if self.rescale == "l2_norm":
            u = l2norm(u)
        elif self.rescale == "h0_norm":
            h0_norm = torch.linalg.norm(h0, dim=-1, ord=2).detach()
            u_norm = torch.linalg.norm(u, dim=-1, ord=2).detach()
            u = u / u_norm[..., None] * h0_norm[..., None]

        output = u.view(bs, seq_len, -1) # [bs, seq_len, u_dim * num_heads]
        output = self.out_proj(output)


        return output
