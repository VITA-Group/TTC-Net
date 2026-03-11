import argparse
import gc
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

# Keep autotune disabled by default for deterministic benchmark behavior.
os.environ.setdefault("TRITON_NO_AUTOTUNE", "1")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TTC_NET_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _TTC_NET_ROOT not in sys.path:
    sys.path.append(_TTC_NET_ROOT)

import torch
import torch.nn.functional as F
import triton

from ttc.solvers.fused_lqr import fused_lqr_solve
from ttc.solvers.torch_lqr import direct_lqr_solve, kkt_lqr_solve

SOLVER_MAP: Dict[str, Callable] = {
    "torch": direct_lqr_solve,
    "torch_kkt": kkt_lqr_solve,
    "triton": fused_lqr_solve,
}

@dataclass(frozen=True)
class Config:
    batch_size: int = 1024
    num_heads: int = 1
    dim: int = 64
    horizon: int = 64
    dtype: torch.dtype = torch.float32
    seed: int = 0


def normalize(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x / torch.linalg.norm(x, dim=-1, ord=2, keepdim=True)
    if x.ndim == 4:
        return x / torch.linalg.norm(x, dim=(-2, -1), ord="fro", keepdim=True)
    raise ValueError(f"Expected 3D or 4D tensor, got ndim={x.ndim}")


def measure_running_memory_bytes(
    fn: Callable[[], None],
    rep: int = 10,
    quantiles: Tuple[float, float, float] = (0.5, 0.2, 0.8),
) -> Tuple[float, float, float]:
    peaks = []
    for _ in range(rep):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()

        fn()
        torch.cuda.synchronize()

        peak = torch.cuda.max_memory_allocated()
        peaks.append(max(0, peak - before))

    peaks_tensor = torch.tensor(peaks, dtype=torch.float64)
    q_tensor = torch.tensor(quantiles, dtype=torch.float64)
    qvals = torch.quantile(peaks_tensor, q_tensor)
    return qvals[0].item(), qvals[1].item(), qvals[2].item()


def generate_inputs(
    batch_size: int,
    num_heads: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    horizon: int,
    require_grad: bool,
):
    scale = 1.0

    A = normalize(torch.rand(batch_size, num_heads, dim, device=device)) * scale
    A = A.to(dtype).requires_grad_(require_grad)

    B = (torch.randn(batch_size, dim, dim, device=device) * scale).to(dtype).requires_grad_(require_grad)

    Q = normalize(torch.randn(batch_size, dim, dim, device=device))
    Q = (Q @ Q.transpose(-1, -2) / math.sqrt(dim)).to(dtype).requires_grad_(require_grad)

    Q_f = normalize(torch.randn(batch_size, num_heads, dim, dim, device=device))
    Q_f = (Q_f @ Q_f.transpose(-1, -2) / math.sqrt(dim)).to(dtype).requires_grad_(require_grad)

    R_inv = F.softplus(torch.rand(batch_size, num_heads, dim, device=device)).to(dtype).requires_grad_(require_grad)

    h0 = torch.randn(batch_size, num_heads, dim, device=device).to(dtype).requires_grad_(require_grad)

    log_gamma_A = (-F.softplus(torch.randn(batch_size, num_heads, device=device).to(dtype)) - 1.328).requires_grad_(require_grad)
    log_gamma_B = (-F.softplus(torch.randn(batch_size, num_heads, dim, device=device).to(dtype)) - 1.328).requires_grad_(require_grad)
    log_gamma_Q = (-F.softplus(torch.randn(batch_size, num_heads, dim, device=device).to(dtype)) - 1.328).requires_grad_(require_grad)

    return A, B, Q, R_inv, Q_f, log_gamma_A, log_gamma_B, log_gamma_Q, h0


def run_solver_with_backward(
    solver: Callable,
    A: torch.Tensor,
    B: torch.Tensor,
    Q: torch.Tensor,
    R_inv: torch.Tensor,
    Q_f: torch.Tensor,
    log_gamma_A: torch.Tensor,
    log_gamma_B: torch.Tensor,
    log_gamma_Q: torch.Tensor,
    h0: torch.Tensor,
    grad: torch.Tensor,
    horizon: int,
) -> Dict[str, torch.Tensor]:
    for tensor in [A, B, Q, R_inv, Q_f, log_gamma_A, log_gamma_B, log_gamma_Q, h0]:
        if tensor.grad is not None:
            tensor.grad.zero_()

    u = solver(
        A,
        B,
        (Q + Q.transpose(-1, -2)) / 2,
        R_inv,
        (Q_f + Q_f.transpose(-1, -2)) / 2,
        log_gamma_A,
        log_gamma_B,
        log_gamma_Q,
        h0,
        horizon,
    )
    loss = torch.sum(grad * u)
    loss.backward()

    return {
        "u": u.detach().clone(),
        "loss": loss.detach().clone(),
        "A_grad": A.grad.detach().clone(),
        "B_grad": B.grad.detach().clone(),
        "Q_grad": Q.grad.detach().clone(),
        "R_inv_grad": R_inv.grad.detach().clone(),
        "Q_f_grad": Q_f.grad.detach().clone(),
        "h0_grad": h0.grad.detach().clone(),
        "log_gamma_A_grad": log_gamma_A.grad.detach().clone(),
        "log_gamma_B_grad": log_gamma_B.grad.detach().clone(),
        "log_gamma_Q_grad": log_gamma_Q.grad.detach().clone(),
    }


def safe_bench(fn: Callable[[], Tuple[float, float, float]], provider: str) -> Tuple[float, float, float]:
    try:
        return fn()
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if "triton" in provider:
            raise e
        print(e)
        return 0.0, 0.0, 0.0


def throughput_gbps(batch_size: int, dim: int, horizon: int, ms: float) -> float:
    if ms <= 0 or not math.isfinite(ms):
        return 0.0

    flops_like = 4 * batch_size * (dim**3) * horizon
    return flops_like * 1e-9 / (ms * 1e-3)


def run_correctness_check(cfg: Config, device: torch.device, reference: str = "torch_kkt") -> None:
    print("[Correctness] comparing direct vs fused and backward gradients")
    torch.manual_seed(cfg.seed)

    inputs = generate_inputs(
        batch_size=4,
        num_heads=4,
        dim=16,
        device=device,
        dtype=cfg.dtype,
        horizon=64,
        require_grad=True,
    )
    A, B, Q, R_inv, Q_f, log_gamma_A, log_gamma_B, log_gamma_Q, h0 = inputs

    grad = torch.randn(4, 4, 16, device=device)
    result_map = {
        "torch": run_solver_with_backward(
            direct_lqr_solve, A, B, Q, R_inv, Q_f, log_gamma_A, log_gamma_B, log_gamma_Q, h0, grad, 64
        ),
        "torch_kkt": run_solver_with_backward(
            kkt_lqr_solve, A, B, Q, R_inv, Q_f, log_gamma_A, log_gamma_B, log_gamma_Q, h0, grad, 64
        ),
        "fused": run_solver_with_backward(
            fused_lqr_solve, A, B, Q, R_inv, Q_f, log_gamma_A, log_gamma_B, log_gamma_Q, h0, grad, 64
        ),
    }

    for name, out in result_map.items():
        if name == reference:
            continue
        print(f"\ncompare {reference} and {name}")
        for key, value in out.items():
            diff = torch.max(torch.abs(value - result_map[reference][key])).item()
            print(f"  {key}: {diff:.6e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TTC LQR solvers (direct, dual/KKT, fused Triton).")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--dtype", type=str, choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot-dir", type=str, default="./plots")
    parser.add_argument("--bench-tput", action="store_true")
    parser.add_argument("--bench-mem", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type != "cuda":
        raise RuntimeError("Benchmark script requires CUDA.")

    cfg = Config(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        dim=args.dim,
        horizon=args.horizon,
        dtype=dtype,
        seed=args.seed,
    )
    print(f"Config: {cfg}")

    run_correctness_check(cfg, device)

    common_line_kwargs = {
        "line_arg": "provider",
        "line_vals": ["torch", "torch_kkt", "triton"],
        "line_names": ["Torch", "Torch KKT", "Triton"],
        "styles": [("blue", "-"), ("green", "-"), ("red", "-")],
        "args": {},
    }
    os.makedirs(args.plot_dir, exist_ok=True)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["horizon"],
            x_vals=[2**i for i in range(4, 12)],
            x_log=True,
            **common_line_kwargs,
            ylabel="Throughput (GB/s)",
            y_log=True,
            plot_name="Throughput of Solving LQRs",
        )
    )
    def benchmark_throughput_horizon(horizon, provider):
        A, B, Q, R_inv, Q_f, log_gamma_A, log_gamma_B, log_gamma_Q, h0 = generate_inputs(
            cfg.batch_size,
            cfg.num_heads,
            cfg.dim,
            device,
            cfg.dtype,
            horizon,
            require_grad=True,
        )
        g = torch.randn(cfg.batch_size, cfg.num_heads, cfg.dim, device=device)
        solver = SOLVER_MAP[provider]

        def fn():
            return triton.testing.do_bench(
                lambda: run_solver_with_backward(solver, A, B, Q, R_inv, Q_f, log_gamma_A, log_gamma_B, log_gamma_Q, h0, g, horizon),
                quantiles=[0.5, 0.2, 0.8],
            )

        ms, min_ms, max_ms = safe_bench(fn, provider)
        return (
            throughput_gbps(A.shape[0], A.shape[-1], horizon, ms),
            throughput_gbps(A.shape[0], A.shape[-1], horizon, max_ms),
            throughput_gbps(A.shape[0], A.shape[-1], horizon, min_ms),
        )

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size"],
            x_vals=[2**i for i in range(4, 14)],
            x_log=True,
            **common_line_kwargs,
            ylabel="Throughput (GB/s)",
            y_log=True,
            plot_name="Throughput of Solving LQRs",
        )
    )
    def benchmark_throughput_batch(batch_size, provider):
        A, B, Q, R_inv, Q_f, log_gamma_A, log_gamma_B, log_gamma_Q, h0 = generate_inputs(
            batch_size,
            cfg.num_heads,
            cfg.dim,
            device,
            cfg.dtype,
            cfg.horizon,
            require_grad=True,
        )
        g = torch.randn(batch_size, cfg.num_heads, cfg.dim, device=device)
        solver = SOLVER_MAP[provider]

        def fn():
            return triton.testing.do_bench(
                lambda: run_solver_with_backward(
                    solver,
                    A,
                    B,
                    Q,
                    R_inv,
                    Q_f,
                    log_gamma_A,
                    log_gamma_B,
                    log_gamma_Q,
                    h0,
                    g,
                    cfg.horizon,
                ),
                quantiles=[0.5, 0.2, 0.8],
            )

        ms, min_ms, max_ms = safe_bench(fn, provider)
        return (
            throughput_gbps(A.shape[0], A.shape[-1], cfg.horizon, ms),
            throughput_gbps(A.shape[0], A.shape[-1], cfg.horizon, max_ms),
            throughput_gbps(A.shape[0], A.shape[-1], cfg.horizon, min_ms),
        )

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["horizon"],
            x_vals=[2**i for i in range(4, 12)],
            x_log=False,
            **common_line_kwargs,
            ylabel="Memory Footprint (GB)",
            y_log=False,
            plot_name="Memory Footprint of Solving LQRs",
        )
    )
    def benchmark_memory_horizon(horizon, provider):
        A, B, Q, R_inv, Q_f, log_gamma_A, log_gamma_B, log_gamma_Q, h0 = generate_inputs(
            cfg.batch_size,
            cfg.num_heads,
            cfg.dim,
            device,
            cfg.dtype,
            horizon,
            require_grad=True,
        )
        g = torch.randn(cfg.batch_size, cfg.num_heads, cfg.dim, device=device)
        solver = SOLVER_MAP[provider]

        def fn():
            return measure_running_memory_bytes(
                lambda: run_solver_with_backward(solver, A, B, Q, R_inv, Q_f, log_gamma_A, log_gamma_B, log_gamma_Q, h0, g, horizon),
                quantiles=(0.5, 0.2, 0.8),
            )

        avg_peak, max_peak, min_peak = safe_bench(fn, provider)
        to_gb = lambda x: x * 1e-9
        return to_gb(avg_peak), to_gb(max_peak), to_gb(min_peak)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size"],
            x_vals=[2**i for i in range(4, 14)],
            x_log=False,
            **common_line_kwargs,
            ylabel="Memory Footprint (GB)",
            y_log=False,
            plot_name="Memory Footprint of Solving LQRs",
        )
    )
    def benchmark_memory_batch(batch_size, provider):
        A, B, Q, R_inv, Q_f, log_gamma_A, log_gamma_B, log_gamma_Q, h0 = generate_inputs(
            batch_size,
            cfg.num_heads,
            cfg.dim,
            device,
            cfg.dtype,
            cfg.horizon,
            require_grad=True,
        )
        g = torch.randn(batch_size, cfg.num_heads, cfg.dim, device=device)
        solver = SOLVER_MAP[provider]

        def fn():
            return measure_running_memory_bytes(
                lambda: run_solver_with_backward(
                    solver,
                    A,
                    B,
                    Q,
                    R_inv,
                    Q_f,
                    log_gamma_A,
                    log_gamma_B,
                    log_gamma_Q,
                    h0,
                    g,
                    cfg.horizon,
                ),
                quantiles=(0.5, 0.2, 0.8),
            )

        avg_peak, max_peak, min_peak = safe_bench(fn, provider)
        to_gb = lambda x: x * 1e-9
        return to_gb(avg_peak), to_gb(max_peak), to_gb(min_peak)

    if args.bench_tput:
        os.makedirs(os.path.join(args.plot_dir, "benchmark_lqr_flops_vs_horizon.png"), exist_ok=True)
        benchmark_throughput_horizon.run(
            print_data=True,
            save_path=os.path.join(args.plot_dir, "benchmark_lqr_flops_vs_horizon.png"),
        )

        os.makedirs(os.path.join(args.plot_dir, "benchmark_lqr_flops_vs_bs.png"), exist_ok=True)
        benchmark_throughput_batch.run(
            print_data=True,
            save_path=os.path.join(args.plot_dir, "benchmark_lqr_flops_vs_bs.png"),
        )
    if args.bench_mem:
        os.makedirs(os.path.join(args.plot_dir, "benchmark_lqr_mem_vs_horizon.png"), exist_ok=True)
        benchmark_memory_horizon.run(
            print_data=True,
            save_path=os.path.join(args.plot_dir, "benchmark_lqr_mem_vs_horizon.png"),
        )

        os.makedirs(os.path.join(args.plot_dir, "benchmark_lqr_mem_vs_bs.png"), exist_ok=True)
        benchmark_memory_batch.run(
            print_data=True,
            save_path=os.path.join(args.plot_dir, "benchmark_lqr_mem_vs_bs.png"),
        )


if __name__ == "__main__":
    main()
