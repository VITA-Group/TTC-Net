# TTC-Net: Beyond Test-Time Training

This repository contains the core implementation for:

**Beyond Test-Time Training: Learning to Reason via Hardware-Efficient Optimal Control**  
Paper: https://arxiv.org/abs/2603.09221  
Project page: https://vita-group.github.io/TTC-Net/

---

## Overview

TTC (Test-Time Control) is an LLM layer that takes initial memory states as inputs and outputs the solution to a receding-horizon optimal control problem with hardware-efficient LQR solvers.

The codebase provides:
- `TTCLayer` for plugging TTC into neural architectures.
- Multiple LQR solver backends:
  - `riccati` (direct PyTorch)
  - `kkt` (dual/KKT PyTorch)
  - `fused` (Triton fused kernel)

---

## Environment

The code in `ttc/` depends on:

```
torch==2.8.0+cu128
triton==3.5.1
fla==0.3.1
```

## Usage

You can use TTC directly from the package:

```python
import torch
from ttc import TTCLayer

x = torch.randn(2, 128, 8, 32, device="cuda", dtype=torch.float32)  # [batch, seq, num_heads, in_dim]

ttc = TTCLayer(
    in_dim=32,
    out_dim=32,
    h_dim=16,
    num_heads=8,
    b_rank=16,
    q_rank=16,
    solver_impl="fused",
).cuda()

y = ttc(x, T=64)
```


## Citation

If you find this repository useful, please cite:

```bibtex
@article{wang2026beyond,
  title   = {Beyond Test-Time Training: Learning to Reason via Hardware-Efficient Optimal Control},
  author  = {Wang, Peihao and Yang, Shan and Wang, Xijun and Xiao, Tesi and Liu, Xin and Yu, Changlong and Lou, Yu and Li, Pan and Wang, Atlas and Lin, Ming and Vidal, Rene},
  journal = {arXiv preprint arXiv:2603.09221},
  year    = {2026}
}
```
