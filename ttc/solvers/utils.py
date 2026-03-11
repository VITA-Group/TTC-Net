import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

import packaging.version
import importlib.metadata

def require_version(package: str, min_version: str):
    try:
        pkg_version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return False
    return packaging.version.Version(pkg_version) >= packaging.version.Version(min_version)

def config_pruner(configs, named_args,tile_dim_pairs, **kwargs):
    pruned_configs = []
    for c in configs:
        valid = True
        for ts_name, dim_name in tile_dim_pairs:
            ts = c.kwargs.get(ts_name, None)
            dim = named_args.get(dim_name, kwargs.get(dim_name, None))
            if ts is None or dim is None:
                continue
            if ts > dim:
                valid = False
                break
        if valid:
            pruned_configs.append(c)
    return pruned_configs

@triton.jit
def ldexp_scale(X, scaler, beta=1.):
    k_f = tl.where(scaler > 0, tl.ceil(tl.log2(scaler / beta)), 0.0)     # float exponent
    k_i = k_f.cast(tl.int32)
    # return tl.extra.libdevice.ldexp(X, -k_i)
    return libdevice.ldexp(X, -k_i)

@triton.jit
def _normalize_axis_(n_dims: tl.constexpr, idx: tl.constexpr) -> tl.constexpr:
    # Python-like: negative indices wrap from the end
    # e.g., for n_dims=3: -1 -> 2, -2 -> 1, -3 -> 0
    tl.static_assert((0 <= idx and idx < n_dims) or (idx < 0 and idx >= -n_dims))
    ax = idx if (idx >= 0) else (n_dims + idx)
    return ax  # constexpr

@triton.jit(do_not_specialize=['idx'])
def _indicator_(n_dims: tl.constexpr,
                axis: tl.constexpr,            # may be negative or non-negative
                idx,
                dim: tl.constexpr):
    # Normalize idx once
    # ax = _normalize_axis_(n_dims, idx)
    # ax = idx
    tl.static_assert(axis < n_dims)
    tl.device_assert(idx < dim)

    y = tl.arange(0, dim)
    y = tl.where(y == idx, 1, 0)
    for n in tl.static_range(0, n_dims):
        if n != axis:
            y = tl.expand_dims(y, n)
    return y


@triton.jit(do_not_specialize=['idx'])
def _put_slice_(x,
                n_dims: tl.constexpr,
                axis: tl.constexpr,            # may be negative/positive
                idx,
                dim: tl.constexpr,
                input_slice):
    # Broadcast rules apply: input_slice should be broadcast-compatible with x,
    # typically same rank as x and size-1 on the sliced axis (or full shape).
    ind = _indicator_(n_dims, axis, idx, dim)
    return tl.where(ind == 1, input_slice, x)


@triton.jit(do_not_specialize=['idx'])
def _take_slice_(x,
                 n_dims: tl.constexpr,
                 axis: tl.constexpr,           # may be negative/positive
                 idx,
                 dim: tl.constexpr,
                 keep_dim: tl.constexpr = True):
    # ax = _normalize_axis_(n_dims, idx)
    # ax = idx
    ind = _indicator_(n_dims, axis, idx, dim)
    y = tl.sum(x * ind, axis)                  # reduce along the sliced axis
    if keep_dim:
        y = tl.expand_dims(y, axis)            # keep a singleton axis there
    return y


# ===== test kernels (branch on constexpr idx and pass M/N directly) =====
@triton.jit
def take2d_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr,
                  idx: tl.constexpr, pos: tl.constexpr):
    # build full (M,N) local tensor (row-major contiguous)
    r = tl.arange(0, M)[:, None]
    c = tl.arange(0, N)[None, :]
    x_ptrs = x_ptr + r * N + c
    x = tl.load(x_ptrs)

    if idx == 0:
        # row slice → shape (1, N)
        y = _take_slice_(x, 2, 0, pos, M, True)
        # out is (1, N) contiguous → linear index is just columns
        out_ptrs = out_ptr + c
        tl.store(out_ptrs, y)
    else:
        # col slice → shape (M, 1)
        y = _take_slice_(x, 2, 1, pos, N, True)
        # out is (M, 1) contiguous → linear index is just rows
        out_ptrs = out_ptr + r
        tl.store(out_ptrs, y)

@triton.jit
def put2d_kernel(x_ptr, in_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr,
                 idx: tl.constexpr, pos: tl.constexpr):
    r = tl.arange(0, M)[:, None]
    c = tl.arange(0, N)[None, :]
    x_ptrs = x_ptr + r * N + c
    x = tl.load(x_ptrs)

    if idx == 0:
        # input slice (1, N)
        in_ptrs = in_ptr + c
        in_slice = tl.load(in_ptrs)
        y = _put_slice_(x, 2, 0, pos, M, in_slice)
    else:
        # input slice (M, 1)
        in_ptrs = in_ptr + r
        in_slice = tl.load(in_ptrs)
        y = _put_slice_(x, 2, 1, pos, N, in_slice)

    # store full (M,N)
    out_ptrs = out_ptr + r * N + c
    tl.store(out_ptrs, y)

# ===== PyTorch harness =====
def run_once(M, N, dtype=torch.float32, idx=0, pos=None, device="cuda"):
    assert idx in (0, 1)
    if pos is None:
        pos = 0
    assert 0 <= pos < (M if idx == 0 else N)

    x = torch.randn((M, N), device=device, dtype=dtype).contiguous()

    # ----- test take -----
    out_take = torch.empty((1, N), device=device, dtype=dtype) if idx == 0 \
               else torch.empty((M, 1), device=device, dtype=dtype)
    take2d_kernel[(1,)](x, out_take, M, N, idx, pos, num_warps=1, num_stages=1)

    with torch.no_grad():
        ref_take = x[pos:pos+1, :] if idx == 0 else x[:, pos:pos+1]
        ok_take = torch.allclose(out_take, ref_take, atol=1e-5 if dtype==torch.float32 else 1e-3, rtol=0)

    # ----- test put -----
    in_slice = torch.randn_like(out_take).contiguous()
    out_put = torch.empty_like(x)
    put2d_kernel[(1,)](x, in_slice, out_put, M, N, idx, pos, num_warps=1, num_stages=1)

    with torch.no_grad():
        ref_put = x.clone()
        if idx == 0:
            ref_put[pos, :] = in_slice[0, :]
        else:
            ref_put[:, pos] = in_slice[:, 0]
        ok_put = torch.allclose(out_put, ref_put, atol=1e-5 if dtype==torch.float32 else 1e-3, rtol=0)

    return ok_take, ok_put

def run_suite():
    torch.cuda.synchronize()
    cases = [
        (4, 8, torch.float32),
        (8, 16, torch.float32),
        (16, 4, torch.float16),
    ]
    for (M, N, dtype) in cases:
        for idx in (0, 1):
            for pos in (0, (M-1 if idx==0 else N-1), max(0, (M if idx==0 else N)//2 - 1)):
                ok_take, ok_put = run_once(M, N, dtype=dtype, idx=idx, pos=pos)
                print(f"[M={M}, N={N}, dtype={dtype}, idx={idx}, pos={pos}] "
                      f"take={'OK' if ok_take else 'FAIL'} | put={'OK' if ok_put else 'FAIL'}")
    torch.cuda.synchronize()

if __name__ == "__main__":
    run_suite()
