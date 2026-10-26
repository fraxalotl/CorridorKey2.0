"""Compute-device abstraction: Apple Silicon (MPS), CUDA, or CPU.

The runtime was written against CUDA. Everything device-specific now goes
through this module so the same code runs on Apple Silicon Macs via PyTorch's
Metal (MPS) backend:

  - device selection (`resolve_device`)
  - mixed-precision autocast and host-to-device copies (`autocast`,
    `non_blocking`)
  - cache flush / sync (`empty_cache`, `synchronize`)
  - memory budget for automatic Hann window selection (`memory_budget_gb`)
  - where idle modules are parked between phases (`offload_device`)
  - attention and area-resize helpers that work around MPS limits
    (`sdpa`, `area_resize`)

Import this module before anything else imports torch when possible so the
MPS CPU fallback for unimplemented operators is enabled.
"""

from __future__ import annotations

import contextlib
import math
import os
import platform
import subprocess

# Let any operator MPS does not implement run on the CPU instead of raising.
# Has to be set before torch initializes the MPS backend.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn.functional as F


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def resolve_device(name: str = "auto") -> torch.device:
    """Pick the compute device. `auto` prefers MPS, then CUDA, then CPU."""
    name = (name or "auto").lower()
    if name == "auto":
        if mps_available() and mps_supports_bf16():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "mps":
        if not mps_available():
            built = getattr(torch.backends.mps, "is_built", lambda: False)()
            raise RuntimeError(
                "MPS device requested but not available. "
                + ("This PyTorch build has MPS, but no Metal device was found "
                   "(Apple Silicon and macOS 13.3+ are required)."
                   if built else
                   "This PyTorch build was compiled without MPS support."))
        if not mps_supports_bf16():
            raise RuntimeError(
                "CorridorKey on Apple Silicon needs macOS 14 (Sonoma) or newer "
                "for bfloat16 support in Metal.")
        return torch.device("mps")
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available.")
        return torch.device(name)
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device: {name!r} (use auto, mps, cuda, or cpu)")


def _device(device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def is_unified_memory(device) -> bool:
    """True when GPU and CPU share physical memory (Apple Silicon)."""
    return _device(device).type in ("mps", "cpu")


def offload_device(device) -> torch.device:
    """Where to park modules that are idle in the current phase.

    On CUDA, moving idle modules to the CPU frees VRAM. On Apple Silicon the
    GPU and CPU share the same memory, so a CPU copy frees nothing and only
    costs transfer time; idle modules stay where they are.
    """
    device = _device(device)
    return device if is_unified_memory(device) else torch.device("cpu")


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------


def mps_supports_bf16() -> bool:
    # bfloat16 on Metal needs macOS 14 (Sonoma) or newer.
    check = getattr(torch.backends.mps, "is_macos_or_newer", None)
    if check is not None:
        try:
            return bool(check(14, 0))
        except Exception:
            pass
    try:
        return int(platform.mac_ver()[0].split(".")[0]) >= 14
    except (ValueError, IndexError):
        return False


def autocast_dtype(device) -> torch.dtype:
    """Mixed-precision dtype. The network was trained and validated in bfloat16
    (asinh HDR domain), and the frame readers hand it bfloat16 tensors, so
    every backend runs under bfloat16 autocast. float16 is deliberately not
    substituted: its narrower exponent range is not what the model was
    validated with."""
    return torch.bfloat16


def autocast(device):
    """`torch.autocast` for the device's backend."""
    device = _device(device)
    return torch.autocast(device.type, dtype=autocast_dtype(device))


def non_blocking(device) -> bool:
    """Async host-to-device copies are only used on CUDA. The MPS backend has
    had correctness bugs with non_blocking copies from short-lived CPU
    tensors, and on unified memory there is no PCIe transfer to overlap."""
    return _device(device).type == "cuda"


# ---------------------------------------------------------------------------
# Memory / synchronization
# ---------------------------------------------------------------------------


def empty_cache(device) -> None:
    device = _device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def synchronize(device) -> None:
    device = _device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _macos_available_ram_bytes() -> int | None:
    """Free + reclaimable RAM on macOS from `vm_stat` (no extra dependency)."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             timeout=5, check=True).stdout
    except Exception:
        return None
    page = 16384
    counts = {}
    for line in out.splitlines():
        if "page size of" in line:
            try:
                page = int(line.split("page size of")[1].split()[0])
            except (IndexError, ValueError):
                pass
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            try:
                counts[key.strip()] = int(val.strip().rstrip("."))
            except ValueError:
                pass
    pages = sum(counts.get(k, 0) for k in (
        "Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))
    return pages * page if pages else None


def memory_budget_gb(device) -> float:
    """Memory this process may use on `device`, including what it already holds."""
    device = _device(device)
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        # Memory this process already holds (the loaded model) is part of the budget.
        return (free_bytes + torch.cuda.memory_reserved(device)) / 2**30
    if device.type == "mps":
        # Metal's recommended working-set size is the GPU's practical ceiling
        # (by default roughly 2/3-3/4 of system RAM). Other apps share the same
        # memory, so also cap by what is currently free plus what we hold.
        limit = torch.mps.recommended_max_memory()
        held = torch.mps.driver_allocated_memory()
        avail = _macos_available_ram_bytes()
        if avail is not None:
            limit = min(limit, avail + held)
        return limit / 2**30
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        return pages / 2**30
    except (ValueError, OSError, AttributeError):
        return 8.0


# ---------------------------------------------------------------------------
# MPS-safe kernels
# ---------------------------------------------------------------------------


def _mps_attn_budget_bytes() -> int:
    mb = int(os.environ.get("CORRIDORKEY_MPS_ATTN_BUDGET_MB", "1024"))
    return max(64, mb) * 2**20


def _slice_mask(mask, q_ndim: int, batch: slice | None, rows: slice | None):
    if mask is None:
        return None
    if batch is not None and mask.dim() == q_ndim and mask.shape[0] > 1:
        mask = mask[batch]
    if rows is not None and mask.dim() >= 2 and mask.shape[-2] > 1:
        mask = mask[..., rows, :]
    return mask


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
         attn_mask: torch.Tensor | None = None, dropout_p: float = 0.0,
         is_causal: bool = False) -> torch.Tensor:
    """`F.scaled_dot_product_attention` that bounds the score matrix on MPS.

    CUDA has fused flash / memory-efficient kernels that never materialize the
    (Lq x Lk) score matrix. The MPS kernel does, so full-resolution attention
    at /16 (tens of thousands of tokens per frame) would need tens of GB.
    Attention rows are independent, so splitting the batch and query dims and
    concatenating is exact. On other devices this is a direct passthrough.
    """
    if q.device.type != "mps" or is_causal or dropout_p:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal)
    return chunked_sdpa(q, k, v, attn_mask, _mps_attn_budget_bytes())


def chunked_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 attn_mask: torch.Tensor | None, budget: int) -> torch.Tensor:
    """Exact SDPA computed in pieces whose float32 score matrix fits `budget` bytes."""
    Lq, Lk = q.shape[-2], k.shape[-2]
    lead = q.shape[:-2]
    batch = math.prod(lead) if lead else 1
    if batch * Lq * Lk * 4 <= budget:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    # Split along dim 0 first so each piece holds whole (heads x Lq) blocks,
    # then along queries if one batch entry alone is still over budget.
    if q.dim() > 2 and q.shape[0] > 1 and k.shape[0] == q.shape[0]:
        per_item = (batch // q.shape[0]) * Lq * Lk * 4
        step = max(1, budget // max(1, per_item))
        if step < q.shape[0]:
            outs = []
            for b0 in range(0, q.shape[0], step):
                bs = slice(b0, min(b0 + step, q.shape[0]))
                outs.append(chunked_sdpa(
                    q[bs], k[bs], v[bs],
                    _slice_mask(attn_mask, q.dim(), bs, None), budget))
            return torch.cat(outs, dim=0)

    rows = max(1, budget // max(1, batch * Lk * 4))
    if rows >= Lq:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    out = q.new_empty(*lead, Lq, v.shape[-1])
    for r0 in range(0, Lq, rows):
        rs = slice(r0, min(r0 + rows, Lq))
        out[..., rs, :] = F.scaled_dot_product_attention(
            q[..., rs, :], k, v,
            attn_mask=_slice_mask(attn_mask, q.dim(), None, rs))
    return out


def _area_matrix(n_in: int, n_out: int, device, dtype) -> torch.Tensor:
    """(n_out, n_in) averaging matrix with adaptive_avg_pool's bin edges."""
    i = torch.arange(n_out)
    start = (i * n_in) // n_out
    end = ((i + 1) * n_in + n_out - 1) // n_out
    cols = torch.arange(n_in)
    m = ((cols[None, :] >= start[:, None]) & (cols[None, :] < end[:, None])).double()
    m = m / (end - start).double()[:, None]
    return m.to(device=device, dtype=dtype)


def area_resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """`F.interpolate(x, size, mode="area")` that also works on MPS.

    Area interpolation is adaptive average pooling, which MPS only supports
    when the input size is an integer multiple of the output size. MoGe's
    7/32 input scale is not, so on MPS non-divisible resizes use the same
    pooling bins expressed as two small matrix multiplies (separable rows /
    columns), which matches the CPU/CUDA result to float rounding.
    """
    h_out, w_out = int(size[0]), int(size[1])
    H, W = x.shape[-2:]
    if x.device.type != "mps" or (H % h_out == 0 and W % w_out == 0):
        return F.interpolate(x, size=(h_out, w_out), mode="area")
    dtype = x.dtype if x.dtype in (torch.float32, torch.float16, torch.bfloat16) else torch.float32
    xf = x.to(torch.float32)
    mh = _area_matrix(H, h_out, x.device, torch.float32)
    mw = _area_matrix(W, w_out, x.device, torch.float32)
    y = torch.matmul(mh, xf)               # (..., h_out, W)
    y = torch.matmul(y, mw.transpose(0, 1))  # (..., h_out, w_out)
    return y.to(dtype)


def describe(device) -> str:
    device = _device(device)
    if device.type == "mps":
        chip = "Apple Silicon"
        try:
            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5).stdout.strip() or chip
        except Exception:
            pass
        return (f"mps ({chip}, {torch.mps.recommended_max_memory() / 2**30:.1f} GB "
                f"Metal working set)")
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return "cpu (expect this to be very slow)"
