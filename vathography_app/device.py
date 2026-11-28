# Vathography App. Copyright (C) 2026 Karim Joseph Nassar (Vathography).
# Licensed under AGPL-3.0-only with additional permissions under its section 7: see LICENSE.
r"""device.py — the GPU the worker runs on: NVIDIA (CUDA) or Apple silicon (Metal, PyTorch `mps`). Imported by the
worker only (it imports torch).

  pick()                 torch.device: cuda, else mps, else an error naming what the app needs
  non_blocking(dev)      host -> device copies asynchronous (CUDA only)
  synchronize / empty_cache
  attention(dev)         CUDA: the kernel priority (cuDNN first) · mps: attention in head chunks (exact per head) so
                         no single attention call materialises more than one chunk's scores
  is_mps_oom(e)          PyTorch's "MPS backend out of memory" error
  mem_point(dev, label)  mps: one memory line after synchronize + empty_cache (in use / Metal total / recommended)
  Peak(dev)              the memory line of each photo: CUDA = torch's peak counters · mps = Metal has no
                         peak counter, so the driver's allocation is POLLED at every progress event, + the process's
                         peak resident memory (getrusage)
  system_line(dev)       one log line: OS, chip / GPU, total memory, torch, device
"""
import contextlib

import torch
import torch.nn.functional as F

from . import platform_io

# attention kernels, best first. Flash is missing from the Windows torch build, so cuDNN runs (2.6x faster per call
# than PyTorch's default "efficient" kernel; a 33 MP photo 141 -> 115 s, map rms 0.0029 vs it, judged similar).
from torch.nn.attention import SDPBackend, sdpa_kernel
KERNELS_BEST_FIRST = [getattr(SDPBackend, k) for k in ("FLASH_ATTENTION", "CUDNN_ATTENTION", "EFFICIENT_ATTENTION",
                                                       "MATH")]

# mps: heads per attention call. One tile input = 12 626 tokens (the global input about as many); 24 heads at once =
# 3.8e9 score elements, over the 2^31 element limit of Metal kernels if materialised; 4 heads = 6.4e8 (~1.3 GB bf16).
MPS_HEADS_PER_CALL = 4


def pick():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if platform_io.IS_MAC:
        raise RuntimeError("PyTorch cannot use this Mac's graphics processor (Metal). The app needs a Mac with Apple "
                           "silicon on macOS 14 or newer.")
    raise RuntimeError("No NVIDIA graphics card PyTorch can use was found (or its driver is missing). The app needs one.")


def non_blocking(dev):
    return torch.device(dev).type == "cuda"


def synchronize(dev):
    t = torch.device(dev).type
    if t == "cuda":
        torch.cuda.synchronize(dev)
    elif t == "mps":
        torch.mps.synchronize()


def empty_cache(dev=None):
    t = torch.device(dev).type if dev is not None else ("cuda" if torch.cuda.is_available() else
                                                         "mps" if torch.backends.mps.is_available() else "cpu")
    if t == "cuda":
        torch.cuda.empty_cache()
    elif t == "mps":
        torch.mps.empty_cache()


def _chunked_sdpa(orig, chunk):
    """scaled_dot_product_attention over [B, H, L, D] in chunks of `chunk` heads: each head's attention is independent,
    so the result is the same per head (kernel order aside). Masks broadcast over heads are sliced with them."""
    def sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kw):
        if query.dim() != 4 or query.shape[1] <= chunk or enable_gqa:
            return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                        enable_gqa=enable_gqa, **kw)
        outs = []
        for h in range(0, query.shape[1], chunk):
            m = attn_mask
            if m is not None and m.dim() == 4 and m.shape[1] > 1:
                m = m[:, h:h + chunk]
            outs.append(orig(query[:, h:h + chunk], key[:, h:h + chunk], value[:, h:h + chunk], attn_mask=m,
                             dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw))
        return torch.cat(outs, dim=1)
    return sdpa


@contextlib.contextmanager
def attention(dev):
    t = torch.device(dev).type
    if t == "cuda":
        with sdpa_kernel(KERNELS_BEST_FIRST, set_priority=True):
            yield "cuDNN first"
        return
    if t == "mps":
        orig = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = _chunked_sdpa(orig, MPS_HEADS_PER_CALL)
        try:
            yield f"PyTorch default, {MPS_HEADS_PER_CALL} heads per call"
        finally:
            F.scaled_dot_product_attention = orig
        return
    yield "PyTorch default"


MPS_OOM_TEXT = "MPS backend out of memory"      # PyTorch's text (a RuntimeError on mps, never torch.OutOfMemoryError)


def is_mps_oom(e):
    return MPS_OOM_TEXT in str(e)


def mem_point(dev, label, say):
    """mps only: one memory line. synchronize + empty_cache first, so PyTorch's cache is out of the numbers; with
    PYTORCH_DEBUG_MPS_ALLOCATOR=1 (set by cli on a Mac) PyTorch also prints its own split ("MPS allocated", "other
    allocations") to stderr at that empty_cache. CUDA / CPU: nothing."""
    if torch.device(dev).type != "mps":
        return
    try:
        torch.mps.synchronize()
        torch.mps.empty_cache()
        g = 2 ** 30
        use, tot = torch.mps.current_allocated_memory() / g, torch.mps.driver_allocated_memory() / g
        rss = _rss_peak_gib()
        say(f"memory {label}: in use {use:.2f} | Metal total {tot:.2f} | outside the tensors {tot - use:.2f} | "
            f"recommended {torch.mps.recommended_max_memory() / g:.1f} GiB"
            + (f" | process peak resident {rss:.2f} GiB" if rss is not None else ""))
    except Exception as e:                       # a reading never stops the photo
        say(f"memory {label}: not readable ({type(e).__name__}: {e})")


def _rss_peak_gib():
    """The process's peak resident memory (macOS reports bytes, Linux KiB); None on Windows."""
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2 ** 30 if platform_io.IS_MAC else r / 2 ** 20
    except Exception:
        return None


class Peak:
    """Per-photo memory peak. reset() before the photo, poll() at every progress event, line() after."""

    def __init__(self, dev):
        self.dev = torch.device(dev)
        self.driver = 0

    def reset(self):
        if self.dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.driver = 0
        self.poll()

    def poll(self):
        if self.dev.type == "mps":
            try:
                self.driver = max(self.driver, torch.mps.driver_allocated_memory())
            except Exception:
                pass

    def line(self):
        if self.dev.type == "cuda":
            return (f"GPU peak allocated {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} / reserved "
                    f"{torch.cuda.max_memory_reserved() / 2 ** 30:.2f} GiB")
        if self.dev.type == "mps":
            self.poll()
            rss = _rss_peak_gib()
            return (f"Metal peak driver memory {self.driver / 2 ** 30:.2f} GiB (polled) | process peak resident "
                    f"{rss:.2f} GiB" if rss is not None else f"Metal peak driver memory {self.driver / 2 ** 30:.2f} GiB (polled)")
        return "CPU"


def system_line(dev):
    dev = torch.device(dev)
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        gpu = f"{p.name}, {p.total_memory / 2 ** 30:.1f} GiB, compute {p.major}.{p.minor}"
    elif dev.type == "mps":
        try:
            rec = f", Metal recommended working set {torch.mps.recommended_max_memory() / 2 ** 30:.1f} GiB"
        except Exception:
            rec = ""
        gpu = f"{platform_io.chip_name() or 'Apple silicon'} (Metal){rec}"
    else:
        gpu = "none"
    try:
        ram = f"{platform_io.ram_gb():.0f} GB"
    except Exception:
        ram = "?"
    return f"system: {platform_io.os_line()} | GPU {gpu} | memory {ram} | torch {torch.__version__} | device {dev.type}"
