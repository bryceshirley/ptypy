"""
Throwaway profiler for FastFirstBornTorch._batch_update at production settings.

Item 0 of the perf review: measure whether the batch is launch/sync-bound or
FLOP-bound at the winning recipe (batch 1, N=10, M=2, 512x512, probe on).

It drives the REAL runner (run_fastborn_100um_het.py) so the setup is identical
to production, and monkeypatches _batch_update HERE ONLY (no engine edits). It
warms the cuFFT plan cache + allocator for a few batches, then:
  - times one batch (sync-bracketed perf_counter)  -> wall time
  - runs the next batch under torch.profiler          -> kernel/sync breakdown
then prints the report and exits before the epoch finishes.

Usage (env ptypy_full, PYTHONPATH=/home/dnz75396/ptypy):
    python profile_batch_update.py
"""
import os
import sys
import time
import runpy

import torch
from torch.profiler import profile, ProfilerActivity
try:
    from torch.autograd import DeviceType
except Exception:
    DeviceType = None

from ptypy.custom.fastfirstborntorch import FastFirstBornTorch

WARMUP = 8                      # batches run untouched to reach steady state
RUNNER = "/home/dnz75396/ptypy/run_fastborn_100um_het.py"
# identical to the F1cold winning recipe, but niter=1 and a profiling label
ARGV = ["run_fastborn_100um_het.py",
        "F1prof", "1", "1", "max", "0.5", "1", "10", "3", "100",
        "0", "0", "0.0", "stxm", "1", "0", "0", "625", "none", "0"]

_orig = FastFirstBornTorch._batch_update
_state = {"n": 0, "wall_ms": None}


def _fmt_us(us):
    return "%.3f ms" % (us / 1000.0)


def _report(prof, wall_ms):
    kas = prof.key_averages()

    # --- total device kernel time + launch count (robust across versions) ---
    total_cuda_us = 0.0
    n_launch = 0
    fft_us = 0.0
    sync_names = ("item", "_local_scalar_dense", "memcpy dtoh",
                  "cudastreamsynchronize", "cudadevicesynchronize",
                  "cudamemcpyasync")
    n_sync = 0

    kernels = None
    if DeviceType is not None:
        try:
            evts = prof.events()
            kernels = [e for e in evts if e.device_type == DeviceType.CUDA]
        except Exception:
            kernels = None

    if kernels:
        n_launch = len(kernels)
        for e in kernels:
            t = float(getattr(e, "self_cuda_time_total", 0.0) or 0.0)
            total_cuda_us += t
            if "fft" in e.name.lower():
                fft_us += t
        # syncs are CPU-side ops; count them from the full event list
        for e in prof.events():
            nm = e.name.lower()
            if any(s in nm for s in sync_names):
                n_sync += 1
    else:
        # fallback: aggregate from key_averages
        for ka in kas:
            t = float(getattr(ka, "self_cuda_time_total", 0.0) or 0.0)
            if t > 0:
                total_cuda_us += t
                n_launch += int(ka.count)
                if "fft" in ka.key.lower():
                    fft_us += t
            if any(s in ka.key.lower() for s in sync_names):
                n_sync += int(ka.count)

    cuda_ms = total_cuda_us / 1000.0
    ratio = cuda_ms / wall_ms if wall_ms else float("nan")

    print("\n" + "=" * 72)
    print("ITEM 0 PROFILE  -  FastFirstBornTorch._batch_update (B=1, N=10, M=2, 512x512)")
    print("=" * 72)
    print("wall time (sync-bracketed) : %.3f ms" % wall_ms)
    print("CUDA kernel time (self)    : %.3f ms" % cuda_ms)
    print("RATIO cuda/wall            : %.3f      (<0.30 => launch/sync-bound)" % ratio)
    print("kernel launches            : %d" % n_launch)
    print("host<->device syncs        : %d   (aten::item / DtoH / stream sync)" % n_sync)
    print("FFT share of kernel time   : %.1f%%  (%.3f ms)"
          % (100.0 * fft_us / total_cuda_us if total_cuda_us else 0.0,
             fft_us / 1000.0))
    print("=" * 72)

    print("\n--- TOP 15 by self CUDA time ---")
    try:
        print(kas.table(sort_by="self_cuda_time_total", row_limit=15))
    except Exception as ex:
        print("(cuda-time table unavailable: %s)" % ex)

    print("\n--- TOP 15 by call count ---")
    try:
        print(kas.table(sort_by="count", row_limit=15))
    except Exception as ex:
        print("(count table unavailable: %s)" % ex)


def _wrapped(self, idx):
    _state["n"] += 1
    n = _state["n"]

    if n < WARMUP:
        return _orig(self, idx)

    if n == WARMUP:
        # sanity: confirm we are profiling the production path
        N = int(self.p.number_of_slices)
        assert N == 10, "expected N=10, got %d" % N
        assert int(self._pint) == 1 and int(self.p.probe_update_start) == 0 \
            and not self._probe_frozen, "probe path not active on every batch"
        print("[profile] warmup done; N=%d, pint=%d, probe active" % (N, self._pint))
        return _orig(self, idx)

    if n == WARMUP + 1:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = _orig(self, idx)
        torch.cuda.synchronize()
        _state["wall_ms"] = (time.perf_counter() - t0) * 1000.0
        print("[profile] timed batch: %.3f ms wall" % _state["wall_ms"])
        return out

    if n == WARMUP + 2:
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False, with_stack=False) as prof:
            _orig(self, idx)
            torch.cuda.synchronize()
        _report(prof, _state["wall_ms"])
        sys.stdout.flush()
        os._exit(0)          # hard exit: skip the rest of the epoch + reporting

    return _orig(self, idx)


FastFirstBornTorch._batch_update = _wrapped

if __name__ == "__main__":
    sys.argv = ARGV
    runpy.run_path(RUNNER, run_name="__main__")
