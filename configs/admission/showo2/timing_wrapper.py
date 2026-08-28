"""External instrumentation harness for T210 R2 smoke reruns.

Runs an unmodified Show-o2 inference script (inference_mmu.py /
inference_t2i.py) via runpy, exactly as `python3 <script> <args...>` would,
and records wall-clock/memory measurements by hooking two generic, public
library entry points that every Show-o2 inference script happens to call
once at a natural phase boundary:

  - torch.nn.Module.to(...)  -> first call after process start marks the
    "model ready" boundary (all inference scripts do
    `model = Showo2Qwen2_5.from_pretrained(...).to(device)`).
  - wandb.log(...)           -> first call marks the "output ready" boundary
    (all inference scripts log the generated/understood media via
    wandb.log as their sole recorded output).

No Show-o2 source file is read, patched, or copied by this wrapper; it only
monkeypatches two third-party library entry points before executing the
official script unchanged. This is an external measurement method, not a
modification of the audited code path.

Usage:
  python3 timing_wrapper.py <stats_out.json> <script.py> [script args...]
"""
import sys
import time
import json
import resource
import atexit

import torch

TIMES = {"process_start": time.time()}
_state = {"model_ready_done": False, "output_ready_done": False}

_orig_to = torch.nn.Module.to


def _patched_to(self, *args, **kwargs):
    result = _orig_to(self, *args, **kwargs)
    if not _state["model_ready_done"]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        TIMES["model_ready"] = time.time()
        _state["model_ready_done"] = True
    return result


torch.nn.Module.to = _patched_to

def _mark_output_ready():
    if not _state["output_ready_done"]:
        TIMES["output_ready"] = time.time()
        _state["output_ready_done"] = True


try:
    import wandb

    _orig_log = wandb.log

    def _patched_log(*args, **kwargs):
        _mark_output_ready()
        return _orig_log(*args, **kwargs)

    wandb.log = _patched_log

    # Show-o2 logs media via the Run instance (run.log(...)), which does not
    # go through the module-level wandb.log function patched above — hook
    # the Run class method too so either call style is captured.
    from wandb.sdk.wandb_run import Run as _WandbRun

    _orig_run_log = _WandbRun.log

    def _patched_run_log(self, *args, **kwargs):
        _mark_output_ready()
        return _orig_run_log(self, *args, **kwargs)

    _WandbRun.log = _patched_run_log
except ImportError:
    pass


def _report(stats_path):
    TIMES["process_end"] = time.time()
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    stats = {
        "times": TIMES,
        "peak_rss_kib": rusage.ru_maxrss,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        stats["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        stats["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        stats["device_count"] = torch.cuda.device_count()
        stats["device_name_0"] = torch.cuda.get_device_name(0)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


def main():
    import os

    stats_path = sys.argv[1]
    target_script = sys.argv[2]
    atexit.register(_report, stats_path)
    sys.argv = [target_script] + sys.argv[3:]
    # Mirror `python3 <target_script>`'s sys.path[0] behavior: runpy.run_path
    # does not do this automatically, but the target script relies on its
    # own directory being importable (e.g. `from models import ...`).
    sys.path.insert(0, os.path.dirname(os.path.abspath(target_script)))
    import runpy

    runpy.run_path(target_script, run_name="__main__")


if __name__ == "__main__":
    main()
