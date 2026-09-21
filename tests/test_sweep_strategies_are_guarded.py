"""Whatever the sweeps benchmark, the library's real-hardware invariance suite must check.

`lowrank_r1` was once in `sweep.yaml` and in neither `tests/gpu` nor `rehearsal.yaml`, so
it was benchmarked on rented hardware by a suite that never checked it was device-count
invariant. It is also the only strategy the sweep has failed on.

The guarded list lives in the library's `tests/gpu`, which is not part of the installed
package, so this needs a checkout of the library (`library_repo` in conftest.py) and skips
without one. Subset rather than equality: guarding a strategy the sweep does not run is
fine, the reverse is not.
"""

import importlib.util
import pathlib
import sys

import yaml

PHASE2 = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "phase2"


def test_every_benchmarked_strategy_is_also_guarded(library_repo):
    gpu_test = library_repo / "tests" / "gpu" / "test_device_invariance_gpu.py"
    spec = importlib.util.spec_from_file_location("gpu_invariance", gpu_test)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gpu_invariance"] = mod
    spec.loader.exec_module(mod)

    guarded = set(mod.NAMES)
    for name in ("sweep.yaml", "rehearsal.yaml"):
        wanted = set(yaml.safe_load((PHASE2 / name).read_text())["strategies"])
        assert wanted <= guarded, (
            f"{name} benchmarks {sorted(wanted - guarded)}, which the library's tests/gpu "
            "never checks for device-count invariance")
