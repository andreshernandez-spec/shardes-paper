#!/usr/bin/env python
"""Turn probe records into the cost of the matched ES runs, per GPU type.

    python cost.py results-a100 results-h100

A full-rank ES member is one batch of P prompts decoded at that member's weights, so
the probe's wall time per member batch is the unit of cost: a run of R rollouts at P
prompts per member is R / P member batches. That already contains prefill, decoding
and the tail of the slowest sequence, which is the point of measuring rather than
dividing tokens by a throughput figure. Added on top: one weight rewrite per member
(measured) and the uptime overhead RunPod bills beyond measured time (~20%, the
runbook's figure). The ES update itself is not measured here and is left out.

Lengths come from the start checkpoint. RL moved them during training (Olmo 3 IF's
per-step mean fell to ~93 late; Tulu 3.1 stayed at ~310-400), and ES may too, so the
table is the cost at start-of-run lengths, and says so.

Prices are the list prices read on 2026-09-25, per GPU-hour.
"""

import argparse
import json
from pathlib import Path

PROBE = Path(__file__).resolve().parent

PRICES = {  # (community, secure) USD per GPU-hour, RunPod list, 2026-09-25
    "A100": (1.39, 1.59),
    "H100": (2.69, 3.49),
}
UPTIME_OVERHEAD = 1.2
RUNS = {  # rollouts in the matched ES run
    "olmo3_if": {"step 2000 (released)": 512_000, "step 500": 128_000},
    "tulu31": {"step 1920 (released)": 1_474_560, "step 640": 491_520},
}


def gpu_class(record: dict) -> str:
    name = record["env"]["torch_device"]
    for k in PRICES:
        if k in name:
            return k
    raise ValueError(f"no price for {name}")


def seconds_per_member(record: dict) -> float:
    walls = [m["wall_seconds"] + m["weight_rewrite"]["seconds"] for m in record["members"]]
    return sum(walls) / len(walls)


def table(dirs) -> list:
    rows = []
    for d in dirs:
        for path in sorted((PROBE / d).glob("*.json")):
            rec = json.loads(path.read_text())
            if rec.get("smoke"):
                continue
            gpu, P = gpu_class(rec), rec["cell"]["prompts_per_member"]
            per_member = seconds_per_member(rec)
            for label, rollouts in RUNS[rec["setting"]].items():
                gpu_h = rollouts / P * per_member / 3600 * UPTIME_OVERHEAD
                lo, hi = PRICES[gpu]
                rows.append({
                    "gpu": gpu, "setting": rec["setting"], "run": label,
                    "util": rec["cell"]["gpu_memory_utilization"], "P": P,
                    "s_per_member": per_member, "gpu_hours": gpu_h,
                    "usd": (gpu_h * lo, gpu_h * hi),
                    "mean_len": rec["lengths"]["mean"],
                    "cap_hits": rec["lengths"]["cap_hits"],
                    "sequences": rec["lengths"]["sequences"],
                })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args(argv)
    print("| gpu | setting | matched to | util | P | s/member | mean len | cap hits | "
          "GPU-h | USD |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in table(args.dirs):
        print(f"| {r['gpu']} | {r['setting']} | {r['run']} | {r['util']:.2f} | {r['P']} | "
              f"{r['s_per_member']:.1f} | {r['mean_len']:.0f} | "
              f"{r['cap_hits']}/{r['sequences']} | {r['gpu_hours']:.1f} | "
              f"{r['usd'][0]:.0f}-{r['usd'][1]:.0f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
