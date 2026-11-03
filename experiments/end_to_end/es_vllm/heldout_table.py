#!/usr/bin/env python
"""The held-out table: one row per model, from the JSONs `heldout.py` wrote.

    python -m es_vllm.heldout_table                        # from experiments/end_to_end
    python -m es_vllm.heldout_table --dir runs/heldout-121-160

Differences to the DPO start use the unpaired standard error, sqrt(se_a^2 + se_b^2): the
records keep means, not per-prompt rewards, so the pairing cannot be used. That
overstates the error of a difference, never understates it.
"""

import argparse
import json
from pathlib import Path

E2E = Path(__file__).resolve().parent.parent
ORDER = ["dpo-start", "rl-step40", "rl-step80", "rl-step120",
         "es-s5e-4", "es-s1e-3", "es-s2e-3", "es-random"]
BASE = "dpo-start"


def load(d: Path) -> dict:
    recs = {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))}
    rank = {m: i for i, m in enumerate(ORDER)}
    return {m: recs[m] for m in sorted(recs, key=lambda m: (rank.get(m, len(ORDER)), m))}


def table(recs: dict) -> str:
    sources = sorted({s for r in recs.values() for s in r["by_source"]})
    base = recs.get(BASE)
    head = (["model", "reward", "vs DPO"] + sources + ["mean len", "capped", "prompts"])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m, r in recs.items():
        if base is None or m == BASE:
            diff = ""
        else:
            d = r["reward"] - base["reward"]
            se = (r["reward_se"] ** 2 + base["reward_se"] ** 2) ** 0.5
            diff = f"{d:+.3f} ({d / se:+.1f} SE)"
        cells = [m, f"{r['reward']:.3f} +- {r['reward_se']:.3f}", diff]
        cells += [f"{r['by_source'][s]['reward']:.2f}" if s in r["by_source"] else ""
                  for s in sources]
        cells += [f"{r['mean_len']:.0f}", str(r["capped"]), str(r["prompts"])]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("runs/heldout-121-160"))
    args = ap.parse_args(argv)
    print(table(load(E2E / args.dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
