#!/usr/bin/env python
"""The numbers of runs/qiu-matched/README.md, from the run's records and the trace.

    python -m es_vllm.qiu_match_table runs/qiu-matched     # from experiments/end_to_end
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np

HIS = {0: 10.0, 100: 35.6, 200: 41.2, 300: 48.0, 400: 54.2, 500: 60.5}  # Table 1 base, Figure 7 (read off the plot)


def lines(p: Path) -> list:
    return [json.loads(x) for x in p.read_text().splitlines()]


def solved(e) -> np.ndarray:
    return np.array(e["answer"]) > 0


def main(root: str) -> int:
    d = Path(root) / "qwen3b-s1"
    log, ev = lines(d / "log.jsonl"), {e["iteration"]: e for e in lines(d / "eval.jsonl")}
    print(f"updates {len(log)}, tests {len(ev)}, every check ok {all(c['ok'] for r in log for c in r['checks'])}, "
          f"grading timeouts {max(r['timeouts'] for r in log)}, median seconds per update {np.median([r['seconds'] for r in log]):.0f}")

    print("\n| updates | their evaluator | our running test | their evaluator and ours differ on | Qiu et al. |")
    print("|---|---|---|---|---|")
    for k in sorted(HIS):
        h = json.loads((d / f"qiu-eval/ckpt-{k}/summary.json").read_text())["eval_stats"]["accuracy"]
        rows = json.load(gzip.open(d / f"qiu-eval/ckpt-{k}/eval_detailed_results.json.gz"))
        his = np.array([r["reward_info"]["answer_reward"] > 0 for r in rows])
        print(f"| {k} | {h:.2f}% | {100 * solved(ev[k]).mean():.2f}% | {int((his != solved(ev[k])).sum())} of 2,000 | {HIS[k]}% |")

    print("\n| updates | test, points per 10 updates | test answers gained, lost | train solved at the end |")
    print("|---|---|---|---|")
    for a, b in ((0, 100), (100, 200), (200, 300), (300, 400), (400, 500)):
        g = np.arange(a, b + 1, 10)
        slope = np.polyfit(g, [100 * ev[k]["accuracy"] for k in g], 1)[0] * 10
        x, y = solved(ev[a]), solved(ev[b])
        train = np.mean([(np.array(log[i]["rewards"]) >= 1).mean() for i in range(b - 10, b)])
        print(f"| {a} to {b} | {slope:+.2f} | {int((~x & y).sum())}, {int((x & ~y).sum())} | {train:.3f} |")
    print(f"mean of the tests after 410 to 500 updates: {100 * np.mean([ev[k]['accuracy'] for k in range(410, 501, 10)]):.2f}%")

    print("\n| update | members' answers off the majority (of 200) | sd of members' net (problems) | corr(fitness, net) | problems that split the members |")
    print("|---|---|---|---|---|")
    for r in [log[i] for i in (0, 100, 200, 300, 400, 499)]:
        s = np.array(r["rewards"]) >= 1.0
        maj = s.mean(0) >= 0.5
        net = (s & ~maj).sum(1) - (~s & maj).sum(1)
        per = s.sum(0)
        print(f"| {r['iteration']} | {(s != maj).sum(1).mean():.1f} | {net.std():.1f} | "
              f"{np.corrcoef(r['fitness'], net)[0, 1]:.2f} | {int(((per > 0) & (per < len(s))).sum())} |")

    for p in ("fp32", "bf16"):
        c = json.loads((d / f"decode-check-{p}.json").read_text())
        v = c["variants"]
        print(f"\ndecode check {p}: HF accuracy {c['hf']['accuracy']:.3f}; HF rebatched identical {c['hf_rebatched']['identical']}/200; "
              + "; ".join(f"{k} identical {x['identical']}/200 (part at {x['first_difference_median']}), accuracy {x['accuracy']:.3f}"
                          for k, x in v.items()) + f"; processor applied {c['processor_applied']}; passes {c['passes']}")

    t = json.loads((Path(root) / "trace-his.json").read_text())
    it = t["his"][0]["tensors"]
    mats = {n: v for n, v in it.items() if "layernorm" not in n}
    print(f"\ntrace: reproduced {all(v['reproduced'] for v in it.values())}; iteration {t['seconds'][0]:.0f} s; "
          f"cosine of the meant update across layers {min(min(x.values()) if isinstance(x, dict) else x for x in t['his'][0]['shared'].values()):.4f}; "
          f"matrices: lost {min(v['lost_share'] for v in mats.values()):.2f} to {max(v['lost_share'] for v in mats.values()):.2f}, "
          f"got/meant {min(v['got_norm'] / v['meant_norm'] for v in mats.values()):.3f} to {max(v['got_norm'] / v['meant_norm'] for v in mats.values()):.3f}, "
          f"cosine {min(v['cosine'] for v in mats.values()):.2f} to {max(v['cosine'] for v in mats.values()):.2f}, "
          f"residue/meant {min(v['residue_over_meant'] for v in mats.values()):.2f} to {max(v['residue_over_meant'] for v in mats.values()):.2f}, "
          f"residue share {min(v['residue_share'] for v in mats.values()):.2f} to {max(v['residue_share'] for v in mats.values()):.2f}; "
          f"layer norms moved {max(v['moved_share'] for n, v in it.items() if 'layernorm' in n):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
