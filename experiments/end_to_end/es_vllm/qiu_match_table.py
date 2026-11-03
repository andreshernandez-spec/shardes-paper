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
    if (Path(root) / "qwen3b-s1-decay").exists():
        decay(Path(root))
    if (Path(root) / "qwen3b-grpo-s0").exists():
        grpo(Path(root))
    return 0


def grpo(root: Path) -> None:
    """S6c (docs/end_to_end/14-grpo-matched.md) beside arm B, by answers seen and by
    training seconds (the logs' own times: decoding and updating, no tests or setup)."""
    g, b = root / "qwen3b-grpo-s0", root / "qwen3b-s1"
    lg, lb = lines(g / "log.jsonl"), lines(b / "log.jsonl")
    eg = {e["iteration"]: e for e in lines(g / "eval.jsonl")}
    eb = {e["iteration"]: e for e in lines(b / "eval.jsonl")}
    per_step, per_update = lg[0]["rollouts"], lb[0]["rollouts"]
    cg = np.concatenate([[0], np.cumsum([sum(r["seconds"].values()) for r in lg])])
    cb = np.concatenate([[0], np.cumsum([r["seconds"] for r in lb])])
    print(f"\nS6c: steps {len(lg)}, answers per step {per_step} (ES per update {per_update}), grading timeouts "
          f"{max(r['timeouts'] for r in lg)}, median seconds per step {np.median([sum(r['seconds'].values()) for r in lg]):.0f} "
          f"(ES per update {np.median([r['seconds'] for r in lb]):.0f}), training hours {cg[-1] / 3600:.1f}")

    def cut(e):
        return 1 - np.mean(e["stopped"])
    print("\n| answers | GRPO step | GRPO | hours | cut at 1,024 | ES update | ES | hours | cut at 1,024 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k in sorted(eg):
        u = k * per_step // per_update
        print(f"| {k * per_step:,} | {k} | {100 * eg[k]['accuracy']:.2f}% | {cg[k] / 3600:.1f} | {100 * cut(eg[k]):.0f}% | "
              f"{u} | {100 * eb[u]['accuracy']:.2f}% | {cb[u] / 3600:.1f} | {100 * cut(eb[u]):.0f}% |")
    ks = (320, 400, 480)
    us = [k * per_step // per_update for k in ks]
    diff = np.mean([solved(eg[k]) for k in ks], 0) - np.mean([solved(eb[u]) for u in us], 0)
    print(f"D (GRPO after {ks} steps minus ES after {us} updates): {100 * diff.mean():+.2f} points, "
          f"paired standard error over the 2,000 problems {100 * diff.std(ddof=1) / np.sqrt(len(diff)):.2f}")
    for k in (240, 480):
        h = json.loads((g / f"qiu-eval/ckpt-{k}/summary.json").read_text())["eval_stats"]["accuracy"]
        print(f"his evaluator, GRPO after {k} steps: {h:.2f}%")
    e = eg[max(eg)]
    st = np.array(e["stopped"])
    print(f"GRPO after {max(eg)} steps: accuracy {100 * solved(e)[st].mean():.1f}% on the answers that end, "
          f"{100 * solved(e)[~st].mean():.1f}% on the {int((~st).sum())} cut at the cap")

    print("\n| hours of training | GRPO | ES (its last test by then) |")
    print("|---|---|---|")
    for s in sorted(eg)[1:]:
        u = max(k for k in eb if cb[k] <= cg[s])
        print(f"| {cg[s] / 3600:.1f} | {100 * eg[s]['accuracy']:.2f}% (step {s}) | {100 * eb[u]['accuracy']:.2f}% (update {u}) |")
    for thr in (0.45, 0.50, 0.56):
        u = min(k for k in eb if eb[k]["accuracy"] >= thr)
        s = min(k for k in eg if eg[k]["accuracy"] >= thr)
        print(f"first test at or above {100 * thr:.0f}%: ES after {u} updates ({u * per_update:,} answers, "
              f"{cb[u] / 3600:.1f} h), GRPO after {s} steps ({s * per_step:,} answers, {cg[s] / 3600:.1f} h)")
    for name, log in (("GRPO", lg), ("ES", lb)):
        print(f"{name}: training problems solved in the last 80 steps or 10 updates "
              f"{np.mean([r['solved'] if name == 'GRPO' else (np.array(r['rewards']) >= 1).mean() for r in log[-(80 if name == 'GRPO' else 10):]]):.3f}")


def decay(root: Path) -> None:
    """S6b (docs/end_to_end/13-step-decay.md) beside arm B."""
    d, b = root / "qwen3b-s1-decay", root / "qwen3b-s1"
    ld, lb = lines(d / "log.jsonl"), lines(b / "log.jsonl")
    ed = {e["iteration"]: e for e in lines(d / "eval.jsonl")}
    eb = {e["iteration"]: e for e in lines(b / "eval.jsonl")}
    print(f"\nS6b: updates {ld[0]['iteration']} to {ld[-1]['iteration']}, every check ok "
          f"{all(c['ok'] for r in ld for c in r['checks'])}, grading timeouts {max(r['timeouts'] for r in ld)}, "
          f"median seconds per update {np.median([r['seconds'] for r in ld]):.0f}, "
          f"step at the last update {ld[-1]['lr'] / ld[0]['lr']:.3f} of the first")
    print("\n| updates | step (of alpha) | decay arm | arm B | difference |")
    print("|---|---|---|---|---|")
    for k in range(200, 351, 10):
        lr = next((r["lr"] / ld[0]["lr"] for r in ld if r["iteration"] == k), None)
        print(f"| {k} | {'' if lr is None else f'{lr:.2f}'} | {100 * ed[k]['accuracy']:.2f}% | {100 * eb[k]['accuracy']:.2f}% | "
              f"{100 * (ed[k]['accuracy'] - eb[k]['accuracy']):+.2f} |")
    ks = range(310, 351, 10)
    diff = np.mean([solved(ed[k]) for k in ks], 0) - np.mean([solved(eb[k]) for k in ks], 0)
    print(f"D (mean of the running tests after 310 to 350): {100 * diff.mean():+.2f} points "
          f"(decay {100 * np.mean([ed[k]['accuracy'] for k in ks]):.2f}%, arm B {100 * np.mean([eb[k]['accuracy'] for k in ks]):.2f}%), "
          f"paired standard error over the 2,000 problems {100 * diff.std(ddof=1) / np.sqrt(len(diff)):.2f}")
    g = np.arange(200, 351, 10)
    for name, ev in (("decay", ed), ("arm B", eb)):
        x, y = solved(ev[200]), solved(ev[350])
        print(f"{name}: slope 200 to 350 {10 * np.polyfit(g, [100 * ev[k]['accuracy'] for k in g], 1)[0]:+.3f} points per 10 updates, "
              f"test answers gained {int((~x & y).sum())}, lost {int((x & ~y).sum())}")
    print(f"the two arms' answers after 350 updates differ on {int((solved(ed[350]) != solved(eb[350])).sum())} of 2,000")
    for k in (300, 350):
        h = json.loads((d / f"qiu-eval/ckpt-{k}/summary.json").read_text())["eval_stats"]["accuracy"]
        print(f"his evaluator, decay checkpoint {k}: {h:.2f}%")

    def train(log, a, z):
        return np.mean([(np.array(r["rewards"]) >= 1).mean() for r in log if a <= r["iteration"] < z])

    def split(r):
        s = np.array(r["rewards"]) >= 1
        per = s.sum(0)
        return int(((per > 0) & (per < len(s))).sum())
    print(f"training problems solved over updates 340 to 349: decay {train(ld, 340, 350):.3f}, arm B {train(lb, 340, 350):.3f}; "
          f"problems that split the members at update 349: decay {split(ld[-1])}, arm B {split(lb[349])}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
