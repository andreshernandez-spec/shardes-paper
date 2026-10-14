#!/usr/bin/env python
"""A training run on a probed setting, at the rule's prescription (docs/end_to_end/10).

    python -m es_vllm.probe_run --setting qwen3b/countdown --prediction runs/probe/qwen3b-countdown \
        --out runs/probe-run/qwen3b-countdown-s0 --seed 0 --iterations 30 --eval-every 5
    python -m es_vllm.probe_run --analyze runs/probe-run/qwen3b-countdown-s0
    (--smoke: tiny blocks, 8 members, wiring only; --step overrides the prediction's)

Members as the probe's: 64 mirrored rank-1 members at sigma 5e-4 on the attention and MLP
matrices, served as LoRA adapters built from their seeds; generation g of seed s uses the
seeds `(s, g, pair, leaf)`, so seed 0's first generation is the probe's own members.
Each iteration the members decode the train block greedily (the same 192 prompts every
iteration, as the published Countdown runs, or fresh prompts from the task's stream with
`--prompts stream`), the update `step (alpha / N) sum_m z_m E_m` is added to an f32 copy
of the perturbed leaves, and the engine's bf16 weights are written from it in place and
checked bit for bit. `step` is the multiple of the unit update, the rule's
`step_multiple` from the setting's prediction.json unless given. Every `eval_every`
updates the validation, test and off-task blocks are decoded greedily by the current
model and scored; the test block is the one the probe reserved (its digest must match).
A restart replays the log and checks the f32 copy's digest.

The rule's prediction, copied into run.json: a gain of `step G` per update on the task
and a loss of `step^2 |C|` from the random part, so after g updates the validation
change is about `g step G - g step^2 |C|` and the off-task block moves `g step G_offtask`;
the target (a tenth of the range) is reached after `updates` of them. `--analyze` puts
the observed changes beside these.
"""

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

PKG = Path(__file__).resolve().parent
E2E = PKG.parent
sys.path.insert(0, str(E2E))
sys.path.insert(0, str(E2E.parent))
import harness  # noqa: E402
from es_vllm import lowrank, probe  # noqa: E402
from provenance import env_block  # noqa: E402

SIGMA, ALPHA, RANK = probe.SIGMA, probe.ALPHA, probe.RANK
N, ITERATIONS, EVAL_EVERY, CHUNK = 64, 30, 5, 32
STREAM_FIRST_STEP = 160          # the probe's GSM8K blocks come from stream steps 0 to 159


def lines(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def append(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def digest(rows) -> str:
    return hashlib.sha256("".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in rows).encode()).hexdigest()


def select_stream(rows: list, protected: set, need: int, per: int) -> list:
    """`need` distinct GSM8K prompts from the stream's rows in order, none of them in
    `protected` (the probe's blocks) and none repeated: the stream passes over its rows
    several times, so a range of steps alone promises neither."""
    from tulu31_data import row_hash  # noqa: PLC0415

    out, seen = [], set()
    for r in rows:
        if r["dataset"] != "gsm8k":
            continue
        h = row_hash(r)
        if h in protected or h in seen:
            continue
        seen.add(h)
        out.append(r)
        if len(out) == need:
            break
    if len(out) < need:
        raise SystemExit(f"{len(out)} distinct unprotected GSM8K prompts in the stream rows given, {need} needed")
    assert not {row_hash(r) for r in out} & protected and len({row_hash(r) for r in out}) == need
    return [out[g * per:(g + 1) * per] for g in range(need // per)]


def stream_batches(task, iterations: int, per: int) -> list:
    """Fresh prompts per iteration from the task's stream: distinct, and none of the
    probe's train, validation, test or off-task prompts."""
    from es_vllm.run_tulu import heldout_rows  # noqa: PLC0415
    from tulu31_data import row_hash  # noqa: PLC0415

    if task.name != "gsm8k":
        raise SystemExit(f"{task.name} has no prompt stream; use --prompts repeated")
    need = iterations * per
    protected = {row_hash(r) for rows_ in task.blocks.values() for r in rows_}
    rows = heldout_rows(STREAM_FIRST_STEP, STREAM_FIRST_STEP + (need * 5) // 48 + 40)
    return select_stream(rows, protected, need, per)


def prediction_block(path: Path) -> dict:
    """What the run is held against, from the probe's prediction.json."""
    p = json.loads((path / "prediction.json").read_text())
    env = json.loads((path / "env.json").read_text())
    u, r = p["update"], p["rule"]
    if r["es"] is None:
        raise SystemExit(f"{path}: the rule gave no prescription ({r['call']})")
    source = str(path.relative_to(E2E)) if path.is_relative_to(E2E) else str(path)   # no machine paths in records
    return {"source": source, "setting": p["setting"], "call": r["call"], "step_multiple": r["es"]["step_multiple"],
            "updates_to_target": r["es"]["updates"], "fitted_net_per_update": r["es"]["fitted_net_per_update"],
            "G": u["G"], "G_se": u["G_se"], "C": u["C"], "C_se": u["C_se"], "G_offtask": u["G_offtask"],
            "unit_norm": u["unit_norm"], "ratio_es_over_rl_rollouts": r["ratio_es_over_rl_rollouts"],
            "block_digests": env["block_digests"], "blocks": env["blocks"]}


def run(args) -> int:
    from vllm.lora.request import LoRARequest  # noqa: PLC0415

    from es_vllm.worker import ESWorker  # noqa: PLC0415

    cfg = probe.setting(args.setting, args.smoke)
    pred = prediction_block(E2E / args.prediction) if args.prediction else None
    # the probe's blocks, validation and off-task included (a probe may score larger ones)
    task = probe.Task(cfg["task"], cfg["template"], args.smoke, args.es_at_scale,
                      block=pred["blocks"]["validation"] if pred and not args.smoke else None)
    if pred is not None and not args.smoke:
        if pred["setting"] != args.setting:
            raise SystemExit(f"the prediction is for {pred['setting']}, the run for {args.setting}")
        if pred["block_digests"] != task.digests():
            raise SystemExit("the blocks are not the probe's")
    step = args.step if args.step is not None else (pred["step_multiple"] if pred else None)
    if step is None:
        raise SystemExit("--step or --prediction")
    n = 8 if args.smoke else args.members
    chunk = min(CHUNK, n)                                   # adapters per generate call
    out = E2E / args.out
    out.mkdir(parents=True, exist_ok=True)
    # what the weights depend on; the iteration count can grow on a resume
    conf = {"setting": args.setting, "repo": cfg["repo"], "revision": cfg["revision"], "seed": args.seed,
            "sigma": SIGMA, "alpha": ALPHA, "rank": RANK, "step": step, "n": n, "prompts": args.prompts,
            "ranking": args.ranking, "shrink": args.shrink,
            "eval_every": 1 if args.smoke else args.eval_every, "smoke": args.smoke}
    if (out / "run.json").exists():
        if json.loads((out / "run.json").read_text())["config"] != conf:
            raise SystemExit(f"{out} holds a run with another configuration")
    else:
        harness.write_atomic(out / "run.json", {
            "config": conf, "iterations": args.iterations, "engine": {"chunk": chunk}, "prediction": pred,
            "blocks": {b: len(v) for b, v in task.blocks.items()}, "block_digests": task.digests(),
            "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "env": env_block(E2E, ["runs"], ("vllm", "torch"))})
    log, e_log = out / "log.jsonl", out / "eval.jsonl"
    done, e_done = lines(log), {r["iteration"] for r in lines(e_log)}
    pairs = n // 2

    scorer = task.scorer()                                  # before CUDA: the Countdown grader forks
    mdir = probe.model_dir(cfg)
    shapes = lowrank.leaf_shapes(mdir)
    template = lowrank.write_template(Path(args.workdir) / "adapter", RANK, cfg["repo"])
    lowrank.install_adapters(template, args.seed, shapes, RANK, SIGMA, n=n)
    for k_, fn in lowrank.worker_methods().items():
        setattr(ESWorker, k_, fn)
    llm = probe.engine(cfg, 0.6, enable_lora=True, max_lora_rank=RANK, max_loras=chunk, max_cpu_loras=n,
                       worker_extension_cls="es_vllm.worker.ESWorker")
    rpc = lambda m, *a: llm.collective_rpc(m, args=a)[0]  # noqa: E731
    rpc("lr_setup", mdir, args.seed, RANK, 0, shapes)
    if args.shrink:
        rpc("lr_keep_start")
    t0 = time.perf_counter()
    for rec in done:                                        # replay: the factors regenerate from their seeds
        rpc("lr_factors", args.seed, RANK, pairs, shapes, lowrank.generation_of(rec["iteration"]))
        rpc("lr_step", list(step * lowrank.coefficients(rec["ranked"], n, ALPHA, RANK)))
        if args.shrink:
            rpc("lr_shrink", args.shrink)
    if done:
        if rpc("lr_digest") != done[-1]["digest"]:
            raise SystemExit("the replayed weights are not the logged ones")
        print(f"replayed {len(done)} updates {time.perf_counter() - t0:.0f}s", flush=True)
    if not rpc("lr_check")["ok"]:
        raise SystemExit("engine weights are not the f32 copy")

    tok = probe.tokenizer(cfg)
    if args.prompts == "repeated":
        batches = [task.blocks["train"]] * args.iterations
    else:
        batches = stream_batches(task, args.iterations, len(task.blocks["train"]))
    ev_rows, ev_kinds, ev_block = [], [], []
    for b in ("validation", "test", "offtask"):
        ev_rows += task.blocks[b]
        ev_kinds += [probe.row_kind(r) for r in task.blocks[b]]
        ev_block += [b] * len(task.blocks[b])
    ev_ids = [task.render(tok, r) for r in ev_rows]

    def evaluate(g):
        if g in e_done or (g % conf["eval_every"] and g != args.iterations):
            return
        t1 = time.perf_counter()
        o = probe.decode(llm, ev_ids, task.params(ev_rows))
        r = probe.score_outs(scorer, o, ev_rows)
        norm = probe.normalized(r, ev_kinds)
        bl = np.asarray(ev_block)
        append(e_log, {"iteration": g, "rewards": r, "kinds": ev_kinds, "block": ev_block,
                       "mean_len": float(np.mean([len(x.outputs[0].token_ids) for x in o])),
                       "normalized_by_block": {b: float(norm[bl == b].mean()) for b in ("validation", "test", "offtask") if (bl == b).any()},
                       "seconds": time.perf_counter() - t1})
        print(f"eval after {g} updates: " + " ".join(f"{b} {norm[bl == b].mean():.3f}" for b in ("validation", "test", "offtask") if (bl == b).any()), flush=True)

    for g in range(len(done), args.iterations):
        evaluate(g)
        t1 = time.perf_counter()
        rows = batches[g]
        ids = [task.render(tok, r) for r in rows]
        rewards, lengths = [], []
        for c0 in range(0, n, chunk):
            part = list(range(c0, min(n, c0 + chunk)))
            o = probe.decode(llm, [i for _ in part for i in ids], task.params(rows * len(part)),
                             [LoRARequest(f"g{g}m{m}", g * n + m + 1, str(template)) for m in part for _ in ids])
            r = probe.score_outs(scorer, o, rows * len(part))
            for q in range(len(part)):
                rewards.append(r[q * len(ids):(q + 1) * len(ids)])
                lengths.append(float(np.mean([len(x.outputs[0].token_ids) for x in o[q * len(ids):(q + 1) * len(ids)]])))
        decode_s = time.perf_counter() - t1
        fitness = np.asarray(rewards).mean(1)
        # the random-only control: the real fitnesses in a seeded random order, so the
        # update has the real one's norm and none of its direction
        ranked = fitness if args.ranking == "real" else fitness[np.random.default_rng((args.seed, g)).permutation(n)]
        k = list(step * lowrank.coefficients(ranked, n, ALPHA, RANK))
        rpc("lr_factors", args.seed, RANK, pairs, shapes, lowrank.generation_of(g))
        norm = rpc("lr_norm", k)
        rpc("lr_step", k)
        if args.shrink:
            rpc("lr_shrink", args.shrink)
        check = rpc("lr_check")
        if not check["ok"]:
            raise SystemExit(f"engine weights are not the f32 copy after update {g + 1}")
        append(log, {"iteration": g, "fitness": fitness.tolist(), "ranked": ranked.tolist(), "mean_fitness": float(fitness.mean()),
                     "sd_fitness": float(fitness.std()), "rewards": rewards, "mean_len": lengths,
                     "prompt_digest": digest(rows), "rollouts": n * len(rows), "update_norm": norm,
                     "digest": rpc("lr_digest"), "check": check["ok"], "decode_seconds": decode_s,
                     "seconds": time.perf_counter() - t1})
        print(f"[{g}] fitness {fitness.mean():.4f} sd {fitness.std():.4f} |update| {norm:.2f} "
              f"{time.perf_counter() - t1:.0f}s", flush=True)
    evaluate(args.iterations)
    scorer.close()
    return 0


CLASS_WORKS, CLASS_MARGINAL, RETAINED_LOSS, COLLAPSE = 0.10, 0.05, -0.02, -0.02


def observed_class(evals: list) -> dict:
    """The outcome class of a run (docs/end_to_end/10, corrections item 6 as amended on
    2026-10-14): the checkpoint is selected on the validation block, the evaluation with
    the largest validation change among those whose retained (off-task) block is within
    -0.02, and the class is the test block's change at that checkpoint: "works" at 0.10,
    "marginal" at 0.05, "fails" otherwise. "Works" therefore means that one evaluated
    checkpoint qualifies, whatever happens later. The class the test block's own peak
    would earn (the definition of 2026-10-13, which selected on the test block) and the
    peak ignoring the retained block are kept beside it, and "collapsed" is set when any
    block falls more than 0.02 below its start at any evaluation."""
    def cls(x):
        return "works" if x >= CLASS_WORKS else "marginal" if x >= CLASS_MARGINAL else "fails"

    sel, best_val = None, None
    peak_ok, at_peak_ok, peak_any, at_peak_any, collapsed = 0.0, None, 0.0, None, False
    for r in evals:
        if r["iteration"] == 0:
            continue
        test = r["by_block"].get("test", {}).get("change", 0.0)
        val = r["by_block"].get("validation", {}).get("change", 0.0)
        retained_ok = r["by_block"].get("offtask", {"change": 0.0})["change"] >= RETAINED_LOSS
        if retained_ok and (best_val is None or val > best_val):
            sel, best_val = r, val
        if test > peak_any:
            peak_any, at_peak_any = test, r["iteration"]
        if retained_ok and test > peak_ok:
            peak_ok, at_peak_ok = test, r["iteration"]
        if any(v["change"] < COLLAPSE for v in r["by_block"].values()):
            collapsed = True
    test_at_sel = sel["by_block"]["test"]["change"] if sel else 0.0
    return {"class": cls(test_at_sel), "selected_iteration": sel["iteration"] if sel else None,
            "validation_change_at_selected": best_val, "test_change_at_selected": test_at_sel if sel else None,
            "offtask_change_at_selected": sel["by_block"].get("offtask", {}).get("change") if sel else None,
            "class_by_test_peak": cls(peak_ok), "best_test_change_with_retained_within_bound": peak_ok, "at_iteration": at_peak_ok,
            "class_ignoring_retained": cls(peak_any), "best_test_change": peak_any, "at_iteration_ignoring_retained": at_peak_any,
            "collapsed": collapsed,
            "thresholds": {"works": CLASS_WORKS, "marginal": CLASS_MARGINAL, "retained_loss": RETAINED_LOSS,
                           "collapse": COLLAPSE}}


def loop_seconds_to(evals: list, log: list, block: str = "test", threshold: float = CLASS_WORKS):
    """The first evaluation at or above `threshold` on `block`, with the loop seconds
    (an ES iteration's decode and update, a GRPO step's phases) and the rollouts of the
    iterations before it: the compute to that gain, without setup or evaluation."""
    for r in evals:
        if r["by_block"].get(block, {}).get("change", 0.0) >= threshold:
            k = r["iteration"]
            before = [x for x in log if x["iteration"] < k]
            secs = sum(sum(x["seconds"].values()) if isinstance(x["seconds"], dict) else x["seconds"] for x in before)
            return {"iteration": k, "loop_seconds": float(secs), "rollouts": int(sum(x["rollouts"] for x in before)),
                    "validation_change": r["by_block"]["validation"]["change"],
                    "offtask_change": r["by_block"].get("offtask", {}).get("change"),
                    "retained_within_bound": r["by_block"].get("offtask", {"change": 0.0})["change"] >= RETAINED_LOSS}
    return None


def compare(es_root: Path, grpo_root: Path, threshold: float = CLASS_WORKS) -> list:
    """Each ES run against the GRPO arm of its setting at the reference learning rate: the
    loop seconds and rollouts to the first evaluation at `threshold` on the test block,
    from the logs, and their ratios."""
    from es_vllm import grpo_run  # noqa: PLC0415

    grpo = {}
    for d in sorted(grpo_root.iterdir()):
        if (d / "eval.jsonl").exists() and "-lr5e-7-" in d.name:
            a = grpo_run.analyze(d)
            grpo[d.name.split("-lr")[0]] = (d.name, loop_seconds_to(a["eval"], lines(d / "log.jsonl"), threshold=threshold))
    rows = []
    for d in sorted(es_root.iterdir()):
        if not (d / "eval.jsonl").exists():
            continue
        a = analyze(d)
        es = loop_seconds_to(a["eval"], lines(d / "log.jsonl"), threshold=threshold)
        setting = d.name.rsplit("-s", 1)[0]
        ref = grpo.get(setting)
        row = {"run": d.name, "es_iteration": es and es["iteration"], "es_loop_seconds": es and es["loop_seconds"],
               "es_rollouts": es and es["rollouts"], "es_offtask_there": es and es["offtask_change"],
               "es_retained": es and es["retained_within_bound"], "grpo_arm": ref and ref[0],
               "grpo_step": ref and ref[1] and ref[1]["iteration"], "grpo_loop_seconds": ref and ref[1] and ref[1]["loop_seconds"],
               "grpo_rollouts": ref and ref[1] and ref[1]["rollouts"], "grpo_offtask_there": ref and ref[1] and ref[1]["offtask_change"]}
        ok = es and ref and ref[1]
        row["seconds_ratio"] = es["loop_seconds"] / ref[1]["loop_seconds"] if ok else None
        row["rollouts_ratio"] = es["rollouts"] / ref[1]["rollouts"] if ok else None
        rows.append(row)
    return rows


def compare_table(rows: list) -> str:
    cols = (("run", "ES run"), ("es_iteration", "+0.10 at update"), ("es_loop_seconds", "ES loop s"), ("es_rollouts", "ES rollouts"),
            ("es_offtask_there", "off-task there"), ("es_retained", "retained"), ("grpo_step", "GRPO +0.10 at step"),
            ("grpo_loop_seconds", "GRPO loop s"), ("grpo_rollouts", "GRPO rollouts"), ("grpo_offtask_there", "GRPO off-task there"),
            ("seconds_ratio", "seconds, ES/GRPO"), ("rollouts_ratio", "rollouts, ES/GRPO"))

    def fmt(x):
        if x is None:
            return "-"
        if isinstance(x, bool):
            return "yes" if x else "no"
        if isinstance(x, float):
            return f"{x:+.3f}" if abs(x) < 1 else f"{x:,.0f}" if abs(x) >= 100 else f"{x:.2f}"
        return f"{x:,}" if isinstance(x, int) and x >= 1000 else str(x)

    head = "| " + " | ".join(n for _, n in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    return "\n".join([head, sep, *("| " + " | ".join(fmt(r[k]) for k, _ in cols) + " |" for r in rows)])


def analyze(out: Path) -> dict:
    """Observed changes per block and task beside the rule's prediction."""
    run_ = json.loads((out / "run.json").read_text())
    pred, step = run_.get("prediction"), run_["config"]["step"]
    ev = sorted(lines(out / "eval.jsonl"), key=lambda r: r["iteration"])
    log = lines(out / "log.jsonl")
    base = probe.normalized(ev[0]["rewards"], ev[0]["kinds"])
    kinds, blocks = np.asarray(ev[0]["kinds"]), np.asarray(ev[0]["block"])
    rows = []
    for r in ev:
        d = probe.normalized(r["rewards"], r["kinds"]) - base
        g = r["iteration"]
        se = lambda x: float(x.std(ddof=1) / np.sqrt(x.size)) if g else 0.0  # noqa: E731
        row = {"iteration": g, "mean_len": r["mean_len"],
               "by_block": {b: {"reward": float(probe.normalized(r["rewards"], r["kinds"])[blocks == b].mean()),
                                "change": float(d[blocks == b].mean()), "change_se": se(d[blocks == b])}
                            for b in ("validation", "test", "offtask") if (blocks == b).any()},
               "by_task": {t: {"change": float(d[kinds == t].mean()), "change_se": se(d[kinds == t])} for t in sorted(set(kinds))}}
        if pred:
            row["predicted"] = {"validation_change": g * step * pred["G"] - g * step ** 2 * -pred["C"],
                                "gain_first_order": g * step * pred["G"], "random_cost": g * step ** 2 * -pred["C"],
                                "offtask_change": g * step * pred["G_offtask"] if pred["G_offtask"] is not None else None}
        rows.append(row)
    res = {"setting": run_["config"]["setting"], "seed": run_["config"]["seed"], "step": step, "n": run_["config"]["n"],
           "iterations_done": len(log), "rollouts_total": sum(r["rollouts"] for r in log),
           "seconds_total": sum(r["seconds"] for r in log),
           "fitness": [{"iteration": r["iteration"], "mean": r["mean_fitness"], "sd": r["sd_fitness"],
                        "update_norm": r["update_norm"]} for r in log],
           "eval": rows, "prediction": pred, "observed": observed_class(rows) if rows else None}
    if pred and rows:
        last = rows[-1]
        val = last["by_block"].get("validation", {})
        res["at_end"] = {"iteration": last["iteration"], "observed_validation_change": val.get("change"),
                         "observed_validation_change_se": val.get("change_se"),
                         "observed_test_change": last["by_block"].get("test", {}).get("change"),
                         "predicted_validation_change": last["predicted"]["validation_change"],
                         "updates_to_target_predicted": pred["updates_to_target"]}
    return res


def summary(root: Path) -> list:
    """One row per run under `root`: the prescription it ran at, what the blocks did, the
    frozen class."""
    rows = []
    for d in sorted(root.iterdir()):
        if not (d / "eval.jsonl").exists():
            continue
        a = analyze(d)
        last, o = a["eval"][-1], a["observed"]
        fmt = lambda b: f"{last['by_block'][b]['change']:+.3f}" if b in last["by_block"] else "-"  # noqa: E731
        rows.append({"run": d.name, "setting": a["setting"], "seed": a["seed"], "step": a["step"],
                     "call": (a["prediction"] or {}).get("call"), "updates_to_target_predicted": (a["prediction"] or {}).get("updates_to_target"),
                     "updates": a["iterations_done"], "rollouts": a["rollouts_total"],
                     "final_validation": fmt("validation"), "final_test": fmt("test"), "final_offtask": fmt("offtask"),
                     "peak_test": o["best_test_change"], "peak_at": o["at_iteration_ignoring_retained"],
                     "selected_at": o["selected_iteration"], "test_at_selected": o["test_change_at_selected"],
                     "class": o["class"], "class_by_test_peak": o["class_by_test_peak"],
                     "class_ignoring_retained": o["class_ignoring_retained"], "collapsed": o["collapsed"]})
    return rows


def table(rows: list) -> str:
    cols = (("run", "run"), ("call", "preregistered call"), ("step", "step"), ("updates_to_target_predicted", "updates to target, predicted"),
            ("updates", "updates"), ("final_validation", "validation at end"), ("final_test", "test at end"),
            ("final_offtask", "off-task at end"), ("peak_test", "peak test gain"), ("peak_at", "at"),
            ("selected_at", "validation-selected checkpoint"), ("test_at_selected", "test there"),
            ("class", "class"), ("class_by_test_peak", "class by test peak"), ("class_ignoring_retained", "class, task alone"),
            ("collapsed", "collapsed"))

    def fmt(x):
        if x is None:
            return "-"
        if isinstance(x, bool):
            return "yes" if x else "no"
        if isinstance(x, float):
            return f"{x:.3g}"
        return str(x)

    head = "| " + " | ".join(n for _, n in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    return "\n".join([head, sep, *("| " + " | ".join(fmt(r[k]) for k, _ in cols) + " |" for r in rows)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, help="the runs directory: one row per run")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("ES_RUNS", "GRPO_RUNS"),
                    help="each ES run against its setting's GRPO arm: loop seconds and rollouts to +0.10 on the test block")
    ap.add_argument("--setting", help="model/task, as es_vllm.probe")
    ap.add_argument("--prediction", type=Path, help="the setting's probe record (step and the numbers to hold the run against)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--analyze", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="member seeds; 0 shares the probe's first generation")
    ap.add_argument("--members", type=int, default=N)
    ap.add_argument("--step", type=float, help="multiple of the unit update (default: the prediction's)")
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--prompts", choices=("repeated", "stream"), default="repeated")
    ap.add_argument("--ranking", choices=("real", "random"), default="real",
                    help="random: the fitnesses in a seeded random order, an update of the real norm with no direction")
    ap.add_argument("--shrink", type=float, default=0.0,
                    help="after each update, pull the weights this fraction of the way back toward the start")
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/probe-run"))
    ap.add_argument("--es-at-scale", type=Path, default=Path.home() / "private" / "open-source" / "es-at-scale")
    args = ap.parse_args(argv)
    if args.compare:
        es, gr = (p if p.is_absolute() else E2E / p for p in args.compare)
        print(compare_table(compare(es, gr)))
        return 0
    if args.table:
        print(table(summary(args.table if args.table.is_absolute() else E2E / args.table)))
        return 0
    if args.analyze:
        print(json.dumps(analyze(E2E / args.analyze), indent=2))
        return 0
    if args.out is None or args.setting is None:
        ap.error("--setting and --out, or --analyze")
    args.workdir.mkdir(parents=True, exist_ok=True)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
