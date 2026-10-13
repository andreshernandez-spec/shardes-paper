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


def stream_batches(task, iterations: int, per: int) -> list:
    """Fresh prompts per iteration from the task's stream, after the probe's blocks."""
    from es_vllm.run_tulu import heldout_rows  # noqa: PLC0415

    if task.name != "gsm8k":
        raise SystemExit(f"{task.name} has no prompt stream; use --prompts repeated")
    need = iterations * per
    rows = [r for r in heldout_rows(STREAM_FIRST_STEP, STREAM_FIRST_STEP + (need * 5) // 48 + 40) if r["dataset"] == "gsm8k"]
    if len(rows) < need:
        raise SystemExit(f"the stream holds {len(rows)} GSM8K prompts after step {STREAM_FIRST_STEP}, {need} needed")
    return [rows[g * per:(g + 1) * per] for g in range(iterations)]


def prediction_block(path: Path) -> dict:
    """What the run is held against, from the probe's prediction.json."""
    p = json.loads((path / "prediction.json").read_text())
    env = json.loads((path / "env.json").read_text())
    u, r = p["update"], p["rule"]
    if r["es"] is None:
        raise SystemExit(f"{path}: the rule gave no prescription ({r['call']})")
    return {"source": str(path), "setting": p["setting"], "call": r["call"], "step_multiple": r["es"]["step_multiple"],
            "updates_to_target": r["es"]["updates"], "fitted_net_per_update": r["es"]["fitted_net_per_update"],
            "G": u["G"], "G_se": u["G_se"], "C": u["C"], "C_se": u["C_se"], "G_offtask": u["G_offtask"],
            "unit_norm": u["unit_norm"], "ratio_es_over_rl_rollouts": r["ratio_es_over_rl_rollouts"],
            "block_digests": env["block_digests"]}


def run(args) -> int:
    from vllm.lora.request import LoRARequest  # noqa: PLC0415

    from es_vllm.worker import ESWorker  # noqa: PLC0415

    cfg = probe.setting(args.setting, args.smoke)
    task = probe.Task(cfg["task"], cfg["template"], args.smoke, args.es_at_scale)
    pred = prediction_block(E2E / args.prediction) if args.prediction else None
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
    conf = {"setting": args.setting, "repo": cfg["repo"], "revision": cfg["revision"], "seed": args.seed,
            "sigma": SIGMA, "alpha": ALPHA, "rank": RANK, "step": step, "n": n, "prompts": args.prompts,
            "iterations": args.iterations, "eval_every": 1 if args.smoke else args.eval_every, "smoke": args.smoke}
    if (out / "run.json").exists():
        if json.loads((out / "run.json").read_text())["config"] != conf:
            raise SystemExit(f"{out} holds a run with another configuration")
    else:
        harness.write_atomic(out / "run.json", {
            "config": conf, "engine": {"chunk": chunk}, "prediction": pred,
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
    t0 = time.perf_counter()
    for rec in done:                                        # replay: the factors regenerate from their seeds
        rpc("lr_factors", args.seed, RANK, pairs, shapes, lowrank.generation_of(rec["iteration"]))
        rpc("lr_step", list(step * lowrank.coefficients(rec["fitness"], n, ALPHA, RANK)))
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
        k = list(step * lowrank.coefficients(fitness, n, ALPHA, RANK))
        rpc("lr_factors", args.seed, RANK, pairs, shapes, lowrank.generation_of(g))
        norm = rpc("lr_norm", k)
        rpc("lr_step", k)
        check = rpc("lr_check")
        if not check["ok"]:
            raise SystemExit(f"engine weights are not the f32 copy after update {g + 1}")
        append(log, {"iteration": g, "fitness": fitness.tolist(), "mean_fitness": float(fitness.mean()),
                     "sd_fitness": float(fitness.std()), "rewards": rewards, "mean_len": lengths,
                     "prompt_digest": digest(rows), "rollouts": n * len(rows), "update_norm": norm,
                     "digest": rpc("lr_digest"), "check": check["ok"], "decode_seconds": decode_s,
                     "seconds": time.perf_counter() - t1})
        print(f"[{g}] fitness {fitness.mean():.4f} sd {fitness.std():.4f} |update| {norm:.2f} "
              f"{time.perf_counter() - t1:.0f}s", flush=True)
    evaluate(args.iterations)
    scorer.close()
    return 0


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
           "eval": rows, "prediction": pred}
    if pred and rows:
        last = rows[-1]
        val = last["by_block"].get("validation", {})
        res["at_end"] = {"iteration": last["iteration"], "observed_validation_change": val.get("change"),
                         "observed_validation_change_se": val.get("change_se"),
                         "observed_test_change": last["by_block"].get("test", {}).get("change"),
                         "predicted_validation_change": last["predicted"]["validation_change"],
                         "updates_to_target_predicted": pred["updates_to_target"]}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/probe-run"))
    ap.add_argument("--es-at-scale", type=Path, default=Path.home() / "private" / "open-source" / "es-at-scale")
    args = ap.parse_args(argv)
    if args.analyze:
        print(json.dumps(analyze(E2E / args.analyze), indent=2))
        return 0
    if args.out is None or args.setting is None:
        ap.error("--setting and --out, or --analyze")
    args.workdir.mkdir(parents=True, exist_ok=True)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
