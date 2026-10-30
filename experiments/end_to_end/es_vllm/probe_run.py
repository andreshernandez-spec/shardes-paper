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

`--full-rank` runs es-at-scale's Countdown recipe (Qiu et al. 2025) on the same blocks
instead: full-rank Gaussian members on every weight at sigma 1e-3, alpha 5e-4, the
fitness z-scored, no anchor, through shardes in the worker (`worker.ESWorker`, the
backend T2 matched to es-at-scale, runs/countdown-README.md). Each member's weights are
written into the engine in turn and decode the train block; member 0 and the restored
view are checked bit for bit at every update. The probe's prediction is for its rank-1
members, so the run carries none. Not resumable: a killed run starts again.

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
FULL_SIGMA, FULL_ALPHA = 1e-3, 5e-4    # es-at-scale's defaults, as es_vllm/countdown-s1.yaml
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
    """What the run is held against, from the probe's prediction.json (the first batches'
    rule) or, when the probe wrote one, its prescription.json (S4b's anchored step and
    the four predictions), which then sets the step and the shrinkage."""
    p = json.loads((path / "prediction.json").read_text())
    env = json.loads((path / "env.json").read_text())
    u, r = p["update"], p["rule"]
    if (path / "prescription.json").exists():
        q = json.loads((path / "prescription.json").read_text())
        return {"source": str(path.relative_to(E2E)) if path.is_relative_to(E2E) else str(path), "setting": p["setting"],
                "prescription": q, "call": q["P1_learns"], "step_multiple": q["step_multiple"], "shrink": q["shrink"],
                "updates_to_target": None, "fitted_net_per_update": None,
                "G": u["G"] if u else None, "G_se": u["G_se"] if u else None, "C": u["C"] if u else None,
                "C_se": u["C_se"] if u else None, "G_offtask": u["G_offtask"] if u else None,
                "unit_norm": q["unit_norm"], "ratio_es_over_rl_rollouts": None,
                "block_digests": env["block_digests"], "blocks": env["blocks"]}
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
    # the probe's blocks, validation and off-task included (a probe may score larger ones),
    # and its untouched retained-task test block where it recorded one
    task = probe.Task(cfg["task"], cfg["template"], args.smoke, args.es_at_scale,
                      block=pred["blocks"]["validation"] if pred and not args.smoke else None,
                      offtask_test=bool(pred and not args.smoke and "offtask_test" in pred["blocks"]))
    if pred is not None and not args.smoke:
        if pred["setting"] != args.setting:
            raise SystemExit(f"the prediction is for {pred['setting']}, the run for {args.setting}")
        if pred["block_digests"] != task.digests():
            raise SystemExit("the blocks are not the probe's")
    if args.full_rank:
        if args.shrink:
            raise SystemExit("--full-rank has no anchor")
        step, args.shrink = (1.0 if args.step is None else args.step), 0.0
        sigma, alpha, rank = FULL_SIGMA, FULL_ALPHA, None
    else:
        step = args.step if args.step is not None else (pred["step_multiple"] if pred else None)
        if step is None:
            raise SystemExit("--step or --prediction")
        if pred and "shrink" in pred and args.shrink is None:
            args.shrink = pred["shrink"]                 # the prescription's anchor, unless overridden
        args.shrink = args.shrink or 0.0
        sigma, alpha, rank = SIGMA, ALPHA, RANK
    n = 8 if args.smoke else args.members
    chunk = min(CHUNK, n)                                   # adapters per generate call
    out = E2E / args.out
    out.mkdir(parents=True, exist_ok=True)
    # what the weights depend on; the iteration count can grow on a resume
    conf = {"setting": args.setting, "repo": cfg["repo"], "revision": cfg["revision"], "seed": args.seed,
            "sigma": sigma, "alpha": alpha, "rank": rank, "step": step, "n": n, "prompts": args.prompts,
            "ranking": args.ranking, "shrink": args.shrink,
            "eval_every": 1 if args.smoke else args.eval_every, "smoke": args.smoke}
    if args.full_rank:
        conf["noise"] = "full-rank"                         # the rank-1 runs predate the key
    if (out / "run.json").exists():
        if json.loads((out / "run.json").read_text())["config"] != conf:
            raise SystemExit(f"{out} holds a run with another configuration")
    else:
        harness.write_atomic(out / "run.json", {
            "config": conf, "iterations": args.iterations, "engine": {"chunk": chunk},
            "prediction": None if args.full_rank else pred, "blocks_from": pred and pred["source"],
            "blocks": {b: len(v) for b, v in task.blocks.items()}, "block_digests": task.digests(),
            "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "env": env_block(E2E, ["runs"], ("vllm", "torch", "jax", "jaxlib"))})
    log, e_log = out / "log.jsonl", out / "eval.jsonl"
    done, e_done = lines(log), {r["iteration"] for r in lines(e_log)}
    if args.full_rank and done:
        raise SystemExit(f"{out} has a log; a full-rank run is never resumed, move it aside first")
    pairs = n // 2

    scorer = task.scorer()                                  # before CUDA: the Countdown grader forks
    mdir = probe.model_dir(cfg)
    if args.full_rank:
        llm = probe.engine(cfg, 0.6, worker_extension_cls="es_vllm.worker.ESWorker")
        rpc = lambda m, *a: llm.collective_rpc(m, args=a)[0]  # noqa: E731
        print(f"es_init {rpc('es_init', mdir, n, sigma, step * alpha * sigma, args.seed)}", flush=True)
        if not rpc("es_check", None)["ok"]:
            raise SystemExit("engine weights are not the f32 master's view")
    else:
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
        for rec in done:                                    # replay: the factors regenerate from their seeds
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
    for b in ("validation", "test", "offtask", "offtask_test"):
        for r in task.blocks.get(b, []):
            ev_rows.append(r)
            ev_kinds.append(probe.row_kind(r))
            ev_block.append(b)
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
        rewards, lengths, extra = [], [], {}
        if args.full_rank:
            rpc("es_ask")
            for m in range(n):
                rpc("es_perturb", m)
                if m == 0:                                  # the JAX-to-torch handoff (worker.py)
                    extra["checks"] = [{"member": 0, **rpc("es_check", 0)}]
                    if not extra["checks"][0]["ok"]:
                        raise SystemExit(f"engine weights are not member 0 at update {g + 1}")
                o = probe.decode(llm, ids, task.params(rows))
                rewards.append(probe.score_outs(scorer, o, rows))
                lengths.append(float(np.mean([len(x.outputs[0].token_ids) for x in o])))
        else:
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
        if args.full_rank:
            extra["tell"] = rpc("es_tell", ranked.tolist())
            rpc("es_restore")
            check = rpc("es_check", None)
            extra["checks"].append({"member": None, **check})
            norm, w_digest = None, rpc("es_digest")
        else:
            k = list(step * lowrank.coefficients(ranked, n, ALPHA, RANK))
            rpc("lr_factors", args.seed, RANK, pairs, shapes, lowrank.generation_of(g))
            norm = rpc("lr_norm", k)
            rpc("lr_step", k)
            if args.shrink:
                rpc("lr_shrink", args.shrink)
            check, w_digest = rpc("lr_check"), rpc("lr_digest")
        if not check["ok"]:
            raise SystemExit(f"engine weights are not the f32 copy after update {g + 1}")
        append(log, {"iteration": g, "fitness": fitness.tolist(), "ranked": ranked.tolist(), "mean_fitness": float(fitness.mean()),
                     "sd_fitness": float(fitness.std()), "rewards": rewards, "mean_len": lengths,
                     "prompt_digest": digest(rows), "rollouts": n * len(rows), "update_norm": norm,
                     "digest": w_digest, "check": check["ok"], **extra, "decode_seconds": decode_s,
                     "seconds": time.perf_counter() - t1})
        print(f"[{g}] fitness {fitness.mean():.4f} sd {fitness.std():.4f}"
              + ("" if norm is None else f" |update| {norm:.2f}") + f" {time.perf_counter() - t1:.0f}s", flush=True)
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


RETENTION_MARGIN, RETAINED_SELECTION = 0.03, 300   # S4b: the fixed loss margin; the selection half of the off-task block


def retention(evals: list, margin: float = RETENTION_MARGIN, half: int = RETAINED_SELECTION) -> dict:
    """S4b's retention and class (docs/end_to_end/10, "S4b, fresh settings for the verdict:
    design", choice 4): on each retained task family separately, the paired change against
    the evaluation at update 0 with its interval; "within" the margin when the interval's
    lower bound is above -margin, "lost" when its upper bound is below -margin,
    "unresolved" otherwise. The off-task block's first `half` prompts are the selection
    half, the rest the reserved half. The checkpoint is the evaluation with the largest
    validation change among those within the margin on every family of the selection
    half (unresolved families do not disqualify, and are reported); the class is the
    reserved test block's change there (works at +0.10, marginal at +0.05); the final
    retention is read on the reserved half at that checkpoint. Needs the per-prompt
    changes of each evaluation (`by_family_half` from `analyze`)."""
    def cls(x):
        return "works" if x >= CLASS_WORKS else "marginal" if x >= CLASS_MARGINAL else "fails"

    def status(c):
        lo, hi = c["change"] - 2 * c["change_se"], c["change"] + 2 * c["change_se"]
        return "within" if lo > -margin else "lost" if hi < -margin else "unresolved"

    def verdict(statuses):
        if not statuses:
            return None
        return "lost" if "lost" in statuses else "within" if all(v == "within" for v in statuses) else "unresolved"

    sel, best_val, per_eval = None, None, []
    for r in evals:
        fam = r.get("by_family_half") or {}
        if r["iteration"] == 0 or not fam:
            continue
        selection = {f: status(c["selection"]) for f, c in fam.items() if c.get("selection")}
        reserved = {f: status(c["reserved"]) for f, c in fam.items() if c.get("reserved")}
        ok = all(v != "lost" for v in selection.values())
        val = r["by_block"].get("validation", {}).get("change", 0.0)
        per_eval.append({"iteration": r["iteration"], "selection": selection, "reserved": reserved, "eligible": ok})
        if ok and (best_val is None or val > best_val):
            sel, best_val = r, val
    if sel is None:
        return {"class": "fails", "task_class": "fails", "retention_verdict": None, "joint": "fails",
                "selected_iteration": None, "test_change_at_selected": None, "retained_at_selected": None,
                "per_evaluation": per_eval, "margin": margin, "selection_half": half,
                "reserved_half_present": bool(per_eval and per_eval[0]["reserved"]), "untouched_block_present": False}
    fam = sel["by_family_half"]
    # the final reading: the untouched retained-task test block where the run has one (the
    # probe never decodes it), else the reserved half of the off-task block (which the
    # probe's scans and off-task decodes did touch, so it is reserved from checkpoint
    # selection only; the review of 2026-10-21)
    final_key = "test" if any(c.get("test") for c in fam.values()) else "reserved"
    final = {f: {**c[final_key], "status": status(c[final_key])} for f, c in fam.items() if c.get(final_key)}
    task_class = cls(sel["by_block"].get("test", {}).get("change", 0.0))
    rv = verdict([v["status"] for v in final.values()])
    joint = task_class if task_class != "works" else ("works" if rv == "within" else f"works on the task; retention {rv}")
    return {"class": task_class, "task_class": task_class, "retention_verdict": rv, "joint": joint,
            "selected_iteration": sel["iteration"],
            "validation_change_at_selected": best_val, "test_change_at_selected": sel["by_block"].get("test", {}).get("change"),
            "retained_at_selected": final, "retained_on": final_key,
            "retained_within_margin": rv == "within" if final else None,
            "per_evaluation": per_eval, "margin": margin, "selection_half": half, "reserved_half_present": bool(final),
            "untouched_block_present": final_key == "test"}


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

    # the reference arm of a setting: the recipe's learning rate and kl coefficient, seed 0
    grpo = {}
    for d in sorted(grpo_root.iterdir()):
        if not (d / "eval.jsonl").exists():
            continue
        c = json.loads((d / "run.json").read_text())["config"]
        if c["lr"] == probe.RL_LR and c.get("beta", grpo_run.BETA) == grpo_run.BETA and c["seed"] == 0:
            if c["setting"] in grpo:
                raise SystemExit(f"two reference arms for {c['setting']}: {grpo[c['setting']][0]}, {d.name}")
            a = grpo_run.analyze(d)
            grpo[c["setting"]] = (d.name, loop_seconds_to(a["eval"], lines(d / "log.jsonl"), threshold=threshold))
    rows = []
    for d in sorted(es_root.iterdir()):
        if not (d / "eval.jsonl").exists():
            continue
        a = analyze(d)
        es = loop_seconds_to(a["eval"], lines(d / "log.jsonl"), threshold=threshold)
        ref = grpo.get(a["setting"])
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


def samples_seen(out: Path) -> list:
    """Each evaluation of a run with the samples generated before it (the rollouts of the
    iterations before, from the log), the loop seconds they took (an ES update's, a GRPO
    step's phases; no evaluation or setup) and its normalized per-prompt rewards."""
    ev = sorted(lines(out / "eval.jsonl"), key=lambda r: r["iteration"])
    log = lines(out / "log.jsonl")
    rolls = {r["iteration"]: r["rollouts"] for r in log}
    secs = {r["iteration"]: sum(r["seconds"].values()) if isinstance(r["seconds"], dict) else r["seconds"]
            for r in log if "seconds" in r}
    return [{"iteration": r["iteration"], "samples": int(sum(v for i, v in rolls.items() if i < r["iteration"])),
             "seconds": float(sum(v for i, v in secs.items() if i < r["iteration"])),
             "normalized": probe.normalized(r["rewards"], r["kinds"]), "block": np.asarray(r["block"])} for r in ev]


def samples_compare(grpo_dir: Path, es_roots: list, tolerance: float = 0.02) -> list:
    """Data efficiency as the reward reached at equal samples seen (docs/end_to_end/10, S5):
    every ES run of the GRPO run's setting under `es_roots`, at each of its evaluations
    whose sample count a GRPO evaluation matches within `tolerance`, with each method's
    change on the test, validation and off-task blocks and the ES-minus-GRPO difference
    paired over prompts (the two runs score the same blocks, checked by digest), and who
    leads at two paired standard errors on the test block."""
    g_run = json.loads((grpo_dir / "run.json").read_text())
    g_ev = samples_seen(grpo_dir)
    rows = []
    for root in es_roots:
        for d in sorted(root.iterdir()):
            if not (d / "eval.jsonl").exists():
                continue
            run_ = json.loads((d / "run.json").read_text())
            if run_["config"]["setting"] != g_run["config"]["setting"]:
                continue
            if run_["block_digests"] != g_run["block_digests"]:
                raise SystemExit(f"{d.name} and {grpo_dir.name} score different blocks")
            e_ev = samples_seen(d)
            for e in e_ev[1:]:
                near = min(g_ev[1:], key=lambda x: abs(x["samples"] - e["samples"]))
                if abs(near["samples"] - e["samples"]) > tolerance * e["samples"]:
                    continue
                row = {"run": d.name, "samples": e["samples"], "es_iteration": e["iteration"],
                       "grpo_step": near["iteration"], "grpo_samples": near["samples"]}
                for b in ("test", "validation", "offtask"):
                    m = e["block"] == b
                    if not m.any():
                        continue
                    de = e["normalized"][m] - e_ev[0]["normalized"][m]
                    dg = near["normalized"][m] - g_ev[0]["normalized"][m]
                    diff = de - dg
                    row[f"es_{b}"], row[f"grpo_{b}"] = float(de.mean()), float(dg.mean())
                    row[f"diff_{b}"], row[f"diff_{b}_se"] = float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(diff.size))
                se = row["diff_test_se"]
                row["lead"] = "ES" if row["diff_test"] > 2 * se else "GRPO" if row["diff_test"] < -2 * se else "within 2 SE"
                rows.append(row)
    return rows


def samples_table(rows: list) -> str:
    cols = (("run", "ES run"), ("samples", "samples seen"), ("es_iteration", "ES update"), ("grpo_step", "GRPO step"),
            ("es_test", "ES test"), ("grpo_test", "GRPO test"), ("diff_test", "ES minus GRPO"), ("diff_test_se", "paired SE"),
            ("lead", "ahead"), ("es_offtask", "ES off-task"), ("grpo_offtask", "GRPO off-task"))

    def fmt(x):
        if x is None:
            return "-"
        if isinstance(x, float):
            return f"{x:+.3f}"
        return f"{x:,}" if isinstance(x, int) and x >= 1000 else str(x)

    head = "| " + " | ".join(n for _, n in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    return "\n".join([head, sep, *("| " + " | ".join(fmt(r.get(k)) for k, _ in cols) + " |" for r in rows)])


def history(runs: list, upto=None) -> list:
    """Every evaluation of the runs (GRPO or ES, the same blocks, checked by digest) on one
    axis of samples seen, up to `upto`, with the loop seconds before each: each block's
    change against the run's own evaluation at zero and the test change's standard error
    over prompts. `samples_compare`
    pairs only evaluations at equal counts; this keeps every point of each curve."""
    rows, digests = [], None
    for d in runs:
        run_ = json.loads((d / "run.json").read_text())
        if digests is None:
            digests = run_["block_digests"]
        elif run_["block_digests"] != digests:
            raise SystemExit(f"{d.name} scores other blocks than {runs[0].name}")
        ev = samples_seen(d)
        for e in ev:
            if upto is not None and e["samples"] > upto:
                continue
            row = {"samples": e["samples"], "loop_seconds": round(e["seconds"]), "run": d.name, "at": e["iteration"]}
            for b in ("test", "validation", "offtask"):
                m = e["block"] == b
                change = e["normalized"][m] - ev[0]["normalized"][m]
                row[b] = float(change.mean())
                if b == "test":
                    row["test_se"] = float(change.std(ddof=1) / np.sqrt(change.size)) if e["iteration"] else 0.0
            rows.append(row)
    return sorted(rows, key=lambda r: (r["samples"], r["run"]))


def history_table(rows: list) -> str:
    cols = (("samples", "samples seen"), ("loop_seconds", "loop seconds"), ("run", "run"), ("at", "update or step"), ("test", "test"),
            ("test_se", "SE"), ("validation", "validation"), ("offtask", "off-task"))

    def fmt(x):
        if isinstance(x, float):
            return f"{x:+.3f}" if x else "0"
        return f"{x:,}" if isinstance(x, int) and x >= 1000 else str(x)

    head = "| " + " | ".join(n for _, n in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    return "\n".join([head, sep, *("| " + " | ".join(fmt(r[k]) for k, _ in cols) + " |" for r in rows)])


def effective_updates(g: int, gamma: float) -> tuple:
    """After `g` updates with the weights pulled a fraction `gamma` back toward the start
    after each, the displacement is sum_i (1 - gamma)^(g - i + 1) u_i: the number of
    updates the first-order gain counts, and the number the random part's variance
    counts, for responses measured per unit update (before the pull-back). Both are `g`
    without shrinkage."""
    if not gamma:
        return float(g), float(g)
    q = 1.0 - gamma
    return q * (1.0 - q ** g) / gamma, q ** 2 * (1.0 - q ** (2 * g)) / (1.0 - q ** 2)


def effective_updates_from_first(g: int, gamma: float) -> tuple:
    """The same counts for responses measured at the applied step's first displacement,
    (1 - gamma) times the update, which already carries one pull-back: update i then
    enters with weight (1 - gamma)^(g - i), so the gain counts (1 - q^g) / gamma updates
    and the variance (1 - q^2g) / (1 - q^2). At one update both are 1, so the forecast is
    the measured change at the first displacement itself (the review of 2026-10-21: the
    frozen S4b forecasts used `effective_updates` on these responses and counted the
    first pull-back twice)."""
    if not gamma:
        return float(g), float(g)
    q = 1.0 - gamma
    return (1.0 - q ** g) / gamma, (1.0 - q ** (2 * g)) / (1.0 - q ** 2)


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
        # the retained families on the off-task block's selection and reserved halves (S4b),
        # and on the untouched retained-task test block where the run has one (the next
        # preregistration: a block the probe never decodes)
        off = np.flatnonzero(blocks == "offtask")
        if off.size:
            halves = {"selection": off[:RETAINED_SELECTION], "reserved": off[RETAINED_SELECTION:]}
            untouched = np.flatnonzero(blocks == "offtask_test")
            if untouched.size:
                halves["test"] = untouched
            fams = sorted(set(kinds[off]) | set(kinds[untouched]))
            row["by_family_half"] = {str(t): {h: {"change": float(d[i[kinds[i] == t]].mean()), "change_se": se(d[i[kinds[i] == t]]),
                                                  "prompts": int((kinds[i] == t).sum())}
                                               for h, i in halves.items() if (kinds[i] == t).any()}
                                     for t in fams}
        if pred:
            # a random ranking has the random cost and no gain (docs/end_to_end/10, S4a arm 2);
            # shrinkage after each update weights update i by (1 - gamma)^(g - i + 1), so the
            # first-order gain counts eff_g updates and the random part's variance eff_c
            real = run_["config"].get("ranking", "real") == "real"
            gamma = run_["config"].get("shrink") or 0.0
            eff_g, eff_c = effective_updates(g, gamma)
            v = (pred.get("prescription") or {}).get("response_at_first_displacement")
            if v:
                # S4b: the prescription's responses at the applied step's first displacement.
                # `predicted` keeps the frozen convention of the S4b prescriptions (the weights
                # of `effective_updates`, which count the first pull-back twice);
                # `predicted_corrected` uses the weights from exponent zero.
                gain = eff_g * v["odd"] if real else 0.0
                row["predicted"] = {"validation_change": gain + eff_c * v["even"], "gain_first_order": gain,
                                    "random_cost": -eff_c * v["even"], "effective_updates": [eff_g, eff_c],
                                    "offtask_change": None, "from": "prescription",
                                    "convention": "frozen (S4b): weights from exponent one on responses at the first displacement"}
                cg, cc = effective_updates_from_first(g, gamma)
                cgain = cg * v["odd"] if real else 0.0
                row["predicted_corrected"] = {"validation_change": cgain + cc * v["even"], "gain_first_order": cgain,
                                              "random_cost": -cc * v["even"], "effective_updates": [cg, cc],
                                              "convention": "weights from exponent zero (2026-10-21)"}
            else:
                gain = eff_g * step * pred["G"] if real else 0.0
                row["predicted"] = {"validation_change": gain - eff_c * step ** 2 * -pred["C"],
                                    "gain_first_order": gain, "random_cost": eff_c * step ** 2 * -pred["C"],
                                    "effective_updates": [eff_g, eff_c],
                                    "offtask_change": (eff_g * step * pred["G_offtask"] if real else 0.0) if pred["G_offtask"] is not None else None}
        rows.append(row)
    res = {"setting": run_["config"]["setting"], "seed": run_["config"]["seed"], "step": step, "n": run_["config"]["n"],
           "noise": run_["config"].get("noise", "rank-1"),
           "ranking": run_["config"].get("ranking", "real"), "shrink": run_["config"].get("shrink"),
           "iterations_done": len(log), "rollouts_total": sum(r["rollouts"] for r in log),
           "seconds_total": sum(r["seconds"] for r in log),
           "fitness": [{"iteration": r["iteration"], "mean": r["mean_fitness"], "sd": r["sd_fitness"],
                        "update_norm": r["update_norm"]} for r in log],
           "eval": rows, "prediction": pred, "observed": observed_class(rows) if rows else None,
           "retention": retention(rows) if rows else None}
    if pred and rows:
        last = rows[-1]
        val = last["by_block"].get("validation", {})
        res["at_end"] = {"iteration": last["iteration"], "observed_validation_change": val.get("change"),
                         "observed_validation_change_se": val.get("change_se"),
                         "observed_test_change": last["by_block"].get("test", {}).get("change"),
                         "predicted_validation_change": last["predicted"]["validation_change"],
                         "updates_to_target_predicted": pred["updates_to_target"]}
    return res


def variant_of(a: dict):
    """What a run changed against the prescription's population and ranking, or None."""
    parts = [x for x, on in (("full-rank", a["noise"] == "full-rank"), (f"N {a['n']}", a["n"] != probe.N_MEMBERS),
                             ("random ranking", a["ranking"] == "random"),
                             (f"shrink {a['shrink']}", bool(a["shrink"]))) if on]
    return ", ".join(parts) or None


def curves_table(evals: list, predicted: bool) -> str:
    """Per-evaluation changes by block, the answer length, and the first-order prediction
    where the run carried one: the tables of the results READMEs."""
    corrected = predicted and "predicted_corrected" in evals[0]
    cols = ["after", "validation", "test", "off-task", "answer length"] + ["predicted validation"] * predicted \
        + ["forecast, corrected weights"] * corrected
    rows = []
    for r in evals:
        b = r["by_block"]
        f = lambda k: f"{b[k]['change']:+.3f}" if k in b else "-"  # noqa: E731
        row = [str(r["iteration"]), f("validation"), f("test"), f("offtask"), f"{r['mean_len']:.0f}"]
        if predicted:
            row.append(f"{r['predicted']['validation_change']:+.3f}")
        if corrected:
            row.append(f"{r['predicted_corrected']['validation_change']:+.3f}")
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|", *rows])


def curves(root: Path) -> str:
    """Every run under `root`: a heading with what it ran, then its curves table."""
    out = []
    for d in sorted(root.iterdir()):
        if not (d / "eval.jsonl").exists():
            continue
        a = analyze(d)
        v = variant_of(a)
        out.append(f"**{d.name}** (step {a['step']:.3g}{', ' + v if v else ''}, {a['iterations_done']} updates, "
                   f"{a['rollouts_total']:,} rollouts, {a['seconds_total'] / 3600:.1f} h)\n\n"
                   + curves_table(a["eval"], "predicted" in a["eval"][0]))
    return "\n\n".join(out)


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
                     "variant": variant_of(a),
                     "call": (a["prediction"] or {}).get("call"), "updates_to_target_predicted": (a["prediction"] or {}).get("updates_to_target"),
                     "updates": a["iterations_done"], "rollouts": a["rollouts_total"],
                     "final_validation": fmt("validation"), "final_test": fmt("test"), "final_offtask": fmt("offtask"),
                     "peak_test": o["best_test_change"], "peak_at": o["at_iteration_ignoring_retained"],
                     "selected_at": o["selected_iteration"], "test_at_selected": o["test_change_at_selected"],
                     "class": o["class"], "class_by_test_peak": o["class_by_test_peak"],
                     "class_ignoring_retained": o["class_ignoring_retained"], "collapsed": o["collapsed"]})
    return rows


def table(rows: list) -> str:
    cols = (("run", "run"), ("call", "preregistered call"), ("step", "step"), ("variant", "variant"),
            ("updates_to_target_predicted", "updates to target, predicted"),
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
    ap.add_argument("--curves", type=Path, help="the runs directory: each run's per-evaluation table")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("ES_RUNS", "GRPO_RUNS"),
                    help="each ES run against its setting's GRPO arm: loop seconds and rollouts to +0.10 on the test block")
    ap.add_argument("--history", nargs="+", type=Path, metavar="DIR",
                    help="runs on the same blocks: every evaluation of each on one axis of samples seen")
    ap.add_argument("--upto", type=int, help="with --history, the last sample count shown")
    ap.add_argument("--samples", nargs="+", type=Path, metavar="DIR",
                    help="a GRPO run, then directories of ES runs: each ES run of its setting against it at equal samples seen")
    ap.add_argument("--setting", help="model/task, as es_vllm.probe")
    ap.add_argument("--prediction", type=Path, help="the setting's probe record (step and the numbers to hold the run against)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--analyze", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="member seeds; 0 shares the probe's first generation")
    ap.add_argument("--members", type=int, default=N)
    ap.add_argument("--full-rank", action="store_true",
                    help="es-at-scale's Countdown recipe: full-rank members at sigma 1e-3, alpha 5e-4, no anchor")
    ap.add_argument("--step", type=float, help="multiple of the unit update (default: the prediction's)")
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--prompts", choices=("repeated", "stream"), default="repeated")
    ap.add_argument("--ranking", choices=("real", "random"), default="real",
                    help="random: the fitnesses in a seeded random order, an update of the real norm with no direction")
    ap.add_argument("--shrink", type=float, default=None,
                    help="after each update, pull the weights this fraction of the way back toward the start "
                         "(default: the prescription's, else 0; --shrink 0 is the un-anchored control)")
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/probe-run"))
    ap.add_argument("--es-at-scale", type=Path, default=Path.home() / "private" / "open-source" / "es-at-scale")
    args = ap.parse_args(argv)
    if args.compare:
        es, gr = (p if p.is_absolute() else E2E / p for p in args.compare)
        print(compare_table(compare(es, gr)))
        return 0
    if args.history:
        print(history_table(history([p if p.is_absolute() else E2E / p for p in args.history], args.upto)))
        return 0
    if args.samples:
        if len(args.samples) < 2:
            ap.error("--samples needs the GRPO run and at least one directory of ES runs")
        gr, *es = (p if p.is_absolute() else E2E / p for p in args.samples)
        print(samples_table(samples_compare(gr, es)))
        return 0
    if args.table:
        print(table(summary(args.table if args.table.is_absolute() else E2E / args.table)))
        return 0
    if args.curves:
        print(curves(args.curves if args.curves.is_absolute() else E2E / args.curves))
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
