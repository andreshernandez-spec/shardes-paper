#!/usr/bin/env python
"""GRPO trained by us on a probed setting: the measured RL reference (docs/end_to_end/10, S3c).

    python -m es_vllm.grpo_run --setting qwen3b/countdown --out runs/grpo-run/qwen3b-countdown-lr5e-7-s0 --steps 40
    python -m es_vllm.grpo_run --analyze runs/grpo-run/qwen3b-countdown-lr5e-7-s0
    (--smoke: tiny blocks, two steps)

The recipe is the Tulu 3.1 run's (its model card, open-instruct 3f37c29): 48 prompts and
16 samples per step at temperature 1.0, GRPO advantages `(score - group mean) / (group sd
+ 1e-8)`, two mini-batches per batch ("half-m"), micro-batches of 2 with the token mean of
`-A ratio` (clipped at 0.2) plus `beta` times the kl3 estimator against the start, beta
0.01, AdamW at a constant 5e-7, one epoch, no penalty for an unfinished answer. Here on
one GPU: the policy in f32 under gradient checkpointing, a bf16 copy of the start as the
reference, vLLM generating from the policy's weights written into it in bf16 after every
step. Prompts cycle through the probe's 192 train prompts in a seeded order, the prompts
the ES runs repeated. Every `eval_every` steps the validation, reserved test and off-task
blocks are decoded greedily by the current policy, as `probe_run.py` does for ES. Each
phase is timed, so the record carries RL's measured cost per step and per answer, which
the rule's ratios have estimated from the Tulu 8B bench until now.

Not implemented from the recipe: DeepSpeed, several actors, the evaluation dataset; the
optimizer's betas (0.9, 0.95), eps 1e-8, no weight decay and a gradient norm clip of 1.0
are this driver's choices, recorded in run.json.
"""

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import datetime  # noqa: E402
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
from es_vllm import probe, probe_run  # noqa: E402
from provenance import env_block  # noqa: E402

PROMPTS, SAMPLES, MINIBATCHES, MICRO = 48, 16, 2, 2       # the Tulu run: 8 prompts x 6 actors, nspp 16, half-m
LR, BETA, CLIP, MAX_GRAD_NORM = 5e-7, 0.01, 0.2, 1.0
STEPS, EVAL_EVERY = 40, 5
SAMPLE_SEED = 40_000


def lines(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def append(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def advantages(rewards) -> np.ndarray:
    """GRPO's, per prompt over its samples (torch.std's correction, open-instruct 3f37c29)."""
    r = np.asarray(rewards, dtype=float)
    sd = r.std(axis=1, ddof=1)
    return (r - r.mean(axis=1, keepdims=True)) / (sd[:, None] + 1e-8)


def kl3(logp, ref):
    """open-instruct's kl3 estimator per token: exp(ref - logp) - (ref - logp) - 1, at least 0."""
    d = ref - logp
    return d.exp() - d - 1


def worker_methods():
    import torch  # noqa: PLC0415

    def gr_sync(self):
        """The engine's weights become the tensors the trainer put in `worker.SHARED`."""
        from es_vllm import worker as W  # noqa: PLC0415

        weights = W.SHARED.pop("weights")
        loaded = self.model_runner.model.load_weights(weights=list(weights.items()))
        torch.cuda.synchronize()
        return len(loaded) if loaded is not None else -1

    return {"gr_sync": gr_sync}


def logprobs(model, seqs, plens, idx, autocast):
    """log p of each predicted token for the sequences `idx`, right-padded without a
    padding mask (causal attention keeps the real tokens exact; see probe.py's gradient
    phase), with the mask of response positions."""
    import torch  # noqa: PLC0415

    L = max(len(seqs[i]) for i in idx)
    x = torch.zeros((len(idx), L), dtype=torch.long)
    mask = torch.zeros((len(idx), L), dtype=torch.bool)
    for r, i in enumerate(idx):
        x[r, :len(seqs[i])] = torch.tensor(seqs[i])
        mask[r, plens[i]:len(seqs[i])] = True
    x, mask = x.cuda(), mask.cuda()
    if autocast:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x).logits
    else:
        logits = model(input_ids=x).logits
    logp = torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, x[:, 1:, None]).squeeze(-1)
    return logp, mask[:, 1:]


def run(args) -> int:
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    from es_vllm import worker as W  # noqa: PLC0415
    from es_vllm.worker import ESWorker  # noqa: PLC0415

    cfg = probe.setting(args.setting, args.smoke)
    pred = probe_run.prediction_block(E2E / args.prediction) if args.prediction else None
    task = probe.Task(cfg["task"], cfg["template"], args.smoke, args.es_at_scale,
                      block=pred["blocks"]["validation"] if pred and not args.smoke else None)
    P, K, steps = (8, 4, 2) if args.smoke else (args.prompts, args.samples, args.steps)
    eval_every = 1 if args.smoke else args.eval_every
    out = E2E / args.out
    if (out / "log.jsonl").exists():
        raise SystemExit(f"{out} holds a run; this driver does not resume")
    out.mkdir(parents=True, exist_ok=True)
    conf = {"setting": args.setting, "repo": cfg["repo"], "revision": cfg["revision"], "seed": args.seed,
            "prompts_per_step": P, "samples_per_prompt": K, "minibatches": MINIBATCHES, "micro": MICRO,
            "lr": args.lr, "beta": args.beta, "clip": CLIP, "max_grad_norm": MAX_GRAD_NORM, "steps": steps,
            "eval_every": eval_every, "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.0},
            "temperature": 1.0, "smoke": args.smoke}
    if pred is not None and not args.smoke and pred["block_digests"] != task.digests():
        raise SystemExit("the blocks are not the probe's")
    harness.write_atomic(out / "run.json", {
        "config": conf, "prediction": pred, "blocks": {b: len(v) for b, v in task.blocks.items()},
        "block_digests": task.digests(),
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "env": env_block(E2E, ["runs"], ("vllm", "torch", "transformers"))})
    log, e_log = out / "log.jsonl", out / "eval.jsonl"

    scorer = task.scorer()                                   # before CUDA: the Countdown grader forks
    for k_, fn in worker_methods().items():
        setattr(ESWorker, k_, fn)
    llm = probe.engine(cfg, 0.35, worker_extension_cls="es_vllm.worker.ESWorker")   # first: it profiles the GPU
    rpc = lambda m, *a: llm.collective_rpc(m, args=a)[0]  # noqa: E731
    tok = probe.tokenizer(cfg)
    model = AutoModelForCausalLM.from_pretrained(cfg["repo"], revision=cfg["revision"], dtype=torch.float32,
                                                 attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    ref = AutoModelForCausalLM.from_pretrained(cfg["repo"], revision=cfg["revision"], dtype=torch.bfloat16,
                                               attn_implementation="sdpa").cuda().eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    rng = np.random.default_rng(args.seed)
    train = task.blocks["train"]
    order = []

    ev_rows, ev_kinds, ev_block = [], [], []
    for b in ("validation", "test", "offtask"):
        ev_rows += task.blocks[b]
        ev_kinds += [probe.row_kind(r) for r in task.blocks[b]]
        ev_block += [b] * len(task.blocks[b])
    ev_ids = [task.render(tok, r) for r in ev_rows]

    def evaluate(g):
        if g % eval_every and g != steps:
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
        print(f"eval after {g} steps: " + " ".join(f"{b} {norm[bl == b].mean():.3f}" for b in ("validation", "test", "offtask") if (bl == b).any()), flush=True)

    def sync():
        W.SHARED["weights"] = {n: p.detach().to(torch.bfloat16) for n, p in model.named_parameters()
                               if not (tied and n == "lm_head.weight")}
        n = rpc("gr_sync")
        W.SHARED.clear()
        torch.cuda.synchronize()
        return n

    for g in range(steps):
        evaluate(g)
        t = {}
        t0 = time.perf_counter()
        if len(order) < P:                                   # a seeded pass over the train block, then another
            order += rng.permutation(len(train)).tolist()
        rows, order = [train[i] for i in order[:P]], order[P:]
        ids = [task.render(tok, r) for r in rows]
        outs = probe.decode(llm, ids, task.params(rows, n=K, temperature=1.0, seed=SAMPLE_SEED + 1000 * g))
        flat = [o for out_ in outs for o in out_.outputs]
        t["generate"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        rewards = np.asarray(scorer.score([o.text for o in flat], [r for r in rows for _ in range(K)],
                                          [o.finish_reason == "stop" for o in flat])).reshape(P, K)
        t["score"] = time.perf_counter() - t0
        adv = advantages(rewards)
        seqs = [ids[j] + list(o.token_ids) for j, out_ in enumerate(outs) for o in out_.outputs]
        plens = [len(ids[j]) for j in range(P) for _ in range(K)]
        A = adv.reshape(-1)
        n_seq = len(seqs)

        # the sampling policy's log-probabilities, before any update of this step
        t0 = time.perf_counter()
        old = [None] * n_seq
        with torch.no_grad():
            for b0 in range(0, n_seq, MICRO):
                idx = list(range(b0, min(n_seq, b0 + MICRO)))
                lp, m = logprobs(model, seqs, plens, idx, autocast=True)
                for r, i in enumerate(idx):
                    old[i] = lp[r][m[r]].detach().clone()
        torch.cuda.synchronize()
        t["old_logprobs"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        perm = rng.permutation(n_seq)
        losses, kls, gnorms, clipped = [], [], [], []
        for mb in np.array_split(perm, MINIBATCHES):
            micro_batches = [mb[i:i + MICRO].tolist() for i in range(0, len(mb), MICRO)]
            tot_loss = tot_kl = tot_clip = 0.0
            for idx in micro_batches:
                lp, m = logprobs(model, seqs, plens, idx, autocast=True)
                with torch.no_grad():
                    rp, _ = logprobs(ref, seqs, plens, idx, autocast=False)
                a = torch.tensor([A[i] for i in idx], device="cuda", dtype=torch.float32)[:, None]
                old_t = torch.zeros_like(lp)
                for r, i in enumerate(idx):
                    old_t[r][m[r]] = old[i]
                ratio = (lp - old_t).exp()
                pg = -torch.minimum(a * ratio, a * ratio.clamp(1 - CLIP, 1 + CLIP))
                kl = kl3(lp, rp)
                tokens = m.sum()
                loss = ((pg + args.beta * kl) * m).sum() / tokens
                (loss / len(micro_batches)).backward()
                tot_loss += float(loss) / len(micro_batches)
                tot_kl += float((kl * m).sum() / tokens) / len(micro_batches)
                tot_clip += float((((ratio - 1).abs() > CLIP) & m).sum() / tokens) / len(micro_batches)
                del lp, rp, ratio, pg, kl, loss
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM))
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(tot_loss); kls.append(tot_kl); gnorms.append(gn); clipped.append(tot_clip)
        torch.cuda.synchronize()
        t["train"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        n_loaded = sync()
        t["sync"] = time.perf_counter() - t0
        append(log, {"iteration": g, "rewards": rewards.tolist(), "mean_reward": float(rewards.mean()),
                     "live_prompts": int((rewards.std(axis=1) > 0).sum()),
                     "mean_len": float(np.mean([len(o.token_ids) for o in flat])),
                     "stopped": float(np.mean([o.finish_reason == "stop" for o in flat])),
                     "loss": losses, "kl": kls, "grad_norm": gnorms, "clipped_share": clipped,
                     "tokens": int(sum(len(s) for s in seqs)), "rollouts": n_seq, "weights_loaded": n_loaded,
                     "seconds": t, "prompt_digest": probe_run.digest(rows)})
        print(f"[{g}] reward {rewards.mean():.4f} live {int((rewards.std(axis=1) > 0).sum())}/{P} loss {losses[-1]:+.4f} "
              f"kl {kls[-1]:.2e} |g| {gnorms[-1]:.3f} " + " ".join(f"{k} {v:.0f}s" for k, v in t.items()), flush=True)
    evaluate(steps)
    scorer.close()
    return 0


def analyze(out: Path) -> dict:
    """The curves by block, the frozen class, the measured cost, and the rule's RL-side
    prediction for the setting beside them."""
    run_ = json.loads((out / "run.json").read_text())
    ev = sorted(lines(out / "eval.jsonl"), key=lambda r: r["iteration"])
    log = lines(out / "log.jsonl")
    base = probe.normalized(ev[0]["rewards"], ev[0]["kinds"])
    kinds, blocks = np.asarray(ev[0]["kinds"]), np.asarray(ev[0]["block"])
    rows = []
    for r in ev:
        d = probe.normalized(r["rewards"], r["kinds"]) - base
        g = r["iteration"]
        se = lambda x: float(x.std(ddof=1) / np.sqrt(x.size)) if g else 0.0  # noqa: E731
        rows.append({"iteration": g, "mean_len": r["mean_len"],
                     "by_block": {b: {"reward": float(probe.normalized(r["rewards"], r["kinds"])[blocks == b].mean()),
                                      "change": float(d[blocks == b].mean()), "change_se": se(d[blocks == b])}
                                  for b in ("validation", "test", "offtask") if (blocks == b).any()},
                     "by_task": {t: {"change": float(d[kinds == t].mean()), "change_se": se(d[kinds == t])} for t in sorted(set(kinds))}})
    phases = {k: float(np.mean([r["seconds"][k] for r in log])) for k in (log[0]["seconds"] if log else {})}
    step_s = float(np.mean([sum(r["seconds"].values()) for r in log])) if log else None
    rollouts = sum(r["rollouts"] for r in log)
    pred = run_.get("prediction") or {}
    return {"setting": run_["config"]["setting"], "seed": run_["config"]["seed"], "lr": run_["config"]["lr"],
            "beta": run_["config"].get("beta", BETA), "steps_done": len(log), "rollouts_total": rollouts,
            "reward": [{"iteration": r["iteration"], "mean": r["mean_reward"], "live": r["live_prompts"], "kl": r["kl"][-1],
                        "grad_norm": r["grad_norm"][-1], "mean_len": r["mean_len"]} for r in log],
            "cost": {"seconds_per_step": step_s, "seconds_per_phase": phases,
                     "gpu_seconds_per_answer": step_s / log[0]["rollouts"] if log else None,
                     "training_tokens_per_step": float(np.mean([r["tokens"] for r in log])) if log else None},
            "eval": rows, "observed": probe_run.observed_class(rows) if rows else None,
            "rule_rl_side": {"rollouts_per_step_reference": probe.RL_ROLLOUTS_PER_STEP, "lr_reference": probe.RL_LR,
                             "prediction_source": pred.get("source"), "ratio_es_over_rl_predicted": pred.get("ratio_es_over_rl_rollouts")}}


def summary(root: Path) -> list:
    """One row per arm under `root`: the blocks' changes, the step the test block first
    reached the target, the class, the measured cost."""
    rows = []
    for d in sorted(root.iterdir()):
        if not (d / "eval.jsonl").exists():
            continue
        a = analyze(d)
        last, o = a["eval"][-1], a["observed"]
        first = next((r["iteration"] for r in a["eval"] if r["by_block"].get("test", {}).get("change", 0) >= probe_run.CLASS_WORKS), None)
        fmt = lambda b: f"{last['by_block'][b]['change']:+.3f}" if b in last["by_block"] else "-"  # noqa: E731
        rows.append({"run": d.name, "setting": a["setting"], "lr": a["lr"], "beta": a["beta"], "seed": a["seed"],
                     "steps": a["steps_done"], "rollouts": a["rollouts_total"],
                     "final_validation": fmt("validation"), "final_test": fmt("test"), "final_offtask": fmt("offtask"),
                     "peak_test": o["best_test_change"], "peak_at": o["at_iteration_ignoring_retained"], "target_first_at": first,
                     "selected_at": o["selected_iteration"], "test_at_selected": o["test_change_at_selected"],
                     "class": o["class"], "collapsed": o["collapsed"],
                     "s_per_step": a["cost"]["seconds_per_step"], "gpu_s_per_answer": a["cost"]["gpu_seconds_per_answer"]})
    return rows


def curves(root: Path) -> str:
    """Every arm under `root`: a heading with its recipe, then its curves table."""
    out = []
    for d in sorted(root.iterdir()):
        if not (d / "eval.jsonl").exists():
            continue
        a = analyze(d)
        hours = (a["cost"]["seconds_per_step"] or 0.0) * a["steps_done"] / 3600
        out.append(f"**{d.name}** (lr {a['lr']:g}, beta {a['beta']:g}, seed {a['seed']}, {a['steps_done']} steps, "
                   f"{a['rollouts_total']:,} rollouts, {hours:.1f} h in the loop)\n\n" + probe_run.curves_table(a["eval"], False))
    return "\n\n".join(out)


def table(rows: list) -> str:
    cols = (("run", "arm"), ("lr", "lr"), ("beta", "beta"), ("seed", "seed"), ("steps", "steps"), ("rollouts", "rollouts"),
            ("final_validation", "validation at end"),
            ("final_test", "test at end"), ("final_offtask", "off-task at end"), ("peak_test", "peak test gain"), ("peak_at", "at"),
            ("target_first_at", "test +0.10 first at"), ("selected_at", "validation-selected checkpoint"),
            ("test_at_selected", "test there"), ("class", "class"), ("collapsed", "collapsed"),
            ("s_per_step", "s per step"), ("gpu_s_per_answer", "GPU-s per answer"))

    def fmt(x):
        if x is None:
            return "-"
        if isinstance(x, bool):
            return "yes" if x else "no"
        if isinstance(x, float):
            return f"{x:.3g}"
        return f"{x:,}" if isinstance(x, int) and x >= 1000 else str(x)

    head = "| " + " | ".join(n for _, n in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    return "\n".join([head, sep, *("| " + " | ".join(fmt(r[k]) for k, _ in cols) + " |" for r in rows)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, help="the arms' directory: one row per arm")
    ap.add_argument("--curves", type=Path, help="the arms' directory: each arm's per-evaluation table")
    ap.add_argument("--setting", help="model/task, as es_vllm.probe")
    ap.add_argument("--prediction", type=Path, help="the setting's probe record, to check the blocks and carry its RL-side numbers")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--analyze", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--beta", type=float, default=BETA, help="the kl3 coefficient; 0 removes the anchor to the start")
    ap.add_argument("--prompts", type=int, default=PROMPTS)
    ap.add_argument("--samples", type=int, default=SAMPLES)
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--es-at-scale", type=Path, default=Path.home() / "private" / "open-source" / "es-at-scale")
    args = ap.parse_args(argv)
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
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
