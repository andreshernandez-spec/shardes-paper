#!/usr/bin/env python
"""GRPO on Qiu et al.'s Countdown setup (docs/end_to_end/14-grpo-matched.md), beside S6's ES.

    python -m es_vllm.qiu_grpo --out runs/qiu-matched/qwen3b-grpo-s0 --ckpt /root/qiu-ckpt --workdir /root/grpo
    python -m es_vllm.qiu_grpo --out ... --ckpt ... --workdir ... --resume      # from the last full checkpoint
    python -m es_vllm.qiu_grpo --out ... --ckpt ... --workdir ... --smoke       # laptop: 0.5B, tiny shapes

His problems, prompts, training reward and test rule are qiu_match's (`his_setup`): the
first 200 problems of countdown.json for training, the last 2,000 for testing, the raw
prompt through the slow tokenizer, his `reward_function` on the decoded prompt and answer,
and the test decoded greedily with the model's repetition penalty, HF's padding included.

The GRPO recipe is grpo_run's (S5): temperature 1.0, GRPO advantages per prompt, two
mini-batches, micro-batches of 2, the clipped surrogate plus beta 0.01 times kl3 against
the start, AdamW at a constant 1e-6, gradient norm clip 1.0, at most 1,024 new tokens.
One change: 50 prompts and 15 samples a step (S5: 48 and 16), 750 answers, so that ES's
6,000 an update is exactly 8 steps and every ES test after 10m updates has a GRPO test
after 80m steps at the same number of answers. A step's 50 prompts are a quarter of a
seeded permutation of the 200, so every 4 steps see each problem 15 times.

The running test (eval.jsonl, qiu_match's format) every 80 steps; the policy's bf16
weights under `--ckpt` at the steps in SAVE, for his eval_countdown.py.
"""

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

PKG = Path(__file__).resolve().parent
E2E = PKG.parent
sys.path.insert(0, str(E2E))
sys.path.insert(0, str(E2E.parent))
import harness  # noqa: E402
from es_vllm import grpo_run as G  # noqa: E402
from es_vllm import qiu_match as Q  # noqa: E402
from provenance import env_block  # noqa: E402

P, K, LR, BETA = 50, 15, 1e-6, 0.01
STEPS, EVAL_EVERY, SAVE = 480, 80, (240, 480)
SAMPLE_SEED = 40_000
SMOKE = {"setting": "qwen0.5b/countdown", "train": 8, "test": 16, "test_batch": 8, "max_new": 256,
         "prompts": 4, "samples": 4, "steps": 4, "eval_every": 2, "save": (2, 4)}


def run(args) -> int:
    from vllm import SamplingParams  # noqa: PLC0415

    from es_vllm import probe  # noqa: PLC0415
    from es_vllm.pad_penalty import PadPenalty  # noqa: PLC0415
    from es_vllm.worker import ESWorker  # noqa: PLC0415

    s = {**SMOKE, "steps": args.steps if args.steps != STEPS else SMOKE["steps"]} if args.smoke else {"setting": "qwen3b/countdown", "train": Q.TRAIN, "test": Q.TEST,
                                  "test_batch": Q.TEST_BATCH, "max_new": Q.MAX_NEW, "prompts": P, "samples": K,
                                  "steps": args.steps, "eval_every": EVAL_EVERY, "save": SAVE}
    st = Q.his_setup(s, args.es_at_scale, args.smoke)
    if len(st.train) % s["prompts"]:
        raise SystemExit(f"{s['prompts']} prompts a step do not divide the {len(st.train)} problems")
    out, ckpt = E2E / args.out, Path(args.ckpt)
    state = G.checkpoint_path(args.workdir, out)
    conf = {"setting": s["setting"], "repo": st.cfg["repo"], "revision": st.cfg["revision"], "seed": args.seed,
            "prompts_per_step": s["prompts"], "samples_per_prompt": s["samples"], "minibatches": G.MINIBATCHES,
            "micro": G.MICRO, "lr": LR, "beta": BETA, "clip": G.CLIP, "max_grad_norm": G.MAX_GRAD_NORM,
            "steps": s["steps"], "eval_every": s["eval_every"], "save": list(s["save"]),
            "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.0},
            "temperature": 1.0, "max_new_tokens": s["max_new"], "train": s["train"], "test": s["test"],
            "test_batch": s["test_batch"], "repetition_penalty": st.rep, "stop_token_ids": st.eos,
            "pad_token": st.pad, "smoke": args.smoke}
    saved = None
    if (out / "log.jsonl").exists():
        if not args.resume:
            raise SystemExit(f"{out} holds a run; pass --resume to continue it from its checkpoint")
        if not state.exists():
            raise SystemExit(f"{out} holds a run but there is no checkpoint at {state}")
        saved = G.load_checkpoint(state)
        bad = G.incompatible(saved["config"], conf)
        if bad:
            raise SystemExit(f"the checkpoint's configuration differs on {bad}")
    elif args.resume:
        raise SystemExit(f"{out} holds no run to resume")
    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    env = env_block(E2E, ["runs"], ("vllm", "torch", "transformers"))
    if saved is None:
        run_ = {"config": conf, "data": st.data, "started": now, "env": env}
    else:
        run_ = json.loads((out / "run.json").read_text())
        if run_["data"] != st.data:
            raise SystemExit(f"{out} ran on other data")
        run_.setdefault("resumed", []).append({"at_step": saved["step"], "time": now, "env": env})
    harness.write_atomic(out / "run.json", run_)
    log, e_log, c_log = out / "log.jsonl", out / "eval.jsonl", out / "ckpt.jsonl"

    def note_checkpoint(step, result):
        run_.setdefault("checkpoints", []).append({"step": step, **({"failed": result} if isinstance(result, str)
                                                                     else {"seconds": round(result, 1)})})
        harness.write_atomic(out / "run.json", run_)
        print(f"checkpoint at {step} steps: {result if isinstance(result, str) else f'{result:.0f} s'}", flush=True)

    grader = Q.Grader(st.clone, st.tok)                 # before CUDA: the pool forks
    try:
        for k_, fn in G.worker_methods().items():
            setattr(ESWorker, k_, fn)
        llm = probe.engine(st.cfg, args.gpu, worker_extension_cls="es_vllm.worker.ESWorker",
                           logits_processors=[PadPenalty])
        rpc = lambda m, *a: llm.collective_rpc(m, args=a)[0]  # noqa: E731
        model, ref, opt = G.policy(st.cfg, LR)
        rng = np.random.default_rng(args.seed)
        order, start = [], 0
        if saved is not None:
            model.load_state_dict(saved["model"])
            opt.load_state_dict(saved["opt"])
            rng.bit_generator.state = saved["rng"]
            order, start = list(saved["order"]), int(saved["step"])
            del saved
            dropped = [G.truncate_records(log, start), G.truncate_records(e_log, start),
                       G.truncate_records(c_log, start + 1)]   # a save at `start` holds the resumed weights
            print(f"resumed at {start} steps (dropped {dropped} log, eval and save records, "
                  f"{G.sync_engine(rpc, model)} weights loaded)", flush=True)

        def evaluate(g):
            if g % s["eval_every"] and g != s["steps"]:
                return
            t1 = time.perf_counter()
            sc, n_tok, fin = Q.decode(llm, st.ids["test"], st.params("test"), grader, st.test)
            Q.append(e_log, Q.test_record(g, sc, n_tok, fin, time.perf_counter() - t1))
            print(f"test after {g} steps: accuracy {Q.accuracy(sc):.4f} mean length {np.mean(n_tok):.0f}", flush=True)

        def save(g):
            """The policy's bf16 weights as an HF directory, the engine's own copy hashed beside."""
            if g not in s["save"]:
                return
            from safetensors.torch import save_file  # noqa: PLC0415

            d = ckpt / f"ckpt-{g}"
            if d.exists():
                shutil.rmtree(d)
            d.mkdir()
            for f in Q.TOKENIZER_FILES:
                shutil.copy(st.files / f, d / f)
            save_file({n: w.cpu().contiguous() for n, w in G.bf16_weights(model).items()}, str(d / "model.safetensors"),
                      metadata={"format": "pt"})
            rec = {"iteration": g, "dir": d.name}
            harness.write_atomic(d / "saved.json", rec)  # the evaluator watches for this
            Q.append(c_log, rec)
            print(f"saved after {g} steps: {d}", flush=True)

        train = st.train
        for g in range(start, s["steps"]):
            evaluate(g)
            t, t0 = {}, time.perf_counter()
            if not order:                               # a seeded pass over the 200, then another
                order = rng.permutation(len(train)).tolist()
            idx, order = order[:s["prompts"]], order[s["prompts"]:]
            rows, ids = [train[i] for i in idx], [st.ids["train"][i] for i in idx]
            sp = SamplingParams(n=s["samples"], temperature=1.0, top_p=1.0, max_tokens=s["max_new"],
                                stop_token_ids=st.eos, seed=SAMPLE_SEED + 1000 * g)
            outs = probe.decode(llm, ids, sp)
            t["generate"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            seqs = [ids[j] + list(o.token_ids) for j, out_ in enumerate(outs) for o in out_.outputs]
            grades = grader.score(seqs, [r for r in rows for _ in range(s["samples"])])
            rewards = np.array([x[0] for x in grades]).reshape(len(rows), s["samples"])
            t["score"] = time.perf_counter() - t0
            plens = [len(ids[j]) for j in range(len(rows)) for _ in range(s["samples"])]
            A = G.advantages(rewards).reshape(-1)
            losses, kls, gnorms, clipped = G.update(model, ref, opt, rng, seqs, plens, A, BETA, t)
            t0 = time.perf_counter()
            n_loaded = G.sync_engine(rpc, model)
            t["sync"] = time.perf_counter() - t0
            flat = [o for out_ in outs for o in out_.outputs]
            Q.append(log, {"iteration": g, "problems": idx, "rewards": rewards.tolist(),
                           "answers": [x[1] for x in grades], "mean_reward": float(rewards.mean()),
                           "solved": float(np.mean([x[1] > 0 for x in grades])),
                           "live_prompts": int((rewards.std(axis=1) > 0).sum()),
                           "mean_len": float(np.mean([len(o.token_ids) for o in flat])),
                           "stopped": float(np.mean([o.finish_reason == "stop" for o in flat])),
                           "loss": losses, "kl": kls, "grad_norm": gnorms, "clipped_share": clipped,
                           "rollouts": len(seqs), "weights_loaded": n_loaded, "timeouts": grader.timeouts,
                           "seconds": t})
            print(f"[{g}] reward {rewards.mean():.4f} solved {np.mean([x[1] > 0 for x in grades]):.3f} "
                  f"live {int((rewards.std(axis=1) > 0).sum())}/{len(rows)} kl {kls[-1]:.2e} |g| {gnorms[-1]:.3f} "
                  + " ".join(f"{k} {v:.0f}s" for k, v in t.items()), flush=True)
            save(g + 1)
            if args.checkpoint_every and (g + 1) % args.checkpoint_every == 0:
                note_checkpoint(g + 1, G.save_checkpoint(state, g + 1, model, opt, rng, order, conf))
        evaluate(s["steps"])
    except BaseException:  # noqa: BLE001
        import traceback  # noqa: PLC0415

        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        grader.close()
        os._exit(1)                                     # the pool's workers would otherwise keep the process alive
    grader.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="the run's records, under experiments/end_to_end")
    ap.add_argument("--ckpt", type=Path, required=True, help="bf16 weights for his evaluator, outside the repository")
    ap.add_argument("--workdir", type=Path, required=True, help="the full training state, for --resume")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--checkpoint-every", type=int, default=80)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--gpu", type=float, default=0.35)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--es-at-scale", type=Path, default=Path.home() / "private" / "open-source" / "es-at-scale")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
