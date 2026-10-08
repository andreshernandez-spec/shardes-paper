#!/usr/bin/env python
"""Does one ES update deliver the gain its members' ranking implies?

    python -m es_vllm.step_check --out runs/step-check/start.json   # on the GPU
    python -m es_vllm.step_check --analyze runs/step-check/start.json
    python -m es_vllm.step_check --predict      # the predictions, from runs/grad-check/
    python -m es_vllm.step_check --out ... --smoke                  # wiring

The gradient probe (`grad_check.py`) implies a first-order gain of about 0.021 reward
points per update from the arm's iteration-0 ranking; the arm beat its random-walk control
by 0.0022 per update over 120 (`docs/end_to_end/07`), a tenth. On Countdown T2 realized a
third (runs/grad-check/README.md). Whether ES on Tulu needs about RL's rollouts or about a
hundred times more turns on where that factor goes, so this measures one update directly.

The iteration-0 update `u = theta_1 - theta_0` (what `es_tell` makes, alpha = sigma =
5e-4, N = 16, seed 0) is built from four rankings of the same 16 members: the arm's logged
fitness (192 prompts), the gradient probe's 768-prompt means, the random control's logged
fitness, and the contrastive probe's fitness (`contrastive_check.py`). For each, the start
is moved to `theta_0 + lambda u` and `theta_0 - lambda u` at lambda 4 and 10 and decoded
greedily on 4,608 prompts: the gradient probe's 768 (RL steps 1 to 16) and RL steps 121 to
160 and 481 to 520 (3,840, held out from every ranking). Half the difference of the two
signs is the update's odd part, to first order `lambda` times its gain: the cost of its
random part is even in lambda and cancels. Their mean minus the start is the even part,
to second order `lambda^2` times that cost. At lambda 4 the update is a member's length.

Prediction, committed before the run, per unit update (lambda = 1) on the held-out prompts,
from runs/grad-check/start.json: first-order gain +0.012 for the arm's ranking (its members'
changes on the 576 prompts it did not rank on), +0.03 for the 768-prompt ranking (split
halves; the implied-gain formula gives 0.036), -0.024 for the control's (the same members
with random weights, on all 768); each of these about +-0.02 from the members' own noise.
Even part -0.0038 (the members' mean change over N). If first order holds, the 768
ranking's odd part at lambda 10 is about +0.3; if an update realizes a tenth, about +0.03.
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
REPO = E2E.parent.parent
sys.path.insert(0, str(E2E))
sys.path.insert(0, str(E2E.parent))
import harness  # noqa: E402
import releases as R  # noqa: E402
from provenance import env_block  # noqa: E402

N, P_RUN, SEED, SIGMA, ALPHA = 16, 192, 0, 5e-4, 5e-4
LAMBDAS = (4.0, 10.0)
HELDOUT = ((120, 160), (480, 520))


def rankings() -> dict:
    """Member fitness for each ranking, in member order (higher is better)."""
    first = lambda path: json.loads((E2E / path).read_text().splitlines()[0])["fitness"]  # noqa: E731
    g = json.loads((E2E / "runs/grad-check/start.json").read_text())
    c = json.loads((E2E / "runs/contrastive-check/start.json").read_text())
    from es_vllm.contrastive_check import objective  # noqa: PLC0415

    f0 = objective(c, c["points"][0]["logp"])
    contrast = [float((objective(c, p["logp"]) - f0).mean()) for p in c["points"] if p["kind"] == "member"]
    return {"arm": first("runs/tulu-long-s5e-4/log.jsonl"),
            "probe768": np.asarray(g["members"]["0.0005"]).mean(1).tolist(),
            "control": first("runs/tulu-long-random/log.jsonl"),
            "contrast": contrast}


def worker_methods():
    import jax.numpy as jnp  # noqa: PLC0415
    from shardes.shaping import group_relative  # noqa: PLC0415

    from es_vllm import stream  # noqa: PLC0415

    def updated(self, rewards, lam):
        """Leaf by leaf, `es_tell`'s expression for the master after an update from these
        rewards, with the step scaled by lam: lam = 1 is `es_tell`'s master to the bit."""
        weights = group_relative((-jnp.asarray(rewards, dtype=jnp.float32))[:, None])
        ids = jnp.arange(self._es_n, dtype=jnp.int32)
        for k, name in enumerate(self._es_names):
            yield name, stream.updated_leaf(self._es_master[name], self._es_streams[k], ids, weights,
                                            float(lam) * self._es_lr, self._es_sigma, self._es_n)

    def sc_norm(self, rewards):
        return float(np.sqrt(sum(float(jnp.sum((u - self._es_master[n]) ** 2))
                                 for n, u in updated(self, rewards, 1.0))))

    def sc_along(self, rewards, lam):
        self._load((n, u.astype(stream.COMPUTE)) for n, u in updated(self, rewards, lam))

    return {"sc_norm": sc_norm, "sc_along": sc_along}


def collect(args) -> dict:
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from es_vllm.run_tulu import Verifier, heldout_rows, load_prompts, render  # noqa: PLC0415
    from es_vllm.worker import ESWorker  # noqa: PLC0415

    for k, fn in worker_methods().items():
        setattr(ESWorker, k, fn)
    ranks = rankings()
    if args.smoke:
        repo, revision, cap = "Qwen/Qwen2.5-0.5B-Instruct", None, 64
        rows = [r for b in load_prompts(E2E / "data" / "tulu31", 1, P_RUN) for r in b][:24]
        sets = {"probe": len(rows)}
    else:
        repo, revision, cap = R.TULU31_START.repo, R.TULU31_START.commit, 2048
        probe = [r for b in load_prompts(E2E / "data" / "tulu31", 4, P_RUN) for r in b]
        held = [r for a, b in HELDOUT for r in heldout_rows(a, b)]
        rows, sets = probe + held, {"probe": len(probe), "heldout": len(held)}
    llm = LLM(model=repo, revision=revision, dtype="bfloat16", seed=0,
              gpu_memory_utilization=0.4 if args.smoke else 0.5, max_model_len=4096,
              enable_prefix_caching=False, worker_extension_cls="es_vllm.worker.ESWorker")
    rpc = lambda m, *a: llm.collective_rpc(m, args=a)[0]  # noqa: E731
    model_dir = snapshot_download(repo, revision=revision, allow_patterns=["*.safetensors", "*.json"])
    rpc("es_init", model_dir, N, SIGMA, ALPHA * SIGMA, SEED)
    rpc("es_ask")  # the arm's iteration-0 members
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    ids = [render(tok, r) for r in rows]
    verifier = Verifier(REPO / ".venv-verify" / "bin" / "python")
    params = SamplingParams(temperature=0.0, max_tokens=cap)

    def decode():
        outs = []
        for c in range(0, len(ids), P_RUN):
            outs += llm.generate([{"prompt_token_ids": i} for i in ids[c:c + P_RUN]], params,
                                 use_tqdm=False)
        rewards = verifier.score([{"text": o.outputs[0].text, "ground_truth": r["ground_truth"],
                                   "dataset": r["dataset"],
                                   "stopped": o.outputs[0].finish_reason == "stop"}
                                  for o, r in zip(outs, rows)])
        if len(rewards) != len(rows):  # the work asked for is the work done
            raise SystemExit(f"{len(rewards)} rewards for {len(rows)} prompts")
        return [float(x) for x in rewards], float(np.mean([len(o.outputs[0].token_ids) for o in outs]))

    t0 = time.perf_counter()
    rec: dict = {"sets": sets, "datasets": [r["dataset"] for r in rows], "rankings": ranks,
                 "points": []}

    def point(kind, **kw):
        r, mean_len = decode()
        rec["points"].append({"kind": kind, **kw, "mean_len": mean_len, "rewards": r})
        print(f"{kind} {kw}: {np.mean(r):.3f} {time.perf_counter() - t0:.0f}s", flush=True)

    point("start")
    for name, fit in ranks.items():
        if len(fit) != N:
            raise SystemExit(f"{name}: {len(fit)} fitness values")
        norm = rpc("sc_norm", fit)
        for lam in LAMBDAS:
            for sign in (1, -1):
                rpc("sc_along", fit, sign * lam)
                point("step", ranking=name, lam=sign * lam, update_norm=norm)
    # lambda = 1 with the arm's ranking is the arm's first update: the engine must then hold
    # what `es_tell` makes of the master, to the bit
    rpc("sc_along", ranks["arm"], 1.0)
    rpc("es_tell", ranks["arm"])
    rec["lambda1_is_es_tell"] = rpc("es_check", None)["ok"]
    if not rec["lambda1_is_es_tell"]:
        raise SystemExit("theta_0 + u is not es_tell's update")
    verifier.close()
    rec["seconds"] = time.perf_counter() - t0
    return rec


def analyze(rec: dict) -> dict:
    """Odd and even parts of each update's effect, per unit update, by prompt set."""
    pts = rec["points"]
    start = np.asarray(pts[0]["rewards"])
    n_probe = rec["sets"]["probe"]
    sets = {"probe768": slice(0, n_probe), "heldout": slice(n_probe, None)}
    out = {"start": {k: float(start[s].mean()) for k, s in sets.items()},
           "mean_len_start": pts[0]["mean_len"], "rankings": {}}
    for name in rec["rankings"]:
        res = {}
        for lam in sorted({abs(p["lam"]) for p in pts if p.get("ranking") == name}):
            plus = np.asarray(next(p for p in pts if p.get("ranking") == name and p["lam"] == lam)["rewards"])
            minus = np.asarray(next(p for p in pts if p.get("ranking") == name and p["lam"] == -lam)["rewards"])
            row = {}
            for k, s in sets.items():
                odd = (plus[s] - minus[s]) / 2
                even = (plus[s] + minus[s]) / 2 - start[s]
                if odd.size == 0:
                    continue
                row[k] = {"odd_per_unit": float(odd.mean() / lam),
                          "odd_per_unit_se": float(odd.std(ddof=1) / np.sqrt(odd.size) / lam),
                          "even_per_unit2": float(even.mean() / lam ** 2),
                          "even_per_unit2_se": float(even.std(ddof=1) / np.sqrt(even.size) / lam ** 2),
                          "plus": float(plus[s].mean()), "minus": float(minus[s].mean())}
            res[str(lam)] = row
        out["rankings"][name] = res
    return out


def predict(draws=1000) -> dict:
    """The first-order gains the gradient probe predicts for these rankings, per unit update,
    and how much of each comes from the odd part of the members' effects (what changes sign
    with the noise, `(f(+eps) - f(-eps)) / 2`, from the mirrored members), the only part an
    update can inherit; the even part (`(f(+eps) + f(-eps)) / 2 - f`) cancels in it."""
    from es_vllm.grad_check import variance_parts  # noqa: PLC0415

    g = json.loads((E2E / "runs/grad-check/start.json").read_text())
    c = np.asarray(g["center"])
    plus = np.asarray(g["members"]["0.0005"]) - c[None]
    minus = np.asarray(g["mirrored"]["0.0005"]) - c[None]
    odd, even = (plus - minus) / 2, (plus + minus) / 2
    z = lambda x: (x - x.mean()) / x.std()  # noqa: E731
    ranks = rankings()
    rng = np.random.default_rng(0)
    splits = [rng.permutation(plus.shape[1]) for _ in range(draws)]
    half_n = plus.shape[1] // 2
    v = {k: variance_parts(x) for k, x in (("plus", plus), ("odd", odd), ("even", even))}
    # a ranking on all 768 prompts: the same covariance over a smaller observed spread
    scale = (np.sqrt(v["plus"]["v_member"] + v["plus"]["v_resid"] / half_n)
             / np.sqrt(v["plus"]["v_member"] + v["plus"]["v_resid"] / plus.shape[1]))

    def judged(members, x):
        """A ranking on one half of the prompts, judged on the other, over the splits."""
        return float(np.mean([np.mean(z(plus[members][:, q[:half_n]].mean(1))
                                      * x[members][:, q[half_n:]].mean(1)) for q in splits]))

    parts = {}
    for k, x in (("total", plus), ("odd", odd), ("even", even)):
        full = judged(np.arange(N), x)
        loo = np.asarray([judged(np.delete(np.arange(N), i), x) for i in range(N)])
        parts[k] = {"ranking_on_384_judged_on_384": full, "ranking_on_768": full * scale,
                    "ranking_on_768_se": float(np.sqrt((N - 1) / N * ((loo - loo.mean()) ** 2).sum()))
                    * scale}  # jackknife over members
    return {
        "variance_of_member_effects": {k: {"true": x["v_member"], "residual": x["v_resid"]}
                                       for k, x in v.items()},
        "gain": parts,
        "arm_on_its_other_576": float(np.mean(z(np.asarray(ranks["arm"])) * plus[:, 192:].mean(1))),
        "arm_odd_on_its_other_576": float(np.mean(z(np.asarray(ranks["arm"])) * odd[:, 192:].mean(1))),
        "control_on_768": float(np.mean(z(np.asarray(ranks["control"])) * plus.mean(1))),
        "control_odd_on_768": float(np.mean(z(np.asarray(ranks["control"])) * odd.mean(1))),
        "even_per_unit2": float(plus.mean()) / N}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--analyze", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--predict", action="store_true", help="the gradient probe's predictions")
    args = ap.parse_args(argv)
    if args.predict:
        print(json.dumps(predict(), indent=2))
        return 0
    if args.analyze:
        print(json.dumps(analyze(json.loads((E2E / args.analyze).read_text())), indent=2))
        return 0
    if args.out is None:
        ap.error("--out or --analyze")
    out = E2E / args.out
    if out.exists():
        raise SystemExit(f"{out} exists")
    rec = collect(args)
    harness.write_atomic(out, {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "env": env_block(E2E, ["runs"], ("vllm", "torch", "jax", "jaxlib")),
        "smoke": args.smoke, "seed": SEED, "sigma": SIGMA, "alpha": ALPHA, "n": N, **rec})
    print(json.dumps(analyze(rec), indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
