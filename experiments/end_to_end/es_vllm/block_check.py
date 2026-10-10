#!/usr/bin/env python
"""Per matrix type: how much of an ES update's gain and of its cost each type carries.

    python -m es_vllm.block_check --out runs/block-check          # on the GPU, resumable
    python -m es_vllm.block_check --analyze runs/block-check
    python -m es_vllm.block_check --out ... --smoke                 # wiring

An ES update spreads its random part by parameter count: 28% each to the up, gate and down
projections, 8% each to q and o, 2% each to k and v. GRPO's exact gradient is spread
otherwise (runs/gradient-check/: down 41%, v 28%, o 12%, up 10%, gate 5%, q 2%, k 0.6% of
the matrices' squared norm). If the update's gain is spread like the gradient and its cost
like the parameters, stepping each type by its own multiple of the update raises the best
net per update, `sum_t G_t^2 / (4 C_t)` against `(sum_t G_t)^2 / (4 sum_t C_t)`, by up to
about 5; if the curvature per parameter tracks the gradient's density, by nothing. This is
the one reshaping of the update that prior knowledge could justify (docs/end_to_end/08).

Measured: the N = 1,024 update `u` of runs/lowrank-check/ restricted to one matrix type at
a time, `u_t`, applied as `theta_0 +- mu u_t / |u_t|` at two lengths per type on the 3,840
held-out prompts of RL steps 121 to 160 and 481 to 520, with the full update at the same
lengths on the same engine as the reference. The lengths are scaled by type so that the
perturbation per element stays within 2.5 times the full update's at 111.7 (a type holding
2% of the parameters takes a quarter of the length). Per type, the odd part per unit
length is the slope `a_t` along the type's component and the even part per unit length
squared its curvature `b_t`; the unit update's gain and cost from that type are
`G_t = a_t |u_t|` and `C_t = b_t |u_t|^2`, whose sums should match the full update's.

Prediction, committed before the run: v_proj's share of the gain exceeds its share of the
parameters by at least five times; the reweighted best net is between 1.5 and 3 times the
uniform one; the sums of `G_t` and of `C_t` match the full update's within two standard
errors. The run that follows (docs/end_to_end/09) uses the multipliers only if the
measured ratio is at least 1.3.
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
import releases as R  # noqa: E402
from es_vllm import lowrank  # noqa: E402
from es_vllm.lowrank_check import Verifiers, items  # noqa: E402
from provenance import env_block  # noqa: E402

SEED, SIGMA, ALPHA, RANK, P_RUN, N = 0, 5e-4, 5e-4, 1, 192, 1024
HELDOUT = ((120, 160), (480, 520))
# lengths of mu u_t / |u_t| by type: the full update's scan lengths for the three large
# types, scaled down where a type's parameters are fewer, so the per-element size stays
# within 2.5x of the full update's at 111.7
LENGTHS = {"full": (44.7, 111.7), "down_proj": (44.7, 111.7), "up_proj": (44.7, 111.7),
           "gate_proj": (44.7, 111.7), "q_proj": (32.0, 80.0), "o_proj": (32.0, 80.0),
           "k_proj": (22.0, 56.0), "v_proj": (22.0, 56.0)}
GRADIENT_SHARES = {"down_proj": 0.414, "v_proj": 0.280, "o_proj": 0.124, "up_proj": 0.104,
                   "gate_proj": 0.052, "q_proj": 0.021, "k_proj": 0.006}   # runs/gradient-check/


def setting(smoke: bool) -> dict:
    if smoke:
        return {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "revision": None, "cap": 48, "n": 8,
                "heldout": 16, "verifiers": 2, "gpu": 0.4,
                "lengths": {k: (v[0] / 10,) for k, v in LENGTHS.items()}}
    return {"repo": R.TULU31_START.repo, "revision": R.TULU31_START.commit, "cap": 2048, "n": N,
            "heldout": None, "verifiers": 8, "gpu": 0.5, "lengths": LENGTHS}


def lines(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def append(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def fitness(smoke: bool, n: int) -> list:
    if smoke:
        return np.random.default_rng(0).normal(size=n).tolist()
    mem = json.loads((E2E / "runs/lowrank-check/members.json").read_text())
    return np.asarray(mem["rewards"]).mean(1).tolist()


def collect(args, cfg) -> None:
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from es_vllm.run_tulu import heldout_rows, load_prompts, render  # noqa: PLC0415
    from es_vllm.worker import ESWorker  # noqa: PLC0415

    out = E2E / args.out
    out.mkdir(parents=True, exist_ok=True)
    for k_, fn in lowrank.worker_methods().items():
        setattr(ESWorker, k_, fn)
    n = cfg["n"]
    k = list(lowrank.coefficients(fitness(args.smoke, n), n, ALPHA, RANK))   # the unit update
    model_dir = snapshot_download(cfg["repo"], revision=cfg["revision"],
                                  allow_patterns=["*.safetensors", "*.json"])
    shapes = lowrank.leaf_shapes(model_dir)
    llm = LLM(model=cfg["repo"], revision=cfg["revision"], dtype="bfloat16", seed=0,
              gpu_memory_utilization=cfg["gpu"], max_model_len=4096, enable_prefix_caching=False,
              max_num_seqs=512, worker_extension_cls="es_vllm.worker.ESWorker")
    rpc = lambda m, *a: llm.collective_rpc(m, args=a)[0]  # noqa: E731
    rpc("lr_setup", model_dir, SEED, RANK, n // 2, shapes)
    tok = AutoTokenizer.from_pretrained(cfg["repo"], revision=cfg["revision"])
    if args.smoke:
        rows = [r for b in load_prompts(E2E / "data" / "tulu31", 2, P_RUN) for r in b][P_RUN:][: cfg["heldout"]]
    else:
        rows = [r for a, b in HELDOUT for r in heldout_rows(a, b)]
    ids = [render(tok, r) for r in rows]
    vs = Verifiers(cfg["verifiers"])
    greedy = SamplingParams(temperature=0.0, max_tokens=cfg["cap"])
    log = out / "points.jsonl"
    done = {(r["kind"], r["length"], r["sign"]) for r in lines(log)}

    # the update's norm by type, and the parameters by type
    norms, params = {}, {}
    for t in ("full",) + lowrank.KINDS:
        rpc("lr_scale", None if t == "full" else {kk: (1.0 if kk == t else 0.0) for kk in lowrank.KINDS})
        norms[t] = rpc("lr_norm", k)
        params[t] = int(sum(o * i for nm, (o, i) in shapes if t == "full" or lowrank.kind(nm) == t))
    harness.write_atomic(out / "norms.json", {"n": n, "norms": norms, "params": params,
                                              "lengths": cfg["lengths"]})
    print(json.dumps({"norms": norms}), flush=True)
    t0 = time.perf_counter()

    def point(t, length, sign):
        if (t, length, sign) in done:
            return
        rpc("lr_scale", None if t == "full" else {kk: (1.0 if kk == t else 0.0) for kk in lowrank.KINDS})
        lam = sign * length / norms[t]
        rpc("lr_load", [lam * x for x in k])
        if not rpc("lr_check")["ok"]:
            raise SystemExit(f"engine weights are not the update: {t} {length} {sign}")
        outs = llm.generate([{"prompt_token_ids": i} for i in ids], greedy, use_tqdm=False)
        r = vs.score(items(outs, rows))
        append(log, {"kind": t, "length": length, "sign": sign, "lam": lam, "rewards": r,
                     "mean_len": float(np.mean([len(o.outputs[0].token_ids) for o in outs]))})
        print(f"{t} length {length} sign {sign:+}: {np.mean(r):.3f} {time.perf_counter() - t0:.0f}s",
              flush=True)

    point("full", 0.0, 0)
    for t in ("full",) + lowrank.KINDS:        # full first, then by the gradient's share
        for length in cfg["lengths"][t]:
            for sign in (1, -1):
                point(t, length, sign)
    rpc("lr_scale", None)
    rpc("lr_load", [0.0] * len(k))
    vs.close()


def analyze(out: Path) -> dict:
    meta = json.loads((out / "norms.json").read_text())
    pts = lines(out / "points.jsonl")
    start = np.asarray(next(r for r in pts if r["kind"] == "full" and r["sign"] == 0)["rewards"])
    res: dict = {"start": float(start.mean()), "by_type": {}}
    for t in ("full",) + lowrank.KINDS:
        rows = {}
        for length in meta["lengths"][t]:
            plus = next((r for r in pts if r["kind"] == t and r["length"] == length and r["sign"] == 1), None)
            minus = next((r for r in pts if r["kind"] == t and r["length"] == length and r["sign"] == -1), None)
            if plus is None or minus is None:
                continue
            p, m = np.asarray(plus["rewards"]), np.asarray(minus["rewards"])
            odd, even = (p - m) / 2, (p + m) / 2 - start
            rows[str(length)] = {
                "plus": float(p.mean()), "minus": float(m.mean()),
                "odd": float(odd.mean()), "odd_se": float(odd.std(ddof=1) / np.sqrt(odd.size)),
                "even": float(even.mean()), "even_se": float(even.std(ddof=1) / np.sqrt(even.size)),
                "a": float(odd.mean() / length), "a_se": float(odd.std(ddof=1) / np.sqrt(odd.size) / length),
                "b": float(even.mean() / length ** 2),
                "b_se": float(even.std(ddof=1) / np.sqrt(even.size) / length ** 2)}
        if not rows:
            continue
        far = rows[max(rows, key=float)]
        norm = meta["norms"][t]
        res["by_type"][t] = {"norm": norm, "params": meta["params"][t],
                             "param_share": meta["params"][t] / meta["params"]["full"],
                             "lengths": rows,
                             # the unit update's gain and cost from this type, from the
                             # longer length (smaller errors), with the shorter as a check
                             "G": far["a"] * norm, "G_se": far["a_se"] * norm,
                             "C": far["b"] * norm ** 2, "C_se": far["b_se"] * norm ** 2}
    kinds = [t for t in lowrank.KINDS if t in res["by_type"]]
    if "full" in res["by_type"] and kinds:
        full = res["by_type"]["full"]
        G = {t: res["by_type"][t]["G"] for t in kinds}
        C = {t: res["by_type"][t]["C"] for t in kinds}
        sum_g, sum_c = sum(G.values()), sum(C.values())
        uniform = full["G"] ** 2 / (4 * -full["C"]) if full["C"] < 0 else None
        parts = {t: G[t] ** 2 / (4 * -C[t]) for t in kinds if C[t] < 0 and G[t] > 0}
        reweighted = sum(parts.values())
        res["sums"] = {"G": sum_g, "G_full": full["G"], "G_full_se": full["G_se"],
                       "C": sum_c, "C_full": full["C"], "C_full_se": full["C_se"]}
        res["gain_share"] = {t: G[t] / sum_g for t in kinds} if sum_g else None
        res["cost_share"] = {t: C[t] / sum_c for t in kinds} if sum_c else None
        res["gradient_share"] = GRADIENT_SHARES
        res["best_net"] = {"uniform": uniform, "reweighted": reweighted,
                           "ratio": reweighted / uniform if uniform else None,
                           "types_with_positive_gain": sorted(parts)}
        if uniform:
            lam_u = full["G"] / (2 * -full["C"])
            res["multipliers"] = {t: (G[t] / (2 * -C[t])) / lam_u if C[t] < 0 and G[t] > 0 else 0.0
                                  for t in kinds}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--analyze", type=Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.analyze:
        print(json.dumps(analyze(E2E / args.analyze), indent=2))
        return 0
    if args.out is None:
        ap.error("--out or --analyze")
    out = E2E / args.out
    out.mkdir(parents=True, exist_ok=True)
    env = out / "env.json"
    if not env.exists():
        env.write_text(json.dumps({"date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                                   "smoke": args.smoke, "seed": SEED, "sigma": SIGMA, "alpha": ALPHA,
                                   "env": env_block(E2E, ["runs"], ("vllm", "torch"))}))
    collect(args, setting(args.smoke))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
