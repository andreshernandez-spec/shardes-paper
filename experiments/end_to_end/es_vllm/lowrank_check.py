#!/usr/bin/env python
"""Does an ES update's gain-to-cost ratio grow with the number of members? Low-rank, mirrored.

    python -m es_vllm.lowrank_check --phase members --out runs/lowrank-check/members.json
    python -m es_vllm.lowrank_check --phase steps --members runs/lowrank-check/members.json \
        --out runs/lowrank-check/steps.json
    python -m es_vllm.lowrank_check --analyze runs/lowrank-check/steps.json
    (--smoke on both phases: Qwen 0.5B, 8 members, wiring only)

One dense ES update on Tulu gains 0.004 +- 0.003 reward points per unit and pays 0.0038 for
its random part (runs/step-check/); at 16 members the two cancel. The random part's cost
falls as 1/N while the gain should not depend on N, so the best net per update,
`gain^2 / (4 cost)`, should grow in proportion to N (Andres, 2026-10-09: hundreds of
members, low-rank, as EGGROLL). This measures it at N = 128, 512 and 1,024.

Members (`es_vllm/lowrank.py`): rank-1 perturbations `sigma a b^T` of every attention and
MLP matrix, sigma 5e-4, in mirrored pairs, served as LoRA adapters built in memory from
their seeds. Phase `members`: the start and 1,024 members decode the arm's 192
iteration-0 prompts greedily; a check scores the start's answers to 32 prompts under the
base and members 0 and 1. Phase `steps`, on an engine without LoRA: the update from the
first N members (N = 128, 512, 1,024, nested) is merged into the weights at
`theta_0 + lambda u` and `theta_0 - lambda u`, with lambda set so that `lambda u` has the
dense probe's lengths (44.7 and 111.7), and decoded on the 3,840 held-out prompts of RL
steps 121 to 160 and 481 to 520. The odd part per unit is the gain, the even part per unit
squared the cost (as `step_check.py`). The check is repeated with members 0 and 1 merged
into the weights: their change in log p must match the adapters', or the adapters are not
the perturbation the update assumes.

Prediction, committed before the run. Cost per unit squared is the members' mean change
times `|u_N|^2 / |sigma E|^2`, about 1/N of the dense -0.0038 x 16; the numbers are
computed from phase `members` and written to `predictions.json` before any point is
scored. The gain per unit does not fall with N; at N = 1,024 it is measured to about
+-0.0005, and if it is near the dense 0.004 the best net per update is about 0.07 there,
against 0.001 for the dense 16 and RL's 0.034 per four steps early in its run. If the gain
at N = 1,024 is below 0.001, more members do not help: the ranking on 192 prompts carries
no first-order signal that transfers.
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
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

PKG = Path(__file__).resolve().parent
E2E = PKG.parent
REPO = E2E.parent.parent
sys.path.insert(0, str(E2E))
sys.path.insert(0, str(E2E.parent))
import harness  # noqa: E402
import releases as R  # noqa: E402
from es_vllm import lowrank  # noqa: E402
from provenance import env_block  # noqa: E402

SEED, SIGMA, ALPHA, RANK, P_RUN = 0, 5e-4, 5e-4, 1, 192
NS = (128, 512, 1024)
LENGTHS = (44.7, 111.7)        # the dense probe's |lambda u| at lambda 4 and 10
HELDOUT = ((120, 160), (480, 520))
CHUNK, CHECK, VERIFIERS = 64, 32, 8


def setting(smoke: bool) -> dict:
    if smoke:
        return {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "revision": None, "cap": 64, "ns": (4, 8),
                "prompts": 16, "heldout": 24, "chunk": 4, "check": 4, "verifiers": 2}
    return {"repo": R.TULU31_START.repo, "revision": R.TULU31_START.commit, "cap": 2048,
            "ns": NS, "prompts": P_RUN, "heldout": None, "chunk": CHUNK, "check": CHECK,
            "verifiers": VERIFIERS}


class Verifiers:
    """Several verify servers, each scoring a contiguous slice, in parallel."""

    def __init__(self, n):
        from es_vllm.run_tulu import Verifier  # noqa: PLC0415

        self.vs = [Verifier(REPO / ".venv-verify" / "bin" / "python") for _ in range(n)]

    def score(self, items) -> list:
        size = -(-len(items) // len(self.vs))
        parts = [items[i * size:(i + 1) * size] for i in range(len(self.vs))]
        with ThreadPoolExecutor(len(self.vs)) as ex:
            res = list(ex.map(lambda vp: vp[0].score(vp[1]) if vp[1] else [], zip(self.vs, parts)))
        out = [float(x) for r in res for x in r]
        if len(out) != len(items):  # the work asked for is the work done
            raise SystemExit(f"{len(out)} rewards for {len(items)} answers")
        return out

    def close(self):
        for v in self.vs:
            v.close()


def items(outs, rows) -> list:
    return [{"text": o.outputs[0].text, "ground_truth": r["ground_truth"], "dataset": r["dataset"],
             "stopped": o.outputs[0].finish_reason == "stop"} for o, r in zip(outs, rows)]


def logp_sums(llm, ids, answers, lora=None) -> list:
    """Teacher-forced log p of each fixed answer, summed over its tokens."""
    from vllm import SamplingParams  # noqa: PLC0415

    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    seqs = [p + a for p, a in zip(ids, answers)]
    kw = {"lora_request": [lora] * len(seqs)} if lora is not None else {}
    res = llm.generate([{"prompt_token_ids": s} for s in seqs], params, use_tqdm=False, **kw)
    out = []
    for p, s, o in zip(ids, seqs, res):
        out.append(float(sum(o.prompt_logprobs[pos][s[pos]].logprob for pos in range(len(p), len(s)))))
    return out


def phase_members(args, cfg) -> dict:
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    from vllm.lora.request import LoRARequest  # noqa: PLC0415

    from es_vllm.run_tulu import load_prompts, render  # noqa: PLC0415

    n = max(cfg["ns"])
    rows = load_prompts(E2E / "data" / "tulu31", 1, P_RUN)[0][: cfg["prompts"]]
    model_dir = snapshot_download(cfg["repo"], revision=cfg["revision"],
                                  allow_patterns=["*.safetensors", "*.json"])
    shapes = lowrank.leaf_shapes(model_dir)
    template = lowrank.write_template(Path(args.workdir) / "adapter", RANK, cfg["repo"])
    lowrank.install_adapters(template, SEED, shapes, RANK, SIGMA)
    llm = LLM(model=cfg["repo"], revision=cfg["revision"], dtype="bfloat16", seed=0,
              enable_lora=True, max_lora_rank=RANK, max_loras=cfg["chunk"], max_cpu_loras=n,
              gpu_memory_utilization=0.4 if args.smoke else 0.85, max_model_len=4096,
              enable_prefix_caching=False, max_num_seqs=512)
    tok = AutoTokenizer.from_pretrained(cfg["repo"], revision=cfg["revision"])
    ids = [render(tok, r) for r in rows]
    vs = Verifiers(cfg["verifiers"])
    greedy = SamplingParams(temperature=0.0, max_tokens=cfg["cap"])
    lora = lambda m: LoRARequest(f"member{m}", m + 1, str(template))  # noqa: E731
    t0 = time.perf_counter()

    outs = llm.generate([{"prompt_token_ids": i} for i in ids], greedy, use_tqdm=False)
    start = vs.score(items(outs, rows))
    answers = [list(o.outputs[0].token_ids) for o in outs[: cfg["check"]]]
    print(f"start {np.mean(start):.3f} {time.perf_counter() - t0:.0f}s", flush=True)

    rewards, lens = [None] * n, [None] * n

    def collect(job):
        ms, fut, ln = job
        r = fut.result()
        for k, m in enumerate(ms):
            rewards[m] = r[k * len(rows):(k + 1) * len(rows)]
            lens[m] = ln[k]

    with ThreadPoolExecutor(1) as ex:
        pending = None
        for c0 in range(0, n, cfg["chunk"]):
            ms = list(range(c0, min(n, c0 + cfg["chunk"])))
            outs = llm.generate([{"prompt_token_ids": i} for _ in ms for i in ids], greedy,
                                lora_request=[lora(m) for m in ms for _ in ids], use_tqdm=False)
            ln = [float(np.mean([len(o.outputs[0].token_ids) for o in outs[k * len(ids):(k + 1) * len(ids)]]))
                  for k in range(len(ms))]
            if pending is not None:
                collect(pending)
            pending = (ms, ex.submit(vs.score, items(outs, rows * len(ms))), ln)
            print(f"members {c0}-{ms[-1]} decoded {time.perf_counter() - t0:.0f}s", flush=True)
        collect(pending)
    vs.close()
    check = {"base": logp_sums(llm, ids[: cfg["check"]], answers),
             **{f"member{m}": logp_sums(llm, ids[: cfg["check"]], answers, lora(m)) for m in (0, 1)}}
    fit = np.asarray(rewards).mean(1)
    print(f"members: mean {fit.mean():.3f} sd {fit.std():.3f} vs start {np.mean(start):.3f}", flush=True)
    return {"phase": "members", "repo": cfg["repo"], "revision": cfg["revision"], "n": n,
            "rank": RANK, "sigma": SIGMA, "seed": SEED, "prompts": len(rows),
            "datasets": [r["dataset"] for r in rows], "leaves": len(shapes),
            "perturbed_params": int(sum(o * i for _, (o, i) in shapes)),
            "start": start, "rewards": rewards, "mean_len": lens, "check": check,
            "check_answers": answers, "seconds": time.perf_counter() - t0}


def predictions(mem: dict, norms: dict, member_norm: float, ns) -> dict:
    """What phase `steps` should find, from the members alone (written before it scores)."""
    r = np.asarray(mem["rewards"])
    start = np.asarray(mem["start"])
    mbar = float(r.mean() - start.mean())
    rng = np.random.default_rng(0)
    out = {"members_mean_change": mbar, "member_norm": member_norm, "by_n": {}}
    for n in ns:
        odd = (r[0:n:2] - r[1:n:2]) / 2                     # pairs x prompts
        gains = []
        for _ in range(2000):  # a ranking on half the prompts, judged on the other half
            q = rng.permutation(r.shape[1])
            a, b = q[: len(q) // 2], q[len(q) // 2:]
            k = lowrank.coefficients(r[:n, a].mean(1), n, ALPHA, RANK)
            # pair j's odd effect is sigma / sqrt(r) times the gradient along a_j b_j^T
            gains.append(float(np.sum(k * odd[:, b].mean(1)) * np.sqrt(RANK) / SIGMA))
        out["by_n"][str(n)] = {
            "unit_norm": norms[n], "lambdas": [L / norms[n] for L in LENGTHS],
            "cost_per_unit2": mbar * norms[n] ** 2 / member_norm ** 2,
            "gain_per_unit_half_ranking": float(np.mean(gains))}
    return out


def phase_steps(args, cfg) -> dict:
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from es_vllm.run_tulu import heldout_rows, load_prompts, render  # noqa: PLC0415
    from es_vllm.worker import ESWorker  # noqa: PLC0415

    mem = json.loads((E2E / args.members).read_text())
    for k, fn in lowrank.worker_methods().items():
        setattr(ESWorker, k, fn)
    n_max = max(cfg["ns"])
    if mem["n"] != n_max or mem["repo"] != cfg["repo"]:
        raise SystemExit("the members record is for another setting")
    model_dir = snapshot_download(cfg["repo"], revision=cfg["revision"],
                                  allow_patterns=["*.safetensors", "*.json"])
    shapes = lowrank.leaf_shapes(model_dir)
    if args.smoke:
        rows = [r for b in load_prompts(E2E / "data" / "tulu31", 2, P_RUN) for r in b][P_RUN:][: cfg["heldout"]]
    else:
        rows = [r for a, b in HELDOUT for r in heldout_rows(a, b)]
    llm = LLM(model=cfg["repo"], revision=cfg["revision"], dtype="bfloat16", seed=0,
              gpu_memory_utilization=0.4 if args.smoke else 0.5, max_model_len=4096,
              enable_prefix_caching=False, max_num_seqs=512,
              worker_extension_cls="es_vllm.worker.ESWorker")
    rpc = lambda m, *a: llm.collective_rpc(m, args=a)[0]  # noqa: E731
    pairs = n_max // 2
    rpc("lr_setup", model_dir, SEED, RANK, pairs, shapes)
    tok = AutoTokenizer.from_pretrained(cfg["repo"], revision=cfg["revision"])
    zeros = [0.0] * pairs
    t0 = time.perf_counter()

    # the members merged into the weights: same change in log p as their adapters?
    p_ids = [render(tok, r) for r in load_prompts(E2E / "data" / "tulu31", 1, P_RUN)[0][: len(mem["check_answers"])]]
    one = lambda m: [lowrank.sign(m) * SIGMA / np.sqrt(RANK) if j == m // 2 else 0.0  # noqa: E731
                     for j in range(pairs)]
    merged = {"base": logp_sums(llm, p_ids, mem["check_answers"])}
    for m in (0, 1):
        rpc("lr_load", one(m))
        if not rpc("lr_check")["ok"]:
            raise SystemExit(f"engine weights are not member {m} merged")
        merged[f"member{m}"] = logp_sums(llm, p_ids, mem["check_answers"])
    member_norm = float(np.mean([rpc("lr_norm", one(m)) for m in (0, 2, 4, 6)]))

    fitness = np.asarray(mem["rewards"]).mean(1)
    units = {n: list(lowrank.coefficients(fitness, n, ALPHA, RANK)) + [0.0] * (pairs - n // 2)
             for n in cfg["ns"]}
    norms = {n: rpc("lr_norm", units[n]) for n in cfg["ns"]}
    pred = predictions(mem, norms, member_norm, cfg["ns"])
    harness.write_atomic(Path(args.workdir) / "predictions.json", pred)  # before any point
    print(json.dumps(pred, indent=1), flush=True)

    ids = [render(tok, r) for r in rows]
    vs = Verifiers(cfg["verifiers"])
    greedy = SamplingParams(temperature=0.0, max_tokens=cfg["cap"])
    rec: dict = {"phase": "steps", "repo": cfg["repo"], "rank": RANK, "sigma": SIGMA,
                 "alpha": ALPHA, "seed": SEED, "ns": list(cfg["ns"]), "heldout": len(rows),
                 "datasets": [r["dataset"] for r in rows], "check_merged": merged,
                 "check_adapters": mem["check"], "predictions": pred, "points": []}

    def point(kind, **kw):
        outs = llm.generate([{"prompt_token_ids": i} for i in ids], greedy, use_tqdm=False)
        r = vs.score(items(outs, rows))
        rec["points"].append({"kind": kind, **kw, "rewards": r,
                              "mean_len": float(np.mean([len(o.outputs[0].token_ids) for o in outs]))})
        print(f"{kind} {kw}: {np.mean(r):.3f} {time.perf_counter() - t0:.0f}s", flush=True)

    rpc("lr_load", zeros)
    point("start")
    for n in cfg["ns"]:
        for lam in pred["by_n"][str(n)]["lambdas"]:
            for s in (1, -1):
                rpc("lr_load", [s * lam * k for k in units[n]])
                if not rpc("lr_check")["ok"]:
                    raise SystemExit(f"engine weights are not the update at N {n}, lambda {s * lam}")
                point("step", n=n, lam=s * lam, update_norm=norms[n])
    rpc("lr_load", zeros)
    if not rpc("lr_check")["ok"]:
        raise SystemExit("engine weights are not the start after the steps")
    vs.close()
    rec["seconds"] = time.perf_counter() - t0
    return rec


def analyze(rec: dict, mem: dict | None = None) -> dict:
    """Per N: the odd part per unit (gain) and even part per unit squared (cost) on the held
    out prompts, against the predictions, and the best net per update `gain^2 / (4 cost)`."""
    from es_vllm.grad_check import variance_parts  # noqa: PLC0415

    pts = rec["points"]
    start = np.asarray(pts[0]["rewards"])
    out = {"start_heldout": float(start.mean()), "by_n": {}}
    for n in rec["ns"]:
        p = rec["predictions"]["by_n"][str(n)]
        res = {"unit_norm": p["unit_norm"], "predicted_cost_per_unit2": p["cost_per_unit2"],
               "predicted_gain_per_unit_half_ranking": p["gain_per_unit_half_ranking"], "lambdas": {}}
        for lam in p["lambdas"]:
            plus = np.asarray(next(q for q in pts if q.get("n") == n and q["lam"] == lam)["rewards"])
            minus = np.asarray(next(q for q in pts if q.get("n") == n and q["lam"] == -lam)["rewards"])
            odd, even = (plus - minus) / 2, (plus + minus) / 2 - start
            res["lambdas"][f"{lam:.2f}"] = {
                "length": lam * p["unit_norm"],
                "gain_per_unit": float(odd.mean() / lam),
                "gain_per_unit_se": float(odd.std(ddof=1) / np.sqrt(odd.size) / lam),
                "cost_per_unit2": float(even.mean() / lam ** 2),
                "cost_per_unit2_se": float(even.std(ddof=1) / np.sqrt(even.size) / lam ** 2),
                "plus": float(plus.mean()), "minus": float(minus.mean())}
        far = res["lambdas"][f"{p['lambdas'][-1]:.2f}"]
        if far["cost_per_unit2"] < 0:
            res["best_net_per_update"] = far["gain_per_unit"] ** 2 / (4 * -far["cost_per_unit2"])
        out["by_n"][str(n)] = res
    ca, cm = rec["check_adapters"], rec["check_merged"]
    out["check"] = {}
    for m in ("member0", "member1"):
        x = np.subtract(ca[m], ca["base"])
        y = np.subtract(cm[m], cm["base"])
        out["check"][m] = {"corr_adapter_merged": float(np.corrcoef(x, y)[0, 1]),
                           "slope_merged_on_adapter": float(np.polyfit(x, y, 1)[0]),
                           "mean_change_adapter": float(x.mean()), "mean_change_merged": float(y.mean())}
    if mem is not None:
        r, s = np.asarray(mem["rewards"]), np.asarray(mem["start"])
        odd, even = (r[0::2] - r[1::2]) / 2, (r[0::2] + r[1::2]) / 2 - s[None]
        out["members"] = {"start": float(s.mean()), "mean_change": float(r.mean() - s.mean()),
                          "member_sd": float(r.mean(1).std(ddof=1)),
                          "pairs_changed": float((r != s[None]).mean()),
                          "odd": variance_parts(odd), "even": variance_parts(even)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("members", "steps"))
    ap.add_argument("--members", type=Path, help="phase steps: the members record")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--analyze", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/lowrank-check"))
    args = ap.parse_args(argv)
    if args.analyze:
        rec = json.loads((E2E / args.analyze).read_text())
        mem_path = args.members or args.analyze.with_name("members.json")
        mem = json.loads((E2E / mem_path).read_text()) if (E2E / mem_path).exists() else None
        print(json.dumps(analyze(rec, mem), indent=2))
        return 0
    if args.out is None or args.phase is None:
        ap.error("--phase and --out, or --analyze")
    out = E2E / args.out
    if out.exists():
        raise SystemExit(f"{out} exists")
    args.workdir.mkdir(parents=True, exist_ok=True)
    cfg = setting(args.smoke)
    rec = phase_members(args, cfg) if args.phase == "members" else phase_steps(args, cfg)
    harness.write_atomic(out, {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "env": env_block(E2E, ["runs"], ("vllm", "torch")), "smoke": args.smoke, **rec})
    if args.phase == "steps":
        print(json.dumps(analyze(rec), indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
