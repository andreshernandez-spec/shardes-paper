#!/usr/bin/env python
"""ES's updates against RL's exact gradient: is the estimate right, and what is its ceiling?

    python -m es_vllm.gradient_check --phase sample   --out runs/gradient-check   # vLLM
    python -m es_vllm.gradient_check --phase gradient --out runs/gradient-check   # torch
    python -m es_vllm.gradient_check --phase project  --out runs/gradient-check   # JAX+torch
    python -m es_vllm.gradient_check --phase scan     --out runs/gradient-check   # vLLM
    python -m es_vllm.gradient_check --analyze runs/gradient-check
    (--smoke on each phase: Qwen 0.5B, synthetic rankings, wiring only)

An ES update is `u = (alpha / N) sum_m z_m E_m`. Writing each member's z-score as
`rho (g . E_m) / |g|` plus noise, `E[u] = alpha rho g_hat`: a step of `alpha rho` along the
gradient and a random part of `alpha sqrt(d / N)` beside it. Everything measured so far
(runs/step-check/, runs/lowrank-check/) is consistent with that, with rho about 0.2
inferred from the members' reliability. This measures the two things left open, against
the gradient RL actually follows:

1. `sample`: the RL run's batch for the 192 iteration-0 prompts (RL steps 1 to 4): 16
   answers each at temperature 1.0, the run's verifiers, GRPO's advantages
   `(score - group mean) / (group sd + 1e-8)` (open-instruct 3f37c29).
2. `gradient`: GRPO's exact gradient `g` at the start on that batch: the token mean over
   each micro-batch of 2 of `-A_i log p(y_t)` as the run forms it, averaged over
   micro-batches (f32 parameters, bf16 autocast). The ratio is 1 and the KL term's gradient
   is 0 at the start, so this is the whole gradient of the run's first step.
3. `project`: `u . g_hat` for the updates already measured: the dense N = 16 updates of
   `step_check.py` (four rankings of the same members), the low-rank N = 128, 512 and
   1,024 updates of `lowrank_check.py`, the eight updates of `lowrank_run.py` and their
   sum, and the RL run's own displacement to step 120. `rho_hat = u . g_hat / alpha` is
   the estimator's correlation with RL's gradient (noise about `1 / sqrt(N)` when the
   ranking carries nothing; the control ranking is that case).
4. `scan`: the held-out reward (3,840 prompts of RL steps 121 to 160 and 481 to 520)
   at `theta_0 +- lambda g_hat` for lambda 1, 2, 4, 8, `g_hat` restricted to the matrices
   ES perturbs, and at `theta_0 +- lambda s_hat`, `s_hat` the unit sign vector (Adam's
   first step's direction), for lambda 1 and 4. The odd part per unit effective length
   is `D`, the reward per unit length along RL's gradient. A perfectly ranked ES gains
   `alpha D` per unit update, so its best net at N members is `(alpha D)^2 / (4 c_N)`,
   with `c_N` the measured cost per unit squared (`runs/lowrank-check/`).

Prediction, committed before the run. `rho_hat` for the N = 1,024 update between 0.1 and
0.3 (consistent with the inferred 0.2 and a greedy fitness that tracks the sampled-reward
gradient); near 0 means ES climbs a different hill than RL; the control's `rho_hat`
within +-0.5. `D` along `g_hat` above 10: ES's measured 0.0047 per unit at N = 1,024
already implies a directional derivative of at least 9.4 along its useful component
(0.0047 / alpha with rho <= 1). If `D` is about 10, the current fitness is near its ceiling
and only N helps; if `D` is 50 or more, a better fitness could gain 25 times or more per
update.
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
from provenance import env_block  # noqa: E402

SEED, SIGMA, ALPHA, RANK, P_RUN, K = 0, 5e-4, 5e-4, 1, 192, 16
SAMPLE_SEED = 30_000              # as contrastive_check.py
MICRO = 2                         # the RL run's per_device_train_batch_size
GRAD_LAMBDAS, SIGN_LAMBDAS = (1.0, 4.0, 2.0, 8.0), (1.0, 4.0)   # the key points first
HELDOUT = ((120, 160), (480, 520))
RUN_STEP, RUN_N, RUN_ITERATIONS = 15.4, 512, 8       # lowrank_run.py
CHECK_NS = (128, 512, 1024)                           # lowrank_check.py


def setting(smoke: bool) -> dict:
    if smoke:
        return {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "revision": None, "cap": 48, "prompts": 8,
                "k": 4, "heldout": 16, "verifiers": 2, "gpu": 0.4,
                "grad_lambdas": (1.0,), "sign_lambdas": (1.0,)}
    return {"repo": R.TULU31_START.repo, "revision": R.TULU31_START.commit, "cap": 2048,
            "prompts": P_RUN, "k": K, "heldout": None, "verifiers": 8, "gpu": 0.45,
            "grad_lambdas": GRAD_LAMBDAS, "sign_lambdas": SIGN_LAMBDAS}


def append(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def lines(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def get_dir(repo, rev):
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    return snapshot_download(repo, revision=rev, allow_patterns=["*.safetensors", "*.json"])


def model_dirs(cfg, smoke):
    from es_vllm.heldout import rl_revision  # noqa: PLC0415

    start = get_dir(cfg["repo"], cfg["revision"])
    rl = get_dir("Qwen/Qwen2.5-0.5B", None) if smoke else get_dir(R.TULU31_RL.repo, rl_revision("step_120"))
    return start, rl


# ------------------------------------------------------------------ 1. the RL batch

def phase_sample(args, cfg) -> None:
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from es_vllm.lowrank_check import Verifiers  # noqa: PLC0415
    from es_vllm.run_tulu import load_prompts, render  # noqa: PLC0415

    rows = load_prompts(E2E / "data" / "tulu31", 1, P_RUN)[0][: cfg["prompts"]]
    tok = AutoTokenizer.from_pretrained(cfg["repo"], revision=cfg["revision"])
    ids = [render(tok, r) for r in rows]
    llm = LLM(model=cfg["repo"], revision=cfg["revision"], dtype="bfloat16", seed=0,
              gpu_memory_utilization=cfg["gpu"], max_model_len=4096, enable_prefix_caching=False)
    k = cfg["k"]
    params = [SamplingParams(n=k, temperature=1.0, top_p=1.0, max_tokens=cfg["cap"], seed=SAMPLE_SEED + j)
              for j in range(len(rows))]
    t0 = time.perf_counter()
    outs = llm.generate([{"prompt_token_ids": i} for i in ids], params, use_tqdm=False)
    vs = Verifiers(cfg["verifiers"])
    rewards = vs.score([{"text": o.text, "ground_truth": r["ground_truth"], "dataset": r["dataset"],
                         "stopped": o.finish_reason == "stop"} for out, r in zip(outs, rows) for o in out.outputs])
    vs.close()
    rew = np.asarray(rewards).reshape(len(rows), k)
    sd = rew.std(axis=1, ddof=1)                                   # torch.std's correction
    adv = (rew - rew.mean(axis=1, keepdims=True)) / (sd[:, None] + 1e-8)
    samples = [{"prompt": j, "tokens": list(o.token_ids), "reward": float(rew[j, s]),
                "advantage": float(adv[j, s]), "stopped": o.finish_reason == "stop"}
               for j, out in enumerate(outs) for s, o in enumerate(out.outputs)]
    rec = {"prompts": len(rows), "k": k, "datasets": [r["dataset"] for r in rows],
           "prompt_tokens": ids, "samples": samples, "live_groups": int((sd > 0).sum()),
           "mean_reward": float(rew.mean()),
           "mean_len": float(np.mean([len(s["tokens"]) for s in samples])),
           "seconds": time.perf_counter() - t0}
    harness.write_atomic(E2E / args.out / "samples.json", rec)
    print(f"{len(samples)} samples, {rec['live_groups']} of {len(rows)} groups live, mean reward "
          f"{rec['mean_reward']:.3f}, mean length {rec['mean_len']:.0f}, {rec['seconds']:.0f}s", flush=True)


# ------------------------------------------------------------------ 2. GRPO's gradient

def leaf_kind(name: str) -> str:
    for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
              "embed_tokens", "lm_head"):
        if k in name:
            return k
    return "norm"


def phase_gradient(args, cfg) -> None:
    import torch  # noqa: PLC0415
    from safetensors.torch import save_file  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    rec = json.loads((E2E / args.out / "samples.json").read_text())
    seqs = [rec["prompt_tokens"][s["prompt"]] + s["tokens"] for s in rec["samples"]]
    plens = [len(rec["prompt_tokens"][s["prompt"]]) for s in rec["samples"]]
    adv = [s["advantage"] for s in rec["samples"]]
    model = AutoModelForCausalLM.from_pretrained(cfg["repo"], revision=cfg["revision"],
                                                 dtype=torch.float32, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    order = np.random.default_rng(0).permutation(len(seqs))     # the run shuffles its batch
    micro = [order[i:i + MICRO] for i in range(0, len(order), MICRO)]
    t0, tokens = time.perf_counter(), 0
    for b, idx in enumerate(micro):
        L = max(len(seqs[i]) for i in idx)
        x = torch.zeros((len(idx), L), dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        weight = torch.zeros((len(idx), L))                       # A_i on response positions
        for r, i in enumerate(idx):
            s = seqs[i]
            x[r, :len(s)], mask[r, :len(s)] = torch.tensor(s), 1
            weight[r, plens[i]:len(s)] = adv[i]
        x, mask, weight = x.cuda(), mask.cuda(), weight.cuda()
        resp = mask.bool() & (torch.arange(L, device="cuda")[None]
                              >= torch.tensor([plens[i] for i in idx], device="cuda")[:, None])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x, attention_mask=mask).logits
        logp = torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, x[:, 1:, None]).squeeze(-1)
        n_resp = resp[:, 1:].sum()
        # masked_mean(-A ratio, response mask) at ratio 1, averaged over micro-batches
        loss = -(weight[:, 1:] * logp).sum() / n_resp / len(micro)
        loss.backward()
        tokens += int(mask.sum())
        if b % 200 == 0:
            print(f"micro-batch {b}/{len(micro)} {time.perf_counter() - t0:.0f}s", flush=True)
    grads = {n: p.grad.detach() for n, p in model.named_parameters() if p.grad is not None}
    stats, kinds = {}, {}
    for n, g in grads.items():
        s2 = float((g.double() ** 2).sum())
        stats[n] = s2
        kinds[leaf_kind(n)] = kinds.get(leaf_kind(n), 0.0) + s2
    matrices = {n for n, _ in lowrank.leaf_shapes(get_dir(cfg["repo"], cfg["revision"]))}
    total = sum(stats.values())
    save_file({n: g.contiguous().cpu() for n, g in grads.items()}, str(args.workdir / "grad.safetensors"))
    harness.write_atomic(E2E / args.out / "gradient.json", {
        "sequences": len(seqs), "micro_batches": len(micro), "micro": MICRO, "tokens": tokens,
        "norm": float(np.sqrt(total)), "norm_matrices": float(np.sqrt(sum(stats[n] for n in matrices))),
        "share_of_norm2_by_kind": {k: v / total for k, v in kinds.items()},
        "share_of_norm2_matrices": sum(stats[n] for n in matrices) / total,
        "leaves": len(grads), "seconds": time.perf_counter() - t0})
    print(f"|g| {np.sqrt(total):.4g}, matrices {np.sqrt(sum(stats[n] for n in matrices)):.4g}, "
          f"{tokens} tokens, {time.perf_counter() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ 3. projections

def rankings(smoke: bool) -> dict:
    """The fitness vectors behind the updates to project; synthetic in smoke mode."""
    if smoke:
        rng = np.random.default_rng(0)
        return {"dense": {"random": rng.normal(size=16).tolist()},
                "check": rng.normal(size=8).tolist(), "check_ns": (4, 8),
                "run": [rng.normal(size=4).tolist() for _ in range(2)], "run_n": 4}
    step = json.loads((E2E / "runs/step-check/start.json").read_text())
    check = json.loads((E2E / "runs/lowrank-check/members.json").read_text())
    run = lines(E2E / "runs/lowrank-run/log.jsonl")
    return {"dense": step["rankings"],
            "check": np.asarray(check["rewards"]).mean(1).tolist(), "check_ns": CHECK_NS,
            "run": [r["fitness"] for r in run], "run_n": RUN_N}


def phase_project(args, cfg) -> None:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415
    from shardes.shaping import group_relative  # noqa: PLC0415

    from es_vllm import stream  # noqa: PLC0415
    from es_vllm.worker import load_master  # noqa: PLC0415

    start_dir, rl_dir = model_dirs(cfg, args.smoke)
    ranks = rankings(args.smoke)
    grad_path = str(args.workdir / "grad.safetensors")
    rl_index, start_index = lowrank_index(rl_dir), lowrank_index(start_dir)
    with safe_open(grad_path, framework="pt", device="cpu") as st:
        grad_names = set(st.keys())

    def read(path, name, device="cuda"):
        with safe_open(path, framework="pt", device=device) as st:
            return st.get_tensor(name).float()

    t0 = time.perf_counter()
    shapes = lowrank.leaf_shapes(start_dir)
    names_m = [n for n, _ in shapes]
    # low-rank factors of every generation, pair-major (one regeneration each), kept on the host
    gens = [(None, max(ranks["check_ns"]) // 2)] + [(g, ranks["run_n"] // 2) for g in range(1, len(ranks["run"]))]
    A = {n: [] for n in names_m}
    B = {n: [] for n in names_m}
    for gen, pairs in gens:
        for j in range(pairs):
            f = lowrank.pair_factors(SEED, j, shapes, RANK, gen)
            for n in names_m:
                A[n].append(f[n][0][:, 0])
                B[n].append(f[n][1][:, 0])
    A = {n: np.stack(v, 1) for n, v in A.items()}      # (out, all pairs)
    B = {n: np.stack(v, 1) for n, v in B.items()}      # (in, all pairs)
    offsets = np.cumsum([0] + [p for _, p in gens])     # pair columns per generation
    print(f"factors: {offsets[-1]} pairs x {len(names_m)} leaves, {time.perf_counter() - t0:.0f}s", flush=True)

    # coefficients per pair column: lowrank_check's N (generation 0) and the run's generations
    coef = {}
    for n_ in ranks["check_ns"]:
        c = np.zeros(offsets[-1])
        c[: n_ // 2] = lowrank.coefficients(ranks["check"], n_, ALPHA, RANK)
        coef[f"check_N{n_}"] = c
    for g, fit in enumerate(ranks["run"]):
        c = np.zeros(offsets[-1])
        c[offsets[g]:offsets[g] + ranks["run_n"] // 2] = lowrank.coefficients(fit, ranks["run_n"], ALPHA, RANK)
        coef[f"run_g{g}"] = c
    coef["run_sum"] = RUN_STEP * sum(coef[f"run_g{g}"] for g in range(len(ranks["run"])))

    # per leaf: a_j^T g b_j and a_j^T Delta b_j for every pair column, and the Gram matrices
    # (a_j . a_l)(b_j . b_l) that give every low-rank update's norm and mutual products
    proj_g = np.zeros(offsets[-1])
    proj_d = np.zeros(offsets[-1])
    gram = np.zeros((offsets[-1], offsets[-1]))
    g2_m, d2_m, gd_m = 0.0, 0.0, 0.0
    for n in names_m:
        g = read(grad_path, n)
        d = read(rl_index[n], n) - read(start_index[n], n)
        a = torch.from_numpy(A[n]).cuda()
        b = torch.from_numpy(B[n]).cuda()
        proj_g += (a * (g @ b)).sum(0).double().cpu().numpy()
        proj_d += (a * (d @ b)).sum(0).double().cpu().numpy()
        gram += ((a.T @ a) * (b.T @ b)).double().cpu().numpy()
        g2_m += float((g.double() ** 2).sum())
        d2_m += float((d.double() ** 2).sum())
        gd_m += float((g.double() * d.double()).sum())
        del g, d, a, b
    print(f"low-rank projections {time.perf_counter() - t0:.0f}s", flush=True)
    out = {"g_norm_matrices": float(np.sqrt(g2_m)), "delta_rl_norm_matrices": float(np.sqrt(d2_m)),
           "cos_g_delta_matrices": gd_m / np.sqrt(g2_m * d2_m), "lowrank": {}}
    for name, c in coef.items():
        norm = float(np.sqrt(c @ gram @ c))
        dot_g, dot_d = float(c @ proj_g), float(c @ proj_d)
        out["lowrank"][name] = {"norm": norm, "dot_g": dot_g, "dot_delta": dot_d,
                                "cos_g": dot_g / (norm * np.sqrt(g2_m)),
                                "cos_delta": dot_d / (norm * np.sqrt(d2_m)),
                                "rho_hat": dot_g / np.sqrt(g2_m) / ALPHA}

    # dense N = 16 updates over every leaf, as es_tell makes them
    master = load_master(start_dir)
    names = stream.leaf_names(master)
    streams = stream.streams(jax.random.fold_in(jax.random.key(SEED), 0), len(names))   # es_ask, generation 0
    ids = jnp.arange(16, dtype=jnp.int32)
    dense = {k: {"dot_g": 0.0, "dot_delta": 0.0, "norm2": 0.0} for k in ranks["dense"]}
    weights = {k: group_relative((-jnp.asarray(v, dtype=jnp.float32))[:, None]) for k, v in ranks["dense"].items()}
    g2, d2, gd, skipped = 0.0, 0.0, 0.0, []
    for k_, n in enumerate(names):
        if n not in grad_names:        # e.g. a tied lm_head: not a parameter of its own
            skipped.append(n)
            continue
        g = read(grad_path, n)
        d = read(rl_index[n], n) - read(start_index[n], n)
        gj = jnp.array(jax.dlpack.from_dlpack(g))      # copies: JAX owns what it sums
        dj = jnp.array(jax.dlpack.from_dlpack(d))
        gj.block_until_ready()
        dj.block_until_ready()
        g2 += float((g.double() ** 2).sum())
        d2 += float((d.double() ** 2).sum())
        gd += float((g.double() * d.double()).sum())
        for name, w in weights.items():
            u = stream.updated_leaf(master[n], streams[k_], ids, w, ALPHA * SIGMA, SIGMA, 16) - master[n]
            dense[name]["dot_g"] += float(jnp.sum(u * gj))
            dense[name]["dot_delta"] += float(jnp.sum(u * dj))
            dense[name]["norm2"] += float(jnp.sum(u * u))
            del u
        del gj, dj, g, d
        torch.cuda.synchronize()
    print(f"dense projections {time.perf_counter() - t0:.0f}s", flush=True)
    out.update({"g_norm": float(np.sqrt(g2)), "delta_rl_norm": float(np.sqrt(d2)),
                "cos_g_delta": gd / np.sqrt(g2 * d2), "leaves_without_gradient": skipped, "dense": {}})
    for name, v in dense.items():
        norm = float(np.sqrt(v["norm2"]))
        out["dense"][name] = {"norm": norm, "dot_g": v["dot_g"], "dot_delta": v["dot_delta"],
                              "cos_g": v["dot_g"] / (norm * np.sqrt(g2)),
                              "cos_delta": v["dot_delta"] / (norm * np.sqrt(d2)),
                              "rho_hat": v["dot_g"] / np.sqrt(g2) / ALPHA}
    out["seconds"] = time.perf_counter() - t0
    harness.write_atomic(E2E / args.out / "projections.json", out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("lowrank", "dense")}, indent=1), flush=True)
    for part in ("dense", "lowrank"):
        for name, v in out[part].items():
            print(f"{part} {name}: |u| {v['norm']:.4g} rho_hat {v['rho_hat']:+.4f} "
                  f"cos(u, g) {v['cos_g']:+.2e} cos(u, delta) {v['cos_delta']:+.2e}", flush=True)


def lowrank_index(model_dir) -> dict:
    from safetensors import safe_open  # noqa: PLC0415

    index = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as st:
            for n in st.keys():
                index[n] = str(f)
    return index


# ------------------------------------------------------------------ 4. the scan

def worker_methods():
    import torch  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415

    def gd_setup(self, model_dir, grad_path, shapes):
        """The f32 base of the perturbed matrices and the unit gradient restricted to them."""
        names = [n for n, _ in shapes]
        index = lowrank_index(model_dir)
        self._gd_base, self._gd_dir = {}, {}
        with safe_open(grad_path, framework="pt", device="cuda") as st:
            norm2 = sum(float((st.get_tensor(n).double() ** 2).sum()) for n in names)
            for n in names:
                self._gd_dir[n] = st.get_tensor(n).float() / float(np.sqrt(norm2))
        for n in names:
            with safe_open(index[n], framework="pt", device="cuda") as st:
                self._gd_base[n] = st.get_tensor(n).float()
        nnz = sum(int((v != 0).sum()) for v in self._gd_dir.values())
        self._gd_sign_scale = 1.0 / float(np.sqrt(nnz))
        self._gd_rows = None
        return {"norm_matrices": float(np.sqrt(norm2)), "nonzero": nnz}

    def rows(self):
        if self._gd_rows is None:
            self._gd_rows = {}
            for name, param in self.model_runner.model.named_parameters():
                parts = [name]
                for packed, pieces in lowrank.PACKED.items():
                    if f".{packed}." in name:
                        parts = [name.replace(packed, p) for p in pieces]
                if not all(p in self._gd_base for p in parts):
                    continue
                off = 0
                for p in parts:
                    self._gd_rows[p] = (param, off)
                    off += self._gd_base[p].shape[0]
        return self._gd_rows

    def gd_load(self, lam, kind):
        """The engine's matrices become bf16(base + lam * direction); returns the effective
        length, |bf16(base + lam direction) - base|, which the rounding can shorten."""
        eff2 = 0.0
        for n in sorted(self._gd_base):
            base = self._gd_base[n]
            d = self._gd_dir[n] if kind == "grad" else torch.sign(self._gd_dir[n]) * self._gd_sign_scale
            w = (base + float(lam) * d).to(torch.bfloat16)
            eff2 += float(((w.float() - base).double() ** 2).sum())
            param, off = rows(self)[n]
            param.data[off:off + w.shape[0]].copy_(w)
            del w, d
        torch.cuda.synchronize()
        return float(np.sqrt(eff2))

    return {"gd_setup": gd_setup, "gd_load": gd_load}


def phase_scan(args, cfg) -> None:
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from es_vllm.lowrank_check import Verifiers, items  # noqa: PLC0415
    from es_vllm.run_tulu import heldout_rows, load_prompts, render  # noqa: PLC0415
    from es_vllm.worker import ESWorker  # noqa: PLC0415

    for k, fn in worker_methods().items():
        setattr(ESWorker, k, fn)
    start_dir, _ = model_dirs(cfg, args.smoke)
    shapes = lowrank.leaf_shapes(start_dir)
    llm = LLM(model=cfg["repo"], revision=cfg["revision"], dtype="bfloat16", seed=0,
              gpu_memory_utilization=cfg["gpu"], max_model_len=4096, enable_prefix_caching=False,
              max_num_seqs=512, worker_extension_cls="es_vllm.worker.ESWorker")
    rpc = lambda m, *a: llm.collective_rpc(m, args=a)[0]  # noqa: E731
    meta = rpc("gd_setup", start_dir, str(args.workdir / "grad.safetensors"), shapes)
    tok = AutoTokenizer.from_pretrained(cfg["repo"], revision=cfg["revision"])
    if args.smoke:
        rows = [r for b in load_prompts(E2E / "data" / "tulu31", 2, P_RUN) for r in b][P_RUN:][: cfg["heldout"]]
    else:
        rows = [r for a, b in HELDOUT for r in heldout_rows(a, b)]
    ids = [render(tok, r) for r in rows]
    vs = Verifiers(cfg["verifiers"])
    greedy = SamplingParams(temperature=0.0, max_tokens=cfg["cap"])
    log = E2E / args.out / "scan.jsonl"
    done = {(r["kind"], r["lam"]) for r in lines(log)}
    harness.write_atomic(E2E / args.out / "scan-meta.json", {**meta, "heldout": len(rows),
                                                             "datasets": [r["dataset"] for r in rows]})
    t0 = time.perf_counter()

    def point(kind, lam):
        if (kind, lam) in done:
            return
        eff = rpc("gd_load", lam, kind)
        outs = llm.generate([{"prompt_token_ids": i} for i in ids], greedy, use_tqdm=False)
        r = vs.score(items(outs, rows))
        append(log, {"kind": kind, "lam": lam, "effective_length": eff, "rewards": r,
                     "mean_len": float(np.mean([len(o.outputs[0].token_ids) for o in outs]))})
        print(f"{kind} lambda {lam:+}: effective {eff:.3f}, reward {np.mean(r):.3f} "
              f"{time.perf_counter() - t0:.0f}s", flush=True)

    point("start", 0.0)
    for lam in cfg["grad_lambdas"]:
        for s in (1, -1):
            point("grad", s * lam)
    for lam in cfg["sign_lambdas"]:
        for s in (1, -1):
            point("sign", s * lam)
    vs.close()


# ------------------------------------------------------------------ analysis

def analyze(out: Path) -> dict:
    res = {}
    for name in ("samples", "gradient", "projections", "scan-meta"):
        p = out / f"{name}.json"
        if p.exists():
            r = json.loads(p.read_text())
            res[name] = {k: v for k, v in r.items() if k not in ("samples", "prompt_tokens", "datasets")}
    scan = lines(out / "scan.jsonl")
    if scan:
        start = np.asarray(next(r for r in scan if r["kind"] == "start")["rewards"])
        res["scan"] = {"start": float(start.mean()), "points": []}
        for kind in ("grad", "sign"):
            for lam in sorted({abs(r["lam"]) for r in scan if r["kind"] == kind}):
                plus = next((r for r in scan if r["kind"] == kind and r["lam"] == lam), None)
                minus = next((r for r in scan if r["kind"] == kind and r["lam"] == -lam), None)
                if plus is None or minus is None:
                    continue
                p, m = np.asarray(plus["rewards"]), np.asarray(minus["rewards"])
                odd, even = (p - m) / 2, (p + m) / 2 - start
                eff = (plus["effective_length"] + minus["effective_length"]) / 2
                res["scan"]["points"].append({
                    "kind": kind, "lam": lam, "effective_length": eff,
                    "plus": float(p.mean()), "minus": float(m.mean()),
                    "odd": float(odd.mean()), "odd_se": float(odd.std(ddof=1) / np.sqrt(odd.size)),
                    "even": float(even.mean()), "even_se": float(even.std(ddof=1) / np.sqrt(even.size)),
                    "D_per_effective_length": float(odd.mean() / eff),
                    "D_se": float(odd.std(ddof=1) / np.sqrt(odd.size) / eff),
                    "even_per_effective_length2": float(even.mean() / eff ** 2)})
        grad = [q for q in res["scan"]["points"] if q["kind"] == "grad"]
        if grad:
            d = min(grad, key=lambda q: q["effective_length"])
            # a perfectly ranked ES at N members: gain alpha D per unit, cost c_N per unit^2
            res["scan"]["ceiling"] = {
                "D": d["D_per_effective_length"], "at_effective_length": d["effective_length"],
                "gain_per_unit_perfect_ranking": ALPHA * d["D_per_effective_length"],
                "measured_gain_per_unit_N1024": 0.0047,
                "implied_rho_if_es_direction_as_steep": 0.0047 / (ALPHA * d["D_per_effective_length"]),
                "best_net_per_update_perfect_ranking_N1024": (ALPHA * d["D_per_effective_length"]) ** 2 / (4 * 7.7e-5)}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("sample", "gradient", "project", "scan"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--analyze", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/gradient-check"))
    args = ap.parse_args(argv)
    if args.analyze:
        print(json.dumps(analyze(E2E / args.analyze), indent=2))
        return 0
    if args.out is None or args.phase is None:
        ap.error("--phase and --out, or --analyze")
    (E2E / args.out).mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)
    env = E2E / args.out / "env.json"
    if not env.exists():
        env.write_text(json.dumps({"date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                                   "smoke": args.smoke,
                                   "env": env_block(E2E, ["runs"], ("vllm", "torch", "transformers", "jax"))}))
    cfg = setting(args.smoke)
    {"sample": phase_sample, "gradient": phase_gradient, "project": phase_project,
     "scan": phase_scan}[args.phase](args, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
