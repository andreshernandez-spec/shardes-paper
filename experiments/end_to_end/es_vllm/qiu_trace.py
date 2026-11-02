#!/usr/bin/env python
"""Their ES trainer, unchanged, for a few iterations, with hooks that record what its
updates do to a few matrices (docs/end_to_end/12, the diagnosis after update 200).

    .venv-qiu-eval/bin/python experiments/end_to_end/es_vllm/qiu_trace.py --run --iterations 3 \
        --es-at-scale /root/es-at-scale --trace /root/qiu-trace
    .venv-qiu-eval/bin/python experiments/end_to_end/es_vllm/qiu_trace.py --analyze --trace /root/qiu-trace \
        --ckpt /root/qiu-ckpt --record experiments/end_to_end/runs/qiu-matched/trace.json

The trainer (archive/es_fine-tuning_countdown.py at the pinned commit) is loaded under
another module name, so its `main()` does not start on import, and its parser reads the
README's Countdown arguments (`--data_sample 200`, Qwen2.5-3B-Instruct) with
`--gpu_threads 4`: four model copies, the members in batches of four, the first copy
running a quarter of them, as each of the README's four processes does with its own copy.
Its data path is relative to the script, which the archive moved, so it is loaded
through a directory of symlinks. The hooks change nothing it computes:

- the model loader keeps each model it loads;
- `np.random.randint`, called once per iteration for the members' seeds, records them and
  the tracked matrices of the first copy (the weights after the last update), and ends the
  run after `--iterations` iterations;
- `process_seed` (one member: add its noise, decode the 200 prompts, score, subtract the
  noise) records the member's reward and, on the first copy, what the add and subtract
  left behind.

`--analyze` recomputes each update from the recorded seeds and rewards with their own
arithmetic (bf16 noise, a bf16 sum, added in place) and checks it gives the recorded
weights bit for bit; then it measures the update they meant (the same sum done exactly)
against the one the weights got, the residue of the add and subtract, and how the update
is shared across layers. The same tensors from our checkpoints (qiu_match.py) give ours.
"""

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

E2E = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(E2E))
import releases as R  # noqa: E402

LAYERS = (0, 17, 35)
KINDS = ("self_attn.q_proj.weight", "self_attn.o_proj.weight", "mlp.gate_proj.weight",
         "mlp.down_proj.weight", "input_layernorm.weight")
TRACKED = [f"model.layers.{i}.{k}" for i in LAYERS for k in KINDS]
SIGMA, ALPHA, N = 1e-3, 5e-4, 30                 # their constants, checked against the module


class Done(Exception):
    pass


def load_trainer(clone: Path, threads: int, cache: str):
    arch = clone / "archive"
    d = Path(tempfile.mkdtemp(prefix="qiu-trainer-"))
    (d / "es_fine-tuning_countdown.py").symlink_to(arch / "es_fine-tuning_countdown.py")
    (d / "data").symlink_to(arch / "countdown" / "data")
    sys.path[:0] = [str(arch / "countdown")]
    sys.argv = ["es_fine-tuning_countdown.py", "--data_sample", "200", "--model_name", R.QWEN25_3B.repo,
                "--gpu_threads", str(threads), "--hf_cache_dir", cache]
    spec = importlib.util.spec_from_file_location("qiu_trainer", d / "es_fine-tuning_countdown.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if (mod.SIGMA, mod.ALPHA, mod.POPULATION_SIZE) != (SIGMA, ALPHA, N):
        raise SystemExit(f"their constants are {mod.SIGMA, mod.ALPHA, mod.POPULATION_SIZE}")
    return mod


def run(args) -> None:
    import numpy as np
    import torch
    from huggingface_hub import constants, model_info

    trace = Path(args.trace)
    trace.mkdir(parents=True, exist_ok=True)
    mod = load_trainer(Path(args.es_at_scale), args.threads, constants.HF_HUB_CACHE)
    models, rec = [], {"seeds": [], "rewards": [], "residue": [], "seconds": [],
                       "model_sha": model_info(R.QWEN25_3B.repo).sha, "pinned": R.QWEN25_3B.commit,
                       "threads": args.threads, "iterations": args.iterations}
    state = {"start": None, "pre": None, "t0": time.time()}

    def tracked(model):
        p = dict(model.named_parameters())
        return {n: p[n].detach().clone() for n in TRACKED}

    loader = mod.AutoModelForCausalLM.from_pretrained

    def load(*a, **k):
        m = loader(*a, **k)
        models.append(m)
        return m
    mod.AutoModelForCausalLM.from_pretrained = load

    member = mod.process_seed

    def process_seed(seed_args):
        out = member(seed_args)
        t = len(rec["seeds"]) - 1
        rec["rewards"][t][out[0]] = out[1]
        if seed_args[2] is models[0]:                    # the first copy, now restored
            now = tracked(models[0])
            rec["residue"][t].append({"member": out[0], **{
                n: {"changed": int((now[n] != state["start"][n]).sum()),
                    "abs_sum": float((now[n].double() - state["start"][n].double()).abs().sum())}
                for n in TRACKED}})
            state["pre"] = now
        return out
    mod.process_seed = process_seed

    draw = np.random.randint

    def randint(*a, **k):
        if k.get("size") != N:
            return draw(*a, **k)
        t = len(rec["seeds"])
        now = tracked(models[0])
        torch.save({n: v.cpu() for n, v in now.items()}, trace / f"start-{t}.pt")
        if t:
            torch.save({n: v.cpu() for n, v in state["pre"].items()}, trace / f"pre-{t - 1}.pt")
            rec["seconds"].append(time.time() - state["t0"])
            state["t0"] = time.time()
            (trace / "trace.json").write_text(json.dumps(rec))
            print(f"traced iteration {t - 1}: {rec['seconds'][-1]:.0f}s", flush=True)
        if t == args.iterations:
            raise Done
        state["start"], state["pre"] = now, now
        seeds = draw(*a, **k)
        rec["seeds"].append([int(s) for s in seeds])
        rec["rewards"].append([None] * N)
        rec["residue"].append([])
        return seeds
    np.random.randint = randint

    try:
        mod.main()
    except Done:
        pass
    print("trace written", trace, flush=True)


def library_pin() -> dict:
    """The shardes commit this checkout pins. The tracer runs in the evaluator's venv and
    never imports the library, but a record stamps it so provenance_audit.py reads the
    record as one written after the split."""
    req = (E2E.parent.parent / "requirements.txt").read_text()
    sha = re.search(r"shardes @ \S+@([0-9a-f]{40})", req).group(1)
    return {"commit": sha, "source": "pin, not imported"}


def stats(u, d) -> dict:
    """The update meant (u) against the one the weights got (d), both float64."""
    nz = u != 0
    return {"meant_norm": float(u.norm()), "got_norm": float(d.norm()),
            "cosine": float((u * d).sum() / (u.norm() * d.norm())) if d.norm() > 0 else 0.0,
            "error_over_meant": float((d - u).norm() / u.norm()),
            "lost_share": float(((d == 0) & nz).sum() / nz.sum()),     # meant to move, did not
            "moved_share": float((d != 0).float().mean())}


def analyze(args) -> dict:
    import numpy as np
    import torch
    from safetensors import safe_open

    trace = Path(args.trace)
    rec = json.loads((trace / "trace.json").read_text())
    out = {"his": [], "ours": {}, "model_sha": rec["model_sha"], "pinned": rec["pinned"],
           "threads": rec["threads"], "seconds": rec["seconds"], "tracked": TRACKED,
           "rewards": rec["rewards"], "env": {"torch": torch.__version__,
           "commit": subprocess.run(["git", "-C", str(E2E), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "shardes": library_pin()}}
    for t in range(len(rec["seconds"])):
        start = torch.load(trace / f"start-{t}.pt")
        pre = torch.load(trace / f"pre-{t}.pt")
        after = torch.load(trace / f"start-{t + 1}.pt")
        r = np.array(rec["rewards"][t], dtype=np.float32)
        z = (r - r.mean()) / (r.std() + 1e-8)                    # their normalization
        it = {"iteration": t, "tensors": {}}
        meant = {}
        for name in TRACKED:
            w = pre[name].cuda()
            upd, exact = torch.zeros_like(w), torch.zeros(w.shape, dtype=torch.float64, device="cuda")
            for n, s in enumerate(rec["seeds"][t]):
                g = torch.Generator(device="cuda")
                g.manual_seed(int(s))
                e = torch.randn(w.shape, generator=g, device="cuda", dtype=w.dtype)
                exact += float(z[n]) * e.double()
                e.mul_(float(z[n]))
                upd.add_(e)
            upd.div_(N)
            mine = w.clone()
            mine.add_(ALPHA * upd)
            u = exact * (ALPHA / N)
            d = after[name].cuda().double() - w.double()
            res = w.double() - start[name].cuda().double()
            meant[name] = u
            it["tensors"][name] = {"reproduced": bool(torch.equal(mine, after[name].cuda())), **stats(u, d),
                                   "residue_over_meant": float(res.norm() / u.norm()),
                                   "residue_share": float((res != 0).float().mean())}
        it["shared"] = {k: {f"{a}-{b}": float((meant[f"model.layers.{a}.{k}"] * meant[f"model.layers.{b}.{k}"]).sum()
                                         / (meant[f"model.layers.{a}.{k}"].norm() * meant[f"model.layers.{b}.{k}"].norm()))
                            for a, b in ((0, 17), (0, 35))} for k in KINDS}
        q, o = meant["model.layers.0.self_attn.q_proj.weight"], meant["model.layers.0.self_attn.o_proj.weight"]
        it["shared"]["q_proj-o_proj layer 0"] = float((q * o).sum() / (q.norm() * o.norm()))
        out["his"].append(it)
        print(f"iteration {t}: reproduced {all(v['reproduced'] for v in it['tensors'].values())}", flush=True)

    if args.ckpt:                                                 # ours, from qiu_match.py's checkpoints
        ck = sorted((int(p.name.split("-")[1]), p) for p in Path(args.ckpt).glob("ckpt-*") if (p / "saved.json").exists())

        def load(p):
            with safe_open(str(p / "model.safetensors"), framework="pt", device="cuda") as f:
                return {n: f.get_tensor(n).double() for n in TRACKED}
        w = {k: load(p) for k, p in ck}
        for (a, _), (b, _) in zip(ck, ck[1:]):
            d = {n: w[b][n] - w[a][n] for n in TRACKED}
            out["ours"][f"{a}-{b}"] = {
                "norm": {n: float(d[n].norm()) for n in TRACKED},
                "relative": {n: float(d[n].norm() / w[a][n].norm()) for n in TRACKED},
                "shared": {k: {f"{x}-{y}": float((d[f"model.layers.{x}.{k}"] * d[f"model.layers.{y}.{k}"]).sum()
                                               / (d[f"model.layers.{x}.{k}"].norm() * d[f"model.layers.{y}.{k}"].norm()))
                               for x, y in ((0, 17), (0, 35))} for k in KINDS}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--es-at-scale", default="/root/es-at-scale")
    ap.add_argument("--ckpt", help="--analyze: qiu_match.py's checkpoint directory, for ours")
    ap.add_argument("--record", help="--analyze: the record, relative to the repository root")
    args = ap.parse_args(argv)
    if args.run:
        run(args)
    if args.analyze:
        rec = analyze(args)
        path = E2E.parent.parent / args.record
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=1) + "\n")
        print("record", path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
