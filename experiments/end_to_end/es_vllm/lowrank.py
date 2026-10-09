"""Low-rank ES members on vLLM: each member a LoRA adapter built from its seed, in memory.

A member's perturbation of a matrix leaf of shape (out, in) is `s sigma a b^T / sqrt(r)`,
`a` (out, r) and `b` (in, r) standard normal (EGGROLL's, `shardes.strategies.lowrank`),
on the attention and MLP matrices of every layer; embeddings and norms are not perturbed.
Members come in mirrored pairs: member `2j` has `s = +1`, member `2j + 1` has `s = -1`,
and both use pair `j`'s factors. The factors of pair `j` on leaf `l` (leaves in sorted
order) come from numpy's PCG64 seeded with `(seed, j, l)`, so every process and machine
regenerates the same ones.

As a LoRA adapter (PEFT layout, `delta = lora_B @ lora_A` at scaling 1) a member is
`lora_A = b^T`, `lora_B = s sigma a / sqrt(r)`. vLLM serves many adapters in one batch on
one copy of the base weights, so evaluating a member costs about what an ordinary request
does. `install_adapters` makes vLLM build an adapter from its seed when a request names
it, instead of reading one from disk.

The update from member fitnesses `f` (higher is better) is `u = (alpha / N) sum_m z_m E_m`
with `z` the z-score of `f` and `E_m` the unit-scale perturbation, the dense backend's
convention: `u = (alpha / (N sqrt(r))) sum_j (z_2j - z_2j+1) a_j b_j^T`.
"""

import json
import math
import re
from pathlib import Path

import numpy as np

TARGETS = re.compile(r"model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight")
PACKED = {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "gate_up_proj": ("gate_proj", "up_proj")}


def leaf_shapes(model_dir) -> list:
    """[(name, (out, in))] of the perturbed leaves, sorted by name, from the checkpoint."""
    from safetensors import safe_open  # noqa: PLC0415

    shapes = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="np") as st:
            for name in st.keys():
                if TARGETS.fullmatch(name):
                    shapes[name] = tuple(st.get_slice(name).get_shape())
    if not shapes:
        raise SystemExit(f"no perturbable leaves in {model_dir}")
    return sorted(shapes.items())


def pair_factors(seed: int, pair: int, shapes: list, r: int) -> dict:
    """{leaf: (a (out, r), b (in, r))}, float32, for one mirrored pair."""
    out = {}
    for i, (name, (o, n)) in enumerate(shapes):
        g = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, pair, i])))
        out[name] = (g.standard_normal((o, r), dtype=np.float32),
                     g.standard_normal((n, r), dtype=np.float32))
    return out


def sign(member: int) -> float:
    return 1.0 if member % 2 == 0 else -1.0


def adapter_tensors(seed: int, member: int, shapes: list, r: int, sigma: float) -> dict:
    """Member `member`'s perturbation as PEFT LoRA tensors (torch, float32, CPU)."""
    import torch  # noqa: PLC0415

    scale = sign(member) * sigma / math.sqrt(r)
    t = {}
    for name, (a, b) in pair_factors(seed, member // 2, shapes, r).items():
        mod = name[: -len(".weight")]
        t[f"base_model.model.{mod}.lora_A.weight"] = torch.from_numpy(np.ascontiguousarray(b.T))
        t[f"base_model.model.{mod}.lora_B.weight"] = torch.from_numpy(scale * a)
    return t


def write_template(path: Path, r: int, base: str) -> Path:
    """The one adapter directory every member's request names; only its config is read."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps({
        "r": r, "lora_alpha": r, "lora_dropout": 0.0, "bias": "none", "peft_type": "LORA",
        "task_type": "CAUSAL_LM", "base_model_name_or_path": base,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                           "down_proj"]}))
    return path


def install_adapters(template: Path, seed: int, shapes: list, r: int, sigma: float) -> None:
    """Make vLLM build member `lora_int_id - 1` from its seed when a request names
    `template`. The engine must run in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0)."""
    import os  # noqa: PLC0415

    from vllm.lora.lora_model import LoRAModel  # noqa: PLC0415

    tpl = os.path.realpath(template)

    def from_local_checkpoint(cls, lora_dir, expected_lora_modules, peft_helper, *,
                              lora_model_id=None, device="cuda", dtype=None,
                              model_vocab_size=None, weights_mapper=None,
                              tensorizer_config_dict=None, skip_prefixes=None,
                              moe_ep_spec=None):
        if os.path.realpath(lora_dir) != tpl or lora_model_id is None:
            raise SystemExit(f"unexpected adapter {lora_dir} ({lora_model_id})")
        return cls.from_lora_tensors(
            lora_model_id=lora_model_id,
            tensors=adapter_tensors(seed, lora_model_id - 1, shapes, r, sigma),
            peft_helper=peft_helper, device=device, dtype=dtype,
            model_vocab_size=model_vocab_size, weights_mapper=weights_mapper,
            skip_prefixes=skip_prefixes)

    LoRAModel.from_local_checkpoint = classmethod(from_local_checkpoint)


def coefficients(fitness, n: int, alpha: float, r: int) -> np.ndarray:
    """Per pair, the update's coefficient on `a_j b_j^T` from the first `n` members."""
    f = np.asarray(fitness[:n], dtype=np.float64)
    z = (f - f.mean()) / f.std() if f.std() > 0 else np.zeros_like(f)
    return alpha / (n * math.sqrt(r)) * (z[0::2] - z[1::2])


def worker_methods():
    import torch  # noqa: PLC0415

    def lr_setup(self, model_dir, seed, r, pairs, shapes):
        """The perturbed leaves' base weights and `pairs` pairs' factors, f32 on the GPU."""
        from safetensors import safe_open  # noqa: PLC0415

        names = {n for n, _ in shapes}
        self._lr_base = {}
        for f in sorted(Path(model_dir).glob("*.safetensors")):
            with safe_open(str(f), framework="pt", device="cpu") as st:
                for name in st.keys():
                    if name in names:
                        self._lr_base[name] = st.get_tensor(name).to("cuda", torch.float32)
        self._lr_r = int(r)
        a_all = {n: np.empty((o, pairs * r), np.float32) for n, (o, _) in shapes}
        b_all = {n: np.empty((i, pairs * r), np.float32) for n, (_, i) in shapes}
        for j in range(pairs):
            for n, (a, b) in pair_factors(seed, j, shapes, r).items():
                a_all[n][:, j * r:(j + 1) * r] = a
                b_all[n][:, j * r:(j + 1) * r] = b
        self._lr_a = {n: torch.from_numpy(x).cuda() for n, x in a_all.items()}
        self._lr_b = {n: torch.from_numpy(x).cuda() for n, x in b_all.items()}
        return len(self._lr_base)

    def delta(self, name, k):
        """sum_j k_j a_j b_j^T on one leaf, f32."""
        kk = torch.as_tensor(k, dtype=torch.float32, device="cuda").repeat_interleave(self._lr_r)
        p = kk.numel()
        return (self._lr_a[name][:, :p] * kk) @ self._lr_b[name][:, :p].T

    def lr_norm(self, k):
        return float(math.sqrt(sum(float((delta(self, n, k) ** 2).sum()) for n in self._lr_base)))

    def lr_load(self, k):
        """The engine's perturbed leaves become `base + sum_j k_j a_j b_j^T`, in bf16."""
        self._lr_k = list(k)
        for n in sorted(self._lr_base):
            w = (self._lr_base[n] + delta(self, n, k)).to(torch.bfloat16)
            self.model_runner.model.load_weights(weights=[(n, w)])
            torch.cuda.synchronize()
            del w

    def lr_check(self):
        """Bit-for-bit: the engine's perturbed leaves against the last `lr_load`."""
        bad, seen = [], 0
        for name, param in self.model_runner.model.named_parameters():
            parts = [name]
            for packed, pieces in PACKED.items():
                if f".{packed}." in name:
                    parts = [name.replace(packed, p) for p in pieces]
            if not all(p in self._lr_base for p in parts):
                continue
            got, off = param.detach(), 0
            for p in parts:
                w = (self._lr_base[p] + delta(self, p, self._lr_k)).to(torch.bfloat16)
                if not torch.equal(got[off:off + w.shape[0]], w):
                    bad.append(p)
                off += w.shape[0]
                seen += 1
        return {"checked": seen, "mismatched": bad[:10], "ok": seen == len(self._lr_base) and not bad}

    return {"lr_setup": lr_setup, "lr_norm": lr_norm, "lr_load": lr_load, "lr_check": lr_check}
