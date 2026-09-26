"""vLLM worker extension: shardes' ES state lives here, beside the engine's weights.

Loaded through `worker_extension_cls` and driven with `llm.collective_rpc`. With
`VLLM_ENABLE_V1_MULTIPROCESSING=0` and one GPU, the engine, this worker and the driver
share one process, and JAX shares the GPU with torch (`XLA_PYTHON_CLIENT_PREALLOCATE=
false`).

What each call does, in shardes' terms (`ShardedES` with `SeedRegenerated`, one device):

- `es_init`: the f32 master, from the same safetensors vLLM loaded, keyed by HF names.
- `es_ask`: `base_key = fold_in(key, generation)`, members `arange(n)`, as `ask` does.
- `es_perturb(i)`: member `i`'s bf16 weights, leaf by leaf (`stream.member_leaf`), written
  into the engine through its own `load_weights`, which knows the fused q/k/v and
  gate/up layouts.
- `es_restore`: the view (master cast to bf16) written back the same way. Restore never
  subtracts noise: a bf16 add and subtract leaves residue.
- `es_tell(rewards)`: `ShardedES.tell` on fitness `-reward` shaped `(n, 1)` with
  `group_relative` (Qiu's z-score), leaf by leaf (`stream.updated_leaf`).
- `es_check(i)`: reads the engine's weights back and compares them bit for bit with
  member `i` (or the view, for `None`), through the same fused layouts.

JAX arrays reach torch by DLPack into buffers the engine then copies from; nothing
mutates a JAX-owned buffer from torch.
"""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from shardes.shaping import group_relative

from . import stream

PACKED = {  # vLLM fused parameter -> the HF leaves it holds, in order along dim 0
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}


def _torch(x):
    return torch.utils.dlpack.from_dlpack(x)


def load_master(model_dir: str) -> dict:
    """Every tensor of the checkpoint as an f32 JAX array on the GPU, one at a time."""
    from safetensors import safe_open  # noqa: PLC0415

    master = {}
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors in {model_dir}")
    for f in files:
        with safe_open(str(f), framework="pt", device="cuda") as st:
            for name in st.keys():
                t = st.get_tensor(name)
                master[name] = jnp.asarray(jax.dlpack.from_dlpack(t)).astype(jnp.float32)
                del t
    return master


class ESWorker:
    # ---------------------------------------------------------------- state

    def es_init(self, model_dir: str, n: int, sigma: float, lr: float, seed: int) -> dict:
        self._es_master = load_master(model_dir)
        self._es_names = stream.leaf_names(self._es_master)
        self._es_n, self._es_sigma, self._es_lr = int(n), float(sigma), float(lr)
        self._es_key = jax.random.key(seed)
        self._es_generation = 0
        self._es_base_key = None
        engine = {name for name, _ in self.model_runner.model.named_parameters()}
        return {"leaves": len(self._es_names),
                "params": int(sum(v.size for v in self._es_master.values())),
                "engine_params": len(engine)}

    def es_ask(self) -> int:
        self._es_base_key = jax.random.fold_in(self._es_key, self._es_generation)
        self._es_streams = stream.streams(self._es_base_key, len(self._es_names))
        self._es_generation += 1
        return self._es_generation

    # ---------------------------------------------------------------- weights

    def _member(self, i):
        for k, name in enumerate(self._es_names):
            leaf = stream.member_leaf(self._es_master[name], self._es_streams[k], i,
                                      self._es_sigma)
            yield name, _torch(leaf)

    def _view(self):
        for name in self._es_names:
            yield name, _torch(self._es_master[name].astype(stream.COMPUTE))

    def _load(self, weights) -> None:
        self.model_runner.model.load_weights(weights=weights)
        torch.cuda.synchronize()

    def es_perturb(self, i: int) -> None:
        self._load(self._member(jnp.int32(i)))

    def es_restore(self) -> None:
        self._load(self._view())

    # ---------------------------------------------------------------- update

    def es_tell(self, rewards: list) -> dict:
        reward = jnp.asarray(rewards, dtype=jnp.float32)
        if reward.shape != (self._es_n,):
            raise ValueError(f"{reward.shape} rewards for {self._es_n} members")
        weights = group_relative((-reward)[:, None])  # tell descends; (n, 1) is a z-score
        ids = jnp.arange(self._es_n, dtype=jnp.int32)
        for k, name in enumerate(self._es_names):
            self._es_master[name] = stream.updated_leaf(
                self._es_master[name], self._es_streams[k], ids, weights,
                self._es_lr, self._es_sigma, self._es_n)
        return {"weights_abs_sum": float(jnp.abs(weights).sum())}

    # ---------------------------------------------------------------- checks

    def _expected(self, i, name):
        """The bf16 array the engine should hold for HF leaf `name`, one leaf at a time."""
        if i is None:
            return _torch(self._es_master[name].astype(stream.COMPUTE))
        k = self._es_index[name]
        return _torch(stream.member_leaf(self._es_master[name], self._es_streams[k],
                                         jnp.int32(i), self._es_sigma))

    def es_check(self, i=None) -> dict:
        """Bit-for-bit comparison of the engine's weights with member `i` or the view."""
        self._es_index = {name: k for k, name in enumerate(self._es_names)}
        seen, bad = set(), []
        for name, param in self.model_runner.model.named_parameters():
            parts = [name]
            for packed, pieces in PACKED.items():
                if f".{packed}." in name:
                    parts = [name.replace(packed, p) for p in pieces]
            if not all(p in self._es_index for p in parts):
                continue  # e.g. a tied lm_head, compared through embed_tokens
            got = param.detach()
            offset = 0
            for p in parts:
                w = self._expected(i, p)
                rows = w.shape[0]
                if not torch.equal(got[offset:offset + rows], w):
                    bad.append(p)
                offset += rows
                seen.add(p)
            if offset != got.shape[0]:
                bad.append(f"{name}: {offset} rows expected, engine has {got.shape[0]}")
        missing = sorted(set(self._es_names) - seen)
        return {"checked": len(seen), "mismatched": bad[:10], "unchecked": missing[:10],
                "ok": not bad and not missing}

    def es_digest(self) -> str:
        """sha256 over the view's bytes, leaf by leaf in name order: what a rebuild from
        the fitness log must reproduce."""
        h = hashlib.sha256()
        for name in self._es_names:
            v = np.asarray(self._es_master[name].astype(stream.COMPUTE).view(jnp.uint16))
            h.update(name.encode())
            h.update(v.tobytes())
        return h.hexdigest()

    def es_status(self) -> str:
        return json.dumps({"generation": self._es_generation, "n": self._es_n,
                           "sigma": self._es_sigma, "lr": self._es_lr})
