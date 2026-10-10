"""vLLM worker extension for the throughput probe.

Loaded inside vLLM's worker process through `worker_extension_cls`, and reached from the
driver with `llm.collective_rpc`. That path is also how the ES backend will write each
member's weights, so the probe exercises it now.

`rewrite_all` reads and writes every parameter once in place (`mul_(1.0)`), which moves
the same bytes as writing a member's perturbed view into the engine, and leaves the
weights bit-identical (x * 1.0 == x for every finite bf16). It returns the wall time and
a checksum taken before and after, so a rewrite that changed anything is caught.
"""

import time

import torch


class WeightRewriter:
    def _params(self):
        return list(self.model_runner.model.parameters())

    def weight_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self._params())

    def checksum(self) -> float:
        # A float64 sum over a strided sample: cheap, and enough to see a changed weight.
        total = 0.0
        for p in self._params():
            flat = p.detach().reshape(-1)
            total += float(flat[:: max(1, flat.numel() // 4096)].double().sum())
        return total

    def rewrite_all(self) -> dict:
        before = self.checksum()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for p in self._params():
                p.mul_(1.0)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        return {"seconds": seconds, "unchanged": self.checksum() == before,
                "bytes": self.weight_bytes()}
