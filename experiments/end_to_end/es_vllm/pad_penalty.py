"""HF's repetition penalty on the left padding, for vLLM (qiu_match.py).

HF's `generate` pads a batch on the left and its repetition penalty reads every id in
`input_ids`, padding included, so in a padded row the pad token (Qwen2.5's <|endoftext|>,
also an end token) is penalized like a prompt token. vLLM never pads. A request asks for
the same with `extra_args={"pad_penalty": [token, penalty]}`; vLLM's own repetition
penalty covers the prompt and output tokens, and the pad token is in neither (the prompts
do not hold it and decoding stops on it), so nothing is penalized twice.
"""

import torch
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor

from es_vllm.qiu_match import penalize


class PadPenalty(LogitsProcessor):
    def __init__(self, _, device: torch.device, is_pin_memory: bool):
        self.device = device
        self.rows: dict[int, tuple] = {}
        self.index = (self._tensor([], torch.int64), self._tensor([], torch.int64))
        self.penalty = self._tensor([], torch.float32)

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        if process_dict_updates(self.rows, batch_update,
                                lambda params, _, __: tuple((params.extra_args or {}).get("pad_penalty") or ()) or None):
            rows = list(self.rows)
            self.index = (self._tensor(rows, torch.int64),
                          self._tensor([self.rows[r][0] for r in rows], torch.int64))
            self.penalty = self._tensor([self.rows[r][1] for r in rows], torch.float32)

    def _tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return async_tensor_h2d(data, device=self.device, dtype=dtype)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.rows:
            logits[self.index] = penalize(logits[self.index], self.penalty)
        return logits
