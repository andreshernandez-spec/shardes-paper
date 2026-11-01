"""es_vllm/qiu_match.py without a GPU: which rows HF pads, the pad penalty against HF's own
processor, his problem slices and his grading of a whole decoded sequence."""

import pathlib
import sys

import pytest

E2E = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "end_to_end"
sys.path.insert(0, str(E2E))

from es_vllm import qiu_match as qm  # noqa: E402

CLONE = pathlib.Path.home() / "private" / "open-source" / "es-at-scale"
needs_clone = pytest.mark.skipif(not (CLONE / "archive" / "countdown" / "data" / "countdown.json").exists(),
                                 reason="needs the es-at-scale clone")


def test_padded_rows_are_the_shorter_ones_of_each_batch():
    assert qm.padded([5, 7, 7, 4, 4, 4, 6], 3) == [True, False, False, False, False, False, False]
    assert qm.padded([5, 7, 7, 3, 4, 4, 6], 3) == [True, False, False, True, False, False, False]
    assert qm.padded([2, 3], 8) == [True, False]


def test_answers_are_compared_up_to_their_first_end_token():
    from es_vllm import qiu_decode_check as dc

    assert dc.trim([5, 6, 9, 0, 0], {9, 0}) == [5, 6]       # HF pads after the end token
    assert dc.trim([5, 6, 9], {9}) == [5, 6]                 # vLLM keeps the end token
    assert dc.trim([5, 6], {9}) == [5, 6]                    # cut at the token limit
    a = dc.agreement([[1, 2, 3], [1, 2, 4], [1, 5], [1]], [[1, 2, 3], [1, 2, 3], [1, 2], [1, 2]])
    assert a == {"identical": 1, "first_difference_median": 1.0}   # parted at 2, 1 and 1


def test_vllm_penalty_plus_pad_penalty_is_hfs_on_a_padded_batch():
    """HF's penalty on a left-padded batch equals its penalty on each unpadded row (what
    vLLM's own penalty does) with `penalize` on the pad token of the padded rows."""
    torch = pytest.importorskip("torch")
    from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

    pad, p, vocab = 0, 1.05, 11
    rows = [[3, 4, 5, 6], [7, 8]]
    batch = torch.tensor([[3, 4, 5, 6], [pad, pad, 7, 8]])
    scores = torch.randn(2, vocab, generator=torch.Generator().manual_seed(0)) * 3
    scores[0, pad], scores[1, pad] = 2.5, -1.5          # a positive and a negative pad logit
    hf = RepetitionPenaltyLogitsProcessor(p)(batch, scores.clone())
    ours = torch.stack([RepetitionPenaltyLogitsProcessor(p)(torch.tensor([r]), scores[i:i + 1].clone())[0]
                        for i, r in enumerate(rows)])
    flags = qm.padded([len(r) for r in rows], 2)
    idx = torch.tensor([i for i, f in enumerate(flags) if f])
    ours[idx, pad] = qm.penalize(ours[idx, pad], torch.full((len(idx),), p))
    assert flags == [False, True]
    assert torch.equal(hf, ours)
    assert hf[1, pad].item() == pytest.approx(-1.5 * p) and hf[0, pad] == 2.5   # only the padded row's pad moved


@needs_clone
def test_his_slices():
    train, test = qm.problems(CLONE, qm.TRAIN, qm.TEST)
    assert len(train) == 200 and len(test) == 2000
    assert train[0]["context"].endswith("<think>") and not {r["id"] for r in train} & {r["id"] for r in test}
    assert sum(not r["target"].isdigit() for r in test) == 58 and all(r["target"].isdigit() for r in train)


@needs_clone
def test_his_grading_of_the_whole_sequence():
    """The decoded text holds the prompt, whose own tags earn every answer 0.06 of format;
    a full-format correct answer gets 1.1; a non-integer target gets no answer point."""
    class Echo:
        def decode(self, ids, skip_special_tokens):
            return ids

    qm._G.update(tok=Echo(), **qm.his_functions(CLONE))
    train, test = qm.problems(CLONE, qm.TRAIN, qm.TEST)
    row = train[0]                                       # [44 19 35] to 98
    good = row["context"] + " 44 + 19 + 35 = 98 </think>\n<answer> 44 + 19 + 35 </answer>"
    assert qm._grade(good, row["context"], row["target"]) == pytest.approx((1.1, 1.0, 1.0))
    assert qm._grade(row["context"] + " no idea", row["context"], row["target"]) == pytest.approx((0.06, 0.0, 0.6))
    odd = next(r for r in test if not r["target"].isdigit())
    nums = odd["context"][odd["context"].find("[") + 1:odd["context"].find("]")].split()
    ans = odd["context"] + f" </think>\n<answer> {' + '.join(nums)} </answer>"
    assert qm._grade(ans, odd["context"], odd["target"])[1] == 0.0
    assert qm.accuracy([(1.1, 1.0, 1.0), (0.06, 0.0, 0.6)]) == 0.5
