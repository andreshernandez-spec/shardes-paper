"""`paper_evidence.py`'s results macros: rounded once, half up, from the exact value.

The abstract once said 1.5x for an exact 1.449: it rounded the section's already-rounded
1.45 a second time. The macros round the exact value, and `round` would not do: it rounds
half to even, so 14.5 would become 14.
"""

import importlib.util
import pathlib
import sys

PHASE2 = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "phase2"
sys.path.insert(0, str(PHASE2))
spec = importlib.util.spec_from_file_location("paper_evidence", PHASE2 / "paper_evidence.py")
pe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pe)


def test_the_exact_value_is_rounded_once():
    assert pe.rounded(1.44938, 1) == "1.4"
    assert pe.rounded(1.44938, 2) == "1.45"


def test_halves_round_up_not_to_even():
    assert pe.rounded(14.5, 0) == "15"
    assert pe.rounded(2.25, 1) == "2.3"


def test_a_span_is_a_latex_range():
    assert pe.span(4.4325, 14.528, 0) == "4--15"


def platform(resolution, ideal_misses=()):
    resolved = sum(row[3] for row in resolution)
    audit = {"measured_components": {"correct": resolved, "resolved_correct": resolved},
             "ideal_isolated": {"correct": resolved - len(ideal_misses),
                                "resolved_misses": [list(m) for m in ideal_misses]},
             "ideal_in_context": {"correct": resolved, "resolved_misses": []}}
    return {"drawn_full_rank_speedup": [1.45, 2.4], "dense_speedup": [1.8, 2.4],
            "large_lowrank_split_slower_pct": [4.4, 14.5], "resolution": resolution,
            "large_lowrank_naive_ms": [-0.49, 0.18], "large_lowrank_measured_ms": [-1.63, -0.84],
            "configurations": len(resolution), "resolved": resolved, "audit": audit}


def block(gpu, tpu):
    return {"8x A100-SXM4-80GB (NVLink)": platform(gpu), "TPU v5e-8 (ICI)": platform(tpu),
            "matched_precision": {"rank1_over_dense": [0.43, 0.17], "count": 7}}


def test_the_unresolved_count_covers_only_narrow_low_rank():
    rows = [["iid_gaussian", 2048, 256, True], ["mirrored_lr1", 2048, 256, True],
            ["mirrored_lr1", 512, 256, False], ["lowrank_r1", 512, 1024, True]]
    text = pe.results_latex(block(rows, rows))
    assert "\\newcommand{\\NarrowLowRank}{four}" in text
    assert "\\newcommand{\\NarrowLowRankUnresolved}{two}" in text


def test_an_unresolved_comparison_the_text_calls_resolved_stops_the_build():
    import pytest
    rows = [["seed_regenerated", 512, 256, False], ["mirrored_lr1", 512, 256, False]]
    with pytest.raises(SystemExit, match="seed_regenerated"):
        pe.results_latex(block(rows, rows))


def test_a_resolved_miss_of_eq_2_stops_the_build():
    import pytest
    rows = [["mirrored_lr1", 512, 256, False], ["lowrank_r1", 512, 256, True]]
    b = block(rows, rows)
    b["TPU v5e-8 (ICI)"]["audit"]["measured_components"]["resolved_correct"] = 0
    with pytest.raises(SystemExit, match="every miss of Eq. 2 is an unresolved"):
        pe.results_latex(b)


def versions(ideal_misses):
    """A100 has the mirrored full-rank variant and fewer unresolved comparisons, as the text says."""
    gpu = platform([["mirrored_seed", 2048, 256, True], ["mirrored_lr1", 512, 256, True],
                    ["lowrank_r1", 2048, 256, True]], ideal_misses)
    tpu = platform([["mirrored_lr1", 512, 256, False], ["lowrank_r1", 2048, 256, True]])
    tm = {k: {"configurations": p["configurations"], "correct": p["audit"]["measured_components"]["correct"]}
          for k, p in (("GPU", gpu), ("TPU", tpu))}
    return gpu, tpu, tm


def test_the_simpler_versions_named_misses_must_be_the_measured_ones():
    import pytest
    named = [("lowrank_r1", 2048, 256), ("mirrored_lr1", 512, 256)]
    assert pe.simpler_versions(*versions(named))["IdealLowRankHiGPU"] == "+0.18"
    with pytest.raises(SystemExit, match="names the resolved A100 misses"):
        pe.simpler_versions(*versions(named[:1]))
