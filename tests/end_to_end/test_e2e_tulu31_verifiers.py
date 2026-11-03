"""Tulu 3.1's vendored verifiers: the dispatch and each task on known answers.

The MATH verifier needs sympy with antlr 4.11 (as the run had); where the environment
lacks them these tests skip rather than pass on the verifier's silent False.
"""

import hashlib
import importlib
import json
import pathlib
import sys

import pytest

E2E = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "end_to_end"
sys.path.insert(0, str(E2E))

VENDORED = {  # sha256 of the originals at 3f37c29; only ground_truth_utils' imports differ
    "if_functions.py": "ec9b8d402ac5032470157afaebc55bbf65b1094b5a93778946fd21c2db0a25c4",
    "math_utils.py": "06106b69de337701093053807149d245dbdbc2f1cf82f102d1e72c7a171e2d88",
}


@pytest.mark.parametrize("name,digest", sorted(VENDORED.items()))
def test_vendored_files_are_the_originals(name, digest):
    data = (E2E / "verifiers" / "tulu31" / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest


@pytest.fixture(scope="module")
def v():
    pytest.importorskip("sympy")
    return importlib.import_module("verifiers.tulu31")


def test_gsm8k_takes_the_last_number(v):
    assert v.reward("so 3 + 4 = 7. The answer is 1,234", "1234", "gsm8k") == 10.0
    assert v.reward("The answer is 1234, not 5", "1234", "gsm8k") == 0.0


def test_ifeval_runs_the_named_function(v):
    gt = json.dumps({"func_name": "validate_lowercase"})
    assert v.reward("all lower case here", gt, "ifeval") == 10.0
    assert v.reward("Not All Lower", json.dumps({"func_name": "validate_lowercase"}),
                    "ifeval") == 0.0


def test_unknown_dataset_and_missing_ground_truth_score_zero(v):
    assert v.reward("7", "7", "somethingelse") == 0.0
    assert v.reward("7", None, "gsm8k") == 0.0


def test_math_needs_the_latex_parser(v):
    try:
        v.check_environment()
    except ImportError:
        pytest.skip("antlr4-python3-runtime 4.11 not installed; the run had it")
    assert v.reward(r"so the answer is $\boxed{\frac{1}{2}}$", r"\frac{1}{2}", "MATH") == 10.0
    assert v.reward(r"so the answer is $\boxed{3}$", r"\frac{1}{2}", "MATH") == 0.0
