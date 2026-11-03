"""Tulu 3.1's rewards, as the GRPO run computed them (open-instruct 3f37c29).

`reward` is `model_utils.apply_verifiable_reward` for one response, without torch: the
response text is decoded with special tokens skipped by the caller, the row's `dataset`
field picks the verifier, and a verified answer scores `verify_reward` (10), anything
else 0. A response that never emitted eos scores `penalty_reward_value` (0.0) instead,
which the caller applies (`--non_stop_penalty`).

The three verifier modules beside this file are copied verbatim from open-instruct at
3f37c29 except for their package imports; see README.md. The MATH verifier needs
sympy's LaTeX parser, and on an ImportError it returns False without a word, so
`check_environment` exists to fail loudly instead.
"""

from .ground_truth_utils import verify_gsm8k_sample, verify_ifeval_sample, verify_math_sample

VERIFY_REWARD = 10
PENALTY_REWARD_VALUE = 0.0


def reward(text: str, ground_truth, dataset: str, verify_reward: int = VERIFY_REWARD) -> float:
    if ground_truth is None:
        return 0.0
    d = dataset.lower()
    if d == "gsm8k":
        ok = verify_gsm8k_sample(text, ground_truth)
    elif d == "math":
        ok = verify_math_sample(text, ground_truth)
    elif d == "ifeval":
        ok = verify_ifeval_sample(text, ground_truth)
    else:
        ok = False
    return float(verify_reward) if ok else 0.0


def check_environment() -> dict:
    """The run had antlr4-python3-runtime 4.11.0 and sympy 1.13.1. Without the parser the
    MATH verifier silently marks LaTeX answers wrong, so refuse to run without it."""
    import sympy
    from sympy.parsing.latex import parse_latex

    parse_latex(r"\frac{1}{2}")  # raises ImportError without antlr 4.11
    return {"sympy": sympy.__version__}
