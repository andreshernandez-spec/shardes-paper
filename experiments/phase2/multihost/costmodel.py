"""The contraction time model, shared by the two E18 predictors.

    t_A - t_B = C(N, d) (D - 1) / D + ag(4N) - ar(4P)

`eval` cancels between the placements: `how` changes what is communicated and not what is
computed, so `ask` and `apply` are identical and the difference isolates the contraction
(`docs/11-cost-model.md`). Going from a measured D=8 anchor to a predicted D:

    delta(D) = delta_8 + [ar_D(4P) - ar_8(4P)] - C (1/8 - 1/D)

with delta = t_B - t_A throughout. The bracket the predictors used before this existed
put the contraction term in [0, |delta_8|], because `C` had never been measured. With
`contraction_isolation.py` on disk it is a number, and the prediction stops being a range.

`contraction_bracket` keeps the old behaviour, so a missing record degrades to the coarse
prediction rather than to no prediction, and every record says which one it used.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACTION = HERE.parent / "results-contraction"


def params_bytes(d_model: int) -> int:
    """Six square float32 matrices: the block's parameter payload."""
    return 4 * 6 * d_model * d_model


def ladder_seconds(alpha: float, beta: float, nbytes: float) -> float:
    return alpha + nbytes / beta


def ladder_payload_bytes(size_bytes: int, devices: int, rec: dict | None = None) -> int:
    """What one preflight ladder point labelled `size_bytes` actually all-reduced.

    Records that carry `ladder_payload_bytes` say so themselves. Every record before that
    field existed (all of E18's) summed a (devices, n/devices + 1) array over its sharded
    axis, so the reduced vector was n/devices + 1 floats: 1/devices of the label. Checked
    against the compiled HLO on 8 devices: the "1 MiB" point lowers to f32[32769].
    """
    if rec is not None and "ladder_payload_bytes" in rec:
        return rec["ladder_payload_bytes"][str(size_bytes)]
    n_f32 = max(size_bytes // 4, 2)
    return 4 * (n_f32 // devices + 1)


def ladder_alpha_beta(rec: dict, devices: int) -> tuple[float, float]:
    """(alpha, beta) of a preflight record, beta at the payload the ladder really moved.

    The record's own `beta_bytes_per_second` divides the labelled 100 MiB by the time,
    which overstates the fabric by `devices` for every record without
    `ladder_payload_bytes`. The frozen E18 predictions used that figure, and stay as made.
    """
    t = rec["allreduce_seconds_by_bytes"]
    big, small = 100 * 2**20, 8
    moved = (ladder_payload_bytes(big, devices, rec)
             - ladder_payload_bytes(small, devices, rec))
    return t[str(small)], moved / max(t[str(big)] - t[str(small)], 1e-9)


def measured_contraction(strategy: str, d_model: int, n: int, devices: int = 8,
                         kind: str | None = None) -> float | None:
    """`C` in seconds from `contraction_isolation.py`, or None if that cell never ran.

    Matched on the anchor's device count, not the predicted one: `C` is the REPLICATED
    contraction, which is the same work on every device and does not depend on D. Any
    A100 record for the cell will do, so `kind` is optional.
    """
    if not CONTRACTION.is_dir():
        return None
    pattern = f"d={d_model}__N={n}__s={strategy}__*__D{devices}.json"
    for path in sorted(CONTRACTION.glob(pattern)):
        rec = json.loads(path.read_text())
        if "failed" in rec or rec.get("contraction_seconds") is None:
            continue
        if kind and rec.get("config", {}).get("strategy") != strategy:
            continue
        return rec["contraction_seconds"]
    return None


def predict_delta(delta_8: float, bump: float, devices: int,
                  contraction: float | None) -> dict:
    """delta(D) = t_B - t_A at `devices`, from the D=8 anchor and the fabric penalty.

    Returns a point when `contraction` is measured and the old bracket when it is not.
    The sign is what H2, H3 and G2 are judged on; the magnitude is reported either way so
    a miss is visible.
    """
    if contraction is not None:
        gain = contraction * (1.0 / 8.0 - 1.0 / devices)  # B's extra split past D=8
        point = delta_8 + bump - gain
        return {"delta_predicted": point, "delta_bracket": [point, point],
                "contraction_seconds": contraction, "contraction_source": "measured",
                "predicted_sign_B_minus_A": "B_wins" if point < 0 else "A_wins"}
    lo, hi = delta_8 + bump - abs(delta_8), delta_8 + bump
    mid = (lo + hi) / 2
    return {"delta_predicted": mid, "delta_bracket": [lo, hi],
            "contraction_seconds": None, "contraction_source": "bracketed",
            "predicted_sign_B_minus_A": "B_wins" if mid < 0 else "A_wins"}


def flip_bandwidth(ag: float, alpha: float, d_model: int, devices: int,
                   contraction: float | None) -> float | None:
    """The beta at which this cell's sign flips, or None without a measured `C`.

    Setting delta(D) = 0 and solving the fabric term for beta:

        alpha + 4P / beta = ag + C (D - 1) / D

    A negative or zero denominator means the contraction alone already exceeds what the
    latency floor costs, so no achievable bandwidth flips it: returns None, which the
    caller reports as "no flip".
    """
    if contraction is None:
        return None
    budget = ag + contraction * (devices - 1) / devices - alpha
    if budget <= 0:
        return None
    return params_bytes(d_model) / budget
