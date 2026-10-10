# Gain and cost of an ES update by matrix type, 2026-10-10

Stage A1 of `docs/end_to_end/09`. `es_vllm/block_check.py` at 481597f (the analysis's usable
window and clipped multipliers added afterwards, see the commit), prediction in the
docstring committed before the run, on a Secure Cloud H200 (US-CO-1, driver 580.178.04,
$5.29/h, 20:22 to 22:25 UTC together with stage A2), about $11 of the pod's $11.
Records `points.jsonl`, `norms.json`, `env.json`; logs `pod-log*.txt.gz`. Numbers:

    python -m es_vllm.block_check --analyze runs/block-check

The N = 1,024 update of `../lowrank-check/` restricted to one matrix type at a time,
applied as `theta_0 +- mu u_t / |u_t|` at two lengths per type on the 3,840 held-out
prompts (start 5.461), the full update at 44.7 and 111.7 on the same engine. A length is
usable when neither sign moved the reward by more than two points; v_proj at 56 broke the
model (1.6 and 1.8) and down_proj at 111.7 did not (5.02 and 4.26). From the longest usable
length: the slope `a_t` (odd part per unit length) and curvature `b_t` (even part per unit
length squared) along the type's component, and the unit update's gain `G_t = a_t |u_t|`
and cost `C_t = b_t |u_t|^2` from that type.

| type | share of parameters | length used | gain `G_t` (SE) | cost `C_t` (SE) | share of gain | share of cost | multiplier, clipped |
|---|---|---|---|---|---|---|---|
| down_proj | 27% | 111.7 | 2.47e-3 (2.4e-4) | -3.5e-5 (2.5e-6) | 43% | 47% | 1.17 |
| up_proj | 27% | 111.7 | 1.08e-3 (2.0e-4) | -7.1e-6 (2.0e-6) | 19% | 10% | 2.52 |
| gate_proj | 27% | 111.7 | 4.3e-4 (1.9e-4) | -3.0e-6 (1.8e-6), unresolved | 7% | 4% | 2.34 |
| q_proj | 8% | 80 | -0.4e-4 (1.4e-4) | -2.3e-6 (1.0e-6) | -1% | 3% | 0.25 |
| o_proj | 8% | 80 | 5.6e-4 (1.6e-4) | -9.3e-6 (1.2e-6) | 10% | 13% | 0.99 |
| k_proj | 2% | 56 | 2.9e-4 (1.0e-4) | -0.4e-6 (0.5e-6), unresolved | 5% | 1% | 4.0 |
| v_proj | 2% | 22 | 9.7e-4 (2.8e-4) | -1.7e-5 (3.8e-6) | 17% | 23% | 0.95 |
| sum | | | 5.8e-3 | -7.4e-5 | | | |
| full update | 100% | 111.7 | 4.67e-3 (4.4e-4) | -7.7e-5 (8.5e-6) | | | 1 |

- **The predictions hold.** v_proj's share of the gain is 8.8 times its share of the
  parameters (at least 5 predicted). The best net per update with a multiplier per type,
  `sum_t G_t^2 / (4 C_t)`, is 2.4 times the uniform `G^2 / (4 C)` (1.5 to 3 predicted), 1.5
  over the four types whose cost is resolved. The sums of `G_t` and `C_t` match the full
  update's within 1.6 and 0.3 standard errors; the types measured at shorter lengths have
  slightly larger slopes, as the full update does at 44.7 against 111.7.
- **Where the gain comes from is not where the prior said.** v_proj carries 17% of the
  gain with 2% of the parameters, but its curvature per unit length squared is 11 times
  the full update's (-4.4e-4 against -3.9e-5), so it carries 23% of the cost: in the most
  sensitive matrices the cost tracks the gradient's density, as a Fisher metric predicts,
  and its multiplier is 0.95. The improvement comes from up_proj (19% of the gain, 10% of
  the cost, multiplier 2.5), the removal of q_proj (no gain, 3% of the cost), and the two
  unresolved types, gate and k, whose costs are within two standard errors of zero and
  whose multipliers (2.3 and 10.8, clipped to 4) carry that uncertainty.
- With the clipped multipliers at the uniform best step (30.5 times the unit update), the
  model predicts a net of 0.15 per update against 0.071, with k_proj's 0.03 the least
  certain part. If k_proj's cost is two standard errors larger than measured, its net at
  the clipped multiplier is still positive; gate_proj's would be about zero.
- The multipliers go into stage B as `docs/end_to_end/09` fixed before this run: the ratio
  is above 1.3 and the sums check. `--type-steps '{"q_proj": 0.25, "k_proj": 4.0,
  "v_proj": 0.952, "o_proj": 0.985, "gate_proj": 2.336, "up_proj": 2.522, "down_proj":
  1.167}'`.
