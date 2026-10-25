"""es_vllm/grpo_run.py: the advantages, the kl3 term and the analysis, no GPU."""

import cProfile  # noqa: F401  (torch.save imports it; experiments/phase2/profile.py shadows the stdlib module once a phase-2 test is on sys.path)
import json
import pathlib
import sys

import numpy as np
import pytest

E2E = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "end_to_end"
sys.path.insert(0, str(E2E))

from es_vllm import grpo_run as gr  # noqa: E402


def test_advantages_are_group_normalized():
    r = np.array([[0, 10, 10, 0], [10, 10, 10, 10], [0, 0, 0, 10]], dtype=float)
    a = gr.advantages(r)
    assert np.allclose(a.mean(axis=1), 0) and np.allclose(a[0].std(ddof=1), 1, atol=1e-6)
    assert np.all(a[1] == 0)                                   # a dead group has no advantage
    assert a[2, 3] > 0 and (a[2, :3] < 0).all()


def test_kl3_is_nonnegative_and_zero_at_equality():
    torch = pytest.importorskip("torch")                     # CI runs the rest without it
    lp = torch.tensor([-1.0, -2.0, -0.5])
    assert torch.allclose(gr.kl3(lp, lp), torch.zeros(3))
    k = gr.kl3(lp, lp + torch.tensor([0.3, -0.3, 1.0]))
    assert (k >= 0).all() and k[2] > k[0] > 0


def test_analysis_reports_curves_cost_and_class(tmp_path):
    rng = np.random.default_rng(0)
    arm = tmp_path / "arm"
    arm.mkdir()
    P = 1500
    kinds = ["countdown"] * P + ["gsm8k"] * (P // 2)
    block = ["validation"] * (P // 2) + ["test"] * (P // 2) + ["offtask"] * (P // 2)
    (arm / "run.json").write_text(json.dumps({"config": {"setting": "qwen3b/countdown", "seed": 0, "lr": 5e-7},
                                                   "prediction": {"source": "x", "ratio_es_over_rl_rollouts": 2.8}}))
    evs = []
    off = (rng.random(P // 2) < 0.8) * 10.0                    # the off-task block does not move
    for g in (0, 5, 10):
        on = (rng.random(P) < 0.14 + 0.016 * g) * 1.1
        evs.append({"iteration": g, "rewards": on.tolist() + off.tolist(), "kinds": kinds, "block": block, "mean_len": 300.0})
    (arm / "eval.jsonl").write_text("\n".join(json.dumps(e) for e in evs) + "\n")
    logs = [{"iteration": g, "rewards": [[0.1] * 16] * 48, "mean_reward": 0.1 + 0.01 * g, "live_prompts": 40, "mean_len": 300.0,
             "stopped": 1.0, "loss": [0.0, 0.0], "kl": [0.0, 1e-4], "grad_norm": [0.5, 0.5], "clipped_share": [0.0, 0.01],
             "tokens": 768 * 1000, "rollouts": 768, "weights_loaded": 434,
             "seconds": {"generate": 20.0, "score": 2.0, "old_logprobs": 15.0, "train": 60.0, "sync": 8.0}, "prompt_digest": "d"}
            for g in range(10)]
    (arm / "log.jsonl").write_text("\n".join(json.dumps(r) for r in logs) + "\n")
    a = gr.analyze(arm)
    assert a["steps_done"] == 10 and a["rollouts_total"] == 7680
    assert abs(a["cost"]["seconds_per_step"] - 105.0) < 1e-9 and abs(a["cost"]["gpu_seconds_per_answer"] - 105.0 / 768) < 1e-9
    last = a["eval"][-1]
    assert abs(last["by_block"]["test"]["change"] - 0.16) < 3 * last["by_block"]["test"]["change_se"] + 0.01
    assert a["observed"]["class"] == "works" and not a["observed"]["collapsed"]
    assert a["rule_rl_side"]["rollouts_per_step_reference"] == 768 and a["rule_rl_side"]["ratio_es_over_rl_predicted"] == 2.8
    # one row per arm directory, with the first evaluation at or above the target
    rows = gr.summary(tmp_path)
    assert len(rows) == 1 and rows[0]["target_first_at"] == 10 and rows[0]["class"] == "works" and rows[0]["rollouts"] == 7680
    t = gr.table(rows).splitlines()
    assert len(t) == 3 and "| works | no |" in t[2] and "| 7,680 |" in t[2]
    c = gr.curves(tmp_path).splitlines()
    assert c[0] == "**arm** (lr 5e-07, beta 0.01, seed 0, 10 steps, 7,680 rollouts, 0.3 h in the loop)"
    assert c[2] == "| after | validation | test | off-task | answer length |" and len(c) == 7


def test_resume_helpers_truncate_records_and_check_the_config(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text("".join(json.dumps({"iteration": g, "x": g}) + "\n" for g in range(6)))
    assert gr.truncate_records(p, 4) == 2
    assert [r["iteration"] for r in gr.lines(p)] == [0, 1, 2, 3]
    assert gr.truncate_records(p, 4) == 0                     # idempotent
    saved = {"setting": "qwen3b/countdown", "lr": 1e-6, "beta": 0.01, "steps": 40, "eval_every": 5, "seed": 0}
    assert gr.incompatible(saved, {**saved, "steps": 480, "eval_every": 10}) == []   # an extension
    assert gr.incompatible(saved, {**saved, "lr": 5e-7, "seed": 1}) == ["lr", "seed"]
    assert gr.checkpoint_path(pathlib.Path("/scratch"), pathlib.Path("runs/s5/arm")) == pathlib.Path("/scratch/arm/checkpoint.pt")


def test_checkpoint_round_trip_restores_model_optimizer_and_sampler(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    opt.step()
    rng.permutation(5)
    path = tmp_path / "w" / "checkpoint.pt"
    secs = gr.save_checkpoint(path, 7, model, opt, rng, [3, 1, 2], {"lr": 1e-3})
    assert path.exists() and not path.with_suffix(".tmp").exists() and secs >= 0
    saved = gr.load_checkpoint(path)
    assert saved["step"] == 7 and saved["order"] == [3, 1, 2] and saved["config"] == {"lr": 1e-3}
    model2 = torch.nn.Linear(3, 2)
    model2.load_state_dict(saved["model"])
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), model2.state_dict().values()))
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    opt2.load_state_dict(saved["opt"])
    assert opt2.state_dict()["state"][0]["step"] == opt.state_dict()["state"][0]["step"]
    rng2 = np.random.default_rng(1)
    rng2.bit_generator.state = saved["rng"]
    assert (rng2.permutation(10) == rng.permutation(10)).all()  # the same next draw
