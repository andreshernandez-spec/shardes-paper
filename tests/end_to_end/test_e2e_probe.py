"""The throughput probe's config and pure parts. No vLLM, no GPU, no network.

The config is checked against the run it imitates: the Olmo 3 cap, stop strings and
context length must be the logged ones, or the probe measures a different workload
from the one ES will run.
"""

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest
import yaml

E2E = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "end_to_end"
PROBE = E2E / "probe"
sys.path.insert(0, str(E2E))
sys.path.insert(0, str(E2E.parent))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


probe = load(PROBE / "probe_throughput.py", "e2e_probe_throughput")
releases = load(E2E / "releases.py", "e2e_releases_probe")
templates = load(E2E / "templates.py", "e2e_templates_probe")
CFG = yaml.safe_load((PROBE / "probe.yaml").read_text())
SAMPLING = yaml.safe_load((PROBE / "probe-sampling.yaml").read_text())


def test_every_setting_names_a_pinned_model_and_template():
    for name, spec in CFG["settings"].items():
        assert isinstance(getattr(releases, spec["model"]), releases.Repo), name
        assert isinstance(getattr(templates, spec["template"]), str), name
        for cell in spec["cells"]:
            assert set(cell) == {"gpu_memory_utilization", "prompts_per_member", "members"}
            assert 0.3 <= cell["gpu_memory_utilization"] <= 0.95


def test_olmo3_cell_matches_the_logged_run():
    logged = json.loads((E2E / "pins" / "wandb-wn9zgjj3-config.json").read_text())["config"]
    spec = CFG["settings"]["olmo3_if"]
    assert spec["max_tokens"] == logged["response_length"]
    assert spec["stop"] == logged["stop_strings"]
    assert spec["add_bos"] == logged["add_bos"]
    assert spec["max_model_len"] == logged["max_prompt_token_length"] + logged["response_length"]
    assert spec["cells"][0]["prompts_per_member"] == logged["num_unique_prompts_rollout"]


def test_cell_names_are_unique():
    names = [probe.cell_name(s, c) for s, spec in CFG["settings"].items()
             for c in spec["cells"]]
    assert len(names) == len(set(names))


def test_summary_counts_cap_hits_and_quantiles():
    s = probe.summarize([10, 20, 30, 16384], ["stop", "stop", "stop", "length"])
    assert s["cap_hits"] == 1 and s["max"] == 16384 and s["sequences"] == 4
    assert s["median"] == 25.0


class StubTokenizer:
    bos_token_id = 7

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, chat_template):
        assert tokenize is False and add_generation_prompt is True
        return "|".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False, "open-instruct encodes without special tokens"
        return {"input_ids": [len(w) for w in text.split("|")]}


@pytest.mark.parametrize("bos", [False, True])
def test_render_adds_bos_only_when_asked(bos):
    ids = probe.render(StubTokenizer(), "t", [{"content": "abc", "role": "user"}], bos)
    assert ids == ([7] if bos else []) + [3]


def test_pod_script_parses():
    subprocess.run(["bash", "-n", str(PROBE / "pod.sh")], check=True)


cost = load(PROBE / "cost.py", "e2e_probe_cost")


def test_cost_counts_member_batches_and_skips_smoke(tmp_path, monkeypatch):
    rec = {"setting": "olmo3_if", "smoke": False,
           "env": {"torch_device": "NVIDIA A100-SXM4-80GB"},
           "cell": {"gpu_memory_utilization": 0.85, "prompts_per_member": 32, "members": 2},
           "spec": {},
           "members": [{"wall_seconds": 30.0, "weight_rewrite": {"seconds": 0.0}}] * 2,
           "lengths": {"mean": 300.0, "cap_hits": 0, "sequences": 64}}
    d = tmp_path / "results-x"
    d.mkdir()
    (d / "a.json").write_text(json.dumps(rec))
    (d / "smoke.json").write_text(json.dumps({**rec, "smoke": True}))
    monkeypatch.setattr(cost, "PROBE", tmp_path)
    rows = [r for r in cost.table(["results-x"]) if r["run"].startswith("step 2000")]
    assert len(rows) == 1, "the smoke record must not count"
    # 512,000 rollouts / 32 per member = 16,000 members x 30 s, x1.2 uptime
    assert rows[0]["gpu_hours"] == pytest.approx(16_000 * 30 / 3600 * 1.2)


def test_sampling_probe_differs_from_the_greedy_one_only_in_sampling():
    greedy, sampled = CFG["settings"]["olmo3_if"], SAMPLING["settings"]["olmo3_if"]
    logged = json.loads((E2E / "pins" / "wandb-wn9zgjj3-config.json").read_text())["config"]
    assert sampled["temperature"] == logged["temperature"]
    extra = {"temperature", "sampling_seed"}
    assert {k: v for k, v in sampled.items() if k not in extra | {"cells"}} == \
        {k: v for k, v in greedy.items() if k != "cells"}
    assert sampled["cells"] == greedy["cells"][:1]
