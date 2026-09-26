"""Phase 0 of the ES-vs-RL paper: pins, templates and the rebuilt IF data path.

No network here. The drivers that touch the hub are tested on their pure parts: the
sampling and iterator logic copied from open-instruct, the pinned identities, and the
shape of the records they write (the provenance audit reads any `commit` key as a
citation, so a stray one would either fail CI or be checked against the wrong repo).
"""

import hashlib
import importlib.util
import json
import pathlib
import re
import sys

import numpy as np
import pytest

E2E = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "end_to_end"
sys.path.insert(0, str(E2E))
sys.path.insert(0, str(E2E.parent))


def load(name):
    spec = importlib.util.spec_from_file_location(f"e2e_{name}", E2E / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"e2e_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


releases = load("releases")
templates = load("templates")
data = load("olmo3_if_data")

HEX40 = re.compile(r"[0-9a-f]{40}")


def test_every_pin_is_a_full_commit_and_unique():
    for rel in releases.ALL_REPOS:
        assert HEX40.fullmatch(rel.commit), rel
        assert rel.kind in ("model", "dataset")
    assert len({r.repo for r in releases.ALL_REPOS}) == len(releases.ALL_REPOS)
    for sha in (releases.OLMO3_IF_OPEN_INSTRUCT, releases.TULU31_OPEN_INSTRUCT):
        assert HEX40.fullmatch(sha)


def test_branch_lists_and_rollouts_match_the_releases():
    assert len(releases.OLMO3_IF_RL_BRANCHES) == 19          # step_100 .. step_1900
    assert releases.OLMO3_IF_RL_BRANCHES[-1] == "step_1900"
    assert len(releases.TULU31_RL_BRANCHES) == 61            # step_40 .. step_2440
    assert releases.OLMO3_IF_ROLLOUTS_PER_STEP * 2000 == 512_000
    assert releases.TULU31_ROLLOUTS_PER_STEP * 1920 == 1_474_560   # the card's episode


def test_templates_are_the_pinned_strings():
    assert hashlib.sha256(templates.OLMO_THINKER.encode()).hexdigest() == \
        templates.OLMO_THINKER_SHA256
    assert hashlib.sha256(templates.TULU.encode()).hexdigest() == templates.TULU_SHA256
    assert templates.OLMO_THINKER.endswith("{% endif %}{% endfor %}")
    assert "<|im_start|>assistant\\n<think>" in templates.OLMO_THINKER.replace("\n", "\\n")


def test_sampling_is_open_instructs_choice_without_replacement():
    idx = data.sample_indices(88_556, 13_314, 42)
    assert len(idx) == len(set(idx.tolist())) == 13_314
    ref = np.random.RandomState(42).choice(88_556, size=13_314, replace=False)
    assert (idx == ref).all()
    with pytest.raises(ValueError):
        data.sample_indices(10, 11, 42)


def test_iterator_covers_each_epoch_once_and_drops_the_remainder():
    n, b = 100, 32   # 96 used per epoch, 4 dropped
    stream = data.prompt_stream(n, b, seed=1, steps=6)
    assert all(len(x) == b for x in stream)
    first, second = sum(stream[:3], []), sum(stream[3:], [])
    assert len(set(first)) == len(first) == 96
    assert len(set(second)) == 96
    assert first != second, "each epoch reshuffles"
    assert data.prompt_stream(n, b, seed=1, steps=6) == stream


def test_iterator_matches_open_instructs_first_shuffle():
    # ShufflingIterator shuffles a copy of arange(n) in place with default_rng(seed).
    n, b = 1000, 32
    expect = np.arange(n)
    np.random.default_rng(1).shuffle(expect)
    got = data.prompt_stream(n, b, seed=1, steps=2)
    assert got == [expect[:32].tolist(), expect[32:64].tolist()]


def test_messages_parse_from_either_storage():
    msgs = [{"content": "hi", "role": "user"}]
    assert data.as_messages(msgs) == msgs
    assert data.as_messages(repr(msgs)) == msgs


def audit_citations(node, path=()):
    """Paths of every `commit` key holding a full sha, as the audit would see them."""
    if isinstance(node, dict):
        v = node.get("commit")
        if isinstance(v, str) and HEX40.fullmatch(v):
            yield path
        for k, child in node.items():
            yield from audit_citations(child, (*path, k))
    elif isinstance(node, list):
        for child in node:
            yield from audit_citations(child, path)


@pytest.mark.parametrize("record", sorted(E2E.rglob("*.json")), ids=lambda p: p.name)
def test_records_cite_commits_only_where_the_audit_expects(record):
    # env.commit is this repository, env.shardes.commit the library. Anything else would
    # be read as a commit of some other repository named by its parent key.
    doc = json.loads(record.read_text())
    for path in audit_citations(doc):
        assert path in ((), ("env",), ("env", "shardes")), f"{record.name}: {path}"
    if "env" in doc:
        assert doc["env"]["shardes"]["commit"] != doc["env"]["commit"], \
            "without a separate library commit the audit reads this as a monorepo record"
