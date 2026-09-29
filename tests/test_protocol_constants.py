"""`experiments/protocol_constants.py`: the paper's protocol counts come from the records.

The failure it exists for: one sentence said the ranking step was timed with seven
repeats, true of the in-context ranking records and not of the 84-configuration benchmark,
which used five. A family that mixes the two must stop the build, not print either number.
"""

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "experiments" / "protocol_constants.py"
spec = importlib.util.spec_from_file_location("protocol_constants", SCRIPT)
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)


def write(directory, name, record):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(record))


@pytest.fixture
def family(tmp_path, monkeypatch):
    """One family over two directories under tmp_path, with no configs."""
    monkeypatch.setitem(pc.FAMILIES, "test", (["a", "b"], "top"))
    return tmp_path


def test_a_uniform_family_gives_its_count(family):
    for d in ("a", "b"):
        write(family / d, "r.json", {"seconds_all": [1.0] * 5})
    assert pc.repeats("test", family, configs={}) == 5


def test_a_family_that_mixes_counts_stops_and_names_both(family):
    write(family / "a", "r.json", {"seconds_all": [1.0] * 5})
    write(family / "b", "r.json", {"seconds_all": [1.0] * 7})
    with pytest.raises(pc.ProtocolError, match=r"a: \{5: 1\}; b: \{7: 1\}"):
        pc.repeats("test", family, configs={})


def test_a_record_that_disagrees_with_its_config_stops(family):
    write(family / "a", "r.json", {"seconds_all": [1.0] * 4})
    write(family / "b", "r.json", {"seconds_all": [1.0] * 4})
    configs = {(family / "a").resolve(): [{"repeats": 5}]}
    with pytest.raises(pc.ProtocolError, match="holds 4 timings, its config says"):
        pc.repeats("test", family, configs=configs)


def test_records_without_timings_are_skipped_not_counted(family):
    write(family / "a", "ok.json", {"seconds_all": [1.0] * 5})
    write(family / "a", "oom.json", {"status": "oom"})
    write(family / "b", "ok.json", {"seconds_all": [1.0] * 5})
    assert pc.repeats("test", family, configs={}) == 5
    assert pc.timed_records("test", family) == 2


def test_warm_up_comes_from_the_record_or_else_its_config(family):
    write(family / "a", "r.json", {"seconds_all": [1.0], "warmup": 3})
    write(family / "b", "r.json", {"seconds_all": [1.0]})
    configs = {(family / "b").resolve(): [{"warmup": 3}]}
    assert pc.warmup("test", family, configs=configs) == 3
    with pytest.raises(pc.ProtocolError, match="no warm-up"):
        pc.warmup("test", family, configs={})


def test_nested_timings_are_found_under_every_all(tmp_path, monkeypatch):
    monkeypatch.setitem(pc.FAMILIES, "nested", (["n"], "nested"))
    write(tmp_path / "n", "r.json",
          {"chain": [1, 9], "A": {"chain": {"1": {"all": [0.0] * 10, "iqr": [0, 1]},
                                            "9": {"all": [0.0] * 10, "iqr": [0, 1]}}}})
    assert pc.repeats("nested", tmp_path, configs={}) == 10


def test_prose_numbers_are_words_up_to_ten():
    assert [pc.spelled(n) for n in (3, 5, 7, 10, 84)] == ["three", "five", "seven", "ten", "84"]


def test_the_committed_macros_match_the_records():
    """The same check CI makes by rebuilding the tables, available without LaTeX."""
    assert pc.OUT.read_text() == pc.render(pc.macros())
