#!/usr/bin/env python
"""The protocol constants the paper states in prose, read from the records that ran.

    python protocol_constants.py            # print them
    python protocol_constants.py --latex    # also write paper/generated/protocol.tex

A count typed into a sentence drifts from the records. The setup section once said the
ranking step was timed with seven repeats: the in-context ranking measurement was, but the
84-configuration ranking benchmark used five, and one sentence had merged the two.

So each macro belongs to one family of measurements, and a sentence has to name the family
it means. A family's repeat count is read from its records (how many timings each record
holds), not from its config: a config says what was intended, a record what happened. The
script stops rather than write a value that is true of only part of a family: if the
records in a family disagree, or a record disagrees with the config that produced it.
Warm-up generations are not recorded in every record, so they come from the record where
it has the field and otherwise from the committed config whose `results_dir` names the
directory.
"""

from __future__ import annotations

import argparse
import collections
from decimal import ROUND_HALF_UP, Decimal
import json
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "paper" / "generated" / "protocol.tex"

#: Where each family's records live, and where each record keeps its timings: "top" is a
#: `seconds_all` list per record, "nested" is every list named `all` inside the record (one
#: per chain length in the ladder and the reconstruction probes).
FAMILIES = {
    "block": (["phase2/results-consistent", "phase2/results-qiu", "phase2/results-tpu-v5e8"], "top"),
    "cost": (["phase2/results-cost", "phase2/results-cost-tpu-v5e8",
              "phase2/results-cost-tpu-v5e8-highest"], "top"),
    "realmodel": (["countdown/results-e17b"], "top"),
    "boundary": (["phase2/multihost/results-e18"], "top"),
    "rankbench": (["phase2/results-barrier-tpu-v5e8"], "top"),
    "ladder": (["phase2/results-ladder"], "nested"),
    "probes": (["phase2/results-contraction"], "nested"),
}

#: Complete-generation benchmarks: evaluate, gather, reconstruct, update, timed as a whole.
GENERATIONS = ["block", "cost", "realmodel", "boundary"]

WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


class ProtocolError(ValueError):
    """A family's records do not support a single value."""


def spelled(n: int) -> str:
    """Numbers up to ten as words, as the prose writes them; larger ones as digits."""
    return WORDS[n] if 0 <= n < len(WORDS) else str(n)


def rounded(x: float, places: int) -> str:
    """Half up from the exact value, as a reader would round it; `round` rounds half to even."""
    return str(Decimal(repr(x)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def configs_by_dir(root: pathlib.Path = HERE) -> dict[pathlib.Path, list[dict]]:
    """Every committed config that names a `results_dir`, keyed by that directory."""
    out: dict[pathlib.Path, list[dict]] = collections.defaultdict(list)
    for path in sorted(root.rglob("*.yaml")):
        try:
            cfg = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        if isinstance(cfg, dict) and isinstance(cfg.get("results_dir"), str):
            out[(path.parent / cfg["results_dir"]).resolve()].append(cfg)
    return out


def records(directory: pathlib.Path):
    for path in sorted(directory.rglob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(record, dict):
            yield path, record


def timings(record: dict, where: str) -> list[int]:
    """The lengths of the timing lists in one record; empty if it holds none (an OOM)."""
    if where == "top":
        values = record.get("seconds_all")
        return [len(values)] if isinstance(values, list) and values else []
    found = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "all" and isinstance(value, list) and value:
                    found.append(len(value))
                else:
                    walk(value)

    walk({k: v for k, v in record.items() if k != "env"})
    return found


def one(values: dict, what: str) -> int:
    """The single value in `values` (source -> Counter of values), or a ProtocolError."""
    seen = collections.Counter()
    for counter in values.values():
        seen.update(counter)
    if len(seen) != 1:
        detail = "; ".join(f"{src}: {dict(c)}" for src, c in values.items())
        raise ProtocolError(f"{what} is not one value: {detail}")
    return next(iter(seen))


def repeats(family: str, root: pathlib.Path = HERE, configs=None) -> int:
    dirs, where = FAMILIES[family]
    configs = configs_by_dir(root) if configs is None else configs
    per_dir = {}
    for rel in dirs:
        directory = (root / rel).resolve()
        declared = {c["repeats"] for c in configs.get(directory, []) if "repeats" in c}
        counts = collections.Counter()
        for path, record in records(directory):
            lengths = timings(record, where)
            counts.update(lengths)
            expected = record.get("repeats", None)
            expected = {expected} if isinstance(expected, int) else declared
            for n in lengths:
                if expected and n not in expected:
                    raise ProtocolError(
                        f"{path.relative_to(root)} holds {n} timings, its config says {sorted(expected)}")
        if not counts:
            raise ProtocolError(f"{rel} has no timed records")
        per_dir[rel] = counts
    return one(per_dir, f"repeats of {family}")


def warmup(family: str, root: pathlib.Path = HERE, configs=None) -> int:
    dirs, where = FAMILIES[family]
    configs = configs_by_dir(root) if configs is None else configs
    per_dir = {}
    for rel in dirs:
        directory = (root / rel).resolve()
        declared = [c["warmup"] for c in configs.get(directory, []) if "warmup" in c]
        counts = collections.Counter()
        for path, record in records(directory):
            if not timings(record, where):
                continue
            if isinstance(record.get("warmup"), int):
                counts[record["warmup"]] += 1
            elif declared:
                counts.update(declared)
            else:
                raise ProtocolError(f"{path.relative_to(root)}: no warm-up in the record or its config")
        per_dir[rel] = counts
    return one(per_dir, f"warm-up of {family}")


def timed_records(family: str, root: pathlib.Path = HERE) -> int:
    dirs, where = FAMILIES[family]
    return sum(1 for rel in dirs for _, r in records((root / rel).resolve()) if timings(r, where))


def macros(root: pathlib.Path = HERE) -> dict[str, str]:
    configs = configs_by_dir(root)

    def across(fn, families, what):
        return one({f: collections.Counter([fn(f, root, configs)]) for f in families}, what)

    return {
        # Section 5: the protocol for every complete-generation benchmark.
        "GenWarmup": spelled(across(warmup, GENERATIONS, "warm-up of complete generations")),
        "GenRepeats": spelled(across(repeats, ["block", "cost", "realmodel"],
                                     "repeats of single-host complete generations")),
        "BoundaryRepeats": spelled(repeats("boundary", root, configs)),
        # Figure 2 and Table 5 captions.
        "BlockRepeats": spelled(repeats("block", root, configs)),
        "RealModelRepeats": spelled(repeats("realmodel", root, configs)),
        # Appendix A: the ranking benchmark, the collective ladder, the reconstruction probes.
        "RankBenchRepeats": spelled(repeats("rankbench", root, configs)),
        "RankBenchConfigs": spelled(timed_records("rankbench", root)),
        "LadderRepeats": spelled(repeats("ladder", root, configs)),
        "ProbeRepeats": spelled(repeats("probes", root, configs)),
    }


def render(values: dict[str, str]) -> str:
    lines = ["% generated by experiments/protocol_constants.py; do not edit"]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--latex", action="store_true", help=f"also write {OUT.relative_to(HERE.parent)}")
    args = ap.parse_args(argv)
    try:
        values = macros()
    except ProtocolError as err:
        print(f"protocol_constants: {err}", file=sys.stderr)
        return 1
    for name, value in values.items():
        print(f"{name:18s} {value}")
    if args.latex:
        OUT.write_text(render(values))
    return 0


if __name__ == "__main__":
    sys.exit(main())
