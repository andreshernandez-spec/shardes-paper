#!/usr/bin/env python
"""Record what every pinned release actually is: commits, weight hashes, duplicates.

    python pin_releases.py --out pins/releases-2026-09-26.json

For each repo in `releases.py`: the pinned commit must still resolve, and the record
keeps the LFS sha256 of every weight or data file at that commit. For the two RL repos
it does the same for every step branch and groups branches whose weights are
byte-identical, because branch labels are not reliable (Olmo 3 RL-Zero IF `step_100`
and `step_1000` hold the same weights; the Code repo's `main` equals `step_300`). A
comparison "at step N" is only as good as the proof that the branch is step N.

It also saves the IF run's full logged W&B config beside the record and checks it
against `releases.OLMO3_IF_LOGGED`, and checks that both open-instruct commits exist.

Network only, no GPU. Refuses to overwrite an existing record: a pin is a measurement
of the hub on a date, not a cache.
"""

import argparse
import datetime
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import harness  # noqa: E402
import releases as R  # noqa: E402
from provenance import env_block  # noqa: E402

WEIGHT_SUFFIXES = (".safetensors",)
DATA_SUFFIXES = (".parquet", ".jsonl", ".json.gz")


def lfs_files(api, repo, kind, revision, suffixes):
    """{path: {"sha256", "size"}} for files ending in `suffixes` at `revision`."""
    out = {}
    for f in api.list_repo_tree(repo, repo_type=kind, revision=revision, recursive=True):
        if getattr(f, "lfs", None) is None or not f.path.endswith(suffixes):
            continue
        out[f.path] = {"sha256": f.lfs.sha256, "size": f.lfs.size}
    if not out:
        raise RuntimeError(f"no {suffixes} files in {repo}@{revision}")
    return out


def fingerprint(files: dict) -> str:
    """One hash for a whole checkpoint: the sorted (path, sha256) pairs."""
    lines = "".join(f"{p}:{v['sha256']}\n" for p, v in sorted(files.items()))
    return hashlib.sha256(lines.encode()).hexdigest()


def branches(api, repo):
    refs = api.list_repo_refs(repo, repo_type="model")
    return {b.name: b.target_commit for b in refs.branches}


def pin_rl_repo(api, rel, expected_branches):
    """Every branch of an RL repo, fingerprinted, with duplicates grouped."""
    refs = branches(api, rel.repo)
    missing = sorted(set(expected_branches) - set(refs))
    if missing:
        raise RuntimeError(f"{rel.repo}: expected branches missing: {missing}")
    if refs.get("main") != rel.commit:
        # main moved since we pinned it. The pinned commit is what we use; say so loudly.
        print(f"WARNING {rel.repo}: main is now {refs.get('main')}, pinned {rel.commit}")
    rows = {}
    for name in ("main", *expected_branches):
        commit = rel.commit if name == "main" else refs[name]
        files = lfs_files(api, rel.repo, "model", commit, WEIGHT_SUFFIXES)
        rows[name] = {"revision": commit, "fingerprint": fingerprint(files), "files": files}
    groups = {}
    for name, row in rows.items():
        groups.setdefault(row["fingerprint"], []).append(name)
    duplicates = [sorted(g) for g in groups.values() if len(g) > 1]
    main_matches = [n for n in groups[rows["main"]["fingerprint"]] if n != "main"]
    return {"branches": rows, "duplicate_groups": duplicates, "main_equals": main_matches}


def github_commit(sha):
    url = f"https://api.github.com/repos/allenai/open-instruct/commits/{sha}"
    with urllib.request.urlopen(url, timeout=30) as r:
        c = json.load(r)
    if c["sha"] != sha:
        raise RuntimeError(f"open-instruct {sha} resolved to {c['sha']}")
    return {"sha": sha, "date": c["commit"]["committer"]["date"],
            "title": c["commit"]["message"].splitlines()[0]}


def wandb_config(entity, project, run):
    query = ('{ project(name:"%s", entityName:"%s") { run(name:"%s") '
             '{ name displayName createdAt config } } }' % (project, entity, run))
    req = urllib.request.Request(
        "https://api.wandb.ai/graphql", data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        run_ = json.load(r)["data"]["project"]["run"]
    cfg = {k: (v["value"] if isinstance(v, dict) and "value" in v else v)
           for k, v in json.loads(run_["config"]).items()}
    return {"name": run_["name"], "display_name": run_["displayName"],
            "created_at": run_["createdAt"], "config": cfg}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    out = (args.out if args.out.is_absolute() else HERE / args.out).resolve()
    out.relative_to(HERE)  # records live beside the driver; raises otherwise
    if out.exists():
        raise SystemExit(f"{out} exists; a pin is never overwritten, pick a new name")
    wandb_out = out.with_name(f"wandb-{R.OLMO3_IF_WANDB[2]}-config.json")
    # The logged config names commits of its own (none today); keep the audit off them.
    if wandb_out.exists():
        raise SystemExit(f"{wandb_out} exists")

    from huggingface_hub import HfApi  # noqa: PLC0415
    api = HfApi()

    outputs = [str(out.relative_to(HERE)), str(wandb_out.relative_to(HERE))]
    record = {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "env": env_block(HERE, outputs, ("huggingface_hub",)),
        "repos": {}, "rl": {}, "open_instruct": {}, "checks": {},
    }

    for rel in R.ALL_REPOS:
        suffixes = WEIGHT_SUFFIXES if rel.kind == "model" else DATA_SUFFIXES
        record["repos"][rel.repo] = {
            "kind": rel.kind, "revision": rel.commit, "note": rel.note,
            "files": lfs_files(api, rel.repo, rel.kind, rel.commit, suffixes),
        }
        print(f"pinned {rel.kind} {rel.repo}@{rel.commit[:10]}", flush=True)

    base_branch = branches(api, R.OLMO3_BASE.repo)[R.OLMO3_BASE_LOGGED_BRANCH]
    base_logged = lfs_files(api, R.OLMO3_BASE.repo, "model", base_branch, WEIGHT_SUFFIXES)
    same = fingerprint(base_logged) == fingerprint(record["repos"][R.OLMO3_BASE.repo]["files"])
    record["checks"]["olmo3_base_main_equals_logged_branch"] = same
    if not same:
        raise SystemExit("Olmo 3 base main is not the checkpoint the IF run started from")

    for rel, expected in ((R.OLMO3_IF_RL, R.OLMO3_IF_RL_BRANCHES),
                          (R.TULU31_RL, R.TULU31_RL_BRANCHES)):
        record["rl"][rel.repo] = pin_rl_repo(api, rel, expected)
        r = record["rl"][rel.repo]
        print(f"{rel.repo}: main equals {r['main_equals'] or 'no branch'}; "
              f"duplicate groups {r['duplicate_groups'] or 'none'}", flush=True)

    for sha in (R.OLMO3_IF_OPEN_INSTRUCT, R.TULU31_OPEN_INSTRUCT):
        record["open_instruct"][sha] = github_commit(sha)

    wb = wandb_config(*R.OLMO3_IF_WANDB)
    mismatches = {k: {"logged": wb["config"].get(k), "releases.py": v}
                  for k, v in R.OLMO3_IF_LOGGED.items() if wb["config"].get(k) != v}
    record["checks"]["olmo3_if_logged_config_mismatches"] = mismatches
    if mismatches:
        raise SystemExit(f"releases.OLMO3_IF_LOGGED disagrees with W&B: {mismatches}")

    harness.write_atomic(wandb_out, wb)
    harness.write_atomic(out, record)
    print(f"wrote {out.relative_to(HERE)} and {wandb_out.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
