#!/usr/bin/env python
"""The reference side of T2: es-at-scale's own train.py, from the same config as ours.

    .venv-esref/bin/python -m es_vllm.countdown_ref --config es_vllm/countdown-s1.yaml
    .venv-esref/bin/python -m es_vllm.countdown_ref --config ... --smoke

Run with es-at-scale's environment (requirements-esref.txt). Builds its command line from
the YAML that `run_countdown.py` reads, so the two sides cannot drift apart, points it at
the pinned model snapshot and the pinned clone, and runs it with one vLLM engine.
Its log, its evaluation outputs and `pip freeze` land in runs/<config stem>-ref/.
`countdown_gate.py --compact` turns the evaluation outputs into the same `eval.jsonl`
that our runs write.
"""

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parent
E2E = PKG.parent
sys.path.insert(0, str(E2E))
import releases as R  # noqa: E402

SMOKE = {"population": 4, "iterations": 2, "max_tokens": 64, "eval_every": 1}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gpu", default="0", help="physical index; es-at-scale sets "
                    "CUDA_VISIBLE_DEVICES from it, overriding any outer setting")
    ap.add_argument("--es-at-scale", type=Path,
                    default=Path.home() / "private" / "open-source" / "es-at-scale")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load((E2E / args.config).read_text())
    if args.smoke:
        cfg = {**cfg, **SMOKE}
    clone = args.es_at_scale.resolve()
    head = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(clone), "status", "--porcelain",
                            "--untracked-files=no"], check=True, capture_output=True,
                           text=True).stdout.strip()
    if head != R.ES_AT_SCALE or dirty:
        raise SystemExit(f"{clone} is at {head[:12]}{' with changes' if dirty else ''}, "
                         f"expected {R.ES_AT_SCALE[:12]}")
    out = E2E / "runs" / (args.config.stem + "-ref" + ("-smoke" if args.smoke else ""))
    if (out / "train.log").exists():
        raise SystemExit(f"{out} has a log; a run is never resumed, move it aside first")
    out.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download  # noqa: PLC0415
    rel = getattr(R, cfg["model"])
    model_dir = snapshot_download(rel.repo, revision=rel.commit)

    # es-at-scale takes one more update after its last evaluation; --n-iterations N runs
    # N + 1 updates with evaluations after 0, eval_every, ..., N.
    cmd = [sys.executable, "es_at_scale/train.py", "--task", "countdown",
           "--model-name", model_dir, "--sigma", str(cfg["sigma"]),
           "--alpha", str(cfg["alpha"]), "--population-size", str(cfg["population"]),
           "--n-iterations", str(cfg["iterations"]), "--eval-freq", str(cfg["eval_every"]),
           "--train-dataset", "datasets/train/countdown",
           "--eval-dataset", "datasets/evaluation_suite/countdown",
           "--batch-size", "200", "--mini-batch-size", "200",
           "--max-tokens", str(cfg["max_tokens"]), "--n-vllm-engines", "1",
           "--use-gpus", args.gpu, "--seed", str(cfg["seed"]), "--logging", "none",
           "--reward-function-timeout", str(cfg["grader_timeout"]),
           "--output-directory", str(out), "--experiment-name", "run"]
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True,
                            capture_output=True, text=True).stdout
    (out / "pip-freeze.txt").write_text(freeze)
    (out / "run.json").write_text(json.dumps(
        {"config": cfg, "config_file": str(args.config), "smoke": args.smoke,
         "model": {"repo": rel.repo, "revision": rel.commit}, "es_at_scale": head,
         "command": cmd, "python": sys.version.split()[0],
         "shardes_paper": subprocess.run(["git", "-C", str(E2E), "rev-parse", "HEAD"],
                                         capture_output=True, text=True).stdout.strip(),
         "started": datetime.datetime.now(datetime.timezone.utc).isoformat()},
        indent=2, sort_keys=True))
    with (out / "train.log").open("w") as log:
        rc = subprocess.run(cmd, cwd=clone, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"es-at-scale exited {rc}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
