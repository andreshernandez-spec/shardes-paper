"""Can a fresh machine rebuild the environment from this repository's SHA alone?

    SHA=$(git rev-parse HEAD)
    sed "s/PINNED_AT_PUSH/$SHA/" bootstrap.py > /tmp/stage/bootstrap.py   # with the metadata
    python experiments/phase1/kaggle/run.py /tmp/stage --accelerator NvidiaTeslaT4

Since the library got its own repository, a run is reproduced by checking THIS
repository out at the commit a record stamps and installing from requirements.txt, which
pins the library. Every kernel committed before that cloned the old single repository and
put its src/ on the path; they are records of how those results were made and still work
as written. This is the template for what comes after, proven on free hardware before a
queued TPU session or a rented node depends on it.

It asserts what it cares about and exits non-zero otherwise, because a Kaggle run that
quietly did less than asked still reports COMPLETE:

  1. the accelerators are real and there are two of them;
  2. the shardes that got installed is the commit requirements.txt pins;
  3. the driver tests pass against it;
  4. a real driver, run on the GPU, writes a record whose provenance names both
     repositories, clean, with the library installed from git and not from a checkout.
"""
import json
import os
import pathlib
import re
import subprocess
import sys

SHA = "PINNED_AT_PUSH"
REPO = "https://github.com/andreshernandez-spec/shardes-paper.git"
#: Outside /kaggle/working on purpose. Everything under the working directory becomes the
#: kernel's output, and `kaggle kernels output` fetches it one file at a time: a clone of
#: this repository is 3,000 small files and took over eight minutes to not finish, against
#: two minutes for the run itself. Copy back what you want kept, and nothing else.
WORK = "/tmp/paper"


def run(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=kw.pop("check", True), **kw)


assert re.fullmatch(r"[0-9a-f]{40}", SHA), "push with the SHA filled in (see the docstring)"
run(["nvidia-smi", "-L"])
run([sys.executable, "-m", "pip", "install", "-q", "-U", "jax[cuda12]>=0.11"])
run(["git", "clone", "-q", REPO, WORK])
run(["git", "checkout", "-q", SHA], cwd=WORK)
run(["git", "log", "--oneline", "-1"], cwd=WORK)
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=WORK)

pin = re.search(r"shardes@([0-9a-f]{40})",
                pathlib.Path(WORK, "requirements.txt").read_text()).group(1)

# A fresh interpreter: this one imported nothing from jax, but the habit is what matters,
# since a process that ran the install has the old packages loaded.
check = f"""
import importlib.metadata as m, json, jax
devs = jax.devices()
print("jax", jax.__version__, [d.device_kind for d in devs])
assert len(devs) == 2 and all(d.platform == "gpu" for d in devs), devs
got = json.loads(m.distribution("shardes").read_text("direct_url.json"))["vcs_info"]["commit_id"]
assert got == "{pin}", (got, "{pin}")
import shardes
print("shardes", shardes.__version__, "at", got)
"""
run([sys.executable, "-c", check])

# The driver tests. conftest pins them to the CPU with eight simulated devices, which is
# what they are for; the count is asserted so an empty collection cannot pass.
tests = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=WORK,
            capture_output=True, text=True, check=False)
print(tests.stdout[-1500:], flush=True)
passed = re.search(r"(\d+) passed", tests.stdout)
assert tests.returncode == 0 and passed and int(passed.group(1)) >= 150, tests.stdout[-400:]

# One real driver on the real GPU, small enough to take seconds.
env = {**os.environ, "JAX_PLATFORMS": "cuda"}
run([sys.executable, "feed_fusion.py", "--d-model", "64", "--population", "16",
     "--repeats", "5", "--warmup", "2"], cwd=f"{WORK}/experiments/phase2", env=env)
record = json.loads(pathlib.Path(
    WORK, "experiments/phase2/results-feed-fusion/d=64__N=16__s=iid_gaussian__how=A.json"
).read_text())["env"]
print(json.dumps({k: record[k] for k in ("commit", "dirty_worktree", "shardes",
                                         "device_kind", "jax")}, indent=1), flush=True)
assert record["commit"] == SHA, record["commit"]
assert record["shardes"] == {**record["shardes"], "commit": pin, "dirty": False,
                             "source": "vcs"}, record["shardes"]
assert record["device_platform"] == "gpu", record["device_platform"]
# Clean, too: the driver's own results directory is exempt, pytest ran without its cache,
# and pip wrote nothing into the clone. A dirty flag here would mean the bootstrap leaves
# something behind that makes every later record on this machine unreproducible.
assert record["dirty_worktree"] is False, "the bootstrap dirtied the checkout"
print("BOOTSTRAP OK")
