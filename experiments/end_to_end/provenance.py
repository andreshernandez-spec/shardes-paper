"""The `env` block for this paper's records, in the shape the provenance audit reads.

`harness.capture_env` asks JAX for its devices, which on a machine with a CUDA jaxlib
claims the GPU. The desk scripts here need no device, and the vLLM drivers must not let
JAX grab memory the engine needs, so they stamp themselves with this instead.

The shape matters: `provenance_audit.py` reads `env.commit` as this repository's commit,
and treats a record without an `env.shardes` block naming a different commit as one
from before the split, checked against the library's history, where it would not be
found. So every record carries the library block, even from a script that does not
import the library. Foreign hashes (Hugging Face, open-instruct) are never stored under a
key named `commit`, which the audit would read as a citation.
"""

import platform
import socket
import sys
from importlib import metadata
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import harness  # noqa: E402


def env_block(here: Path, outputs: Iterable[str], packages: Iterable[str] = ()) -> dict:
    def version(p):
        try:
            return metadata.version(p)
        except metadata.PackageNotFoundError:
            return None

    return {
        "commit": harness.git(here, "rev-parse", "HEAD"),
        "dirty_worktree": harness.worktree_is_dirty(here, list(outputs)),
        "shardes": harness.library_provenance(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": {p: version(p) for p in packages},
    }
