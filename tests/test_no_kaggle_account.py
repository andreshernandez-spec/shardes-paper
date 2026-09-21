"""No tracked file names the Kaggle account.

The Kaggle runs happen on a personal account that is not the identity this repository is
published under. Kernel metadata is committed with a `USERNAME` placeholder and the logs
are ignored, and the account still got into three tracked files: twice as a kernel id in
a results README and once as an email address in the runbook. Once pushed, that cannot be
taken back out of the history.

This cannot name the account to look for it, so it looks for the two shapes it arrived
in: a kernel id with a real owner, and an address nobody decided to publish.
"""

import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: An owner starts with a letter or a digit, which keeps `./shardes-lib` and other relative
#: paths out of it.
KERNEL_ID = re.compile(r"(?<![A-Za-z0-9_.-])([A-Za-z0-9][A-Za-z0-9_.-]*)/shardes-[a-z0-9-]+")
#: The placeholder the runner fills in at push time, and the GitHub owner of
#: `andreshernandez-spec/shardes-paper`, which has the same shape.
OWNERS = {"USERNAME", "andreshernandez-spec"}

EMAIL = re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z][A-Za-z.]*")
#: Published on purpose: the author address on the manuscript, the identity the commits
#: carry, and the two ssh remotes, which are shaped like addresses.
ADDRESSES = {"andres.hernandez@gmx.net", "andres.hernandez.deml@gmail.com",
             "git@github.com", "git@specgithub.com"}


def _tracked_text():
    names = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True,
                           check=True).stdout.decode().split("\0")
    for name in filter(None, names):
        try:
            yield name, (ROOT / name).read_text()
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue    # a figure, or a file deleted in the working tree


def test_the_scan_reads_the_repository():
    """An empty scan passes everything below. The kernel metadata alone is 16 placeholders."""
    text = dict(_tracked_text())
    assert len(text) > 2000, len(text)
    assert sum(len(KERNEL_ID.findall(t)) for t in text.values()) >= 16


def test_every_kernel_id_has_the_placeholder_for_an_owner():
    found = sorted({f"{name}: {m.group(0)}" for name, text in _tracked_text()
                    for m in KERNEL_ID.finditer(text) if m.group(1) not in OWNERS})
    assert not found, ("a kernel id names an account; write the kernel's slug and its "
                       "directory instead:\n  " + "\n  ".join(found))


def test_every_address_is_one_that_was_published_on_purpose():
    found = sorted({f"{name}: {m.group(0)}" for name, text in _tracked_text()
                    for m in EMAIL.finditer(text)
                    if m.group(0) not in ADDRESSES and not m.group(0).endswith("@example.com")})
    assert not found, "an address nobody decided to publish:\n  " + "\n  ".join(found)
