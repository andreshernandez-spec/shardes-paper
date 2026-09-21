"""requirements.txt pins the library, and three things parse that line.

CI checks the library out at the pin, the bootstrap kernel asserts the installed commit
against it, and the weekly job measures how far the library's main is past it. All three
read a 40-character commit out of the file, so a pin written any other way (a branch, a
tag, a short hash) breaks them, and a tag or a branch would also stop one commit of this
repository from fixing one commit of the library.
"""

import pathlib
import re

REQUIREMENTS = pathlib.Path(__file__).resolve().parent.parent / "requirements.txt"


def _pins():
    lines = [ln for ln in REQUIREMENTS.read_text().splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    return [ln for ln in lines if re.match(r"shardes\b", ln)]


def test_the_library_is_pinned_once_to_a_full_commit():
    pins = _pins()
    assert len(pins) == 1, pins
    requirement = pins[0].split(" #")[0].strip()
    assert re.fullmatch(
        r"shardes @ git\+https://github\.com/andreshernandez-spec/shardes@[0-9a-f]{40}",
        requirement), requirement


def test_a_release_named_beside_the_pin_looks_like_one():
    """CI resolves the name against the library's tags. Here, only that it is a name CI
    will find: `# v0.1.0`, nothing else on the line."""
    _, _, comment = _pins()[0].partition(" #")
    assert not comment or re.fullmatch(r"\s*v\d+\.\d+\.\d+\S*\s*", comment), comment
