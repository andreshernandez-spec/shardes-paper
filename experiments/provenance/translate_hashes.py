"""Rewrite every commit hash mentioned in a repository's tracked text through the
commit-maps a filter-repo rewrite produced. Abbreviations keep their length. A token is
translated only if the OLD repository resolves it to a commit that the map knows."""
import pathlib, re, subprocess, sys, functools, collections

repo, old_lib, old_paper, map_lib, map_paper = sys.argv[1:6]
maps = {}
for name, path in (("lib", map_lib), ("paper", map_paper)):
    maps[name] = dict(line.split() for line in pathlib.Path(path).read_text().splitlines() if line.strip())
olds = {"lib": old_lib, "paper": old_paper}

@functools.lru_cache(maxsize=None)
def resolve(token):
    hits = []
    for name, old in olds.items():
        r = subprocess.run(["git", "-C", old, "rev-parse", "--verify", "--quiet", f"{token}^{{commit}}"],
                           capture_output=True, text=True)
        full = r.stdout.strip()
        if r.returncode == 0 and full in maps[name] and maps[name][full] != full:
            hits.append((name, full, maps[name][full]))
    if len(hits) > 1:
        raise SystemExit(f"ambiguous token {token}: {hits}")
    return hits[0] if hits else None

TOKEN = re.compile(r"(?<![0-9a-zA-Z])([0-9a-f]{7,40})(?![0-9a-zA-Z])")
names = subprocess.run(["git", "-C", repo, "ls-files", "-z"], capture_output=True, check=True).stdout.decode().split("\0")
changed = collections.Counter(); per_repo = collections.Counter(); digits_only = set(); examples = {}
for name in filter(None, names):
    p = pathlib.Path(repo, name)
    try: text = p.read_text()
    except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError): continue
    def sub(m):
        tok = m.group(1); hit = resolve(tok)
        if not hit: return tok
        which, old, new = hit
        changed[name] += 1; per_repo[which] += 1
        if tok.isdigit(): digits_only.add(tok)
        examples.setdefault(tok, new[:len(tok)])
        return new[:len(tok)]
    out = TOKEN.sub(sub, text)
    if out != text: p.write_text(out)
print(f"{len(changed)} files changed, {sum(changed.values())} hash mentions translated "
      f"({per_repo['lib']} library hashes, {per_repo['paper']} paper-repo hashes), {len(examples)} distinct")
print("all-digit tokens translated (check these are really hashes):", sorted(digits_only) or "none")
by_kind = collections.Counter(pathlib.Path(n).suffix or n for n in changed)
print("by file type:", dict(by_kind))
