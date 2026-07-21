"""Release-time long_description link check: zero relative links, self-repo
links pin the current version tag, and every GitHub link answers 200.
Ref-consistency and --map-ref are scoped to THIS repo's URLs only — third-party
pins (e.g. upstream docs at their own tags) are checked for 200 but never
rewritten or version-matched (Bugbot, e4b PR #31 round 4)."""
import re
import subprocess
import sys
import urllib.request

SKIP_SUBSTRINGS = ()


def _self_slug() -> str:
    url = subprocess.run(["git", "remote", "get-url", "origin"],
                         capture_output=True, text=True, check=True).stdout.strip()
    m = re.search(r"github\.com[:/]([^/]+/[^/.]+)", url)
    assert m, f"cannot derive owner/repo from origin: {url}"
    return m.group(1)


def main() -> int:
    args = sys.argv[1:]
    ref_map = None
    if args[:1] == ["--map-ref"]:
        ref_map = (args[1], args[2])
    slug = _self_slug()
    self_prefix = f"https://github.com/{slug}/"
    text = open("README.md").read()
    targets = re.findall(r"\]\(([^)]+)\)", text)
    rel = [t for t in targets if not t.startswith(("http://", "https://", "#"))]
    assert not rel, f"relative links present (dead on PyPI): {rel}"
    # ref-consistency (SELF-repo links only): every pinned blob/tree ref must be
    # the CURRENT version's tag, else a bump ships docs pointing at the old tag.
    try:
        import tomllib
        want = "v" + tomllib.load(open("pyproject.toml", "rb"))["project"]["version"]
        refs = set(re.findall(
            re.escape(self_prefix) + r"(?:blob|tree)/([^/]+)/", text))
        stale = refs - {want, "main"}
        assert not stale, f"README pins self-repo ref(s) {sorted(stale)} but project.version wants {want}"
    except FileNotFoundError:
        pass
    gh = [t for t in targets if t.startswith("https://github.com/")]
    if ref_map:
        gh = [t.replace(f"/{ref_map[0]}/", f"/{ref_map[1]}/")
              if t.startswith(self_prefix) else t
              for t in gh]
    bad = []
    for t in gh:
        if any(s in t for s in SKIP_SUBSTRINGS):
            continue
        req = urllib.request.Request(t, method="HEAD",
                                     headers={"User-Agent": "link-check"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                if r.status != 200:
                    bad.append((t, r.status))
        except Exception as e:
            bad.append((t, str(e)))
    assert not bad, f"dead GitHub links: {bad}"
    print(f"link check OK: 0 relative, self-repo refs consistent, {len(gh)} GitHub links answer 200")
    return 0


if __name__ == "__main__":
    sys.exit(main())
