"""Release-time long_description link check: zero relative links, and every
GitHub link answers 200. Skip-list via SKIP_SUBSTRINGS for rate-limit noise."""
import re
import sys
import urllib.request

SKIP_SUBSTRINGS = ()

def main() -> int:
    # --map-ref FROM TO: rewrite /blob|tree/FROM/ -> /TO/ before checking.
    # CI runs pre-tag with `--map-ref vX.Y.Z main` (validates the paths); the
    # publish run (release-triggered, tag exists) checks the real URLs unmapped.
    args = sys.argv[1:]
    ref_map = None
    if args[:1] == ["--map-ref"]:
        ref_map = (args[1], args[2])
    text = open("README.md").read()
    targets = re.findall(r"\]\(([^)]+)\)", text)
    rel = [t for t in targets if not t.startswith(("http://", "https://", "#"))]
    assert not rel, f"relative links present (dead on PyPI): {rel}"
    gh = [t for t in targets if t.startswith("https://github.com/")]
    if ref_map:
        gh = [t.replace(f"/{ref_map[0]}/", f"/{ref_map[1]}/") for t in gh]
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
    print(f"link check OK: 0 relative, {len(gh)} GitHub links answer 200")
    return 0

if __name__ == "__main__":
    sys.exit(main())
