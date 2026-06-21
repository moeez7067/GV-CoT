"""One-shot: vendor the full Three.js r0.160.0 module closure locally.

Downloads three.module.js + the addons used by graph_template.html and every
file they transitively import, into D:\\gvcot\\static\\three\\ preserving the
package-relative directory structure. Verifies the closure is complete (no
unresolved imports). Run once; app.py then inlines these at runtime.
"""
import re
import sys
import posixpath
import urllib.request
from pathlib import Path

VERSION  = "0.160.0"
BASE_URL = f"https://unpkg.com/three@{VERSION}/"
OUT_DIR  = Path(r"D:\gvcot\static\three")

ENTRIES = [
    "build/three.module.js",
    "examples/jsm/controls/OrbitControls.js",
    "examples/jsm/postprocessing/EffectComposer.js",
    "examples/jsm/postprocessing/RenderPass.js",
    "examples/jsm/postprocessing/UnrealBloomPass.js",
    "examples/jsm/postprocessing/OutputPass.js",
]

# Find static import / re-export specifiers (and side-effect imports)
_FROM_RE   = re.compile(r"""\bfrom\s*['"]([^'"]+)['"]""")
_SIDE_RE   = re.compile(r"""(?:^|[\n;])\s*import\s*['"]([^'"]+)['"]""")


def resolve(spec: str, current: str):
    """Resolve an import specifier to a package-relative path, or None if external."""
    if spec == "three":
        return "build/three.module.js"
    if spec.startswith("three/addons/"):
        return "examples/jsm/" + spec[len("three/addons/"):]
    if spec.startswith("three/"):
        return spec[len("three/"):]            # e.g. three/src/... (rare)
    if spec.startswith("."):
        d = posixpath.dirname(current)
        return posixpath.normpath(posixpath.join(d, spec))
    return None                                 # other bare specifier — none expected


def fetch(path: str) -> str:
    url = BASE_URL + path
    req = urllib.request.Request(url, headers={"User-Agent": "gvcot-vendor"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = {}                # path -> source text
    queue   = list(ENTRIES)
    unresolved = []

    while queue:
        path = queue.pop(0)
        if path in sources:
            continue
        try:
            print(f"  fetching {path} ...", flush=True)
            src = fetch(path)
        except Exception as e:
            print(f"  !! FAILED {path}: {e}", flush=True)
            sys.exit(1)
        sources[path] = src

        specs = set(_FROM_RE.findall(src)) | set(_SIDE_RE.findall(src))
        for spec in specs:
            r = resolve(spec, path)
            if r is None:
                if spec not in ("three",):
                    unresolved.append((path, spec))
                continue
            if r not in sources and r not in queue:
                queue.append(r)

    # Write all files preserving structure
    total = 0
    for path, src in sources.items():
        dst = OUT_DIR / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src, encoding="utf-8")
        total += len(src.encode("utf-8"))

    # Closure verification: every local import must have been downloaded
    missing = []
    for path, src in sources.items():
        specs = set(_FROM_RE.findall(src)) | set(_SIDE_RE.findall(src))
        for spec in specs:
            r = resolve(spec, path)
            if r is not None and r not in sources:
                missing.append((path, spec, r))

    print()
    print(f"Downloaded {len(sources)} modules, {total/1024:.1f} KB total -> {OUT_DIR}")
    for path in sorted(sources):
        kb = len(sources[path].encode('utf-8')) / 1024
        print(f"   {kb:8.1f} KB  {path}")
    if unresolved:
        print("\n  External (left as-is):")
        for p, s in unresolved:
            print(f"     {s}  (in {p})")
    if missing:
        print("\n!! CLOSURE INCOMPLETE — missing:")
        for p, s, r in missing:
            print(f"     {s} -> {r}  (in {p})")
        sys.exit(1)
    print("\nCLOSURE COMPLETE — every local import resolves to a downloaded file.")


if __name__ == "__main__":
    main()
