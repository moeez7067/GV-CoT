"""Offline Three.js bundling for the GV-CoT 3D graph.

The cinematic scene (graph_template.html) depends on three.module.js plus
several ES-module addons that import each other. To run with NO internet — and
inside Streamlit's sandboxed component iframe — we read the locally-vendored
files in static/three/ (populated by vendor_three.py) and inline them as
self-contained blob-URL modules. If the vendored files are missing we fall back
to the unpkg CDN importmap so the app still works when online.

This module is import-safe (no Streamlit / no side effects) so it can be unit
-tested directly.
"""
import re
import json
import posixpath
import functools
from pathlib import Path

BASE_DIR         = Path(__file__).resolve().parent
STATIC_THREE_DIR = BASE_DIR / "static" / "three"

_THREE_ENTRYPOINTS = (
    "build/three.module.js",
    "examples/jsm/controls/OrbitControls.js",
    "examples/jsm/postprocessing/EffectComposer.js",
    "examples/jsm/postprocessing/RenderPass.js",
    "examples/jsm/postprocessing/UnrealBloomPass.js",
    "examples/jsm/postprocessing/OutputPass.js",
)

_CDN_IMPORTMAP = (
    '<script type="importmap">\n'
    '{"imports":{'
    '"three":"https://unpkg.com/three@0.160.0/build/three.module.js",'
    '"three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"'
    '}}\n</script>'
)

# Matches  from "spec"  /  export ... from "spec"  /  import "spec" (side-effect).
# Does NOT match  import(  or  import.meta  (no quote right after the keyword).
_IMP_RE = re.compile(r"""(\b(?:from|import)\s*)(['"])([^'"]+)(['"])""")


def _three_resolve(spec: str, current: str):
    """Resolve an import specifier to a package-relative path (None if external)."""
    if spec == "three":
        return "build/three.module.js"
    if spec.startswith("three/addons/"):
        return "examples/jsm/" + spec[len("three/addons/"):]
    if spec.startswith("three/"):
        return spec[len("three/"):]
    if spec.startswith("."):
        return posixpath.normpath(posixpath.join(posixpath.dirname(current), spec))
    return None


def _rewrite_imports(code: str, current: str, path_to_index: dict) -> str:
    """Rewrite every local import specifier to a __GVCOT_DEP_<i>__ placeholder."""
    def repl(m):
        pre, q, spec, q2 = m.group(1), m.group(2), m.group(3), m.group(4)
        p = _three_resolve(spec, current)
        if p is not None and p in path_to_index:
            return f"{pre}{q}__GVCOT_DEP_{path_to_index[p]}__{q2}"
        return m.group(0)
    return _IMP_RE.sub(repl, code)


@functools.lru_cache(maxsize=1)
def build_vendored_bundle():
    """Read static/three/, topo-sort the module graph, rewrite imports.
    Returns (modules_list, path_to_index) or (None, None) if files are absent."""
    if not STATIC_THREE_DIR.exists():
        return None, None
    sources = {
        p.relative_to(STATIC_THREE_DIR).as_posix(): p.read_text(encoding="utf-8")
        for p in STATIC_THREE_DIR.rglob("*.js")
    }
    if not all(e in sources for e in _THREE_ENTRYPOINTS):
        return None, None

    def deps(path):
        for m in _IMP_RE.finditer(sources[path]):
            r = _three_resolve(m.group(3), path)
            if r is not None and r in sources:
                yield r

    order, seen = [], set()

    def visit(path):
        if path in seen:
            return
        seen.add(path)
        for d in deps(path):
            visit(d)
        order.append(path)            # post-order → deps precede dependents

    for e in _THREE_ENTRYPOINTS:
        visit(e)

    path_to_index = {path: i for i, path in enumerate(order)}
    modules = [{"code": _rewrite_imports(sources[p], p, path_to_index)} for p in order]
    return modules, path_to_index


def inline_three_offline(html_str: str) -> str:
    """Replace the CDN importmap + entry script with a self-contained blob-URL
    module bundle. Falls back to the CDN importmap if vendored files are absent."""
    m = re.search(
        r'(<script[^>]*id="gvcot-entry"[^>]*>)(.*?)(</script>)',
        html_str, re.DOTALL,
    )
    if not m:
        return html_str  # template not in the expected shape — leave untouched

    modules, path_to_index = build_vendored_bundle()

    if modules is None:
        # ── Fallback: use the CDN, restore an executable module script ──
        html_str = html_str.replace("<!--__THREE_BUNDLE__-->", _CDN_IMPORTMAP)
        html_str = html_str.replace(m.group(1), '<script type="module">')
        return html_str

    # ── Offline: append entry as the final module, emit blob bootstrap ──
    entry_code = _rewrite_imports(m.group(2), "__entry__", path_to_index)
    all_mods = list(modules) + [{"code": entry_code}]
    bundle = {"modules": all_mods, "entry": len(all_mods) - 1}
    bundle_json = json.dumps(bundle, ensure_ascii=False).replace("</", "<\\/")

    boot = (
        '<script id="gvcot-bundle" type="application/json">'
        + bundle_json +
        '</script>\n<script>(function(){'
        'var B=JSON.parse(document.getElementById("gvcot-bundle").textContent);'
        'var u=new Array(B.modules.length);'
        'for(var i=0;i<B.modules.length;i++){'
        'var c=B.modules[i].code.replace(/__GVCOT_DEP_(\\d+)__/g,function(_,d){return u[+d];});'
        'u[i]=URL.createObjectURL(new Blob([c],{type:"text/javascript"}));}'
        'import(u[B.entry]).catch(function(e){console.error("3D bundle error",e);'
        'var d=document.createElement("div");'
        'd.style.cssText="position:fixed;top:12px;left:12px;color:#ff6b6b;'
        'font-family:monospace;font-size:13px;z-index:99999";'
        'd.textContent="3D graph failed to load: "+e;document.body.appendChild(d);});'
        '})();</script>'
    )
    html_str = html_str.replace("<!--__THREE_BUNDLE__-->", "")
    html_str = html_str.replace(m.group(0), boot)
    return html_str
