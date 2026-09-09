"""Freeze the running dashboard into one self-contained HTML file.

Captures every GET the UI makes against a live server, recompresses the
figures, and inlines CSS/JS/data so the page works with no server, no venv
and no runs/ directory -- for handing to someone who just wants to click
around. Writes are stubbed out (it is a read-only snapshot).

    venv\\Scripts\\python.exe -m app.make_snapshot [-o snapshot.html]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

STATIC = Path(__file__).resolve().parent / "static"

# figure ladder: widest/best first, stepped down until the page fits the budget
LADDER = [(1600, 80), (1400, 76), (1200, 72), (1000, 65)]
BUDGET_MB = 13.0          # artifacts cap at 16MB rendered; leave headroom


def norm(url: str) -> str:
    """Key a request by path + sorted decoded params (must match the JS shim)."""
    parts = urllib.parse.urlsplit(url)
    pairs = sorted(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
    return parts.path + "?" + json.dumps(pairs, separators=(",", ":"))


class Grab:
    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def raw(self, path: str) -> bytes:
        with urllib.request.urlopen(self.base + path, timeout=120) as r:
            return r.read()

    def json(self, path: str):
        return json.loads(self.raw(path).decode("utf-8"))


def encode_figure(raw: bytes, width: int, quality: int) -> str:
    from PIL import Image

    im = Image.open(io.BytesIO(raw)).convert("RGB")
    if im.width > width:
        im.thumbnail((width, width * 4), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=quality, method=5)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build(base: str, out: Path) -> None:
    g = Grab(base)
    frozen: dict = {}

    print("capturing api ...")
    for path in ("/api/state", "/api/results", "/api/files",
                 "/api/detect/options", "/api/jobs"):
        frozen[norm(path)] = g.json(path)
        print("  ", path)

    state = frozen[norm("/api/state")]
    files = frozen[norm("/api/files")]

    # every figure the results tab can show, plus every image the files tab previews
    figure_urls, present_paths = [], []
    for run in state["runs"]:
        for name in run["figures"]:
            figure_urls.append(
                "/api/figure?pipeline=%s&mode=%s&name=%s"
                % (run["pipeline"], run["mode"], urllib.parse.quote(name)))
    for sec in files["sections"]:
        for e in sec["entries"]:
            if e["status"] == "present":
                present_paths.append(e["path"])

    print("capturing %d figures ..." % len(figure_urls))
    originals = {}
    for i, u in enumerate(figure_urls, 1):
        originals[u] = g.raw(u)
        print("   %d/%d" % (i, len(figure_urls)), end="\r")
    print()

    # file-tab previews: /api/file/inspect for everything, /api/file/raw for images
    print("capturing %d file previews ..." % len(present_paths))
    raw_for_path = {}
    for p in present_paths:
        q = "/api/file/inspect?path=" + urllib.parse.quote(p, safe="")
        info = g.json(q)
        frozen[norm(q)] = info
        if info.get("type") == "image":
            rp = "/api/file/raw?path=" + urllib.parse.quote(p, safe="")
            raw_for_path[rp] = g.raw(rp)

    # re-encode figures, stepping the ladder down until the page fits
    images: dict = {}
    for width, quality in LADDER:
        images = {}
        for u, raw in originals.items():
            images[norm(u)] = encode_figure(raw, width, quality)
        for u, raw in raw_for_path.items():
            images[norm(u)] = encode_figure(raw, width, quality)
        mb = sum(len(v) for v in images.values()) / 1e6
        print("  figures at %dpx q%d -> %.1f MB" % (width, quality, mb))
        if mb <= BUDGET_MB:
            break

    css = (STATIC / "style.css").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    body = html[html.index("<body>") + 6:html.index("</body>")]
    body = re.sub(r'<script src="[^"]*"></script>', "", body)

    payload = json.dumps({"json": frozen, "img": images},
                         separators=(",", ":"), ensure_ascii=False)
    # </script> inside a JSON string would close the tag early
    payload = payload.replace("</", "<\\/")

    page = (PAGE.replace("__CSS__", css)
                .replace("__BODY__", body)
                .replace("__DATA__", payload)
                .replace("__APP__", js))
    out.write_text(page, encoding="utf-8")
    print("wrote %s (%.1f MB)" % (out, out.stat().st_size / 1e6))


PAGE = """<title>Injection Detection Dashboard</title>
<style>
__CSS__

/* snapshot-only: banner explaining why nothing can be launched from here */
#snap-note {
  background: rgba(57, 135, 229, 0.10); border-bottom: 1px solid var(--border);
  color: var(--ink-2); font: 12.5px/1.5 var(--sans); padding: 7px 20px;
}
#snap-note b { color: var(--text); font-weight: 650; }
</style>
<div id="snap-note">
  <b>Read-only snapshot.</b> Frozen copy of the live dashboard &mdash; every tab,
  table, sort and filter works, but queueing runs and live detection are disabled.
</div>
__BODY__
<script>
/* ---- offline shim: serve the frozen capture instead of the API ---- */
(function () {
  var SNAP = __DATA__;
  function key(u) {
    var url = new URL(u, location.origin);
    var pairs = [];
    url.searchParams.forEach(function (v, k) { pairs.push([k, v]); });
    pairs.sort(function (a, b) {
      return a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : (a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0);
    });
    return url.pathname + "?" + JSON.stringify(pairs);
  }
  function reply(status, body) {
    return Promise.resolve(new Response(JSON.stringify(body), {
      status: status, headers: { "Content-Type": "application/json" }
    }));
  }
  window.fetch = function (u, opts) {
    var method = ((opts && opts.method) || "GET").toUpperCase();
    if (method !== "GET") {
      return reply(503, { detail: "Disabled in this snapshot \\u2014 run the real dashboard to launch jobs or detect." });
    }
    var hit = SNAP.json[key(u)];
    if (hit !== undefined) return reply(200, hit);
    return reply(404, { detail: "Not captured in this snapshot." });
  };

  /* images load via src=, not fetch, so swap those URLs in the DOM */
  function swap(el) {
    var src = el.getAttribute("src");
    if (src && src.charAt(0) === "/") {
      var d = SNAP.img[key(src)];
      if (d) { el.src = d; } else { el.removeAttribute("src"); }
    }
    var href = el.getAttribute("href");
    if (href && href.charAt(0) === "/") {
      el.removeAttribute("href");           /* full-size popout can't open offline */
      el.style.cursor = "default";
      el.title = "Full-size view is only available in the live dashboard";
    }
  }
  function sweep(root) {
    if (root.nodeType !== 1) return;
    if (root.matches("img,a")) swap(root);
    root.querySelectorAll("img,a").forEach(swap);
  }
  new MutationObserver(function (muts) {
    muts.forEach(function (m) { m.addedNodes.forEach(sweep); });
  }).observe(document.documentElement, { childList: true, subtree: true });
  sweep(document.documentElement);
})();
</script>
<script>
__APP__
</script>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8756",
                    help="running dashboard to capture (default: %(default)s)")
    ap.add_argument("-o", "--out", default="snapshot.html", type=Path)
    a = ap.parse_args()
    try:
        build(a.base, a.out)
    except urllib.error.URLError as e:
        sys.exit("could not reach %s (%s) -- start the dashboard first" % (a.base, e))
