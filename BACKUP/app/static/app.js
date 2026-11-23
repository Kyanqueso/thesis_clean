/* thesis_clean dashboard frontend (no build step, no external deps) */
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const MODE_SHORT = { normal: "normal", user_intent_only: "intent-only", context_only: "context-only" };
/* pipeline focus; drives every card on the Analysis tab (persisted, defaults to P3) */
const ALL_PIPES = ["pipeline1_alamsabi", "pipeline2_organic_bipia", "pipeline3_organic_injected"];
let PIPE_FOCUS;
try { PIPE_FOCUS = new Set(JSON.parse(localStorage.getItem("pipeFocus"))); } catch { /* fall through */ }
if (!PIPE_FOCUS || !PIPE_FOCUS.size) PIPE_FOCUS = new Set(["pipeline3_organic_injected"]);

function renderPipeChips() {
  const el = $("#pipe-chips");
  el.innerHTML = `<span class="lbl">Focus — all cards</span>` + ALL_PIPES.map((p) =>
    `<span class="pipe-chip ${PIPE_FOCUS.has(p) ? "sel" : ""}" data-pipe="${p}" data-tip="${esc(pLabel(p))}">${pShort(p)}</span>`).join("");
  $$("#pipe-chips .pipe-chip").forEach((c) => c.addEventListener("click", () => {
    const p = c.dataset.pipe;
    if (PIPE_FOCUS.has(p)) {
      if (PIPE_FOCUS.size === 1) return; // keep at least one selected
      PIPE_FOCUS.delete(p);
    } else {
      PIPE_FOCUS.add(p);
    }
    try { localStorage.setItem("pipeFocus", JSON.stringify([...PIPE_FOCUS])); } catch {}
    renderPipeChips();
    renderMaster();
    renderAblation();
    renderResultsTable();
    renderResources();
    renderFigPicker();
  }));
}
const focusedResults = () => RESULTS.filter((r) => PIPE_FOCUS.has(r.Pipeline));

/* Verdict thresholds, on best ROC-AUC per pipeline per mode. Intent-only scoring
   high means the label is readable without any injection — a corpus artifact, not
   detection. Move these two numbers to change what the strip claims. */
const SHORTCUT_AUC = 0.90;   // intent-only at or above this = corpus shortcut
const CHANCE_AUC = 0.60;     // intent-only at or below this = no signal without the injection
const DETECT_AUC = 0.80;     // normal must clear this to count as detecting


/* Best row per mode for one pipeline, ranked by ROC-AUC, plus the verdict that
   follows from it. The whole row is kept, not just its AUC: the card also shows
   that configuration's accuracy, and both numbers have to describe the SAME
   model — "best AUC" and "best accuracy" are not always the same row. */
function pipelineVerdict(pipeline) {
  const best = {};
  for (const r of RESULTS) {
    if (r.Pipeline !== pipeline) continue;
    const cur = best[r.Mode];
    if (!cur || (r["ROC-AUC"] ?? -1) > (cur["ROC-AUC"] ?? -1)) best[r.Mode] = r;
  }
  const aucOf = (m) => (best[m] ? best[m]["ROC-AUC"] : undefined);
  const normal = aucOf("normal"), intent = aucOf("user_intent_only");
  let verdict = "inconclusive";
  if (intent >= SHORTCUT_AUC) verdict = "shortcut";
  else if (intent <= CHANCE_AUC && normal >= DETECT_AUC) verdict = "detects";
  if (normal === undefined && intent === undefined) verdict = "no runs";
  return { pipeline, best, normal, intent, context: aucOf("context_only"), verdict };
}

const VERDICT_NOTE = {
  detects: "intent alone is near chance — the signal is in the injected context",
  shortcut: "intent alone already separates the classes — corpus artifact",
  inconclusive: "neither clearly detecting nor clearly a shortcut",
  "no runs": "no results for this pipeline yet",
};

// hover text on the verdict tag: the exact rule, built from the thresholds above
// so it cannot drift from what the code decides
const VERDICT_TIP = {
  detects: `DETECTS: intent alone scores ${CHANCE_AUC.toFixed(2)} ROC-AUC or less (near chance), `
    + `and the normal run reaches ${DETECT_AUC.toFixed(2)} or more.\nThe model needs the injected context to tell the classes apart.`,
  shortcut: `SHORTCUT: intent alone scores ${SHORTCUT_AUC.toFixed(2)} ROC-AUC or more.\n`
    + "The classes can be told apart without seeing the injection, so a high score does not prove detection.",
  inconclusive: `INCONCLUSIVE: neither rule applies. Intent alone scores between ${CHANCE_AUC.toFixed(2)} `
    + `and ${SHORTCUT_AUC.toFixed(2)}, or it is near chance but the normal run is below ${DETECT_AUC.toFixed(2)}.`,
};
const VERDICT_BASIS = "\nScores are each mode's best ROC-AUC across every embedding and classifier.";

/* One labelled line per mode: which configuration won, how well it RANKS
   (ROC-AUC) and how often it is RIGHT at the 0.50 threshold (accuracy). Both
   are named in the markup — a bare 0.9818 reads as accuracy when it is not, and
   the two differ by ~5 points on P3. */
function verdictRow(v, mode, label) {
  const row = v.best[mode];
  const num = (x) => (typeof x === "number" ? fmt(x, 4) : "—");
  const clf = row ? `${row.Embeddings} + ${row.Classifier}` : "no run for this mode";
  return `<div class="v-row" data-tip="${esc(`Best ROC-AUC configuration: ${clf}\nAccuracy is from that same model, at the 0.5 threshold.`)}">
    <span class="v-mode">${label}</span>
    <span class="v-metric">ROC-AUC <b>${num(row && row["ROC-AUC"])}</b></span>
    <span class="v-metric">accuracy <b>${num(row && row["Accuracy"])}</b></span>
  </div>`;
}

function renderVerdicts() {
  const cards = ALL_PIPES.map(pipelineVerdict).filter((v) => v.verdict !== "no runs");
  $("#verdict-strip").innerHTML = cards.map((v) => `<div class="verdict ${v.verdict}">
    <div class="v-head"><b>${esc(pShort(v.pipeline))}</b>
      <span class="v-tag"${VERDICT_TIP[v.verdict] ? ` data-tip="${esc(VERDICT_TIP[v.verdict] + VERDICT_BASIS)}"` : ""}>${v.verdict}</span></div>
    <div class="v-name">${esc(pLabel(v.pipeline).replace(/^P\d+\s*·\s*/, ""))}</div>
    <div class="v-nums">
      ${verdictRow(v, "normal", "normal")}
      ${verdictRow(v, "user_intent_only", "intent-only")}
    </div>
    <div class="v-why muted">${VERDICT_NOTE[v.verdict]}</div>
  </div>`).join("");
}

/* in-cell micro bar: value scaled from 0.5 (empty) to 1.0 (full) */
const microbar = (v) => typeof v === "number"
  ? `<span class="microbar"><i style="width:${Math.max(0, Math.min(1, (v - 0.5) / 0.5)) * 100}%"></i></span>`
  : "";

let STATE = null;        // /api/state payload
let RESULTS = [];        // /api/results rows
let DETECT_OPTIONS = []; // /api/detect/options
let openLogJob = null;
let logOffset = 0;
let LOG_TEXT = "";       // full log of the newest job, for the progress scrape
let PREVIEW_FOR = null;  // job id whose results are already previewed

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || body.error || r.statusText);
  return body;
}
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (x, d = 4) => (typeof x === "number" ? x.toFixed(d) : x ?? "");
const pLabel = (p) => (STATE && STATE.pipelines[p]) || p;
const pShort = (p) => p.replace("pipeline", "P").split("_")[0];

/* One floating tooltip for anything carrying data-tip. The browser's own title
   tooltip waits ~1s, is tiny and unstyled; this one allows line breaks and is
   position:fixed so a scrolling table cannot clip it. The first one waits a
   beat, so sweeping the mouse across a row of buttons does not flash a tooltip
   per button; while one is open (or just closed) its neighbours swap in at once. */
const TIP = document.createElement("div");
TIP.className = "tip";
TIP.hidden = true;
document.body.appendChild(TIP);
let tipFor = null, tipTimer = 0, tipWarmUntil = 0;

function showTip(el) {
  if (tipFor !== el || !el.isConnected) return;
  TIP.textContent = el.dataset.tip;
  TIP.hidden = false;
  const r = el.getBoundingClientRect(), t = TIP.getBoundingClientRect();
  const left = Math.max(8, Math.min(innerWidth - t.width - 8, r.left + r.width / 2 - t.width / 2));
  const top = r.bottom + 8 + t.height > innerHeight - 8 ? r.top - t.height - 8 : r.bottom + 8;
  TIP.style.left = `${left}px`;
  TIP.style.top = `${top}px`;
}

document.addEventListener("mouseover", (e) => {
  const el = e.target.closest ? e.target.closest("[data-tip]") : null;
  if (el === tipFor) return;
  clearTimeout(tipTimer);
  if (!TIP.hidden) tipWarmUntil = Date.now() + 400;
  TIP.hidden = true;
  tipFor = el;
  if (!el) return;
  if (Date.now() < tipWarmUntil) showTip(el);
  else tipTimer = setTimeout(() => showTip(el), 350);
});
const hideTip = () => { clearTimeout(tipTimer); TIP.hidden = true; tipFor = null; };
document.addEventListener("mouseleave", hideTip);
document.addEventListener("mousedown", hideTip);   // a click answers the question the tooltip was for
window.addEventListener("scroll", hideTip, true);

/* ================= tabs ================= */
$$(".tab-btn").forEach((b) => b.addEventListener("click", () => {
  $$(".tab-btn").forEach((x) => x.classList.toggle("active", x === b));
  $$("section.tab").forEach((s) => s.classList.toggle("active", s.id === "tab-" + b.dataset.tab));
  $("#sec-nav").hidden = b.dataset.tab !== "results";
  if (b.dataset.tab === "results") loadResults();
  if (b.dataset.tab === "detect") loadDetectOptions();
  if (b.dataset.tab === "files") { loadFiles(); loadPeek(); }
}));

/* ---------- Analysis mini-nav ----------
   Jump links in the sticky header, shown on the Analysis tab only. The section
   currently under the header is highlighted as the page scrolls, and links to
   cards that are hidden (no results yet) are hidden with them. */
const SEC_IDS = ["master-card", "ablation-card", "allrows-card", "resources-card", "figures-card"];
const PAGE_HEADER = $("body > header");

// the header's real height, so a jump lands just below it instead of under it
new ResizeObserver(() =>
  document.documentElement.style.setProperty("--hdr", `${PAGE_HEADER.offsetHeight}px`)).observe(PAGE_HEADER);

$("#sec-nav").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-go]");
  if (!b) return;
  if (b.dataset.go === "top") window.scrollTo({ top: 0, behavior: "smooth" });
  else {
    const card = document.getElementById(b.dataset.go);
    setCollapsed(card, false);   // jumping to a folded card opens it
    card.scrollIntoView({ behavior: "smooth", block: "start" });
  }
});

function updateSecNav() {
  const nav = $("#sec-nav");
  if (nav.hidden) return;
  const shown = SEC_IDS.filter((id) => document.getElementById(id).offsetParent);
  const line = PAGE_HEADER.offsetHeight + 40;
  let current = "top";
  for (const id of shown) if (document.getElementById(id).getBoundingClientRect().top <= line) current = id;
  // scrolled to the very bottom: the last card may be too short to reach the line
  if (shown.length && innerHeight + scrollY >= document.documentElement.scrollHeight - 2) current = shown.at(-1);
  nav.querySelectorAll("button[data-go]").forEach((b) => {
    b.hidden = b.dataset.go !== "top" && !shown.includes(b.dataset.go);
    b.classList.toggle("on", b.dataset.go === current);
    if (b.dataset.go === current) b.setAttribute("aria-current", "true"); else b.removeAttribute("aria-current");
  });
}
let secRaf = 0;
window.addEventListener("scroll", () => {
  if (!secRaf) secRaf = requestAnimationFrame(() => { secRaf = 0; updateSecNav(); });
}, { passive: true });

/* ================= run tab ================= */
/* Embeddings and results ship with the repo, so the dashboard no longer builds
   datasets — it selects cells and trains classifiers over what is already on
   disk. The matrix is the selector; one button runs the selection. */
/* The grid is the picker: each cell is one run, and carries two independent
   facts — selected for this launch, and whether results already exist. */
const SELECTED = new Set();      // "pipeline|mode"
let LAST_RUN_KEYS = new Set();   // what the most recent launch covered
let SEL_INIT = false;

/* What the Analysis tab is a view of. Results are read off disk on every
   request, so the only thing that can go stale is the page: this fingerprints
   the artifacts /api/state reports and reloads the tab when they change — a run
   finishing, or a whole runs/ tree dropped in beside the repo. */
let RESULTS_SIG = null;

function resultsSignature(state) {
  return JSON.stringify([
    state.paths && state.paths.runs_dir,
    state.has_sweep_summary,
    (state.runs || []).map((r) => [
      r.pipeline, r.mode, r.has_results, r.has_predictions,
      r.results_info && r.results_info.mtime, r.results_info && r.results_info.size,
      r.figures.length, r.models.length,
    ]),
  ]);
}

async function refreshState() {
  try { STATE = await fetchJSON("/api/state"); } catch { return; }

  // a changed fingerprint means the tab is showing something that no longer
  // matches disk; reload it when it is open, and let the tab-click handler do
  // it otherwise so a background change costs nothing
  const sig = resultsSignature(STATE);
  if (sig !== RESULTS_SIG) {
    RESULTS_SIG = sig;
    if ($("#tab-results").classList.contains("active")) loadResults();
  }
  // preselect only what can actually run: a pipeline with no dataset on disk
  // would otherwise sit checked and silently do nothing. A flag, not an
  // emptiness test, so clearing every cell by hand stays cleared.
  if (!SEL_INIT) {
    SEL_INIT = true;
    for (const p of Object.keys(STATE.pipelines)) {
      if (!(STATE.datasets[p] && STATE.datasets[p].info)) continue;
      for (const m of STATE.modes) SELECTED.add(`${p}|${m}`);
    }
  }
  renderStartup();
  renderCapabilities();
  renderMatrix();
}

/* Three lines above the hero button: what has to be on disk before a run means
   anything. Counts come from the same /api/state the matrix reads, so the two
   can never disagree about what is there. */
function renderStartup() {
  if (!STATE) return;
  const ds = Object.values(STATE.datasets);
  const nData = ds.filter((x) => x.info).length;
  const nEmb = STATE.runs.filter((r) => Object.values(r.embeddings).some(Boolean)).length;
  const files = `<button class="tablink" data-goto="files">files</button>`;
  $("#startup").innerHTML = [
    `Add the dataset (${nData}/${ds.length}). Check ${files}, or let <b>Full pipeline</b> build it.`,
    `Add embeddings (${nEmb}/${STATE.runs.length}). Check ${files}.`,
    "Press <b>Run</b> and pick what to train.",
  ].map((t) => `<li>${t}</li>`).join("");
  // click the real tab button rather than duplicating the switch logic, so this
  // cannot drift from however tabs actually work
  $$("#startup .tablink").forEach((b) => b.addEventListener("click",
    () => $(`.tab-btn[data-tab="${b.dataset.goto}"]`).click()));
}

// delivered artifacts carry results + figures but no models or per-sample
// probabilities; say so once, plainly, instead of letting features look broken
function renderCapabilities() {
  const c = STATE.capabilities;
  const el = $("#matrix-note");
  if (!c) { el.hidden = true; return; }
  const gaps = [];
  if (!c.n_runs_with_predictions) gaps.push("per-sample probabilities");
  if (!c.n_runs_with_models) gaps.push("saved classifiers");
  if (!gaps.length) { el.hidden = true; return; }
  el.hidden = false;
  el.textContent = `No ${gaps.join(" or ")} on disk yet. Running produces them.`;
}

function renderMatrix() {
  if (!STATE) return;
  const modes = STATE.modes;
  let html = `<div class="mhead"></div>`
    + modes.map((m) => `<button class="mhead-btn" data-col="${m}"
        title="Take the whole ${MODE_SHORT[m]} column">${MODE_SHORT[m]}</button>`).join("");

  for (const p of Object.keys(STATE.pipelines)) {
    const hasData = !!(STATE.datasets[p] && STATE.datasets[p].info);
    html += `<button class="mname" data-row="${p}" title="Take the whole ${esc(pShort(p))} row">
      <b>${esc(pShort(p))}</b>
      <span class="mname-full">${esc(pLabel(p).replace(/^P\d+\s*·\s*/, ""))}</span>
    </button>`;

    for (const m of modes) {
      const run = STATE.runs.find((r) => r.pipeline === p && r.mode === m) || {};
      const key = `${p}|${m}`;
      const why = run.has_results ? "Results already exist" : "Not run yet";
      html += `<button class="mcell${SELECTED.has(key) ? " on" : ""}${
        run.has_results ? " done" : ""}${hasData ? "" : " locked"}"
        data-key="${key}" ${hasData ? "" : "disabled"}
        title="${why}${hasData ? "" : ", and the dataset is missing"}">
        <span class="mcell-mark">${run.has_results ? "✓" : ""}</span>
      </button>`;
    }
  }
  $("#matrix-grid").innerHTML = html;

  // a cell toggles itself; a name toggles its whole row or column
  const take = (keys) => {
    const all = keys.every((k) => SELECTED.has(k));
    keys.forEach((k) => (all ? SELECTED.delete(k) : SELECTED.add(k)));
    renderMatrix();
    updateReviewBtn();
  };
  $$("#matrix-grid .mcell").forEach((b) =>
    b.addEventListener("click", () => take([b.dataset.key])));
  $$("#matrix-grid [data-row]").forEach((b) =>
    b.addEventListener("click", () => take(STATE.modes.map((m) => `${b.dataset.row}|${m}`))));
  $$("#matrix-grid [data-col]").forEach((b) =>
    b.addEventListener("click", () =>
      take(Object.keys(STATE.pipelines).map((p) => `${p}|${b.dataset.col}`))));
}

function updateReviewBtn() {
  const n = SELECTED.size;
  const btn = $("#b-review");
  btn.disabled = !n;
  btn.textContent = n ? `Review ${n} run${n === 1 ? "" : "s"}` : "Nothing picked";
}

/* one place builds the params, so the hero button and every matrix cell can
   never disagree about what the current settings mean */
function runParams() {
  const seed = parseInt($("#f-seed").value, 10);
  const p = {
    models: $$(".f-model").filter((c) => c.checked).map((c) => c.value).join(","),
    force: $("#f-force").checked,
    save_models: $("#f-save-models").checked,
    skip_projections: $("#f-skip-proj").checked,
    qwen3_remote: $("#f-vast").checked,
  };
  // a row limit is what makes it a smoke test, and the backend reroutes those
  // to a separate folder so they cannot overwrite delivered results
  if ($("#f-smoke").checked) {
    p.limit = Math.max(1, parseInt($("#f-limit").value, 10) || 500);
  }
  // the pipeline hardcodes SEED=42 and takes no --seed argument yet; this is
  // sent so the backend can honour it once one exists
  if (seed >= 0) p.seed = seed;
  return p;
}

/* ---- wizard ---- */
function goStep(n) {
  $$(".step").forEach((s) => (s.hidden = +s.dataset.step !== n));
  $("#wiz-dots").hidden = n === 0;
  const pos = n >= 4 ? 2 : n;   // the full-pipeline and resources forms sit where "pick" does
  $$("#wiz-dots i").forEach((d, i) => d.classList.toggle("on", i < pos));
  launchMsg("");
  if (n === 2) { renderMatrix(); updateReviewBtn(); }
  if (n === 3) renderReview();
  if (n === 4) renderFullPlan();
}

/* ---- full pipeline ---- */
function fullParams() {
  return {
    models: $$(".fp-model").filter((c) => c.checked).map((c) => c.value).join(","),
    qwen3_remote: $("#fp-vast").checked,
    skip_done: $("#fp-skip-done").checked,
    rebuild_data: $("#fp-rebuild").checked,
    force: $("#fp-force").checked,
    skip_projections: $("#fp-skip-proj").checked,
    save_models: true,
  };
}

/* The plan is read off the same /api/state the backend decides from, so what
   it says will be skipped is what build_cmds skips. */
function renderFullPlan() {
  if (!STATE) return;
  const p = fullParams();
  const has = (k) => !!(STATE.datasets[k] && STATE.datasets[k].info);
  const data = (what, there) => `<li>${what}
    <span class="muted">${there && !p.rebuild_data ? "already on disk, kept" : "will run"}</span></li>`;
  $("#full-plan").innerHTML =
    data("Download the Alamsabi dataset.", has("pipeline1_alamsabi"))
    + data("Build the organic datasets: collect sources, clean and de-identify the emails.",
        has("pipeline2_organic_bipia") && has("pipeline3_organic_injected"))
    + `<li>Embed all 9 runs with ${esc(p.models.replaceAll(",", ", ") || "nothing picked")}.
        <span class="muted">${p.force ? "recomputed" : "existing embeddings are reused"}${
          p.qwen3_remote ? "; qwen3 on Vast.ai" : ""}</span></li>
       <li>Train and evaluate all 9 runs.
        <span class="muted">${p.skip_done && !p.force ? "finished runs are skipped" : "every run is retrained"}</span></li>
       <li>Build the summary table.</li>`;
  $("#b-confirm-full").disabled = !p.models;
}
$$("#tab-runs [data-step='4'] input").forEach((i) => i.addEventListener("change", renderFullPlan));

// qwen3 on Vast.ai means nothing unless qwen3 is one of the embeddings
for (const [vastBox, models] of [["#f-vast", ".f-model"], ["#fp-vast", ".fp-model"]]) {
  $(vastBox).addEventListener("change", () => {
    if (!$(vastBox).checked) return;
    $$(models).find((c) => c.value === "qwen3").checked = true;
    if (vastBox === "#fp-vast") renderFullPlan();
  });
}

$("#b-full").addEventListener("click", () => goStep(4));
$("#b-confirm-full").addEventListener("click", () => {
  if (!STATE) return;
  if (!confirm("This runs the whole pipeline and can take many hours. Continue?")) return;
  LAST_RUN_KEYS = new Set();
  for (const p of Object.keys(STATE.pipelines))
    for (const m of STATE.modes) LAST_RUN_KEYS.add(`${p}|${m}`);
  queueJob("full", fullParams());
  goStep(0);
});

/* ---- resource benchmark ---- */
function resourceParams() {
  return {
    models: $$(".rs-model").filter((c) => c.checked).map((c) => c.value).join(","),
    stages: $$(".rs-stage").filter((c) => c.checked).map((c) => c.value).join(","),
  };
}
const updateResourcesBtn = () => {
  const p = resourceParams();
  $("#b-confirm-resources").disabled = !p.models || !p.stages;
};
$$("#tab-runs [data-step='5'] input").forEach((i) => i.addEventListener("change", updateResourcesBtn));

$("#b-resources").addEventListener("click", () => { updateResourcesBtn(); goStep(5); });
$("#b-confirm-resources").addEventListener("click", () => {
  LAST_RUN_KEYS = new Set();
  queueJob("resources", resourceParams());
  goStep(0);
});

function renderReview() {
  const p = runParams();
  const byPipe = {};
  for (const k of SELECTED) {
    const [pi, m] = k.split("|");
    (byPipe[pi] = byPipe[pi] || []).push(MODE_SHORT[m]);
  }
  const n = SELECTED.size;
  const rows = Object.keys(STATE.pipelines).filter((pi) => byPipe[pi]).map((pi) =>
    `<div class="rev-row"><b>${esc(pShort(pi))}</b>
      <span>${esc(byPipe[pi].join(", "))}</span></div>`).join("");
  $("#review-body").innerHTML =
    `<h3 class="wiz-title">${n} run${n === 1 ? "" : "s"}</h3>
     <div class="rev-list">${rows}</div>
     <dl class="rev-meta">
       <div><dt>Embeddings</dt><dd>${esc(p.models || "none picked")}</dd></div>
       <div><dt>Dataset</dt><dd>${p.limit ? `first ${p.limit} rows` : "every row"}</dd></div>
       <div><dt>Seed</dt><dd>${p.seed ?? 42}</dd></div>
     </dl>`;
  const btn = $("#b-confirm-run");
  btn.textContent = `Start ${n} run${n === 1 ? "" : "s"}`;
  btn.disabled = !n || !p.models;
}

/* a full nine is exactly the sweep stage, which is cheaper and prints the
   per-step banners the progress bar counts */
function launchSelected(keys, params) {
  if (!keys.size) return;
  LAST_RUN_KEYS = new Set(keys);
  if (keys.size === 9) return queueJob("sweep", params);
  for (const k of keys) {
    const [pipeline, mode] = k.split("|");
    queueJob("run", { ...params, pipeline, mode });
  }
}

function launchMsg(text, cls = "") {
  const el = $("#launch-msg");
  el.textContent = text;
  el.className = "msg " + cls;
}

async function queueJob(stage, params, msgFn = launchMsg) {
  try {
    const job = await fetchJSON("/api/jobs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stage, params }),
    });
    msgFn(`queued ${stage} (${job.id})`, "ok");
    refreshJobs();
  } catch (e) { msgFn(String(e.message || e), "error"); }
}

$("#b-run").addEventListener("click", () => goStep(1));
$("#b-advanced").addEventListener("click", () => goStep(2));
$("#b-review").addEventListener("click", () => goStep(3));
$$("[data-back]").forEach((b) =>
  b.addEventListener("click", () => goStep(+b.dataset.back)));

// "Run all" ignores the advanced controls entirely: fixed, predictable defaults
$("#b-run-all").addEventListener("click", () => {
  if (!STATE) return;
  if (!confirm("This runs all nine configurations with minilm on the full dataset, "
      + "which can take hours. Continue?")) return;
  const all = new Set();
  for (const p of Object.keys(STATE.pipelines))
    for (const m of STATE.modes) all.add(`${p}|${m}`);
  launchSelected(all, { models: "minilm", force: false, save_models: true, skip_projections: false });
  goStep(0);
});

$("#b-confirm-run").addEventListener("click", () => {
  const p = runParams();
  if (!p.models) return launchMsg("Pick at least one embedding under More settings.", "error");
  if (!p.limit && !confirm("This runs on the full dataset and can take hours. Continue?")) return;
  launchSelected(SELECTED, p);
  goStep(0);
});

$("#f-smoke").addEventListener("change", () => $("#f-limit").disabled = !$("#f-smoke").checked);
$("#f-limit").disabled = true;

$("#b-show-analysis").addEventListener("click", () =>
  $$(".tab-btn").find((b) => b.dataset.tab === "results").click());

/* -------- activity strip -------- */
const jobLabel = (j) =>
  j.stage === "sweep" ? "all 9 runs" :
  j.stage === "full" ? "full pipeline" :
  j.stage === "dataset" ? `dataset: ${j.params.which}` :
  j.stage === "resources" ? "resource benchmark" :
  j.stage === "run" ? `${pShort(j.params.pipeline)} / ${MODE_SHORT[j.params.mode]}` : j.stage;

/* Job times arrive as epoch seconds. Rows carry a short local stamp with the
   full date-time on hover, so "when did this run?" survives a page reload. */
const stampOf = (t) => new Date(t * 1000).toLocaleString([],
  { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
const clockOf = (t) => new Date(t * 1000).toLocaleTimeString([],
  { hour: "2-digit", minute: "2-digit" });

async function refreshJobs() {
  let data;
  try { data = await fetchJSON("/api/jobs"); } catch { return; }
  const jobs = data.jobs;
  const running = jobs.find((j) => j.status === "running");
  const queued = jobs.filter((j) => j.status === "queued").length;

  const pill = $("#queue-pill");
  pill.textContent = running ? `running${queued ? ` · ${queued} queued` : ""}` : "idle";
  pill.classList.toggle("busy", !!running || queued > 0);

  // the queue runs one job at a time, so "what is running" is a single line
  $("#activity-line").innerHTML = running
    ? `<span class="spinner"></span><b>${esc(jobLabel(running))}</b>
       <span class="muted">since ${esc(clockOf(running.started))}, ${Math.round(Date.now() / 1000 - running.started)}s${queued ? `, ${queued} queued` : ""}</span>
       <button class="btn small danger" data-cancel="${running.id}">cancel</button>`
    : `<span class="muted">${jobs.length ? "Nothing running." : "Nothing has run yet. Press Run to start one."}</span>`;

  $("#activity-recent").innerHTML = jobs.filter((j) => j.status !== "running").slice(0, 5)
    .map((j) => {
      const dur = j.started && j.finished ? `${Math.round(j.finished - j.started)}s` : "";
      const at = j.finished || j.started || j.created;
      return `<div class="act-row"><span class="status ${j.status}">${j.status}</span>
        <span class="act-what">${esc(jobLabel(j))}</span>
        <time class="act-time" title="${esc(new Date(at * 1000).toLocaleString())}">${esc(stampOf(at))}</time>
        <span class="act-dur">${dur}</span></div>`;
    }).join("");

  // the log belongs to the newest job, so date the disclosure with its start
  const latest = jobs[0];
  if (latest) {
    $("#log-sum").innerHTML = `Log <span class="muted">${esc(stampOf(latest.created))}</span>`;
  }

  $$("[data-cancel]").forEach((b) => b.addEventListener("click", async () => {
    try { await fetchJSON(`/api/jobs/${b.dataset.cancel}/cancel`, { method: "POST" }); } catch {}
    refreshJobs();
  }));

  // one log, always the newest job — no per-row log buttons to hunt through
  const newest = jobs[0];
  $("#log-wrap").hidden = !newest;
  if (!newest) openLogJob = null; // server restarted: the old job is gone, stop polling it
  if (newest && newest.id !== openLogJob) {
    openLogJob = newest.id;
    logOffset = 0;
    LOG_TEXT = "";
    $("#job-log").textContent = "";
  }

  renderProgress(running);
  if (newest && newest.status === "done" && newest.id !== PREVIEW_FOR) {
    PREVIEW_FOR = newest.id;
    loadPreview();
  }
}

/* Progress comes from the job itself (/api/jobs → progress): the scripts print
   ##step / ##progress marker lines and the server folds them into a step count
   plus where the current step is. The time estimate is still a straight-line
   extrapolation over steps of very different lengths, so treat it as rough. */
const fmtDur = (s) => (s >= 3600 ? `${Math.round(s / 360) / 10}h`
  : s >= 60 ? `${Math.round(s / 60)}m` : `${s}s`);

const PHASE_NAME = {
  download: "Downloading", write: "Writing", pull: "Collecting sources",
  anonymize: "Cleaning and de-identifying", build: "Building dataset",
  load: "Loading model", tokenize: "Tokenizing", embed: "Embedding",
  connect: "Connecting to", upload: "Uploading to Vast.ai", setup: "Setting up",
  project: "Projections", train: "Training", report: "Writing results",
  gpu: "Measuring GPU use", classify: "Measuring classifier",
};
const stepName = (label) => label
  .replace(/pipeline(\d)_\w+/, "P$1")
  .replace("user_intent_only", "intent-only").replace("context_only", "context-only")
  .replace("generate_embeddings", "embeddings").replace("train_evaluate_visualize", "training")
  .replace("benchmark_resources", "resources");

function renderProgress(job) {
  const box = $("#prog");
  if (!job) { box.hidden = true; return; }
  const p = job.progress || {};
  const ph = p.phase;
  const total = p.steps || 1;
  const inner = ph && ph.total ? Math.min(1, ph.done / ph.total) : 0;
  const frac = Math.min(1, (Math.max(0, (p.step || 0) - 1) + inner) / total);   // a step marker means STARTED
  const elapsed = Math.max(0, Date.now() / 1000 - job.started);
  const left = frac > 0.02 ? Math.round(elapsed * (1 - frac) / frac) : null;
  box.hidden = false;
  $("#prog-lbl").textContent = p.label ? stepName(p.label) : "Starting…";
  $("#prog-fill").style.width = `${Math.round(frac * 100)}%`;
  $("#prog-txt").textContent = `step ${Math.min(Math.max(1, p.step || 0), total)} of ${total}`
    + (left ? ` · ~${fmtDur(left)} left` : "");

  $("#phase-row").hidden = !ph;
  if (!ph) return;
  $("#phase-lbl").textContent = (PHASE_NAME[ph.phase] || ph.phase) + (ph.item ? ` · ${ph.item}` : "");
  $("#phase-fill").classList.toggle("indet", !ph.total);
  $("#phase-fill").style.width = `${Math.round(inner * 100)}%`;
  $("#phase-txt").textContent = ph.total
    ? `${ph.done.toLocaleString()} / ${ph.total.toLocaleString()} · ${Math.round(inner * 100)}%`
    : "working…";
}

async function loadPreview() {
  let rows;
  try { rows = (await fetchJSON("/api/results")).rows; } catch { return; }
  const best = rows.filter((r) => LAST_RUN_KEYS.has(`${r.Pipeline}|${r.Mode}`))
    .sort((a, b) => (b["ROC-AUC"] ?? 0) - (a["ROC-AUC"] ?? 0)).slice(0, 6);
  if (!best.length) return;
  $("#preview-table").innerHTML =
    `<tr><th>Run</th><th>Emb</th><th>Classifier</th><th class="num">ROC-AUC</th><th class="num">F1</th></tr>`
    + best.map((r) => `<tr><td>${pShort(r.Pipeline)}/${MODE_SHORT[r.Mode]}</td>
        <td>${esc(r.Embeddings)}</td><td>${esc(r.Classifier)}</td>
        <td class="num">${fmt(r["ROC-AUC"])}</td><td class="num">${fmt(r["F1-Score"])}</td></tr>`).join("");
  $("#results-preview").hidden = false;
}

async function pollLog() {
  if (!openLogJob) return;
  try {
    const data = await fetchJSON(`/api/jobs/${openLogJob}/log?offset=${logOffset}`);
    if (data.text) {
      const el = $("#job-log");
      const stick = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
      el.textContent += data.text;
      LOG_TEXT += data.text;
      logOffset = data.offset;
      if (stick) el.scrollTop = el.scrollHeight;
    }
  } catch {}
}

/* ================= results tab ================= */
async function loadResults() {
  if (!STATE) { try { STATE = await fetchJSON("/api/state"); } catch {} }
  try { RESULTS = (await fetchJSON("/api/results")).rows; } catch { RESULTS = []; }
  const empty = !RESULTS.length;
  $("#no-results-card").style.display = empty ? "" : "none";
  for (const id of ["#master-card", "#ablation-card", "#allrows-card", "#resources-card", "#figures-card"])
    $(id).style.display = empty ? "none" : "";
  updateSecNav();
  if (empty) return;
  renderVerdicts();
  renderPipeChips();
  loadMaster();
  renderAblation();
  renderResultsTable();
  loadResources();
  renderFigPicker();
  updateSecNav();
}

function runKey(r) { return `${r.Pipeline}|${r.Mode}`; }

/* ---------- master results: one grid across every pipeline ---------- */
let MASTER = null;

// [TPR column, header, the FPR that cutoff actually gave on test]
const MASTER_LOWFPR = [
  ["TPR@1%FPR", "TPR @1% FPR", "FPR@1%FPR", "1%"],
  ["TPR@0.1%FPR", "TPR @0.1% FPR", "FPR@0.1%FPR", "0.1%"],
];

// the one long classifier name sets the column width for every row
const CLF_SHORT = { LogisticRegression: "LogReg" };
// spelled out once, in the Classifier header tooltips, not on every LogReg cell
const CLF_ABBR = Object.entries(CLF_SHORT)
  .map(([full, short]) => `${short} = ${full.replace(/([a-z])([A-Z])/g, "$1 $2")}`).join(", ");

/* Header hover text for Master results: what each column is and how it is
   computed. Replaces the collapsed "How to read Δ and p" paragraph. */
const MODE_DESC = {
  normal: "The model sees both the intent and the context, as trained.",
  user_intent_only: "The model sees only the intent. The context, where the injection sits, is removed.",
  context_only: "The model sees only the context. The intent is removed.",
};
const COL_TIP = {
  "Pipeline": "How the dataset was built: P1 Alamsabi, P2 organic + BIPIA, P3 organic self-injected.\nHover a row's P1 / P2 / P3 for its full name.",
  "Emb": "The embedding model that turns the text into vectors before classification.",
  "Accuracy": "Accuracy: share of test rows classified correctly.\n(TP + TN) / all rows, at the 0.5 threshold.",
  "Precision": "Precision: of the rows flagged malicious, the share that really are.\nTP / (TP + FP), at the 0.5 threshold.",
  "Recall": "Recall: of the malicious rows, the share that got flagged.\nTP / (TP + FN), at the 0.5 threshold.",
  "F1-Score": "F1: one number balancing precision and recall.\n2 × P × R / (P + R), at the 0.5 threshold.",
  "ROC-AUC": "ROC-AUC: chance that a random malicious row scores higher than a random benign one.\n0.5 = guessing, 1.0 = perfect. Uses no threshold.",
  "PR-AUC": "PR-AUC: area under the precision–recall curve (average precision).\nUses no threshold. On a balanced test set, guessing scores about 0.5.",
  "FPR": "False-positive rate: share of benign rows wrongly flagged as malicious.\nFP / (FP + TN), at the 0.5 threshold.",
  "TPR@1%FPR": "Share of malicious prompts caught when the cutoff is set so only 1% of benign prompts get flagged.\n"
    + "The cutoff is chosen on held-out training rows, then applied to the test set. Hover a value for the false-alarm rate it actually gave.",
  "TPR@0.1%FPR": "Share of malicious prompts caught when the cutoff is set so only 0.1% of benign prompts (1 in 1,000) get flagged.\n"
    + "The cutoff is chosen on held-out training rows, then applied to the test set. Hover a value for the false-alarm rate it actually gave.",
};

const SIG_NOTE = "ns = not statistically significant: McNemar test, Holm-corrected, p ≥ 0.05.\n"
  + "McNemar compares which rows each model gets right at the 0.5 threshold, so it tests accuracy, not the Δ's metric.";

function deltaTip(v, ref) {
  let t = `${MODE_DESC[v]}\nΔ = ${MODE_SHORT[ref]} − ${MODE_SHORT[v]}, in percentage points. Positive = removing it hurt.`;
  if (ref === "normal" && v === "user_intent_only")
    t += "\nRed (≤ 0.5 pp): the intent alone does as well as the full input, so the model is not using the injection. That is a shortcut.";
  if (ref === "normal" && v === "context_only")
    t += "\nNear zero is the healthy result: the signal is in the context.";
  return `${t}\n${SIG_NOTE}`;
}

// the shared fmt() renders a missing value as blank; in this grid a blank cell
// reads as "zero", so absent metrics get an explicit em dash instead
const mfmt = (x, d = 4) => (typeof x === "number" ? x.toFixed(d) : "—");

async function loadMaster() {
  const ref = $("#ms-reference").value || "normal";
  try { MASTER = await fetchJSON(`/api/master?reference=${encodeURIComponent(ref)}`); }
  catch { MASTER = null; }
  if (!MASTER) return;
  const sel = $("#ms-metric");
  if (!sel.options.length) {
    sel.innerHTML = MASTER.metrics.map((m) =>
      `<option value="${esc(m)}"${m === "ROC-AUC" ? " selected" : ""}>${esc(m)}</option>`).join("");
  }
  renderMaster();
}

function masterRows() {
  if (!MASTER) return [];
  let rows = MASTER.rows.filter((r) => PIPE_FOCUS.has(r.Pipeline));
  if ($("#ms-collapse").value === "best") {
    const best = {};
    for (const r of rows) {
      const k = `${r.Pipeline}|${r.Embeddings}`;
      if (!best[k] || (r["ROC-AUC"] ?? -1) > (best[k]["ROC-AUC"] ?? -1)) best[k] = r;
    }
    rows = Object.values(best);
  }
  return rows;
}

// Δ is signed and the sign is the whole point: positive = the ablation cost
// performance. What a NON-positive Δ means depends on which input was removed,
// so only the intent-only column raises an alarm:
//   intent-only  — the intent carries no label information by construction, so
//                  "removing it cost nothing" is expected; "removing it HELPED,
//                  or the model was already near-perfect without the document"
//                  means the classes separate without the injection. Alarm.
//   context-only — dropping the intent and losing nothing is an ordinary
//                  finding (the signal lives in the document), not a defect.
// Colouring both the same way cried wolf on every P1/P3 row.
// The McNemar p used to have its own columns, but the Δ cell already carries
// its verdict ("ns"), so the exact value moved into the cell's hover text.
function degCell(value, sig, variant, p) {
  if (value === null || value === undefined) return `<td class="num">—</td>`;
  const pp = value * 100;
  const alarm = variant === "user_intent_only";
  const cls = pp > 0.5 ? "deg-pos" : (alarm ? "deg-neg" : "deg-quiet");
  const mark = sig === false ? " ns" : "";
  return `<td class="num ${cls}" data-tip="${esc(pTitle(p, sig))}">${pp >= 0 ? "+" : ""}${pp.toFixed(2)}<span class="ns">${mark}</span></td>`;
}

// p is Holm-corrected; at n≈14k it underflows to exactly 0, which toExponential
// would print as "0e+0"
function pTitle(p, sig) {
  if (p === null || p === undefined) return "McNemar p: needs predictions.npz from every mode";
  const txt = p === 0 ? "< 1e-300" : p < 1e-4 ? `= ${p.toExponential(0)}` : `= ${p.toFixed(4)}`;
  return `McNemar p ${txt} (Holm) — ${sig ? "significant" : "not significant"}`;
}

function renderMaster() {
  if (!MASTER) return;
  const metric = $("#ms-metric").value || "ROC-AUC";
  const rows = masterRows();

  /* Delivered runs predate several metrics, so whole column groups arrive empty.
     An em-dash grid reads as a broken table; drop what has no data anywhere. */
  const has = (key) => rows.some((r) => typeof r[key] === "number");
  const q = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"].filter(has);
  const errs = ["FPR"].filter(has);
  const lowFpr = MASTER_LOWFPR.filter(([k]) => has(k));
  const degVars = MASTER.variants.filter((v) => has(`Deg:${v}:${metric}`));
  const dropped = (6 - q.length) + (1 - errs.length) + (MASTER_LOWFPR.length - lowFpr.length)
    + (MASTER.variants.length - degVars.length);

  const qShort = { "Accuracy": "Acc", "Precision": "Prec", "Recall": "Rec",
                   "F1-Score": "F1", "ROC-AUC": "ROC-AUC", "PR-AUC": "PR-AUC" };

  const ref = MASTER.reference;
  const refName = MODE_SHORT[ref];
  const tipAttr = (tip) => ` data-tip="${esc(tip)}"`;
  const th = (label, tip, cls = "") => `<th${cls ? ` class="${cls}"` : ""}${tipAttr(tip)}>${label}</th>`;
  const clfTip = ($("#ms-collapse").value === "best"
    ? `The classifier with the best ROC-AUC in the ${refName} run, for this pipeline × embedding.`
    : "Each classifier trained on these embeddings, one row each.") + `\n${CLF_ABBR}.`;

  // two-tier header: group band on top, columns beneath; every header explains itself on hover
  const band = (n, label, tip) => (n ? `<th colspan="${n}"${tipAttr(tip)}>${label}</th>` : "");
  let head = `<tr class="grp">
    ${band(3, "Configuration", "Which run the row describes.")}
    ${band(q.length, `Detection — ${esc(refName)}`,
      `How well the ${refName} run detects injections.\nThreshold metrics use 0.5; ROC-AUC and PR-AUC use no threshold.`)}
    ${band(errs.length, "Errors", "False alarms: benign rows wrongly flagged as malicious, at the 0.5 threshold.")}
    ${band(lowFpr.length, "Low false-alarm",
      "How many attacks are still caught when the cutoff is moved so false alarms stay rare.\nThe cutoff is set on held-out training rows, never on the test set.")}
    ${band(degVars.length, `Ablation Δ ${esc(metric)} (pp)`,
      `How much ${metric} drops when part of the input is removed: ${refName} − ablated, in percentage points.\n`
      + `Positive = removing it hurt.\n${SIG_NOTE}`)}
    </tr>`;
  head += `<tr>${th("Pipeline", COL_TIP.Pipeline)}${th("Emb", COL_TIP.Emb)}${th("Classifier", clfTip)}
    ${q.map((m) => th(qShort[m], COL_TIP[m], "num")).join("")}
    ${errs.map((e) => th(e, COL_TIP[e], "num")).join("")}
    ${lowFpr.map(([k, l]) => th(l, COL_TIP[k], "num")).join("")}
    ${degVars.map((v) => th(`→${MODE_SHORT[v]}`, deltaTip(v, ref), "num")).join("")}
    </tr>`;

  let body = "", lastPipe = null;
  for (const r of rows) {
    const sep = lastPipe && lastPipe !== r.Pipeline ? " pipe-sep" : "";
    lastPipe = r.Pipeline;
    // no shortcut chip here: the verdict cards name the pipeline and the red
    // intent-only Δ marks the row, so a chip column only cost width
    body += `<tr class="${sep}">
      <td${tipAttr(pLabel(r.Pipeline))}>${esc(pShort(r.Pipeline))}</td>
      <td>${esc(r.Embeddings)}</td>
      <td>${esc(CLF_SHORT[r.Classifier] || r.Classifier)}</td>
      ${q.map((m) => `<td class="num">${mfmt(r[m])}</td>`).join("")}
      ${errs.map((e) => `<td class="num">${mfmt(r[e])}</td>`).join("")}
      ${lowFpr.map(([k, , fk, target]) => `<td class="num"${typeof r[fk] === "number"
        ? tipAttr(`Cutoff aimed at ${target} false alarms; on the test set it gave ${(r[fk] * 100).toFixed(2)}%.`) : ""}>${mfmt(r[k])}</td>`).join("")}
      ${degVars.map((v) => degCell(r[`Deg:${v}:${metric}`], r[`Sig:${v}:significant`], v, r[`Sig:${v}:p_holm`])).join("")}
      </tr>`;
  }
  $("#master-table").innerHTML = head + body;

  // the always-visible key that replaced the collapsed "How to read Δ and p"
  $("#master-key").innerHTML = degVars.length ? `
    <span>Δ = ${esc(refName)} − ablated, in pp · positive = removing it hurt</span>
    ${degVars.includes("user_intent_only") && ref === "normal"
      ? `<span><b class="key-bad">red</b> = intent-only Δ ≤ 0.5 pp: the intent alone is enough → shortcut</span>` : ""}
    <span><span class="ns">ns</span> = not significant (McNemar, Holm, p ≥ 0.05)</span>
    <span class="key-hint">hover a header or a Δ for details</span>` : "";

  const notes = MASTER.notes || [];
  $("#master-notes").innerHTML = notes.length
    ? `<details class="why"><summary>${dropped} empty column${dropped === 1 ? "" : "s"} hidden${
        dropped ? " — " : ""}why these runs have no per-sample metrics</summary>
       ${notes.map((n) => `<p class="muted note">${esc(n)}</p>`).join("")}</details>`
    : "";
  $("#ms-msg").textContent = `${rows.length} row${rows.length === 1 ? "" : "s"}`;
}

function masterCSV() {
  const t = $("#master-table");
  return Array.from(t.rows).map((tr) =>
    Array.from(tr.cells).map((td) => {
      const v = td.textContent.trim().replace(/\s+/g, " ");
      return /[",]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;
    }).join(",")).join("\n");
}

["#ms-metric", "#ms-collapse"].forEach((id) =>
  $(id).addEventListener("change", renderMaster));
$("#ms-reference").addEventListener("change", loadMaster);
$("#ms-copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(masterCSV());
    $("#ms-msg").textContent = "copied to clipboard";
  } catch { $("#ms-msg").textContent = "clipboard blocked — select the table and copy"; }
});

/* Grouped horizontal bars, one group per pipeline x embedding, one bar per mode.
   Categorical slots 1-3 of the validated dark palette (blue/orange/aqua) - the
   three modes are identities, not statuses, so they get categorical hues and the
   verdict strip carries the good/bad judgement.
   An ablation holds the model fixed and changes only the input, so each group is
   ONE classifier — the one Master results shows (best normal-mode ROC-AUC) — and
   the gap between its bars is exactly the table's Δ. Picking the best classifier
   per bar instead mixed models inside a group and disagreed with the table.
   Bars grow out of the 0.50 chance line: right = better than guessing, left =
   worse. The axis only extends below 0.50 when some bar actually goes there. */
const MODE_HUE = { normal: "var(--m-normal)", user_intent_only: "var(--m-intent)", context_only: "var(--m-context)" };
const AB_GEO = { pad: 120, right: 66, bar: 18, gap: 4, group: 26, sep: 22, subsep: 10, top: 34 };
// "All" draws up to 45 groups; thinner bars keep that from running thousands of px
const AB_DENSE = { bar: 12, gap: 3, group: 20 };
let AB_GROUPS = [];   // what the chart drew, for the hover card
let AB_W = 0;         // width it was drawn at, so a resize can redraw

/* Which groups to draw. Classifier: "best" = the Master-results pick (default),
   "all" = every classifier, or one classifier by name. Embeddings toggle like the
   Focus chips, keeping at least one on. Pipelines come from the Focus chips —
   a second pipeline control here could disagree with them. */
let AB_CLF = "best";
const AB_EMB_OFF = new Set();

function renderAbPicker() {
  const rows = focusedResults();
  const clfs = [...new Set(rows.map((r) => r.Classifier))].sort();
  const embs = [...new Set(rows.map((r) => r.Embeddings))].sort();
  if (AB_CLF !== "best" && AB_CLF !== "all" && !clfs.includes(AB_CLF)) AB_CLF = "best";
  const chip = (attr, val, label, sel, tip = "") =>
    `<button class="pipe-chip ${sel ? "sel" : ""}" ${attr}="${esc(val)}" aria-pressed="${sel}"${
      tip ? ` data-tip="${esc(tip)}"` : ""}>${esc(label)}</button>`;
  $("#ab-clfs").innerHTML =
    chip("data-clf", "best", "Best", AB_CLF === "best",
      "The classifier Master results shows for each pipeline × embedding (best normal ROC-AUC)")
    + chip("data-clf", "all", "All", AB_CLF === "all", "Every classifier, one group each")
    + `<span class="pick-div"></span>`
    + clfs.map((k) => chip("data-clf", k, CLF_SHORT[k] || k, AB_CLF === k)).join("");
  $("#ab-embs").innerHTML = embs.map((e) => chip("data-emb", e, e, !AB_EMB_OFF.has(e))).join("");
}

$("#ab-clfs").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-clf]");
  if (b) { AB_CLF = b.dataset.clf; renderAblation(); }
});
$("#ab-embs").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-emb]");
  if (!b) return;
  const emb = b.dataset.emb;
  if (AB_EMB_OFF.has(emb)) AB_EMB_OFF.delete(emb);
  else if ($$("#ab-embs button[aria-pressed=true]").length > 1) AB_EMB_OFF.add(emb);
  renderAblation();
});

/* rounded at the data end (x1), square against the chance baseline (x0) */
function barPath(x0, x1, y, h, r = 4) {
  const w = Math.abs(x1 - x0);
  if (w < 0.5) return "";
  const k = Math.min(r, w, h / 2);
  return x1 > x0
    ? `M${x0},${y} H${x1 - k} a${k},${k} 0 0 1 ${k},${k} V${y + h - k} a${k},${k} 0 0 1 ${-k},${k} H${x0} Z`
    : `M${x0},${y} H${x1 + k} a${k},${k} 0 0 0 ${-k},${k} V${y + h - k} a${k},${k} 0 0 0 ${k},${k} H${x0} Z`;
}

/* Same rows, same order, same pick as Master results with reference = normal:
   pipeline order, then embedding, then the classifier with the best normal ROC-AUC
   ("Best"), every classifier in name order ("All", as the table's "every
   classifier"), or one named classifier. */
function ablationGroups(modes) {
  const auc = {};
  const configs = new Map();
  for (const r of focusedResults()) {
    if (AB_EMB_OFF.has(r.Embeddings)) continue;
    auc[`${r.Pipeline}|${r.Embeddings}|${r.Classifier}|${r.Mode}`] = r["ROC-AUC"];
    const k = `${r.Pipeline}|${r.Embeddings}`;
    if (!configs.has(k)) configs.set(k, { pipeline: r.Pipeline, emb: r.Embeddings, clfs: new Set() });
    configs.get(k).clfs.add(r.Classifier);
  }
  const order = (c) => ALL_PIPES.indexOf(c.pipeline);
  return [...configs.values()]
    .sort((a, b) => order(a) - order(b) || String(a.emb).localeCompare(String(b.emb)))
    .flatMap((c) => {
      const clfs = [...c.clfs].sort();
      let best = null, top = -1;
      for (const k of clfs) {
        const v = auc[`${c.pipeline}|${c.emb}|${k}|normal`];
        if (typeof v === "number" && v > top) { top = v; best = k; }
      }
      const pick = AB_CLF === "best" ? (best ? [best] : [])   // no normal run to anchor on, same as the table
        : AB_CLF === "all" ? clfs
        : clfs.filter((k) => k === AB_CLF);
      return pick.map((clf) => {
        const vals = {};
        for (const m of modes) vals[m] = auc[`${c.pipeline}|${c.emb}|${clf}|${m}`];
        return { pipeline: c.pipeline, emb: c.emb, clf, best: clf === best, vals };
      });
    });
}

function renderAblation() {
  const box = $("#ablation-chart");
  const modes = STATE ? STATE.modes : ["normal", "user_intent_only", "context_only"];
  renderAbPicker();
  const groups = AB_GROUPS = ablationGroups(modes);
  if (!groups.length) {
    box.innerHTML = `<span class="muted">Nothing to show for this selection.</span>`;
    return;
  }
  // drawn at the card's real width, so 1 SVG unit = 1 CSS px and text never scales
  const w = Math.max(560, Math.round(box.clientWidth || 900));
  AB_W = box.clientWidth;
  const { pad, right, sep, subsep, top } = AB_GEO;
  const { bar, gap, group } = groups.length > 12 ? AB_DENSE : AB_GEO;
  const plot = w - pad - right;
  const block = modes.length * bar + (modes.length - 1) * gap;

  // pipelines get the heavy divider; with every classifier drawn, a light one
  // also marks where the embedding changes inside a pipeline
  const newPipe = (i) => i > 0 && groups[i].pipeline !== groups[i - 1].pipeline;
  const newEmb = (i) => i > 0 && AB_CLF === "all" && !newPipe(i) && groups[i].emb !== groups[i - 1].emb;

  const ys = [];
  let y = top;
  groups.forEach((_, i) => {
    if (newPipe(i)) y += sep;
    else if (newEmb(i)) y += subsep;
    ys.push(y);
    y += block + group;
  });
  const h = y - group + 14;

  const seen = groups.flatMap((g) => modes.map((m) => g.vals[m])).filter((v) => typeof v === "number");
  const lo = Math.min(0.5, Math.floor((Math.min(...seen) - 0.05) * 10) / 10);
  const x = (v) => pad + (Math.max(lo, Math.min(1, v)) - lo) / (1 - lo) * plot;
  const base = x(0.5);

  let grid = "";
  for (let i = Math.round(lo * 10); i <= 10; i++) {
    const t = i / 10;
    grid += `<line x1="${x(t)}" y1="${top - 8}" x2="${x(t)}" y2="${h - 6}" class="${t === 0.5 ? "ab-chance" : "ab-grid"}"/>
      <text x="${x(t)}" y="${top - 14}" class="ab-tick${t === 0.5 ? " ab-tick-chance" : ""}" text-anchor="middle">${t === 0.5 ? "0.50 chance" : t.toFixed(1)}</text>`;
  }

  let hits = "", seps = "", body = "";
  groups.forEach((g, gi) => {
    const y0 = ys[gi];
    if (newPipe(gi) || newEmb(gi)) {
      const sy = (ys[gi - 1] + block + y0) / 2;
      seps += `<line x1="0" y1="${sy}" x2="${w}" y2="${sy}" class="${newPipe(gi) ? "ab-sep" : "ab-sep-thin"}"/>`;
    }
    hits += `<rect x="0" y="${y0 - group / 2 + 2}" width="${w}" height="${block + group - 4}" rx="6" class="ab-hit" data-i="${gi}"/>`;
    // two lines: the configuration, then the classifier under it (short name, as in the table);
    // with every classifier on screen, the one Master results picks is marked
    const mark = AB_CLF === "all" && g.best ? `<tspan class="ab-best"> · best</tspan>` : "";
    body += `<text x="${pad - 12}" y="${y0 + block / 2 - 3}" class="ab-label" text-anchor="end">${esc(pShort(g.pipeline))} · ${esc(g.emb)}</text>
      <text x="${pad - 12}" y="${y0 + block / 2 + 14}" class="ab-clf" text-anchor="end">${esc(CLF_SHORT[g.clf] || g.clf)}${mark}</text>`;
    modes.forEach((m, mi) => {
      const v = g.vals[m];
      const by = y0 + mi * (bar + gap);
      const ty = by + bar / 2 + 4.5;
      if (typeof v !== "number") {
        body += `<text x="${base + 6}" y="${ty}" class="ab-val ab-none">no run</text>`;
        return;
      }
      const xv = x(v);
      body += `<path d="${barPath(base, xv, by, bar)}" fill="${MODE_HUE[m]}"/>`;
      body += v >= 0.5
        ? `<text x="${xv + 7}" y="${ty}" class="ab-val">${fmt(v)}</text>`
        : `<text x="${xv - 7}" y="${ty}" class="ab-val" text-anchor="end">${fmt(v)}</text>`;
    });
  });

  const legend = modes.map((m) =>
    `<span class="ab-key" data-tip="${esc(MODE_DESC[m])}"><i style="background:${MODE_HUE[m]}"></i>${MODE_SHORT[m]}</span>`).join("");

  const note = AB_CLF === "best"
    ? "each group is the classifier shown in Master results, so bar gaps equal its Δ"
    : AB_CLF === "all"
      ? `every classifier, one group each · "best" marks the one Master results shows`
      : `${esc(AB_CLF)} in every configuration`;

  box.innerHTML = `<div class="ab-legend">${legend}
      <span class="ab-note">${note}</span></div>
    <svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" class="ab-svg" role="img"
      aria-label="ROC-AUC per pipeline, embedding and classifier, one bar per ablation mode">
      ${hits}${grid}${seps}${body}
    </svg>
    <div class="ab-tip" hidden></div>`;

  const tip = box.querySelector(".ab-tip");
  box.querySelectorAll(".ab-hit").forEach((r) => {
    r.addEventListener("mouseenter", () => {
      tip.innerHTML = abTipHTML(AB_GROUPS[+r.dataset.i], modes);
      tip.hidden = false;
    });
    r.addEventListener("mousemove", (e) => {
      const b = box.getBoundingClientRect();
      const mx = e.clientX - b.left, my = e.clientY - b.top;
      const tw = tip.offsetWidth;
      tip.style.left = `${mx + 16 + tw > b.width ? mx - tw - 16 : mx + 16}px`;
      tip.style.top = `${my + 16}px`;
    });
    r.addEventListener("mouseleave", () => { tip.hidden = true; });
  });
}

/* Hover card: the whole group at once — which model, all three scores, and the
   Δ against normal exactly as the table computes it (normal − ablated, in pp). */
function abTipHTML(g, modes) {
  const ref = g.vals.normal;
  const rows = modes.map((m) => {
    const v = g.vals[m];
    const has = typeof v === "number";
    const d = m !== "normal" && has && typeof ref === "number" ? (ref - v) * 100 : null;
    return `<tr>
      <td><i style="background:${MODE_HUE[m]}"></i>${MODE_SHORT[m]}</td>
      <td class="num">${has ? fmt(v) : "no run"}</td>
      <td class="num tip-d">${d === null ? "" : `Δ ${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(2)} pp`}</td>
      <td class="tip-flag">${has && v < 0.5 ? "below chance" : ""}</td></tr>`;
  }).join("");
  return `<div class="tip-h">${esc(pLabel(g.pipeline))}</div>
    <div class="tip-sub">${esc(g.emb)} · ${esc(g.clf)}${
      g.best && AB_CLF !== "best" ? " (Master results pick)" : ""} · ROC-AUC</div>
    <table>${rows}</table>`;
}

// the chart is drawn to the card's width, so redraw when that width changes
new ResizeObserver(() => {
  const w = $("#ablation-chart").clientWidth;
  if (w && RESULTS.length && Math.abs(w - AB_W) > 4) renderAblation();
}).observe($("#ablation-chart"));

/* ---------- all runs ----------
   Same button pickers as Figures and the ablation card. Pipelines come from the
   Focus chips only: the old Pipeline dropdown could contradict them and show 0
   rows. Rows stay grouped by pipeline (sorting happens inside each group) with
   the Master-results divider, and the classifier Master results shows for each
   pipeline × embedding gets an accent edge. The heatmap is the same rows as a
   classifier × embedding grid per pipeline. */
const RF = { view: "table", mode: "normal", metric: "ROC-AUC", embOff: new Set(), clfOff: new Set() };
// [result key, header, decimals, scale] — headers match Master results
const RF_COLS = [
  ["Pipeline", "Pipeline"], ["Mode", "Mode"], ["Embeddings", "Emb"], ["Classifier", "Classifier"],
  ["Accuracy", "Acc", 4], ["F1-Score", "F1", 4], ["ROC-AUC", "ROC-AUC", 4], ["PR-AUC", "PR-AUC", 4],
  ["Train-Time(s)", "fit s", 1], ["Inference-Time-per-Sample(ms)", "µs/row", 1, 1000],
];
const RF_TIP = {
  "Mode": "Which input the model saw: normal = intent + context, intent-only, or context-only.",
  "Classifier": "The classifier trained on the embeddings.\nA blue left edge marks the one Master results shows (best normal ROC-AUC).\n"
    + `${CLF_ABBR}.`,
  "Inference-Time-per-Sample(ms)": "Total inference time on the test set ÷ number of rows, in microseconds.",
};
const RF_METRICS = [["ROC-AUC", "ROC-AUC"], ["PR-AUC", "PR-AUC"], ["F1-Score", "F1"], ["Accuracy", "Acc"]];
let sortCol = "ROC-AUC", sortAsc = false;

/* the classifier Master results shows per pipeline × embedding: best normal ROC-AUC,
   ties to the first name alphabetically, as the table does */
function masterPicks() {
  const best = {};
  for (const r of [...RESULTS].sort((a, b) => a.Classifier.localeCompare(b.Classifier))) {
    if (r.Mode !== "normal" || typeof r["ROC-AUC"] !== "number") continue;
    const k = `${r.Pipeline}|${r.Embeddings}`;
    if (!best[k] || r["ROC-AUC"] > best[k]["ROC-AUC"]) best[k] = r;
  }
  return new Set(Object.values(best).map((r) => `${r.Pipeline}|${r.Embeddings}|${r.Classifier}`));
}

function renderRfPicker() {
  const rows = RESULTS.filter((r) => PIPE_FOCUS.has(r.Pipeline));
  const modes = STATE ? STATE.modes : ["normal", "user_intent_only", "context_only"];
  const embs = [...new Set(rows.map((r) => r.Embeddings))].sort();
  const clfs = [...new Set(rows.map((r) => r.Classifier))].sort();
  const heat = RF.view === "heat";
  if (heat && RF.mode === "all") RF.mode = "normal";   // a grid holds one mode
  const chip = (attr, val, label, sel, tip = "") =>
    `<button class="pipe-chip ${sel ? "sel" : ""}" ${attr}="${esc(val)}" aria-pressed="${sel}"${
      tip ? ` data-tip="${esc(tip)}"` : ""}>${esc(label)}</button>`;
  $("#rf-view").innerHTML = chip("data-view", "table", "Table", !heat)
    + chip("data-view", "heat", "Heatmap", heat, "One classifier × embedding grid per pipeline, shaded by the chosen metric.");
  $("#rf-modes").innerHTML = modes.map((m) => chip("data-mode", m, MODE_SHORT[m], RF.mode === m, MODE_DESC[m])).join("")
    + (heat ? "" : chip("data-mode", "all", "all", RF.mode === "all", "Rows from all three modes together."));
  $("#rf-metric-grp").hidden = !heat;
  $("#rf-metrics").innerHTML = RF_METRICS.map(([k, l]) => chip("data-metric", k, l, RF.metric === k, COL_TIP[k])).join("");
  $("#rf-embs").innerHTML = embs.map((e) => chip("data-emb", e, e, !RF.embOff.has(e))).join("");
  $("#rf-clfs").innerHTML = clfs.map((k) => chip("data-clf", k, CLF_SHORT[k] || k, !RF.clfOff.has(k))).join("");
}

// toggle chips keep at least one option on, like the Focus chips
function toggleKeep(off, value, box) {
  if (off.has(value)) off.delete(value);
  else if ($$(`${box} button[aria-pressed=true]`).length > 1) off.add(value);
}

$("#rf-pick").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  const d = b.dataset;
  if (d.view) RF.view = d.view;
  else if (d.mode) RF.mode = d.mode;
  else if (d.metric) RF.metric = d.metric;
  else if (d.emb) toggleKeep(RF.embOff, d.emb, "#rf-embs");
  else if (d.clf) toggleKeep(RF.clfOff, d.clf, "#rf-clfs");
  else return;
  renderResultsTable();
});

function renderResultsTable() {
  renderRfPicker();
  const inFocus = RESULTS.filter((r) => PIPE_FOCUS.has(r.Pipeline));
  const rows = inFocus.filter((r) => (RF.mode === "all" || r.Mode === RF.mode)
    && !RF.embOff.has(r.Embeddings) && !RF.clfOff.has(r.Classifier));
  const picks = masterPicks();
  const pickKey = (r) => `${r.Pipeline}|${r.Embeddings}|${r.Classifier}`;

  if (RF.view === "heat") {
    $("#rf-count").textContent = "";
    renderRfHeat(rows, picks);
    return;
  }
  $("#rf-count").textContent = `${rows.length} of ${inFocus.length} rows`;

  const order = (r) => ALL_PIPES.indexOf(r.Pipeline);
  const sorted = rows.slice().sort((a, b) => {
    const g = order(a) - order(b);
    if (g) return g;
    const av = a[sortCol], bv = b[sortCol];
    const c = typeof av === "number" ? av - bv : String(av).localeCompare(String(bv));
    return sortAsc ? c : -c;
  });

  const header = RF_COLS.map(([k, label, d]) => {
    const tip = RF_TIP[k] || COL_TIP[k] || COL_TIP[label] || "";
    const arrow = sortCol === k ? (sortAsc ? " ▲" : " ▼") : "";
    return `<th class="${d !== undefined ? "num" : ""}" data-sort="${k}"${tip ? ` data-tip="${esc(tip)}"` : ""}>${label}${arrow}</th>`;
  }).join("");

  let body = "", lastPipe = null;
  for (const r of sorted) {
    const cls = [lastPipe && lastPipe !== r.Pipeline ? "pipe-sep" : "", picks.has(pickKey(r)) ? "best" : ""]
      .filter(Boolean).join(" ");
    lastPipe = r.Pipeline;
    body += `<tr${cls ? ` class="${cls}"` : ""}>${RF_COLS.map(([k, , d, scale]) => {
      const v = r[k];
      if (d !== undefined) {
        const shown = typeof v === "number" ? (v * (scale || 1)).toFixed(d) : "—";
        return `<td class="num">${shown}${k === "ROC-AUC" ? microbar(v) : ""}</td>`;
      }
      if (k === "Pipeline") return `<td data-tip="${esc(pLabel(v))}">${esc(pShort(v))}</td>`;
      if (k === "Mode") return `<td>${MODE_SHORT[v]}</td>`;
      if (k === "Classifier") return `<td>${esc(CLF_SHORT[v] || v)}</td>`;
      return `<td>${esc(v)}</td>`;
    }).join("")}</tr>`;
  }

  $("#rf-body").innerHTML = `<div class="scroll"><table id="results-table">${`<tr>${header}</tr>`}${body}</table></div>
    <div class="key"><span><i class="key-edge"></i>blue edge = the classifier Master results shows for that pipeline × embedding</span>
      <span class="key-hint">click a header to sort · rows stay grouped by pipeline</span></div>`;
  $$("#results-table [data-sort]").forEach((th) => th.addEventListener("click", () => {
    const c = th.dataset.sort;
    if (sortCol === c) sortAsc = !sortAsc; else { sortCol = c; sortAsc = false; }
    renderResultsTable();
  }));
}

/* One classifier × embedding grid per pipeline, shaded from 0.50 (chance, faint)
   to 1.00 (full) — the same anchor as the ablation chart — so the strongest
   combination is visible at a glance instead of read off 45 rows. */
function renderRfHeat(rows, picks) {
  const m = RF.metric;
  const mLabel = (RF_METRICS.find(([k]) => k === m) || [m, m])[1];
  const pipes = ALL_PIPES.filter((p) => rows.some((r) => r.Pipeline === p));
  const embs = [...new Set(rows.map((r) => r.Embeddings))].sort();
  const clfs = [...new Set(rows.map((r) => r.Classifier))].sort();
  const val = {};
  for (const r of rows) val[`${r.Pipeline}|${r.Embeddings}|${r.Classifier}`] = r[m];
  const alpha = (v) => (0.06 + 0.84 * Math.max(0, Math.min(1, (v - 0.5) / 0.5))).toFixed(3);

  const grids = pipes.map((p) => {
    const cells = clfs.map((k) => `<tr><th class="heat-clf">${esc(CLF_SHORT[k] || k)}</th>${
      embs.map((e) => {
        const key = `${p}|${e}|${k}`;
        const v = val[key];
        if (typeof v !== "number") return `<td class="heat-none">—</td>`;
        const pick = picks.has(key);
        const tip = `${pLabel(p)}\n${e} · ${k} · ${MODE_SHORT[RF.mode]}\n${m} ${v.toFixed(4)}`
          + (v < 0.5 ? "\nBelow chance" : "") + (pick ? "\nShown in Master results" : "");
        return `<td class="${pick ? "pick" : ""}${v < 0.5 ? " below" : ""}" style="background: rgba(57, 135, 229, ${alpha(v)})"
          data-tip="${esc(tip)}">${v.toFixed(3)}</td>`;
      }).join("")}</tr>`).join("");
    return `<div class="heat">
      <div class="heat-h" data-tip="${esc(pLabel(p))}">${esc(pShort(p))}
        <span>${esc(pLabel(p).replace(/^P\d+\s*·\s*/, ""))}</span></div>
      <table class="heat-t"><tr><th></th>${embs.map((e) => `<th>${esc(e)}</th>`).join("")}</tr>${cells}</table>
    </div>`;
  }).join("");

  $("#rf-body").innerHTML = pipes.length
    ? `<div class="heats">${grids}</div>
      <div class="key"><span>${esc(mLabel)}, ${esc(MODE_SHORT[RF.mode])} runs · shade runs from 0.50 (chance, faint) to 1.00 (full)</span>
        <span><i class="key-pick"></i>outlined = the classifier Master results shows</span>
        <span><b class="key-bad">red</b> = below chance</span></div>`
    : `<span class="muted">Nothing to show for this selection.</span>`;
}

/* ---------- resources ----------
   One row per pipeline × embedding × classifier, normal mode. Measured by the runs
   themselves (training run + embedding step) and by Run → Measure resources; per row
   the server picks the newer. The embedding columns are shared by every classifier on
   that embedding, so they span the group's rows. */
let RES = null;
const RS = { rows: "best" };

const RS_TIP = {
  vram: "Most GPU memory PyTorch held while embedding (the whole dataset in a run, 1,000 prompts in "
    + "Measure resources), in MB.\nOpenAI has none: it runs on OpenAI's servers. Blank when neither has "
    + "measured it on this GPU (embeddings made before this was recorded, on the CPU, or on Vast.ai).",
  gpu: "How busy the GPU was while embedding: average / highest, in %.\nMeasured for the whole card, so anything else on the GPU counts too.",
  embdisk: "Disk space of the stored embeddings for the whole dataset, in MB.",
  fit: "Time to train the classifier, in seconds.",
  cpu: "CPU time spent training, in seconds.\nHigher than the training time when several cores work at once.",
  ram: "Most extra RAM training took, in MB, on top of what was already loaded (Python and the embeddings),\n"
    + "helper processes included. Blank for runs trained before this was recorded, until re-measured.",
  p50: "Median time for the classifier to score one prompt, in milliseconds.",
  p99: "99th-percentile time to score one prompt: the slow tail, in milliseconds.",
  model: "Disk space of the saved classifier, in MB.",
};

async function loadResources() {
  try { RES = await fetchJSON("/api/resources"); } catch { RES = null; }
  renderResources();
}

$("#rs-pick").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-rows]");
  if (!b) return;
  RS.rows = b.dataset.rows;
  renderResources();
});

function renderResources() {
  if (!RES) return;
  const chip = (val, label, tip) => `<button class="pipe-chip ${RS.rows === val ? "sel" : ""}" data-rows="${val}"
    aria-pressed="${RS.rows === val}" data-tip="${esc(tip)}">${label}</button>`;
  $("#rs-rows").innerHTML = chip("best", "best per pipeline × embedding",
    "The classifier Master results shows for each pipeline × embedding (best normal ROC-AUC).")
    + chip("all", "every classifier", "All five classifiers for each pipeline × embedding.");

  const picks = masterPicks();
  let rows = RES.rows.filter((r) => PIPE_FOCUS.has(r.Pipeline));
  if (RS.rows === "best") rows = rows.filter((r) => picks.has(`${r.Pipeline}|${r.Embeddings}|${r.Classifier}`));
  const n = (x, d) => (typeof x === "number" ? x.toFixed(d) : "—");
  const th = (label, tip, cls = "num") => `<th class="${cls}" data-tip="${esc(tip)}">${label}</th>`;

  // group consecutive rows of one pipeline × embedding: they share the embedding cells
  const groups = [];
  for (const r of rows) {
    const g = groups.at(-1);
    if (g && g.key === `${r.Pipeline}|${r.Embeddings}`) g.rows.push(r);
    else groups.push({ key: `${r.Pipeline}|${r.Embeddings}`, rows: [r] });
  }
  let body = "", lastPipe = null;
  for (const g of groups) {
    g.rows.forEach((r, i) => {
      const cls = [i === 0 && lastPipe && lastPipe !== r.Pipeline ? "pipe-sep" : "",
                   i === 0 && g.rows.length > 1 ? "grp-top" : "",
                   RS.rows === "all" && picks.has(`${r.Pipeline}|${r.Embeddings}|${r.Classifier}`) ? "best" : ""]
        .filter(Boolean).join(" ");
      const span = g.rows.length > 1 ? ` rowspan="${g.rows.length}"` : "";
      let emb = "";
      if (i === 0) {
        const gpu = !r.emb_local ? `<td class="num muted"${span} colspan="2" data-tip="${esc(RS_TIP.vram)}">n/a — OpenAI's servers</td>`
          : `<td class="num"${span}>${n(r.vram_mb, 0)}</td><td class="num"${span}>${
              typeof r.gpu_mean === "number" ? `${r.gpu_mean.toFixed(0)} / ${n(r.gpu_max, 0)}` : "—"}</td>`;
        emb = `<td${span} data-tip="${esc(pLabel(r.Pipeline))}">${esc(pShort(r.Pipeline))}</td><td${span}>${esc(r.Embeddings)}</td>
          ${gpu}<td class="num"${span} data-tip="${esc(r.emb_rows ? `${r.emb_rows.toLocaleString()} rows × ${r.emb_dims.toLocaleString()} dimensions` : "")}">${n(r.emb_disk_mb, 1)}</td>`;
      }
      body += `<tr${cls ? ` class="${cls}"` : ""}>${emb}
        <td class="rs-clf">${esc(CLF_SHORT[r.Classifier] || r.Classifier)}</td>
        <td class="num">${n(r.fit_s, 1)}</td><td class="num">${n(r.cpu_s, 1)}</td><td class="num">${n(r.ram_mb, 0)}</td>
        <td class="num">${n(r.p50_ms, 2)}</td><td class="num">${n(r.p99_ms, 2)}</td>
        <td class="num">${n(r.model_mb, 2)}</td></tr>`;
    });
    lastPipe = g.rows[0].Pipeline;
  }

  $("#rs-table").innerHTML = rows.length ? `<tr class="grp"><th colspan="2"></th>
      <th colspan="3" data-tip="Turning the text into vectors, on the GPU.">Embedding</th>
      <th colspan="1"></th>
      <th colspan="3" data-tip="Training the classifier once, on the training rows.">Classifier training</th>
      <th colspan="2" data-tip="Scoring one prompt with the trained classifier.">Classifier inference</th>
      <th colspan="1" data-tip="Saved classifier file.">Disk</th></tr>
    <tr>${th("Pipeline", COL_TIP.Pipeline, "")}${th("Emb", COL_TIP.Emb, "")}
      ${th("VRAM MB", RS_TIP.vram)}${th("GPU %", RS_TIP.gpu)}${th("disk MB", RS_TIP.embdisk)}
      ${th("Classifier", `${CLF_ABBR}.`, "")}
      ${th("fit s", RS_TIP.fit)}${th("CPU s", RS_TIP.cpu)}${th("fit RAM MB", RS_TIP.ram)}
      ${th("p50 ms", RS_TIP.p50)}${th("p99 ms", RS_TIP.p99)}${th("model MB", RS_TIP.model)}</tr>${body}`
    : `<tr><td class="muted">Nothing to show for this selection.</td></tr>`;

  const m = RES.machine;
  const measured = RES.n_embedding || RES.n_classifier;
  $("#rs-key").innerHTML = measured
    ? `<span>Measured on ${esc([m?.gpu, m?.cpu, m?.ram_gb ? `${m.ram_gb} GB RAM` : ""].filter(Boolean).join(" · "))}${
        RES.measured_at ? `, ${esc(RES.measured_at.slice(0, 10))}` : ""}</span>
       <span>normal mode · classifiers run on the CPU only</span>
       <span>${RES.n_benchmark ? `${RES.n_benchmark} of ${RES.rows.length} rows from Measure resources, the rest from the runs`
         : "from the runs"} · the newer measurement is shown</span>
       ${RS.rows === "all" ? `<span><i class="key-edge"></i>blue edge = the classifier Master results shows</span>` : ""}`
    : `<span>Not measured yet: only the disk sizes are filled in. Re-run from the Run tab, or use
       Run → Measure resources.</span>`;
}

/* ---------- collapsible cards ----------
   Every Analysis card except Master results folds from its title bar. All runs starts
   folded; whatever the viewer chooses is remembered in this browser. */
const FOLD_DEFAULT = { "ablation-card": false, "allrows-card": true, "resources-card": false, "figures-card": false };
let FOLD = {};
try { FOLD = JSON.parse(localStorage.getItem("cardFold")) || {}; } catch { FOLD = {}; }

function setCollapsed(card, folded) {
  if (!card || !(card.id in FOLD_DEFAULT)) return;
  card.classList.toggle("folded", folded);
  const h = card.querySelector(":scope > h2");
  h.setAttribute("aria-expanded", String(!folded));
  h.querySelector(".fold-pill").textContent = folded ? "Show ▾" : "Hide ▴";
  FOLD[card.id] = folded;
  try { localStorage.setItem("cardFold", JSON.stringify(FOLD)); } catch {}
  updateSecNav();
}

for (const id of Object.keys(FOLD_DEFAULT)) {
  const card = document.getElementById(id);
  const h = card.querySelector(":scope > h2");
  h.classList.add("fold-bar");
  h.setAttribute("role", "button");
  h.tabIndex = 0;
  h.insertAdjacentHTML("beforeend", `<span class="fold-pill"></span>`);
  h.addEventListener("click", () => setCollapsed(card, !card.classList.contains("folded")));
  h.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); h.click(); }
  });
  setCollapsed(card, FOLD[id] ?? FOLD_DEFAULT[id]);
}

/* -------- figures --------
   Two button rows pick the run (pipeline × mode). The pipeline buttons only
   offer pipelines selected in the Focus filter, so Figures follows it like
   every other card. The run's figures lay out as the summary on top, then one
   row per embedding with the same figure type in each column. */
let FIG_SEL = { pipeline: null, mode: "normal" };

// column order, label, and the PNGs' aspect ratio (width / height): each column
// is as wide as its figure's shape, so the three images in a row share a height
const FIG_TYPES = [
  ["confusion_matrices", "Confusion matrices", 4.78,
    "One grid per classifier: how many test rows it got right and wrong at the 0.5 threshold.\n"
    + "Rows = actual label, columns = predicted. Darker = a larger share of that row."],
  ["dimensionality_reduction", "Dimensionality reduction", 2.98,
    "The embeddings squashed to 2-D so they can be seen: PCA, t-SNE and UMAP, on up to 5,000 rows.\n"
    + "Coloured by label. Colours that sit apart = the two classes are easy to tell apart."],
  ["roc_pr_curves", "ROC / PR curves", 2.31,
    "Left: ROC curve, one line per classifier (dashed = chance). Right: precision–recall curve.\n"
    + "Curves hugging the top-left (ROC) or top-right (PR) are better; the legend gives each AUC."],
];
const FIG_TYPE_RE = new RegExp(`^(.+?)_(${FIG_TYPES.map((t) => t[0]).join("|")})\\.png$`);

const figRuns = () => (STATE ? STATE.runs : []).filter((r) => r.figures.length && PIPE_FOCUS.has(r.pipeline));

function renderFigPicker() {
  const runs = figRuns();
  const pipes = ALL_PIPES.filter((p) => runs.some((r) => r.pipeline === p));
  if (!pipes.includes(FIG_SEL.pipeline)) FIG_SEL.pipeline = pipes[0] || null;
  const modes = STATE ? STATE.modes : ["normal", "user_intent_only", "context_only"];
  const has = (m) => runs.some((r) => r.pipeline === FIG_SEL.pipeline && r.mode === m);
  if (!has(FIG_SEL.mode)) FIG_SEL.mode = modes.find(has) || FIG_SEL.mode;

  $("#fig-pipes").innerHTML = pipes.map((p) =>
    `<button class="pipe-chip ${p === FIG_SEL.pipeline ? "sel" : ""}" data-pipe="${p}" data-tip="${esc(pLabel(p))}">${pShort(p)}</button>`).join("");
  $("#fig-modes").innerHTML = modes.map((m) =>
    `<button class="pipe-chip ${m === FIG_SEL.mode ? "sel" : ""}" data-mode="${m}" data-tip="${esc(MODE_DESC[m])}"${
      has(m) ? "" : " disabled"}>${MODE_SHORT[m]}</button>`).join("");
  renderFigures();
}

$("#fig-pipes").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-pipe]");
  if (b) { FIG_SEL.pipeline = b.dataset.pipe; renderFigPicker(); }
});
$("#fig-modes").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-mode]");
  if (b && !b.disabled) { FIG_SEL.mode = b.dataset.mode; renderFigPicker(); }
});

function renderFigures() {
  const el = $("#figures-list");
  const { pipeline, mode } = FIG_SEL;
  const run = figRuns().find((r) => r.pipeline === pipeline && r.mode === mode);
  if (!run) { el.innerHTML = `<span class="muted">No run has figures yet.</span>`; return; }

  const url = (f) => `/api/figure?pipeline=${pipeline}&mode=${mode}&name=${encodeURIComponent(f)}`;
  let n = 0;   // viewer order = reading order
  const fig = (f, style = "") => `<figure${style}><a href="${url(f)}" target="_blank" data-i="${n++}" data-name="${esc(f)}">
      <img loading="lazy" src="${url(f)}" alt="${esc(f)}"></a></figure>`;

  const files = new Set(run.figures);
  const used = new Set();
  const embs = [...new Set(run.figures.map((f) => (f.match(FIG_TYPE_RE) || [])[1]).filter(Boolean))].sort();

  let html = "";
  if (files.has("performance_summary.png")) {
    used.add("performance_summary.png");
    html += `<div class="fig-summary">${fig("performance_summary.png")}</div>`;
  }
  if (embs.length) {
    const flex = (a) => ` style="flex: ${a} 1 0"`;
    html += `<div class="fig-row fig-head"><div class="fig-emb"></div><div class="fig-cells">
      ${FIG_TYPES.map(([, label, a, tip]) => `<div${flex(a)}><span data-tip="${esc(tip)}">${label}</span></div>`).join("")}</div></div>`;
    for (const emb of embs) {
      const cells = FIG_TYPES.map(([type, , a]) => {
        const f = `${emb}_${type}.png`;
        if (!files.has(f)) return `<div class="fig-missing" style="flex: ${a} 1 0; aspect-ratio: ${a}">not generated</div>`;
        used.add(f);
        return fig(f, flex(a));
      }).join("");
      html += `<div class="fig-row fig-emb-row"><div class="fig-emb">${esc(emb.toUpperCase())}</div>
        <div class="fig-cells">${cells}</div></div>`;
    }
  }
  // anything that fits neither slot still shows, rather than silently vanishing
  const rest = run.figures.filter((f) => !used.has(f));
  if (rest.length) {
    html += `<div class="fig-rest">${rest.map((f) => `<div>${fig(f)}<div class="fig-cap">${esc(f)}</div></div>`).join("")}</div>`;
  }
  el.innerHTML = html;
}

/* -------- figure viewer --------
   A plain click opens the figure in a dialog over the page; ctrl/shift/middle
   click keeps the link's normal open-in-new-tab behaviour. ← → walk the run. */
let FIG_LIST = [], FIG_I = 0;

$("#figures-list").addEventListener("click", (e) => {
  const a = e.target.closest("a[data-i]");
  if (!a || e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey) return;
  e.preventDefault();
  FIG_LIST = $$("#figures-list a[data-i]").map((x) => ({ href: x.href, name: x.dataset.name }));
  showFig(+a.dataset.i);
});

function showFig(i) {
  const n = FIG_LIST.length;
  if (!n) return;
  FIG_I = (i + n) % n;
  const f = FIG_LIST[FIG_I];
  $("#fig-img").src = f.href;
  $("#fig-img").alt = f.name;
  $("#fig-title").textContent = `${pShort(FIG_SEL.pipeline)} / ${MODE_SHORT[FIG_SEL.mode]}  ·  ${f.name}`;
  $("#fig-count").textContent = `${FIG_I + 1} / ${n}`;
  $("#fig-open").href = f.href;
  $("#fig-prev").hidden = $("#fig-next").hidden = n < 2;
  const dlg = $("#fig-dlg");
  if (!dlg.open) dlg.showModal();
}

$("#fig-prev").addEventListener("click", () => showFig(FIG_I - 1));
$("#fig-next").addEventListener("click", () => showFig(FIG_I + 1));
$("#fig-close").addEventListener("click", () => $("#fig-dlg").close());
$("#fig-dlg").addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") { e.preventDefault(); showFig(FIG_I - 1); }
  if (e.key === "ArrowRight") { e.preventDefault(); showFig(FIG_I + 1); }
});
/* the backdrop and the empty stage around the image both close it */
$("#fig-dlg").addEventListener("click", (e) => {
  if (e.target === $("#fig-dlg") || e.target.classList.contains("fig-stage")) $("#fig-dlg").close();
});

/* ================= detect tab ================= */
/* A saved model is keyed by pipeline × mode × emb × clf, but this tab exposes
   only two dropdowns. Pipeline/mode are pinned to the headline run — falling
   back to whatever this tree actually has, so a P1-only tree still works — and
   the line under the button names the run that answered. */
const DETECT_PIN = { pipeline: "pipeline3_organic_injected", mode: "normal" };
let DETECT_RUN = null;

const runModels = () => DETECT_OPTIONS.filter(
  (m) => m.pipeline === DETECT_RUN.pipeline && m.mode === DETECT_RUN.mode);

function setBtn(cls, text) {
  const b = $("#b-detect");
  b.className = "detect-btn" + (cls ? " " + cls : "");
  b.textContent = text;
}

function fillClf() {
  const emb = $("#d-emb").value;
  $("#d-clf").innerHTML = runModels().filter((m) => m.emb === emb)
    .map((m) => `<option value="${esc(m.clf)}">${esc(m.clf)}</option>`).join("");
}

async function loadDetectOptions() {
  let opts = {};
  try { opts = await fetchJSON("/api/detect/options"); } catch { opts = {}; }
  DETECT_OPTIONS = opts.models || [];
  const msg = $("#detect-msg");
  // this is the landing tab, so the empty state stays one short line; the
  // Instructions card below already spells out how to produce models
  if (!DETECT_OPTIONS.length) {
    msg.textContent = "No trained classifiers yet — run one from the Run tab.";
    msg.className = "d-note msg warn";
    $("#b-detect").disabled = true;
    $("#b-sample").disabled = true;
    $("#d-emb").hidden = true;
    $("#d-clf").hidden = true;
    return;
  }
  $("#b-detect").disabled = false;
  $("#b-sample").disabled = false;
  $("#d-emb").hidden = false;
  $("#d-clf").hidden = false;
  msg.textContent = "";
  DETECT_RUN = DETECT_OPTIONS.find(
    (m) => m.pipeline === DETECT_PIN.pipeline && m.mode === DETECT_PIN.mode) || DETECT_OPTIONS[0];
  $("#d-emb").innerHTML = [...new Set(runModels().map((m) => m.emb))]
    .map((e) => `<option value="${esc(e)}">${esc(e)}</option>`).join("");
  fillClf();
}
$("#d-emb").addEventListener("change", fillClf);

$("#b-upload").addEventListener("click", () => $("#d-file").click());
$("#d-file").addEventListener("change", async (event) => {
  const input = event.target, file = input.files[0];
  if (!file) return;
  const status = $("#d-file-status");
  const controls = [input, $("#b-upload"), $("#d-context"), $("#b-detect"), $("#b-sample")];
  const disabled = controls.map((control) => control.disabled);
  controls.forEach((control) => { control.disabled = true; });
  status.className = "msg";
  status.textContent = "Reading file…";
  try {
    if (!/\.(txt|html|json|jsonl|csv|tsv|py|js)$/i.test(file.name)) {
      throw new Error("Choose a .txt, .html, .json, .jsonl, .csv, .tsv, .py, or .js text file.");
    }
    if (file.size > 1024 * 1024) throw new Error("Choose a text file no larger than 1 MB.");
    let text;
    try {
      text = new TextDecoder("utf-8", { fatal: true }).decode(await file.arrayBuffer());
    } catch {
      throw new Error("Could not read this file as UTF-8 text. Save it as UTF-8 and try again.");
    }
    if (/[\x00-\x08\x0e-\x1f]/.test(text)) throw new Error("This file contains binary data. Choose a plain text file.");
    if (!text.trim()) throw new Error("This text file is empty.");
    $("#d-context").value = text;
    $("#d-context").dispatchEvent(new Event("input"));
    $("#detect-msg").textContent = "";
    status.textContent = `Loaded ${file.name}. Review the context above, then click Detect.`;
  } catch (error) {
    status.className = "msg error";
    status.textContent = error.message;
  } finally {
    input.value = "";
    controls.forEach((control, i) => { control.disabled = disabled[i]; });
  }
});

$("#b-sample").addEventListener("click", async () => {
  const msg = $("#detect-msg");
  msg.textContent = "sampling…"; msg.className = "d-note msg";
  try {
    const data = await fetchJSON(`/api/sample?pipeline=${DETECT_RUN.pipeline}`);
    const row = data.rows[0];
    $("#d-intent").value = row.user_intent || "";
    $("#d-context").value = row.context || "";
    $("#d-context").dataset.truth = row.label;
    $("#d-file-status").textContent = "";
    msg.textContent = "loaded a labeled row — ground truth stays hidden until you detect";
  } catch (e) { msg.textContent = e.message; msg.className = "d-note msg warn"; }
});

$("#b-detect").addEventListener("click", async () => {
  const msg = $("#detect-msg"), detail = $("#detect-detail");
  const clf = $("#d-clf").value;
  if (!clf) { msg.textContent = "no classifier selected"; msg.className = "d-note msg error"; return; }
  msg.textContent = ""; msg.className = "d-note msg"; detail.textContent = "";
  setBtn("loading", "Loading…");
  $("#b-detect").disabled = true;
  try {
    const res = await fetchJSON("/api/detect", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pipeline: DETECT_RUN.pipeline, mode: DETECT_RUN.mode, emb: $("#d-emb").value, clf,
        user_intent: $("#d-intent").value, context: $("#d-context").value,
      }),
    });
    setBtn(res.verdict, res.verdict === "malicious" ? "Detected MALICIOUS INTENT" : "Prompt is SAFE");
    const truth = $("#d-context").dataset.truth;
    detail.textContent =
      `${(res.probability * 100).toFixed(1)}% confidence · embed ${res.embed_ms} ms · `
      + `classify ${res.classify_ms} ms · ${res.emb} + ${res.clf} trained on `
      + `${pShort(res.pipeline)}/${MODE_SHORT[res.mode]}`
      + (truth ? ` · dataset ground truth: ${truth === "1" ? "malicious" : "benign"}` : "");
  } catch (e) {
    setBtn("", "Detect");
    msg.textContent = e.message; msg.className = "d-note msg error";
  } finally { $("#b-detect").disabled = false; }
});

// editing either field invalidates the verdict sitting on the button
["#d-intent", "#d-context"].forEach((id) => $(id).addEventListener("input", () => {
  $("#d-file-status").textContent = "";
  delete $("#d-context").dataset.truth;
  $("#detect-detail").textContent = "";
  setBtn("", "Detect");
}));

/* ================= files tab ================= */
/* Artifacts grouped by kind then pipeline (see runs.file_tree). Missing files
   are hidden by default — the count in the summary already says how many — and
   a per-group toggle brings them back when you need the expected filenames. */
const openSections = new Set();  // node keys the user expanded
const showMissing = new Set();   // group names whose missing rows are revealed

const EYE = `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor"
  stroke-width="2" stroke-linecap="round" aria-hidden="true"><path
  d="M1.5 12S5 5.5 12 5.5 22.5 12 22.5 12 19 18.5 12 18.5 1.5 12 1.5 12Z"/><circle cx="12" cy="12" r="3"/></svg>`;

const fmtSize = (b) => (b == null ? ""
  : b < 1e3 ? `${b} B` : b < 1e6 ? `${(b / 1e3).toFixed(0)} KB` : `${(b / 1e6).toFixed(1)} MB`);
const nodeCount = (n) => (n.n_expected ? `${n.n_present}/${n.n_expected}` : `${n.n_files} files`);
const nodeCls = (n) => (n.n_expected
  ? (n.n_present === n.n_expected ? "on" : n.n_present ? "part" : "")
  : (n.n_files ? "on" : ""));

function fileRow(e) {
  const name = e.mode ? `${MODE_SHORT[e.mode] || e.mode} / ${e.label}` : e.label;
  const when = e.mtime
    ? new Date(e.mtime * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" }) : "";
  const extra = e.kind === "unexpected" ? ` <span class="badge part">extra</span>` : "";
  const click = e.preview ? ` data-view="${esc(e.path)}"` : "";
  const eye = e.preview
    ? `<button class="eye" tabindex="-1" aria-label="Preview ${esc(e.label)}">${EYE}</button>` : "";
  return `<tr class="file-${e.status}"${click} title="${esc(e.path)}">
    <td class="dot-cell"><i class="dot ${e.status === "present" ? "on" : ""}"
      aria-label="${e.status}"></i></td>
    <td class="mono">${esc(name)}${extra}</td>
    <td class="num">${fmtSize(e.size)}</td>
    <td class="muted">${when}</td>
    <td class="eye-cell">${eye}</td></tr>`;
}

function fileTable(entries, show) {
  const rows = entries.filter((e) => show || e.status === "present");
  if (!rows.length) return `<p class="empty muted">nothing here yet</p>`;
  return `<div class="scroll"><table class="plain files">${rows.map(fileRow).join("")}</table></div>`;
}

function fileNode(node, key, show, cls) {
  const body = node.children.length
    ? node.children.map((c) => fileNode(c, `${key}/${c.name}`, show, "file-sub")).join("")
    : fileTable(node.entries, show);
  const nMissing = node.n_expected - node.n_present;
  const toggle = cls === "file-section" && nMissing
    ? `<button class="ghost" data-missing="${esc(node.name)}">${show ? "hide" : "show"} ${nMissing} missing</button>`
    : "";
  return `<details class="${cls}" data-key="${esc(key)}" ${openSections.has(key) ? "open" : ""}>
    <summary>${esc(node.name)}
      <span class="badge ${nodeCls(node)}">${nodeCount(node)}</span>${toggle}</summary>
    <div class="file-body">${body}</div></details>`;
}

let FILES_SIG = "";   // last payload rendered; the tab re-polls every 5s
let FILES_DATA = null;

/* Re-render only when the tree actually changed. Replacing innerHTML on a timer
   drops hover state and can detach a row mid-click. */
async function loadFiles() {
  let data;
  try { data = await fetchJSON("/api/files"); }
  catch (e) { $("#files-list").innerHTML = `<span class="msg error">${esc(e.message)}</span>`; return; }
  const sig = JSON.stringify(data);
  if (sig === FILES_SIG) return;
  FILES_SIG = sig;
  FILES_DATA = data;
  renderFiles();
}

function renderFiles() {
  $("#files-list").innerHTML = FILES_DATA.groups
    .map((g) => fileNode(g, g.name, showMissing.has(g.name), "file-section")).join("");

  $$("#files-list details").forEach((d) => d.addEventListener("toggle", () => {
    if (d.open) openSections.add(d.dataset.key); else openSections.delete(d.dataset.key);
  }));
  /* the toggle lives inside <summary>, whose default action is open/close */
  $$("#files-list [data-missing]").forEach((b) => b.addEventListener("click", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    const g = b.dataset.missing;
    if (showMissing.has(g)) showMissing.delete(g); else showMissing.add(g);
    renderFiles();
  }));
  $$("#files-list tr[data-view]").forEach((r) =>
    r.addEventListener("click", () => previewFile(r.dataset.view)));
}

/* metrics want 4 decimals; counts like Test-Samples must not become 14000.0000 */
const cell = (v) => (typeof v !== "number" ? v ?? ""
  : Number.isInteger(v) ? v : fmt(v, 4));

function previewBody(info) {
  if (info.type === "image") {
    return `<img class="preview-img" src="/api/file/raw?path=${encodeURIComponent(PEEK_PATH)}">`;
  }
  if (info.type === "table") {
    const head = `<tr>${info.cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>`;
    const body = info.rows.map((r) =>
      `<tr>${r.map((v) => `<td>${esc(cell(v))}</td>`).join("")}</tr>`).join("");
    return `<div class="scroll"><table class="plain">${head}${body}</table></div>
      <p class="note muted">${esc(info.note || "")}</p>`;
  }
  return `<pre class="log">${esc(info.text)}</pre>`;
}

let PEEK_PATH = null;

async function previewFile(path) {
  const dlg = $("#peek-dlg");
  PEEK_PATH = path;
  $("#peek-path").textContent = path;
  $("#peek-body").innerHTML = `<span class="muted">loading…</span>`;
  if (!dlg.open) dlg.showModal();
  try {
    const info = await fetchJSON(`/api/file/inspect?path=${encodeURIComponent(path)}`);
    if (PEEK_PATH !== path) return;  // a newer peek won the race
    $("#peek-body").innerHTML = previewBody(info);
  } catch (e) {
    $("#peek-body").innerHTML = `<span class="msg error">${esc(e.message)}</span>`;
  }
}

$("#peek-close").addEventListener("click", () => $("#peek-dlg").close());
/* clicking the backdrop lands on the dialog itself, never on its children */
$("#peek-dlg").addEventListener("click", (ev) => {
  if (ev.target === $("#peek-dlg")) $("#peek-dlg").close();
});

/* ================= dataset peek (files tab) ================= */
function fillPeekSelect() {
  const el = $("#pk-pipeline");
  if (!STATE || el.dataset.filled) return;
  el.innerHTML = Object.keys(STATE.pipelines)
    .map((p) => `<option value="${p}">${esc(pLabel(p))}</option>`).join("");
  el.dataset.filled = "1";
}

async function loadPeek() {
  fillPeekSelect();
  const msg = $("#pk-msg"), table = $("#peek-table");
  const pipeline = $("#pk-pipeline").value;
  if (!pipeline) return;
  msg.textContent = "sampling…";
  msg.className = "msg";
  try {
    const d = await fetchJSON(`/api/sample?pipeline=${pipeline}&n=${$("#pk-n").value}`);
    msg.textContent = `${d.rows.length} random rows`;
    const hasAttack = d.rows.some((r) => r.attack_category != null);
    const trunc = (t, n) => { t = String(t ?? ""); return t.length > n ? t.slice(0, n) + "…" : t; };
    const rows = d.rows.map((r) => `<tr>
      <td>${r.label ? `<span class="badge part">malicious</span>` : `<span class="badge on">benign</span>`}</td>
      <td class="wrapcell mono" style="max-width:220px">${esc(trunc(r.user_intent, 110))}</td>
      <td class="wrapcell" style="max-width:520px"><details><summary class="mono" style="cursor:pointer">${esc(trunc(r.context, 170))}</summary><pre class="log" style="max-height:220px">${esc(r.context)}</pre></details></td>
      ${hasAttack ? `<td class="mono">${esc(r.attack_category ?? "—")}</td><td class="mono">${esc(r.injection_position ?? "—")}</td><td class="mono">${esc(r.category ?? r.source ?? "—")}</td>` : `<td class="mono">${esc(r.source ?? "—")}</td>`}
    </tr>`).join("");
    table.innerHTML = `<tr><th>Label</th><th>User intent</th><th>Context (click to expand)</th>
      ${hasAttack ? "<th>Attack cat.</th><th>Position</th><th>Source</th>" : "<th>Source</th>"}</tr>${rows}`;
  } catch (e) {
    table.innerHTML = "";
    msg.textContent = e.message;
    msg.className = "msg";
  }
}
$("#pk-resample").addEventListener("click", loadPeek);
$("#pk-pipeline").addEventListener("change", loadPeek);
$("#pk-n").addEventListener("change", loadPeek);

/* ================= settings ================= */
const SET_INPUT = {
  OPENAI_API_KEY: "#s-openai-key", OPENAI_TPM: "#s-openai-tpm",
  VAST_SSH: "#s-vast-ssh", VAST_SSH_KEY: "#s-vast-key",
  VAST_API_KEY: "#s-vast-api", VAST_INSTANCE_ID: "#s-vast-id",
};
const SET_SECRET = ["OPENAI_API_KEY", "VAST_API_KEY"];

function fillSettings(cur) {
  for (const [name, sel] of Object.entries(SET_INPUT)) {
    const el = $(sel);
    if (!SET_SECRET.includes(name)) { el.value = cur[name].value; continue; }
    // a secret never comes back: show that one is saved, and its last characters
    el.value = "";
    el.placeholder = cur[name].set ? `saved (${cur[name].value}) · type to replace` : "not set";
    $(`[data-clear="${name}"]`).hidden = !cur[name].set;
  }
  // on by default until an API key exists, so adding the key is all it takes
  $("#s-vast-autostop").checked = cur.VAST_AUTO_STOP.value === "1" || !cur.VAST_API_KEY.set;
}

function setMsg(sel, text, cls = "") {
  $(sel).textContent = text;
  $(sel).className = "msg " + cls;
}

async function saveSettings(extra = {}) {
  const body = { VAST_AUTO_STOP: $("#s-vast-autostop").checked ? "1" : "", ...extra };
  for (const [name, sel] of Object.entries(SET_INPUT)) {
    const v = $(sel).value.trim();
    if (SET_SECRET.includes(name) && !v) continue;   // blank = leave the saved one alone
    body[name] = v;
  }
  try {
    fillSettings(await fetchJSON("/api/settings", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    }));
    setMsg("#settings-msg", "Saved. The next job uses these.", "ok");
    return true;
  } catch (e) {
    setMsg("#settings-msg", String(e.message || e), "error");
    return false;
  }
}

$("#b-settings").addEventListener("click", async () => {
  for (const m of ["#settings-msg", "#t-openai", "#t-vast"]) setMsg(m, "");
  $("#settings-dlg").showModal();
  try { fillSettings(await fetchJSON("/api/settings")); }
  catch (e) { setMsg("#settings-msg", String(e.message || e), "error"); }
});
$("#settings-close").addEventListener("click", () => $("#settings-dlg").close());
$("#settings-form").addEventListener("submit", (e) => { e.preventDefault(); saveSettings(); });
$$("[data-clear]").forEach((b) => b.addEventListener("click", () => saveSettings({ [b.dataset.clear]: "" })));
// the test runs against what is saved, so save first: testing a typed-but-unsaved
// key would pass and then the job would use the old one
$$("[data-test]").forEach((b) => b.addEventListener("click", async () => {
  const out = `#t-${b.dataset.test}`;
  if (!(await saveSettings())) return setMsg(out, "");
  setMsg(out, "testing…");
  b.disabled = true;
  try {
    const r = await fetchJSON(`/api/settings/test/${b.dataset.test}`, { method: "POST" });
    setMsg(out, r.detail, r.ok ? "ok" : "error");
  } catch (e) { setMsg(out, String(e.message || e), "error"); }
  b.disabled = false;
}));

/* ================= boot & polling ================= */
/* Simulation is the landing tab, so its model list must load at boot: the
   tab-click handler that fills it never fires for the tab you arrive on. */
loadDetectOptions();
refreshState();
refreshJobs();
setInterval(() => {
  const runsActive = $("#tab-runs").classList.contains("active");
  refreshJobs();
  if (runsActive) pollLog();
}, 2000);
setInterval(refreshState, 5000);
setInterval(() => { if ($("#tab-files").classList.contains("active")) loadFiles(); }, 5000);
