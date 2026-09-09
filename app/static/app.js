/* thesis_clean dashboard frontend (no build step, no external deps) */
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const MODE_SHORT = { normal: "normal", user_intent_only: "intent-only", context_only: "context-only" };
/* pipeline focus for the headline/ablation cards (persisted; defaults to P3) */
const ALL_PIPES = ["pipeline1_alamsabi", "pipeline2_organic_bipia", "pipeline3_organic_injected"];
let PIPE_FOCUS;
try { PIPE_FOCUS = new Set(JSON.parse(localStorage.getItem("pipeFocus"))); } catch { /* fall through */ }
if (!PIPE_FOCUS || !PIPE_FOCUS.size) PIPE_FOCUS = new Set(["pipeline3_organic_injected"]);

function renderPipeChips() {
  const el = $("#pipe-chips");
  el.innerHTML = `<span class="lbl">Focus</span>` + ALL_PIPES.map((p) =>
    `<span class="pipe-chip ${PIPE_FOCUS.has(p) ? "sel" : ""}" data-pipe="${p}" title="${esc(pLabel(p))}">${pShort(p)}</span>`).join("");
  $$(".pipe-chip").forEach((c) => c.addEventListener("click", () => {
    const p = c.dataset.pipe;
    if (PIPE_FOCUS.has(p)) {
      if (PIPE_FOCUS.size === 1) return; // keep at least one selected
      PIPE_FOCUS.delete(p);
    } else {
      PIPE_FOCUS.add(p);
    }
    try { localStorage.setItem("pipeFocus", JSON.stringify([...PIPE_FOCUS])); } catch {}
    renderPipeChips();
    renderHeadline();
    renderAblation();
  }));
}
const focusedResults = () => RESULTS.filter((r) => PIPE_FOCUS.has(r.Pipeline));

/* in-cell micro bar: value scaled from 0.5 (empty) to 1.0 (full) */
const microbar = (v) => typeof v === "number"
  ? `<span class="microbar"><i style="width:${Math.max(0, Math.min(1, (v - 0.5) / 0.5)) * 100}%"></i></span>`
  : "";

let STATE = null;        // /api/state payload
let RESULTS = [];        // /api/results rows
let DETECT_OPTIONS = []; // /api/detect/options
let openLogJob = null;
let logOffset = 0;

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

/* ================= tabs ================= */
$$(".tab-btn").forEach((b) => b.addEventListener("click", () => {
  $$(".tab-btn").forEach((x) => x.classList.toggle("active", x === b));
  $$("section.tab").forEach((s) => s.classList.toggle("active", s.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "results") loadResults();
  if (b.dataset.tab === "detect") loadDetectOptions();
  if (b.dataset.tab === "files") { loadFiles(); loadPeek(); }
}));

/* ================= runs tab ================= */
async function refreshState() {
  try { STATE = await fetchJSON("/api/state"); } catch { return; }
  renderPaths();
  renderDatasets();
  renderCapabilities();
  renderMatrix();
  fillPipelineSelects();
}

function renderPaths() {
  const p = STATE.paths;
  if (!p) return;
  const mark = (ok) => (ok ? "on" : "");
  const write = p.split_write
    ? ` · <span class="badge">writes → ${esc(p.runs_write_dir)}</span>` : "";
  $("#paths-line").innerHTML =
    `reading <span class="badge ${mark(p.data_exists)}">${esc(p.data_dir)}</span>
     and <span class="badge ${mark(p.runs_exists)}">${esc(p.runs_dir)}</span>${write}`;
}

// delivered artifacts carry results + figures but no models or per-sample
// probabilities; say so once, plainly, instead of letting features look broken
function renderCapabilities() {
  const c = STATE.capabilities;
  const el = $("#matrix-note");
  if (!c) { el.style.display = "none"; return; }
  const gaps = [];
  if (!c.n_runs_with_predictions) gaps.push("per-sample probabilities (predictions.npz)");
  if (!c.n_runs_with_models) gaps.push("saved classifiers (models/)");
  if (!gaps.length) { el.style.display = "none"; return; }
  el.style.display = "";
  el.innerHTML = `This tree has results for ${c.n_runs_with_results}/9 runs but no ${gaps.join(" and ")}. `
    + `Tables and figures work; live detection and curve/breakdown APIs need a run redone here `
    + `(Queue run, “save models” on) — the delivered artifacts do not include them.`;
}

function renderDatasets() {
  const rows = Object.entries(STATE.datasets).map(([key, d]) => {
    const info = d.info
      ? `<span class="badge on">present</span> <span class="muted">${d.info.size_mb} MB</span>`
      : `<span class="badge">missing</span>`;
    return `<tr><td>${esc(d.label)}</td><td class="muted">${esc(d.file)}</td><td>${info}</td></tr>`;
  }).join("");
  $("#datasets-table").innerHTML = `<tr><th>Pipeline</th><th>File</th><th>Status</th></tr>${rows}`;
}

function renderMatrix() {
  const modes = STATE.modes;
  let html = `<tr><th>Pipeline</th>${modes.map((m) => `<th>${MODE_SHORT[m]}</th>`).join("")}</tr>`;
  for (const p of Object.keys(STATE.pipelines)) {
    html += `<tr><td>${esc(pLabel(p))}</td>`;
    for (const m of modes) {
      const run = STATE.runs.find((r) => r.pipeline === p && r.mode === m);
      const nEmb = Object.values(run.embeddings).filter(Boolean).length;
      const embCls = nEmb === 3 ? "on" : nEmb > 0 ? "part" : "";
      const cell = [
        `<span class="badge ${embCls}">E ${nEmb}/3</span>`,
        `<span class="badge ${run.has_results ? "on" : ""}">R</span>`,
        `<span class="badge ${run.figures.length ? "on" : ""}">F ${run.figures.length}</span>`,
        `<span class="badge ${run.models.length ? "on" : ""}">M ${run.models.length}</span>`,
      ].join("");
      html += `<td>${cell}</td>`;
    }
    html += "</tr>";
  }
  $("#matrix-table").innerHTML = html;
}

function fillPipelineSelects() {
  const opts = Object.keys(STATE.pipelines)
    .map((p) => `<option value="${p}">${esc(pLabel(p))}</option>`).join("");
  for (const id of ["#f-pipeline", "#d-sample-pipeline"]) {
    const el = $(id);
    if (el && !el.dataset.filled) { el.innerHTML = opts; el.dataset.filled = "1"; }
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

$("#b-ds-alamsabi").addEventListener("click", () => queueJob("dataset", { which: "alamsabi" }));
$("#b-ds-organic").addEventListener("click", () => queueJob("dataset", { which: "organic" }));

$("#b-run").addEventListener("click", () => {
  const models = $$(".f-model").filter((c) => c.checked).map((c) => c.value).join(",");
  if (!models) return launchMsg("select at least one embedding model", "error");
  const params = {
    pipeline: $("#f-pipeline").value,
    mode: $("#f-mode").value,
    models,
    force: $("#f-force").checked,
    save_models: $("#f-save-models").checked,
    skip_projections: $("#f-skip-proj").checked,
  };
  const limit = parseInt($("#f-limit").value, 10);
  if (limit > 0) params.limit = limit;
  queueJob("run", params);
});

$("#b-sweep").addEventListener("click", () => {
  if (!confirm("Queue the FULL sweep: 3 pipelines × 3 modes, all selected embedding models. This can take many hours. Continue?")) return;
  const models = $$(".f-model").filter((c) => c.checked).map((c) => c.value).join(",");
  const params = {
    force: $("#f-force").checked,
    save_models: $("#f-save-models").checked,
    skip_projections: $("#f-skip-proj").checked,
  };
  if (models) params.models = models;
  const limit = parseInt($("#f-limit").value, 10);
  if (limit > 0) params.limit = limit;
  queueJob("sweep", params);
});

/* -------- jobs table + log -------- */
async function refreshJobs() {
  let data;
  try { data = await fetchJSON("/api/jobs"); } catch { return; }
  const jobs = data.jobs;
  const busy = jobs.some((j) => j.status === "running" || j.status === "queued");
  const pill = $("#queue-pill");
  pill.textContent = busy ? `${jobs.filter((j) => j.status === "running").length} running / ${jobs.filter((j) => j.status === "queued").length} queued` : "idle";
  pill.classList.toggle("busy", busy);

  if (!jobs.length) {
    $("#jobs-table").innerHTML = `<tr><td class="muted">No jobs yet — queue one above.</td></tr>`;
    return;
  }
  const rows = jobs.map((j) => {
    const dur = j.started ? Math.round(((j.finished || Date.now() / 1000) - j.started)) : 0;
    const what = j.stage === "run" ? `${pShort(j.params.pipeline)} / ${MODE_SHORT[j.params.mode]}` :
                 j.stage === "sweep" ? "full sweep" :
                 j.stage === "dataset" ? `dataset: ${j.params.which}` : j.stage;
    const extra = [j.params.models, j.params.limit ? `limit=${j.params.limit}` : ""].filter(Boolean).join(" · ");
    const cancel = (j.status === "running" || j.status === "queued")
      ? `<button class="btn small danger" data-cancel="${j.id}">cancel</button>` : "";
    return `<tr>
      <td class="muted">${j.id}</td><td>${esc(what)}</td><td class="muted">${esc(extra)}</td>
      <td><span class="status ${j.status}">${j.status}</span></td>
      <td class="num">${dur ? dur + "s" : ""}</td>
      <td><button class="btn small" data-log="${j.id}">log</button> ${cancel}</td>
    </tr>`;
  }).join("");
  $("#jobs-table").innerHTML = `<tr><th>id</th><th>Job</th><th></th><th>Status</th><th class="num">Time</th><th></th></tr>${rows}`;

  $$("[data-cancel]").forEach((b) => b.addEventListener("click", async () => {
    try { await fetchJSON(`/api/jobs/${b.dataset.cancel}/cancel`, { method: "POST" }); } catch {}
    refreshJobs();
  }));
  $$("[data-log]").forEach((b) => b.addEventListener("click", () => {
    if (openLogJob === b.dataset.log) { openLogJob = null; $("#job-log").style.display = "none"; return; }
    openLogJob = b.dataset.log;
    logOffset = 0;
    $("#job-log").textContent = "";
    $("#job-log").style.display = "block";
    pollLog();
  }));
}

async function pollLog() {
  if (!openLogJob) return;
  try {
    const data = await fetchJSON(`/api/jobs/${openLogJob}/log?offset=${logOffset}`);
    if (data.text) {
      const el = $("#job-log");
      const stick = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
      el.textContent += data.text;
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
  for (const id of ["#headline-card", "#ablation-card", "#allrows-card", "#figures-card"])
    $(id).style.display = empty ? "none" : "";
  if (empty) return;
  renderPipeChips();
  renderHeadline();
  renderAblation();
  fillResultFilters();
  renderResultsTable();
  fillFigRunSelect();
}

function runKey(r) { return `${r.Pipeline}|${r.Mode}`; }

function renderHeadline() {
  const best = {};
  for (const r of focusedResults()) {
    const k = runKey(r);
    if (!best[k] || r["ROC-AUC"] > best[k]["ROC-AUC"]) best[k] = r;
  }
  const rows = Object.values(best)
    .sort((a, b) => b["ROC-AUC"] - a["ROC-AUC"])
    .map((r, i) => {
      // near-perfect intent-only means the class is readable without the injection: corpus shortcut
      const shortcut = r.Mode === "user_intent_only" && r["ROC-AUC"] >= 0.99
        ? `<span class="chip shortcut" title="Near-perfect from the user intent alone — the classes are separable without any injection content (corpus-style shortcut)">shortcut</span>` : "";
      return `<tr class="${i === 0 ? "best" : ""}">
      <td class="wrapcell">${esc(pLabel(r.Pipeline))}</td><td>${MODE_SHORT[r.Mode]}${shortcut}</td>
      <td>${esc(r.Embeddings)}</td><td>${esc(r.Classifier)}</td>
      <td class="num">${fmt(r.Accuracy)}</td><td class="num">${fmt(r["F1-Score"])}</td>
      <td class="num">${fmt(r["ROC-AUC"])}${microbar(r["ROC-AUC"])}</td><td class="num">${fmt(r["PR-AUC"])}</td>
      <td class="num">${fmt(r["Inference-Time-per-Sample(ms)"], 4)}</td>
    </tr>`;
    }).join("");
  $("#headline-table").innerHTML =
    `<tr><th>Pipeline</th><th>Mode</th><th>Emb</th><th>Classifier</th>
      <th class="num">Acc</th><th class="num">F1</th><th class="num">ROC-AUC</th>
      <th class="num">PR-AUC</th><th class="num">ms/sample</th></tr>${rows}`;
}

function renderAblation() {
  // best ROC-AUC per pipeline × embedding × mode
  const cell = {};
  const pipes = new Set(), embs = new Set();
  for (const r of focusedResults()) {
    pipes.add(r.Pipeline); embs.add(r.Embeddings);
    const k = `${r.Pipeline}|${r.Embeddings}|${r.Mode}`;
    if (!cell[k] || r["ROC-AUC"] > cell[k]) cell[k] = r["ROC-AUC"];
  }
  const modes = STATE ? STATE.modes : ["normal", "user_intent_only", "context_only"];
  // heat tint: intent-only cells go red as they rise above chance (shortcut exposure);
  // normal/context cells go green (legitimate detection strength)
  const tint = (v, mode) => {
    if (v === undefined) return "";
    const strength = Math.max(0, Math.min(1, (v - 0.5) / 0.5));
    return mode === "user_intent_only"
      ? `background: rgba(208, 59, 59, ${(strength * 0.34).toFixed(3)})`
      : `background: rgba(12, 163, 12, ${(strength * 0.22).toFixed(3)})`;
  };
  let html = `<tr><th>Pipeline</th><th>Emb</th>${modes.map((m) => `<th class="num">${MODE_SHORT[m]}</th>`).join("")}<th class="num">Δ(normal−context)</th></tr>`;
  for (const p of pipes) {
    for (const e of embs) {
      const vals = modes.map((m) => cell[`${p}|${e}|${m}`]);
      if (vals.every((v) => v === undefined)) continue;
      const delta = (vals[0] !== undefined && vals[2] !== undefined) ? vals[0] - vals[2] : undefined;
      html += `<tr><td>${esc(pShort(p))}</td><td>${esc(e)}</td>
        ${vals.map((v, i) => `<td class="num" style="${tint(v, modes[i])}">${v === undefined ? "—" : fmt(v)}</td>`).join("")}
        <td class="num">${delta === undefined ? "—" : (delta >= 0 ? "+" : "") + fmt(delta)}</td></tr>`;
    }
  }
  $("#ablation-table").innerHTML = html;
}

function fillResultFilters() {
  const fill = (id, values, labelFn = (x) => x) => {
    const el = $(id);
    const cur = el.value;
    el.innerHTML = `<option value="">all</option>` +
      values.map((v) => `<option value="${esc(v)}">${esc(labelFn(v))}</option>`).join("");
    el.value = cur;
  };
  fill("#rf-pipeline", [...new Set(RESULTS.map((r) => r.Pipeline))], pShort);
  fill("#rf-mode", [...new Set(RESULTS.map((r) => r.Mode))], (m) => MODE_SHORT[m]);
  fill("#rf-emb", [...new Set(RESULTS.map((r) => r.Embeddings))]);
  fill("#rf-clf", [...new Set(RESULTS.map((r) => r.Classifier))]);
}
for (const id of ["#rf-pipeline", "#rf-mode", "#rf-emb", "#rf-clf"])
  $(id).addEventListener("change", renderResultsTable);

let sortCol = "ROC-AUC", sortAsc = false;

function renderResultsTable() {
  const f = {
    Pipeline: $("#rf-pipeline").value, Mode: $("#rf-mode").value,
    Embeddings: $("#rf-emb").value, Classifier: $("#rf-clf").value,
  };
  let rows = RESULTS.filter((r) => Object.entries(f).every(([k, v]) => !v || r[k] === v));
  rows = rows.slice().sort((a, b) => {
    const av = a[sortCol], bv = b[sortCol];
    const c = typeof av === "number" ? av - bv : String(av).localeCompare(String(bv));
    return sortAsc ? c : -c;
  });
  $("#rf-count").textContent = `${rows.length} of ${RESULTS.length} rows`;
  const cols = ["Pipeline", "Mode", "Embeddings", "Classifier", "Accuracy", "F1-Score", "ROC-AUC", "PR-AUC", "Train-Time(s)", "Inference-Time-per-Sample(ms)"];
  const numCols = new Set(cols.slice(4));
  const header = cols.map((c) =>
    `<th class="${numCols.has(c) ? "num" : ""}" data-sort="${c}">${c === "Inference-Time-per-Sample(ms)" ? "ms/sample" : c}${sortCol === c ? (sortAsc ? " ▲" : " ▼") : ""}</th>`).join("");
  const body = rows.map((r) => `<tr>${cols.map((c) => {
    let v = r[c];
    if (c === "Pipeline") v = pShort(v);
    if (c === "Mode") v = MODE_SHORT[v];
    const bar = c === "ROC-AUC" ? microbar(v) : "";
    return `<td class="${numCols.has(c) ? "num" : ""}">${numCols.has(c) ? fmt(v, c.includes("Time") ? 3 : 4) + bar : esc(v)}</td>`;
  }).join("")}</tr>`).join("");
  $("#results-table").innerHTML = `<tr>${header}</tr>${body}`;
  $$("#results-table [data-sort]").forEach((th) => th.addEventListener("click", () => {
    const c = th.dataset.sort;
    if (sortCol === c) sortAsc = !sortAsc; else { sortCol = c; sortAsc = false; }
    renderResultsTable();
  }));
}

/* -------- figures -------- */
function fillFigRunSelect() {
  const runsWithFigs = (STATE ? STATE.runs : []).filter((r) => r.figures.length);
  const el = $("#fig-run");
  const cur = el.value;
  el.innerHTML = runsWithFigs.map((r) =>
    `<option value="${r.pipeline}|${r.mode}">${pShort(r.pipeline)} / ${MODE_SHORT[r.mode]}</option>`).join("");
  if ([...el.options].some((o) => o.value === cur)) el.value = cur;
  renderFigures();
}
$("#fig-run").addEventListener("change", renderFigures);

function renderFigures() {
  const sel = $("#fig-run").value;
  const el = $("#figures-list");
  if (!sel) { el.innerHTML = `<span class="muted">No run has figures yet.</span>`; return; }
  const [pipeline, mode] = sel.split("|");
  const run = (STATE ? STATE.runs : []).find((r) => r.pipeline === pipeline && r.mode === mode);
  if (!run || !run.figures.length) {
    el.innerHTML = `<span class="muted">No figures for this run.</span>`;
    return;
  }
  el.innerHTML = run.figures.map((f) =>
    `<figure><a href="/api/figure?pipeline=${pipeline}&mode=${mode}&name=${encodeURIComponent(f)}" target="_blank">
      <img loading="lazy" src="/api/figure?pipeline=${pipeline}&mode=${mode}&name=${encodeURIComponent(f)}"></a>
      <figcaption>${esc(f)}</figcaption></figure>`).join("");
}

/* ================= detect tab ================= */
async function loadDetectOptions() {
  let opts = {};
  try { opts = await fetchJSON("/api/detect/options"); } catch { opts = {}; }
  DETECT_OPTIONS = opts.models || [];
  const note = $("#detect-note");
  const sel = $("#d-model");
  const btn = $("#b-detect");
  if (!DETECT_OPTIONS.length) {
    note.textContent = opts.reason
      ? `${opts.reason}. ${opts.fix}.`
      : "No saved models found. Run a pipeline with “save models” checked (Runs tab), then come back.";
    sel.innerHTML = `<option>— none available —</option>`;
    btn.disabled = true;
    return;
  }
  btn.disabled = false;
  note.textContent = "Scores one (intent, context) pair with a trained classifier. minilm embeds locally in ms; openai needs an API key; qwen3 loads a 4B model into VRAM on first use.";
  sel.innerHTML = DETECT_OPTIONS.map((m, i) =>
    `<option value="${i}">${pShort(m.pipeline)} / ${MODE_SHORT[m.mode]} · ${m.emb} + ${m.clf}</option>`).join("");
}

$("#b-sample").addEventListener("click", async () => {
  const msg = $("#detect-msg");
  msg.textContent = "sampling…"; msg.className = "msg";
  try {
    const data = await fetchJSON(`/api/sample?pipeline=${$("#d-sample-pipeline").value}`);
    const row = data.rows[0];
    $("#d-intent").value = row.user_intent || "";
    $("#d-context").value = row.context || "";
    $("#d-context").dataset.truth = row.label;
    msg.textContent = `loaded a labeled row (ground truth hidden until you detect)`;
  } catch (e) { msg.textContent = e.message; msg.className = "msg warn"; }
});

$("#b-detect").addEventListener("click", async () => {
  const m = DETECT_OPTIONS[parseInt($("#d-model").value, 10)];
  const msg = $("#detect-msg");
  if (!m) { msg.textContent = "no model selected"; msg.className = "msg error"; return; }
  msg.textContent = "embedding + scoring…"; msg.className = "msg";
  $("#b-detect").disabled = true;
  try {
    const res = await fetchJSON("/api/detect", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pipeline: m.pipeline, mode: m.mode, emb: m.emb, clf: m.clf,
        user_intent: $("#d-intent").value, context: $("#d-context").value,
      }),
    });
    msg.textContent = "";
    $("#detect-result").style.display = "";
    const badge = $("#verdict-badge");
    badge.textContent = res.verdict.toUpperCase();
    badge.className = "verdict " + res.verdict;
    $("#gauge-fill").style.width = (res.probability * 100).toFixed(1) + "%";
    $("#gauge-label").textContent = (res.probability * 100).toFixed(1) + "%";
    $("#detect-timing").textContent = `embed ${res.embed_ms} ms · classify ${res.classify_ms} ms · ${res.emb} + ${res.clf}, trained on ${pShort(res.pipeline)}/${MODE_SHORT[res.mode]}`;
    const truth = $("#d-context").dataset.truth;
    $("#detect-truth").textContent = truth !== undefined && truth !== ""
      ? `dataset ground truth for this row: ${truth === "1" ? "malicious" : "benign"}` : "";
  } catch (e) {
    msg.textContent = e.message; msg.className = "msg error";
  } finally { $("#b-detect").disabled = false; }
});
["#d-intent", "#d-context"].forEach((id) => $(id).addEventListener("input", () => {
  delete $("#d-context").dataset.truth;
  $("#detect-truth").textContent = "";
}));

/* ================= files tab ================= */
const openSections = new Set();

async function loadFiles() {
  let data;
  try { data = await fetchJSON("/api/files"); }
  catch (e) { $("#files-list").innerHTML = `<span class="msg error">${esc(e.message)}</span>`; return; }

  $("#files-list").innerHTML = data.sections.map((s) => {
    const anyPresent = s.entries.some((e) => e.status === "present");
    const isOpen = openSections.size ? openSections.has(s.name) : anyPresent;
    const cls = s.n_expected && s.n_present === s.n_expected ? "on" : s.n_present ? "part" : "";
    const count = s.n_expected ? `${s.n_present}/${s.n_expected} expected` : `${s.entries.length} files`;
    const nExtra = s.entries.filter((e) => e.kind === "unexpected").length;
    const rows = s.entries.map((e) => {
      const status = e.status === "present"
        ? `<span class="badge on">present</span>` : `<span class="badge">missing</span>`;
      const tag = e.kind === "unexpected" ? ` <span class="badge part">unexpected</span>`
        : e.kind === "optional" ? ` <span class="badge">output</span>` : "";
      const when = e.mtime ? new Date(e.mtime * 1000).toLocaleString() : "";
      const view = e.status === "present"
        ? `<button class="btn small" data-view="${esc(e.path)}">view</button>` : "";
      return `<tr class="file-${e.status}"><td>${status}${tag}</td>
        <td class="mono">${esc(e.path)}</td>
        <td class="num">${e.size_mb != null ? e.size_mb + " MB" : ""}</td>
        <td class="muted">${when}</td><td>${view}</td></tr>`;
    }).join("");
    return `<details class="file-section" data-name="${esc(s.name)}" ${isOpen ? "open" : ""}>
      <summary>${esc(s.name)} <span class="badge ${cls}">${count}</span>
        ${nExtra ? `<span class="badge part">${nExtra} unexpected</span>` : ""}</summary>
      <div class="scroll"><table class="plain">${rows}</table></div>
    </details>`;
  }).join("");

  $$(".file-section").forEach((d) => d.addEventListener("toggle", () => {
    if (d.open) openSections.add(d.dataset.name); else openSections.delete(d.dataset.name);
  }));
  $$("#files-list [data-view]").forEach((b) => b.addEventListener("click", () => previewFile(b.dataset.view)));
}

async function previewFile(path) {
  const card = $("#file-preview-card");
  card.style.display = "";
  $("#file-preview-title").textContent = path;
  $("#file-preview").innerHTML = `<span class="muted">loading…</span>`;
  try {
    const info = await fetchJSON(`/api/file/inspect?path=${encodeURIComponent(path)}`);
    $("#file-preview").innerHTML = info.type === "image"
      ? `<img class="preview-img" src="/api/file/raw?path=${encodeURIComponent(path)}">`
      : `<pre class="log">${esc(info.text)}</pre>`;
  } catch (e) {
    $("#file-preview").innerHTML = `<span class="msg error">${esc(e.message)}</span>`;
  }
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

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

/* ================= boot & polling ================= */
refreshState();
refreshJobs();
setInterval(() => {
  const runsActive = $("#tab-runs").classList.contains("active");
  refreshJobs();
  if (runsActive) pollLog();
}, 2000);
setInterval(refreshState, 5000);
setInterval(() => { if ($("#tab-files").classList.contains("active")) loadFiles(); }, 5000);
