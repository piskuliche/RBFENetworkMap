"use strict";
/* The knob explorer.
 *
 * Every control on this page is built from /api/schema, which is generated from the CLI's
 * own argument parser. Nothing here knows the name of a single knob, and that is on
 * purpose: a flag added to `rbfenet plan` shows up in this form without anyone editing
 * this file, and the command line the page offers to copy is exactly the run it displays.
 */

const el = (tag, attrs = {}, ...kids) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (v === true) node.setAttribute(k, "");
    else if (v !== false && v != null) node.setAttribute(k, v);
  }
  for (const kid of kids) if (kid != null) node.append(kid);
  return node;
};

const api = async (path, options) => {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
};
const post = (path, body) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

const state = { schema: null, values: {}, run: null, poll: null, fields: new Map() };

/* ---------------------------------------------------------------- the form */

/* Promoted to the top because they are the knobs that change the shape of the network
 * rather than the feasibility of an edge. Purely an ordering hint: every one of them is
 * still in its own group below, and nothing depends on this list being complete. */
const COMMON = [
  "planner", "edges_per_ligand", "min_cycle_coverage", "max_diameter",
  "cbfe", "max_softcore_atoms", "design", "cluster_by",
];

function control(field) {
  const current = state.values[field.dest];
  const value = current === undefined ? field.default : current;
  const set = (v) => {
    if (v === "" || v === null) delete state.values[field.dest];
    else state.values[field.dest] = v;
    refreshForm();
  };

  if (field.widget === "bool") {
    const box = el("input", { type: "checkbox" });
    box.checked = Boolean(value);
    box.onchange = () => set(box.checked ? true : null);
    return box;
  }
  if (field.widget === "choice" || field.widget === "plugin") {
    const select = el("select");
    if (field.optional) select.append(el("option", { value: "", text: "(unset)" }));
    const names = field.widget === "plugin"
      ? state.schema.plugins[field.plugin_kind].map((p) => [p.name, p.available])
      : field.choices.map((c) => [c, true]);
    for (const [name, available] of names) {
      const option = el("option", { value: name, text: available ? name : `${name} (not installed)` });
      if (!available) option.disabled = true;
      select.append(option);
    }
    select.value = value == null ? "" : String(value);
    select.onchange = () => set(select.value);
    return select;
  }
  if (field.widget === "repeatable") {
    const list = Array.isArray(value) ? value : [];
    const box = el("input", { type: "text", value: list.join(" "), placeholder: field.metavar || "" });
    box.onchange = () => {
      const parts = box.value.split(/\s+/).filter(Boolean);
      set(parts.length ? parts : null);
    };
    return box;
  }
  const box = el("input", {
    type: field.widget === "int" || field.widget === "float" ? "number" : "text",
    value: value == null ? "" : String(value),
    placeholder: field.optional ? "(unset)" : "",
  });
  if (field.widget === "float") box.step = "any";
  box.onchange = () => {
    if (box.value === "") return set(null);
    set(field.widget === "int" ? parseInt(box.value, 10) : field.widget === "float" ? parseFloat(box.value) : box.value);
  };
  return box;
}

function fieldNode(field) {
  const node = el("div", { class: "field" });
  const label = el("label", { text: field.flag });
  const row = el("div", { class: "row" }, control(field));
  if (state.schema.export_knobs.includes(field.dest)) {
    row.append(el("span", { class: "badge export", text: "export only", title:
      "Read by the Amber exporter when writing runconfigs. It cannot move an edge, so the network above will not change." }));
  }
  node.append(label, row, el("div", { class: "help", text: field.help }));
  state.fields.set(field.dest, node);
  return node;
}

function buildForm() {
  const form = document.getElementById("form");
  form.replaceChildren();
  state.fields.clear();

  const byDest = new Map();
  for (const group of state.schema.groups) for (const field of group.fields) byDest.set(field.dest, field);

  const common = el("details", { class: "group", open: true }, el("summary", { text: "common" }));
  const commonBody = el("div", { class: "group-body" });
  for (const dest of COMMON) if (byDest.has(dest)) commonBody.append(fieldNode(byDest.get(dest)));
  common.append(commonBody);
  form.append(common);

  for (const group of state.schema.groups) {
    const details = el("details", { class: "group" }, el("summary", { text: group.title }));
    const body = el("div", { class: "group-body" });
    for (const field of group.fields) body.append(fieldNode(field));
    details.append(body);
    form.append(details);
  }
  refreshForm();
}

/* Recomputed on every change, because which knobs matter depends on the planner chosen. */
function refreshForm() {
  const schema = state.schema;
  const planner = state.values.planner || "mst";
  const known = schema.planner_knobs[planner];
  const active = known ? new Set([...known, ...schema.pipeline_knobs, ...schema.export_knobs]) : null;

  const byDest = new Map();
  for (const group of schema.groups) for (const field of group.fields) byDest.set(field.dest, field);

  for (const [dest, node] of state.fields) {
    const field = byDest.get(dest);
    const value = state.values[dest];
    const changed = value !== undefined && JSON.stringify(value) !== JSON.stringify(field.default);
    node.classList.toggle("changed", changed);

    const inactive = active !== null && !active.has(dest);
    node.classList.toggle("inactive", inactive);
    let badge = node.querySelector(".badge.inactive-badge");
    if (inactive && !badge) {
      node.querySelector(".row").append(el("span", {
        class: "badge inactive-badge",
        text: `ignored by ${planner}`,
        title: `The ${planner} planner never reads this knob. It is accepted and has no effect.`,
      }));
    } else if (!inactive && badge) {
      badge.remove();
    } else if (inactive && badge) {
      badge.textContent = `ignored by ${planner}`;
    }
  }
  renderWarnings();
}

/* ------------------------------------------------------------- the warnings */

function renderWarnings() {
  const box = document.getElementById("warnings");
  const notes = [];
  const v = state.values;
  const n = state.nLigands || 0;

  /* Peak memory during mapping, from the relation --jobs' own help text states: roughly
   * 40 MB per second of --mcs-timeout per job, because FindMCS allocates monotonically
   * and frees nothing until it returns. The default 60 s x 8 jobs is some 20 GB, which is
   * how a 47-ligand run once reached 30 GB RSS and exhausted swap. */
  const jobs = v.jobs ?? 1;
  const timeout = v.mcs_timeout ?? 60;
  const gb = (40 * timeout * jobs) / 1024;
  if (gb >= 8) {
    notes.push(el("div", { class: gb >= 24 ? "warn bad" : "warn" },
      `Mapping may need about ${gb.toFixed(0)} GB at peak: roughly 40 MB per second of `,
      el("code", { text: "--mcs-timeout" }), ` per job, and you have ${timeout} s x ${jobs} jobs. `,
      `Lower either if that is more than this machine has.`));
  }

  if (n >= 40 && (v.prefilter ?? "none") === "none" && (v.pair_evaluation ?? "eager") === "eager") {
    const pairs = (n * (n - 1)) / 2;
    const note = el("div", { class: "warn" },
      `${n} ligands is ${pairs} pairs to map. `,
      el("button", { id: "large-preset" }, "Set --prefilter fingerprint and --pair-evaluation adaptive"),
      " ");
    notes.push(note);
  }

  if ((v.pair_evaluation ?? "eager") === "adaptive" && (v.planner ?? "mst") !== "mst") {
    notes.push(el("div", { class: "warn" },
      el("code", { text: "--pair-evaluation adaptive" }),
      ` is honoured only by the mst planner. Under ${v.planner} it falls back to eager and maps the whole pool.`));
  }

  if (v.compat) {
    notes.push(el("div", { class: "warn" },
      el("code", { text: `--compat ${v.compat}` }),
      " pins every algorithmic knob. Setting any of them alongside it is refused, not merged."));
  }

  box.replaceChildren(...notes);
  const preset = document.getElementById("large-preset");
  if (preset) {
    preset.onclick = () => {
      /* Set them visibly in the form rather than defaulting them behind the user's back:
       * the copied command line has to be the command that produced the picture. */
      state.values.prefilter = "fingerprint";
      state.values.pair_evaluation = "adaptive";
      buildForm();
    };
  }
}

/* --------------------------------------------------------------- the result */

const STATS = [
  ["n_edges", "Edges"], ["n_ligands", "Ligands"], ["cost", "Cost", 2], ["gpu_hours", "GPU-h", 0],
  ["n_cycles", "Cycles ≤4"], ["diameter", "Diameter"],
];

function renderRun(run) {
  const status = document.getElementById("status");
  status.className = `status ${run.state}`;
  if (run.state === "running") {
    const { done, total } = run.progress;
    const bar = el("progress", { value: done, max: Math.max(total, 1) });
    status.replaceChildren(el("div", { text: `Mapping ${done} / ${total} pairs…` }), bar);
  } else if (run.state === "error") {
    status.replaceChildren(el("strong", { text: "Cannot plan that. " }), document.createTextNode(run.error));
  } else if (run.state === "cancelled") {
    status.replaceChildren(document.createTextNode("Cancelled."));
  } else {
    const cache = run.cache;
    status.replaceChildren(document.createTextNode(
      `Planned in ${run.seconds.toFixed(1)} s. ${cache.hits} mapping(s) reused, ${cache.misses} computed.`));
  }

  document.getElementById("cancel").disabled = run.state !== "running";

  const metrics = document.getElementById("metrics");
  if (run.metrics) {
    metrics.replaceChildren(...STATS.map(([key, label, digits]) => {
      const raw = run.metrics[key];
      const text = raw == null ? "—" : digits == null ? String(raw) : Number(raw).toFixed(digits);
      return el("div", { class: "stat" }, el("div", { class: "value", text }), el("div", { class: "label", text: label }));
    }));
  } else {
    metrics.replaceChildren();
  }

  if (detail.runId !== run.id) {
    /* A new run means a new diagram, and possibly a different partition behind the same
     * edge key once a soft-core knob has moved. */
    detail.runId = run.id;
    detail.rejectedFor = null;
    resetDetail();
    document.getElementById("rejected").replaceChildren();
  }

  if (run.svg) document.getElementById("diagram").innerHTML = run.svg;
  if (run.state === "done" && detail.rejectedFor !== run.id) {
    detail.rejectedFor = run.id;
    loadRejected(run.id);
  }

  const block = document.getElementById("command-block");
  block.hidden = run.state !== "done";
  if (run.state === "done") {
    document.getElementById("command").textContent = run.command;
    document.getElementById("report").href = `/api/run/${run.id}/report`;
    document.getElementById("download").href = `/api/run/${run.id}/network.json`;
  }

  if (run.unmet && run.unmet.length) {
    const warn = el("div", { class: "warn" }, el("strong", { text: "Unmet constraints: " }),
      document.createTextNode(run.unmet.join("; ")));
    document.getElementById("warnings").append(warn);
  }
}

const PIN_COLUMNS = [
  ["n_edges", "edges"], ["cost", "cost", 2], ["gpu_hours", "GPU-h", 0],
  ["diameter", "diam"], ["n_cycles", "cycles"],
];

function renderPins(pins) {
  const box = document.getElementById("pins");
  if (!pins.length) return box.replaceChildren();
  const head = el("tr", {}, el("th", { text: "run" }),
    ...PIN_COLUMNS.map(([, label]) => el("th", { text: label })),
    el("th", { text: "deg min/mean/max" }), el("th", { text: "flags" }), el("th", {}));
  const rows = pins.map((pin) => {
    const m = pin.metrics;
    const cells = PIN_COLUMNS.map(([key, , digits]) => {
      const raw = m[key];
      return el("td", { text: raw == null ? "—" : digits == null ? String(raw) : Number(raw).toFixed(digits) });
    });
    const drop = el("button", { text: "×", title: "Remove this pin" });
    drop.onclick = async () => renderPins((await post("/api/unpin", { run_id: pin.run_id })).pins);
    return el("tr", {},
      el("td", { text: pin.label }), ...cells,
      el("td", { text: `${m.degree.min}/${m.degree.mean.toFixed(1)}/${m.degree.max}` }),
      el("td", { title: pin.command, text: pin.argv.slice(3).join(" ") || "(defaults)" }),
      el("td", {}, drop));
  });
  box.replaceChildren(el("h2", { text: "Pinned runs" }),
    el("table", {}, el("thead", {}, head), el("tbody", {}, ...rows)));
}


/* ------------------------------------------------- the soft-core detail panel */

/* Hovering an edge asks the server for its partition. Kept out of the poll response
 * deliberately: two depictions are ~47 KB, and shipping them for every edge on every tick
 * is what would make the tool feel slow. */
const detail = {
  cache: new Map(),   /* keyed by run, because "a~b" is a different partition once a
                         soft-core knob moves, and the string is the same in every run */
  runId: null,
  rejectedFor: null,
  token: 0,           /* a later hover must win even if an earlier response lands after it */
  timer: null,
  pinned: false,
  showBefore: false,
  indices: false,
  current: null,
};

const CACHE_LIMIT = 50;
const detailKey = (runId, scope, key, indices) => `${runId} ${scope} ${key} ${indices ? 1 : 0}`;

function resetDetail() {
  detail.cache.clear();
  detail.pinned = false;
  detail.current = null;
  detail.showBefore = false;
  const panel = document.getElementById("edge-detail");
  panel.hidden = true;
  panel.classList.remove("pinned");
}

function trimCache() {
  while (detail.cache.size > CACHE_LIMIT) detail.cache.delete(detail.cache.keys().next().value);
}

function renderDetail(data) {
  const panel = document.getElementById("edge-detail");
  panel.hidden = false;
  panel.classList.toggle("pinned", detail.pinned);

  if (data.error) {
    panel.replaceChildren(el("div", { class: "detail-note warn-note", text: data.error }));
    return;
  }

  const badges = [];
  if (data.counterpoised) badges.push("CBFE");
  if (data.synthetic.length) badges.push(`SYN: ${data.synthetic.join(", ")}`);
  for (const reason of data.rejections) badges.push(reason);

  const cost = data.cost == null ? "rejected" : `cost ${data.cost.toFixed(3)}`;
  const counts = data.counterpoised
    ? `${data.n_atoms_1}/${data.n_atoms_2} atoms fully decoupled, no atom mapping`
    : `soft-core ${data.n_softcore_1}/${data.n_softcore_2}, common core ${data.n_common_core}`;
  const who = data.counterpoised ? "protocol" : "mapper";
  const meta = `${cost} · ${counts} · ${who} ${data.mapper}` +
    (badges.length ? " · " + badges.join(" · ") : "");

  const controls = el("div", { class: "detail-controls" });

  if (data.before) {
    const label = detail.showBefore ? "Show the repaired soft-core" : "Show it as the mapper proposed it";
    const toggle = el("button", { text: label });
    toggle.onclick = () => { detail.showBefore = !detail.showBefore; renderDetail(data); };
    controls.append(toggle);
  }

  /* The repair trace names atom indices, so a trace beside pictures without them is half a
   * tool. Re-fetches, because RDKit draws the indices server-side. */
  const box = el("input", { type: "checkbox" });
  box.checked = detail.indices;
  box.onchange = () => {
    detail.indices = box.checked;
    if (detail.current) loadEdge(detail.current.scope, detail.current.key, { now: true });
  };
  controls.append(el("label", {}, box, document.createTextNode("atom indices")));

  const pin = el("button", { text: detail.pinned ? "Unpin" : "Pin" });
  pin.onclick = () => { detail.pinned = !detail.pinned; renderDetail(data); };
  controls.append(pin);

  const head = el("div", { class: "detail-head" },
    el("div", {}, el("h3", { text: data.key }), el("div", { class: "detail-meta", text: meta })),
    controls);

  /* innerHTML, not el(): el() uses createElement and cannot build SVG children. */
  const shown = detail.showBefore && data.before ? data.before : data.after;
  const panes = el("div", { class: "panes" });
  panes.innerHTML = shown.source + shown.target;

  const parts = [head, panes];

  if (detail.showBefore && data.before) {
    parts.push(el("div", { class: "detail-note", text:
      `As the mapper proposed it, before the repair demoted ${data.n_demoted} atom(s) to join ` +
      `${data.regions_before.join("/")} soft-core region(s) into ${data.regions_after.join("/")}.` }));
  }

  /* mapper_failed and no_common_core yield a wholly soft-core mapping, so both molecules
   * draw entirely warm. That is the rejection, not a drawing fault; the report carries the
   * same note, and without it this reads as a bug. */
  if (data.n_common_core === 0 && !data.counterpoised) {
    parts.push(el("div", { class: "detail-note warn-note", text:
      "No common core was found, so every atom is drawn as soft-core. " +
      "That is the rejection itself, not a drawing fault." }));
  }

  if (data.trace.length) {
    const section = el("details", { class: "detail-section" }, el("summary", { text: "soft-core repair trace" }));
    section.append(el("pre", { text: data.trace.join(String.fromCharCode(10)) }));
    parts.push(section);
  }

  const masks = el("details", { class: "detail-section" }, el("summary", { text: "amber masks" }));
  if (data.masks.unavailable) {
    masks.append(el("div", { class: "detail-note", text: data.masks.unavailable }));
  } else {
    const list = el("dl", { class: "masks" });
    for (const name of ["timask1", "scmask1", "timask2", "scmask2"]) {
      list.append(el("dt", { text: name }), el("dd", { text: data.masks[name] || "(empty)" }));
    }
    masks.append(list);
  }
  parts.push(masks);

  panel.replaceChildren(...parts);
}

async function loadEdge(scope, key, { now = false } = {}) {
  const runId = state.run && state.run.id;
  if (!runId) return;
  clearTimeout(detail.timer);
  const token = ++detail.token;

  const go = async () => {
    const cacheKey = detailKey(runId, scope, key, detail.indices);
    let data = detail.cache.get(cacheKey);
    if (!data) {
      const path = scope === "edges" ? "edge" : "candidate";
      const query = detail.indices ? "?indices=1" : "";
      try {
        data = await api(`/api/run/${runId}/${path}/${encodeURIComponent(key)}${query}`);
        detail.cache.set(cacheKey, data);
        trimCache();
      } catch (err) {
        data = { error: err.message };
      }
    }
    if (token !== detail.token) return;   /* a later hover already won */
    detail.current = { scope, key };
    renderDetail(data);
  };

  /* Debounced: sweeping across a dense diagram would otherwise fire a request per edge
   * crossed, and the server spawns a thread for each one. */
  if (now) return go();
  detail.timer = setTimeout(go, 80);
}

function hoverTarget(event) {
  const node = event.target.closest && event.target.closest("[data-edge]");
  return node ? { key: node.dataset.edge, scope: node.dataset.scope || "edges" } : null;
}

function wireHover() {
  /* Delegated, on the containers rather than the edges: renderRun assigns
   * innerHTML = run.svg, which destroys any listener attached to a child. focusin as well
   * as mouseover, since the edges carry tabindex and the rejected rows are buttons. */
  for (const id of ["diagram", "rejected"]) {
    const host = document.getElementById(id);
    for (const type of ["mouseover", "focusin"]) {
      host.addEventListener(type, (event) => {
        if (detail.pinned) return;
        const hit = hoverTarget(event);
        if (hit) loadEdge(hit.scope, hit.key);
      });
    }
    host.addEventListener("click", (event) => {
      const hit = hoverTarget(event);
      if (!hit) return;
      event.preventDefault();
      detail.pinned = true;
      loadEdge(hit.scope, hit.key, { now: true });
    });
  }
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && detail.pinned) {
      detail.pinned = false;
      document.getElementById("edge-detail").classList.remove("pinned");
    }
  });
}

/* Fetched once when a run finishes, never in the 250 ms poll -- which omits even the
 * network SVG for the same reason. */
async function loadRejected(runId) {
  const box = document.getElementById("rejected");
  let summary;
  try {
    summary = await api(`/api/run/${runId}/rejected`);
  } catch (err) {
    return box.replaceChildren();
  }
  if (!summary.total) return box.replaceChildren();

  const groups = summary.groups.map((group) => {
    const rows = el("div", { class: "reject-rows" });
    for (const pair of group.pairs) {
      rows.append(el("button", { text: pair.key, "data-edge": pair.key, "data-scope": "candidates" }));
    }
    const node = el("details", { class: "reject-group" },
      el("summary", {}, document.createTextNode(group.reason + " "),
        el("span", { class: "count", text: `(${group.count})` })),
      rows);
    /* Always state what was left out: a truncated list read as complete is how somebody
     * concludes a pair was never tried. */
    if (group.omitted) {
      node.append(el("div", { class: "reject-omitted", text: `${group.omitted} more not listed.` }));
    }
    return node;
  });
  box.replaceChildren(el("h2", { text: `Rejected pairs (${summary.total})` }), ...groups);
}

/* ------------------------------------------------------------------ driving */

function poll(runId) {
  clearInterval(state.poll);
  state.poll = setInterval(async () => {
    try {
      const run = await api(`/api/run/${runId}`);
      state.run = run;
      renderRun(run);
      if (run.state !== "running") clearInterval(state.poll);
    } catch (err) {
      clearInterval(state.poll);
    }
  }, 250);
}

async function plan() {
  try {
    const run = await post("/api/plan", { values: state.values });
    state.run = run;
    renderRun(run);
    poll(run.id);
  } catch (err) {
    const status = document.getElementById("status");
    status.className = "status error";
    status.replaceChildren(el("strong", { text: "Cannot plan that. " }), document.createTextNode(err.message));
  }
}

async function loadLigands() {
  const raw = document.getElementById("ligand-path").value.trim();
  const summary = document.getElementById("ligand-summary");
  if (!raw) return;
  /* Whitespace-separated, because --ligands is nargs="+" and this box should accept what
   * that accepts. Globs are expanded server-side, where a shell would have done it. */
  const paths = raw.split(/\s+/).filter(Boolean);
  summary.textContent = "loading…";
  try {
    const info = await post("/api/ligands", { paths });
    state.nLigands = info.n_ligands;
    summary.textContent = `${info.n_ligands} ligands: ${info.names.slice(0, 6).join(", ")}${info.names.length > 6 ? "…" : ""}`;
    refreshForm();
  } catch (err) {
    summary.textContent = err.message;
  }
}

async function main() {
  state.schema = await api("/api/schema");
  buildForm();

  document.getElementById("load").onclick = loadLigands;
  document.getElementById("ligand-path").onkeydown = (e) => { if (e.key === "Enter") loadLigands(); };
  document.getElementById("plan").onclick = plan;
  document.getElementById("cancel").onclick = () => state.run && post(`/api/run/${state.run.id}/cancel`);
  document.getElementById("reset").onclick = () => { state.values = {}; buildForm(); };
  document.getElementById("copy").onclick = () =>
    navigator.clipboard.writeText(document.getElementById("command").textContent);
  document.getElementById("pin").onclick = async () => {
    if (!state.run || state.run.state !== "done") return;
    const label = prompt("Label for this pin", `run ${state.run.id.slice(0, 4)}`);
    if (label === null) return;
    await post("/api/pin", { run_id: state.run.id, label });
    renderPins((await api("/api/session")).pins);
  };

  wireHover();

  const session = await api("/api/session");
  if (session.ligands.paths.length) {
    document.getElementById("ligand-path").value = session.ligands.paths[0];
    state.nLigands = session.ligands.loaded.length;
    document.getElementById("ligand-summary").textContent = `${state.nLigands} ligands loaded`;
  }
  renderPins(session.pins);
  refreshForm();
}

main();
