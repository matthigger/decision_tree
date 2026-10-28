// Tree builder, behind the Classification and Regression tabs. The user
// selects a node in the tree diagram (or a region of the plot) and splits
// it by hand or greedily; the plot, the diagram and the impurity readout
// update live. The tree stores only its questions (feat, thr); route()
// sends the samples through it on every draw, so dragging a sample or a
// threshold re-scores the whole tree.

const B_STATE = {
  class: { set: "blobs", n: 25, noise: 0.35, seed: 1 },
  reg: { set: "sine", n: 40, noise: 0.3, seed: 1 },
};
const B_N_RANGE = { class: [5, 100], reg: [5, 200] };
// Diagram geometry: node box, horizontal slot per leaf, vertical level.
const T_W = 108, T_H = 54, T_SLOT = 118, T_LEVEL = 86, T_PAD = 10;
// Scan chart plot box, in its 400 x 168 viewBox.
const SC = { x0: 52, x1: 390, y0: 8, y1: 128 };

// Per mode: samples {X, y}, the tree root and the selected node.
const bt = { class: {}, reg: {} };
let bmode = "class";
// Preorder list of the nodes, from layoutTree(); a node's id indexes it.
let bnodes = [];

const bsvg = document.getElementById("b-stage");
const tsvg = document.getElementById("tree");
const scan = document.getElementById("scan");
const bels = {};
// Resample and the data settings; kept by reference because rebuilding
// the data bar detaches them.
const bTools = document.getElementById("b-tools");

function bcur() { return B_STATE[bmode]; }
function bdat() { return bt[bmode]; }
function bC() { return bmode === "class" ? CLASS_SETS[bcur().set].C : 0; }
function bD() { return bmode === "class" ? 2 : 1; }
function leaf() { return { feat: null, thr: 0, left: null, right: null }; }
function impName() { return bmode === "class" ? "Gini" : "MSE"; }
function fmtImp(v) { return v.toFixed(3); }

/** Full domain box of the plot: one [lo, hi] per feature. */
function domainBox() {
  const p = VIEW[bmode];
  return bmode === "class" ? [p.xdom.slice(), p.ydom.slice()]
    : [p.xdom.slice()];
}

/**
 * Send the samples idx through the subtree at nd, filling in each node's
 * nodeStats() fields plus idx, box (its cell of the domain), depth and
 * parent. An empty node estimates what its parent does.
 */
function route(nd, idx, box, depth = 0, parent = null) {
  const { X, y } = bdat();
  Object.assign(nd, nodeStats(idx, y, bC()), { idx, box, depth, parent });
  if (nd.pred === null) nd.pred = parent ? parent.pred : 0;
  if (nd.feat === null) return;
  const yes = [], no = [];
  for (const i of idx) (X[i][nd.feat] <= nd.thr ? yes : no).push(i);
  const by = box.map(b => b.slice()), bn = box.map(b => b.slice());
  const t = clamp(nd.thr, ...box[nd.feat]);
  by[nd.feat][1] = t;
  bn[nd.feat][0] = t;
  route(nd.left, yes, by, depth + 1, nd);
  route(nd.right, no, bn, depth + 1, nd);
}

function reroute() {
  const d = bdat();
  route(d.root, d.y.map((_, i) => i), domainBox());
}

/** Draw fresh samples; keep the tree unless asked to start over. */
function bRegen(newTree) {
  const s = bcur(), d = bdat();
  Object.assign(d, bmode === "class" ? makeClass(s.set, s.n, s.noise, s.seed)
    : makeReg(s.set, s.n, s.noise, s.seed));
  if (newTree || !d.root) d.root = d.sel = leaf();
  bInvalidate();
}

/** Midpoint of the two middle values of feature f in the node's samples. */
function medianThr(nd, f) {
  const v = nd.idx.map(i => bdat().X[i][f]).sort((a, b) => a - b);
  if (!v.length) return (nd.box[f][0] + nd.box[f][1]) / 2;
  return (v[(v.length - 1) >> 1] + v[v.length >> 1]) / 2;
}

/** Ask "x_f <= thr?" at nd, giving a leaf two leaf children. */
function setSplit(nd, f, thr) {
  if (nd.feat === null) {
    nd.left = leaf();
    nd.right = leaf();
  }
  nd.feat = f;
  nd.thr = thr;
}

function bestHere(nd) {
  const { X, y } = bdat();
  return bestSplit(nd.idx, X, y, bC(), [...Array(bD()).keys()]);
}

/** Give every impure leaf its best split: the tree grows one level. */
function growLevel() {
  for (const nd of bnodes) {
    if (nd.feat !== null || nd.imp <= EPS) continue;
    const b = bestHere(nd);
    if (b) setSplit(nd, b.feat, b.thr);
  }
}

/** Lay the tree out: leaves take slots left to right, parents center. */
function layoutTree(root) {
  const nodes = [];
  let slots = 0, depth = 0;
  (function walk(nd) {
    nd.id = nodes.length;
    nodes.push(nd);
    depth = Math.max(depth, nd.depth);
    if (nd.feat === null) {
      nd.lx = slots++;
    } else {
      walk(nd.left);
      walk(nd.right);
      nd.lx = (nd.left.lx + nd.right.lx) / 2;
    }
  })(root);
  return { nodes, slots, depth };
}

// ------------------------------------------------------------ scheduling

let bQueued = false;
function bInvalidate() {
  if (bQueued) return;
  bQueued = true;
  requestAnimationFrame(() => {
    bQueued = false;
    bDraw();
  });
}

function bDraw() {
  if (state.mode === "forest") return;
  reroute();
  const lay = layoutTree(bdat().root);
  bnodes = lay.nodes;
  if (bmode === "class") drawPlaneClass();
  else drawPlaneReg();
  drawDiagram(lay);
  drawEditor();
  bLegend();
  bUpdateControls();
}

// ------------------------------------------------------------------ build

function bBuild() {
  const p = VIEW[bmode], w = p.x1 - p.x0, h = p.y1 - p.y0;
  bsvg.setAttribute("viewBox", `0 0 ${p.w} ${p.h}`);
  bsvg.innerHTML = `<clipPath id="bclip"><rect x="${p.x0}" y="${p.y0}"
    width="${w}" height="${h}"/></clipPath>`;
  const axes = node("g", { class: "axes" });
  axes.innerHTML = axesSvg(p, bmode);
  for (const k of ["regions", "band", "curve", "dim", "cuts", "sticks",
    "samples"]) {
    bels[k] = node("g", { class: k, "clip-path": "url(#bclip)" });
  }
  // Samples may sit on the frame, so they are not clipped.
  bels.samples.removeAttribute("clip-path");
  const frame = node("rect", { class: "frame", x: p.x0, y: p.y0, width: w,
    height: h });
  // The wash outside the selected cell covers the other splits and the
  // estimate too, so only the selection stays in focus.
  bsvg.append(bels.regions, axes, bels.band, bels.curve, bels.cuts,
    bels.dim, bels.sticks, frame, bels.samples);

  // One picker for both modes: a regression set switches the builder to
  // MSE. Each mode keeps its own tree while the other is shown.
  // Grid cells: "Data" spans both rows; each row is a group name, then
  // its chips, so the names and the chips line up.
  const group = (label, mode, sets) => `<span
    class="group-name">${label}</span><span class="group">${Object.entries(
    sets).map(([key, s]) => `<button data-mode="${mode}" data-set="${key}"
    role="radio">${s.name}</button>`).join("")}</span>`;
  const sets = document.getElementById("b-sets");
  sets.innerHTML = `<span class="data-label
    tall">Data</span>${group("Classification", "class", CLASS_SETS)}${
    group("Regression", "reg", REG_SETS)}`;
  // Resample and the data settings take the bar's third row.
  sets.append(bTools);
  for (const b of document.querySelectorAll("#b-sets [data-set]")) {
    b.onclick = () => {
      const mode = b.dataset.mode, fresh = B_STATE[mode].set !== b.dataset.set;
      B_STATE[mode].set = b.dataset.set;
      if (mode !== bmode) setMode(mode);
      if (fresh) bRegen(true);
    };
  }
}

/** Plot key: the current set's classes, or the curves. */
function bLegend() {
  document.getElementById("b-legend").innerHTML = bmode === "class"
    ? [...Array(bC()).keys()].map(c => `<span class="key">${dot(c)}<span>
      <i>y</i> = ${c}</span></span>`).join("")
      + `<span class="key"><span class="line-key ink"></span>split</span>`
    : `<span class="key"><span class="line-key"></span><span>tree
      <i>&ycirc;</i>(<i>x</i>)</span></span>
      <span class="key"><span class="line-key dash"></span><span>true
      <i>f</i>(<i>x</i>)</span></span>
      <span class="key"><span class="line-key ink"></span>split</span>`;
}

// -------------------------------------------------------------- the plot

function rgb(c) { return `rgb(${PALE[c].join(",")})`; }

/** Outside of the selected node's cell, washed out. */
function dimPath(box, sx, sy, p) {
  const [bx, by] = box.length === 2 ? box : [box[0], VIEW.reg.ydom];
  return `<path fill-rule="evenodd" d="M${p.x0} ${p.y0}H${p.x1}V${p.y1}
    H${p.x0}Z M${sx(bx[0])} ${sy(by[1])}H${sx(bx[1])}V${sy(by[0])}
    H${sx(bx[0])}Z"/>`;
}

function drawPlaneClass() {
  const p = VIEW.class, { sx, sy } = scales(p), d = bdat(), sel = d.sel;
  let regions = "", cuts = "";
  for (const nd of bnodes) {
    const [bx, by] = nd.box;
    if (nd.feat === null) {
      regions += `<rect x="${sx(bx[0])}" y="${sy(by[1])}"
        width="${sx(bx[1]) - sx(bx[0])}" height="${sy(by[0]) - sy(by[1])}"
        fill="${rgb(nd.pred)}"/>`;
      continue;
    }
    const b = nd.box[nd.feat];
    if (nd.thr <= b[0] || nd.thr >= b[1]) continue;
    const [x1, x2, y1, y2] = nd.feat === 0
      ? [sx(nd.thr), sx(nd.thr), sy(by[0]), sy(by[1])]
      : [sx(bx[0]), sx(bx[1]), sy(nd.thr), sy(nd.thr)];
    const cls = nd === sel ? "cut sel" : "cut";
    cuts += `<line class="${cls}" x1="${x1}" x2="${x2}" y1="${y1}"
      y2="${y2}"/><line class="cut-hit" data-id="${nd.id}" x1="${x1}"
      x2="${x2}" y1="${y1}" y2="${y2}"/>`;
  }
  bels.regions.innerHTML = regions;
  const [bx, by] = sel.box;
  bels.dim.innerHTML = !sel.depth ? "" : dimPath(sel.box, sx, sy, p)
    + `<rect class="sel-box" x="${sx(bx[0])}" y="${sy(by[1])}"
      width="${sx(bx[1]) - sx(bx[0])}" height="${sy(by[0]) - sy(by[1])}"/>`;
  bels.cuts.innerHTML = cuts;
  const inSel = new Set(sel.idx);
  bels.samples.innerHTML = d.X.map((x, i) => `<circle class="s${
    inSel.has(i) ? "" : " out"}" data-i="${i}" cx="${sx(x[0]).toFixed(1)}"
    cy="${sy(x[1]).toFixed(1)}" r="6" fill="${COLORS[d.y[i]]}"/>`).join("");
}

function drawPlaneReg() {
  const p = VIEW.reg, { sx, sy } = scales(p), d = bdat(), sel = d.sel;
  const f = REG_SETS[bcur().set].f;
  let truth = "";
  for (let j = 0; j <= 300; j++) {
    const x = p.xdom[0] + j / 300 * (p.xdom[1] - p.xdom[0]);
    truth += `${j ? "L" : "M"}${sx(x).toFixed(1)} ${sy(f(x)).toFixed(1)}`;
  }
  // The leaves, left to right, are the steps of the estimate.
  let est = "";
  for (const nd of bnodes) {
    if (nd.feat !== null) continue;
    const [a, b] = nd.box[0];
    est += `${est ? "L" : "M"}${sx(a).toFixed(1)} ${sy(nd.pred).toFixed(1)}
      H${sx(b).toFixed(1)}`;
  }
  bels.curve.innerHTML = `<path class="truth" d="${truth}"/>
    <path class="est" d="${est}" stroke="${CURVE}"/>`;

  const [a, b] = sel.box[0];
  bels.band.innerHTML = `<rect class="sel-band" x="${sx(a)}" y="${p.y0}"
    width="${sx(b) - sx(a)}" height="${p.y1 - p.y0}"/>`;
  bels.dim.innerHTML = !sel.depth ? "" : dimPath(sel.box, sx, sy, p);

  // Residuals of the selected node's samples about the estimate that its
  // impurity measures: its own mean, or after a split each side's mean.
  const split = sel.feat !== null;
  const mean = i => (!split ? sel.pred
    : d.X[i][0] <= sel.thr ? sel.left.pred : sel.right.pred);
  let sticks = sel.idx.map(i => `<line class="stick"
    x1="${sx(d.X[i][0])}" x2="${sx(d.X[i][0])}" y1="${sy(d.y[i])}"
    y2="${sy(mean(i))}"/>`).join("");
  if (split) {
    const t = clamp(sel.thr, a, b);
    sticks += `<line class="before" x1="${sx(a)}" x2="${sx(b)}"
      y1="${sy(sel.pred)}" y2="${sy(sel.pred)}"/>`;
    for (const [nd, u, v] of [[sel.left, a, t], [sel.right, t, b]]) {
      if (nd.n) {
        sticks += `<line class="after" x1="${sx(u)}" x2="${sx(v)}"
          y1="${sy(nd.pred)}" y2="${sy(nd.pred)}"/>`;
      }
    }
  }
  bels.sticks.innerHTML = sticks;

  let cuts = "";
  for (const nd of bnodes) {
    if (nd.feat === null) continue;
    const [lo, hi] = nd.box[0];
    if (nd.thr <= lo || nd.thr >= hi) continue;
    // The steps already show every split; only the selected one is inked.
    const x = sx(nd.thr), cls = nd === sel ? "cut sel" : "cut faint";
    cuts += `<line class="${cls}" x1="${x}" x2="${x}" y1="${p.y0}"
      y2="${p.y1}"/><line class="cut-hit" data-id="${nd.id}" x1="${x}"
      x2="${x}" y1="${p.y0}" y2="${p.y1}"/>`;
    if (nd === sel) {
      cuts += `<path class="handle" data-id="${nd.id}"
        d="M${x} ${p.y1} l-8 13 h16 z"/>`;
    }
  }
  bels.cuts.innerHTML = cuts;
  const inSel = new Set(sel.idx);
  bels.samples.innerHTML = d.X.map((x, i) => `<circle class="s${
    inSel.has(i) ? "" : " out"}" data-i="${i}" cx="${sx(x[0]).toFixed(1)}"
    cy="${sy(d.y[i]).toFixed(1)}" r="5.5" fill="${COLORS[0]}"/>`).join("");
}

// ----------------------------------------------------------- the diagram

/** Question of a split node, as SVG text. */
function questionSvg(nd) {
  return `${xTspan(nd.feat, bD())} ≤ ${fmt(nd.thr)}`;
}

function drawDiagram({ nodes, slots, depth }) {
  const sel = bdat().sel;
  const W = Math.max(slots, 5) * T_SLOT + 2 * T_PAD;
  const H = depth * T_LEVEL + T_H + 2 * T_PAD;
  const ox = T_PAD + (W - 2 * T_PAD - slots * T_SLOT) / 2;
  const cx = nd => ox + (nd.lx + 0.5) * T_SLOT;
  const ty = nd => T_PAD + nd.depth * T_LEVEL;
  // Edges on the path from the root to the selected node are inked; nodes
  // off that path and outside the selected subtree fade.
  const onPath = new Set();
  for (let nd = sel; nd; nd = nd.parent) onPath.add(nd);
  const near = new Set(onPath);
  (function walk(nd) {
    near.add(nd);
    if (nd.feat !== null) {
      walk(nd.left);
      walk(nd.right);
    }
  })(sel);

  let edges = "", boxes = "";
  for (const nd of nodes) {
    if (nd.feat !== null) {
      for (const [ch, word] of [[nd.left, "yes"], [nd.right, "no"]]) {
        const x1 = cx(nd), y1 = ty(nd) + T_H, x2 = cx(ch), y2 = ty(ch);
        const cls = onPath.has(ch) ? "edge on" : "edge";
        edges += `<g${near.has(ch) ? "" : ` class="far"`}><line
          class="${cls}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/><text
          class="edge-label" x="${(x1 + x2) / 2
          + (word === "yes" ? -8 : 8)}" y="${(y1 + y2) / 2 + 4}"
          text-anchor="${word === "yes" ? "end" : "start"}">${word}</text></g>`;
      }
    }
    const x = cx(nd) - T_W / 2, y = ty(nd), isLeaf = nd.feat === null;
    const fill = !isLeaf ? "#fff"
      : bmode === "class" ? rgb(nd.pred) : "#ffedd5";
    let head, mid = "";
    if (!isLeaf) head = questionSvg(nd);
    else if (bmode === "class") head = `<tspan class="sym">ŷ</tspan> = ${nd.pred}`;
    else head = `<tspan class="sym">ŷ</tspan> = ${fmt(nd.pred)}`;
    if (bmode === "class") {
      // Class mix as a stacked bar.
      let u = x + 10;
      const bw = T_W - 20;
      for (let c = 0; c < nd.counts.length; c++) {
        const w = nd.n ? bw * nd.counts[c] / nd.n : 0;
        if (w > 0) {
          mid += `<rect x="${u}" y="${y + 23}" width="${w}" height="8"
            fill="${COLORS[c]}"/>`;
        }
        u += w;
      }
      if (!nd.n) {
        mid = `<rect x="${x + 10}" y="${y + 23}" width="${bw}" height="8"
          class="empty-bar"/>`;
      }
    } else if (!isLeaf) {
      mid = `<text class="t-mid" x="${cx(nd)}" y="${y + 33}"
        text-anchor="middle"><tspan class="sym">ȳ</tspan> = ${fmt(nd.pred)}
        </text>`;
    }
    boxes += `<g class="tnode${nd === sel ? " sel" : ""}${
      near.has(nd) ? "" : " far"}" data-id="${nd.id}">
      <rect x="${x}" y="${y}" width="${T_W}" height="${T_H}" rx="8"
        fill="${fill}"/>
      <text class="t-head" x="${cx(nd)}" y="${y + 17}"
        text-anchor="middle">${head}</text>${mid}
      <text class="t-foot" x="${cx(nd)}" y="${y + 47}"
        text-anchor="middle">n ${nd.n} · ${impName()} ${
        nd.n ? fmtImp(nd.imp) : "–"}</text></g>`;
  }
  tsvg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  // Natural size up to the card width; a wide tree shrinks to 85% (the
  // smallest readable text), then scrolls sideways.
  tsvg.style.width = `min(100%, ${W}px)`;
  tsvg.style.minWidth = `${Math.round(0.85 * W)}px`;
  tsvg.innerHTML = edges + boxes;
  // Keep a newly selected node in view.
  if (sel !== lastShown) {
    lastShown = sel;
    const wrap = tsvg.parentElement, k = tsvg.clientWidth / W;
    wrap.scrollLeft = cx(sel) * k - wrap.clientWidth / 2;
  }
}
let lastShown = null;

// ------------------------------------------------------------ the editor

/** HTML for a node's class mix (classification) or mean (regression). */
function mixHtml(nd) {
  if (bmode === "class") {
    return nd.counts.map((k, c) => `${dot(c)}${k}`).join(" ");
  }
  return nd.n ? fmt(nd.pred) : "–";
}

function pathHtml(sel) {
  const steps = [];
  for (let nd = sel; nd.parent; nd = nd.parent) {
    const p = nd.parent, x = xName(p.feat, bD());
    steps.unshift(nd === p.left ? `${x} ≤ ${fmt(p.thr)}`
      : `${x} &gt; ${fmt(p.thr)}`);
  }
  const where = steps.length ? steps.join(", then ") : "the root";
  return `<b>Selected:</b> ${where} <span class="muted">(${sel.n}
    sample${sel.n === 1 ? "" : "s"})</span>`;
}

/** A split button's label, with an icon for the line it draws. */
function splitLabel(f, D) {
  if (D === 1) return "Split";
  return `<span class="ico ${f ? "h" : "v"}"></span>Split on ${xName(f, D)}`;
}

function drawEditor() {
  const sel = bdat().sel, isLeaf = sel.feat === null, D = bD();
  const name = impName(), isClass = bmode === "class";
  document.getElementById("node-path").innerHTML = pathHtml(sel);
  document.getElementById("leaf-tools").hidden = !isLeaf;
  document.getElementById("split-tools").hidden = isLeaf;
  document.getElementById("feat-seg").hidden = D === 1;
  document.getElementById("scan-box").hidden = isLeaf || !sel.n;
  for (const b of document.querySelectorAll("[data-split]")) {
    b.hidden = +b.dataset.split >= D;
    b.innerHTML = splitLabel(+b.dataset.split, D);
  }
  const out = document.getElementById("node-readout");

  if (isLeaf) {
    const est = isClass ? `${dot(sel.pred)}<i>y</i> = ${sel.pred}`
      : fmt(sel.pred);
    let h = `<p class="leaf-line">${isClass ? `${mixHtml(sel)} &middot; `
      : ""}${name} <b>${sel.n ? fmtImp(sel.imp) : "–"}</b> &middot;
      estimates <i>ŷ</i> = ${est}</p>`;
    if (sel.n && sel.imp <= EPS) {
      h += `<p class="muted">Pure: no split can lower its ${name}.</p>`;
    }
    out.innerHTML = h;
    return;
  }

  for (const b of document.querySelectorAll("[data-feat]")) {
    b.setAttribute("aria-checked", +b.dataset.feat === sel.feat);
    b.innerHTML = `<span class="ico ${+b.dataset.feat ? "h" : "v"}"></span>${
      xName(+b.dataset.feat, D)} ≤ <i>t</i>`;
  }
  const x = xName(sel.feat, D), t = fmt(sel.thr), L = sel.left;
  const R = sel.right, n = sel.n;
  if (!n) {
    out.innerHTML = `<p class="muted">No training sample reaches this
      node.</p>`;
    return;
  }
  const w = (L.n * L.imp + R.n * R.imp) / n;
  const row = (label, nd) => `<tr><td>${label}</td><td>${nd.n}</td>
    <td>${mixHtml(nd)}</td><td>${nd.n ? fmtImp(nd.imp) : "–"}</td></tr>`;
  out.innerHTML = `<p class="imp-big">${name} <b>${fmtImp(sel.imp)}</b>
      <span class="arrow">&rarr;</span> <b>${fmtImp(w)}</b>
      <span class="drop">reduction ${fmtImp(sel.imp - w)}</span></p>
    <table class="imp"><tr><th></th><th>n</th><th>${isClass ? "classes"
      : "mean <i>ȳ</i>"}</th><th>${name}</th></tr>
      ${row(`Yes: ${x} ≤ ${t}`, L)}${row(`No: ${x} &gt; ${t}`, R)}</table>
    <p class="calc muted">after = ${L.n}/${n} &times; ${fmtImp(L.imp)} +
      ${R.n}/${n} &times; ${fmtImp(R.imp)} = ${fmtImp(w)}</p>`;
  drawScan(sel, w);
}

/** Weighted impurity against the threshold, for the selected split. */
function drawScan(sel, w) {
  const { X, y } = bdat(), f = sel.feat, P = SC;
  const { steps, best } = scanFeature(sel.idx, X, y, bC(), f);
  const [lo, hi] = sel.box[f];
  let top = sel.imp;
  for (const s of steps) top = Math.max(top, s.w);
  top = niceCeil(1.05 * top || 1);
  const tx = v => P.x0 + (v - lo) / (hi - lo) * (P.x1 - P.x0);
  const wy = v => P.y1 - v / top * (P.y1 - P.y0);
  // Outside the samples' range one side is empty: no change from before.
  let d = `M${tx(lo)} ${wy(sel.imp)}`;
  for (const s of steps) d += `H${tx(s.lo)}V${wy(s.w)}H${tx(s.hi)}`;
  if (steps.length) d += `V${wy(sel.imp)}`;
  d += `H${tx(hi)}`;

  let h = "";
  for (const v of [0, top / 2, top]) {
    h += `<line class="grid" x1="${P.x0}" x2="${P.x1}" y1="${wy(v)}"
      y2="${wy(v)}"/><text class="tick" x="${P.x0 - 5}" y="${wy(v) + 4}"
      text-anchor="end">${+v.toFixed(3)}</text>`;
  }
  const span = hi - lo, stepT = niceCeil(span / 5);
  for (let v = Math.ceil(lo / stepT) * stepT; v <= hi + 1e-9; v += stepT) {
    h += `<text class="tick" x="${tx(v)}" y="${P.y1 + 14}"
      text-anchor="middle">${fmt(v, stepT < 0.1 ? 2 : 1)}</text>`;
  }
  const t = clamp(sel.thr, lo, hi);
  h += `<rect class="frame" x="${P.x0}" y="${P.y0}" width="${P.x1 - P.x0}"
      height="${P.y1 - P.y0}"/>
    <line class="c-before" x1="${P.x0}" x2="${P.x1}" y1="${wy(sel.imp)}"
      y2="${wy(sel.imp)}"/>
    <path class="c-scan" d="${d}"/>
    <text class="axis-label" x="${(P.x0 + P.x1) / 2}" y="${P.y1 + 34}"
      text-anchor="middle">threshold <tspan class="sym">t</tspan> on
      ${xTspan(f, bD())}</text>
    <text class="axis-label" text-anchor="middle" transform="translate(13
      ${(P.y0 + P.y1) / 2}) rotate(-90)">${impName()} after</text>`;
  if (best) {
    h += `<circle class="c-best" cx="${tx(best.thr)}" cy="${wy(best.w)}"
      r="5.5"/>`;
  }
  h += `<line class="c-k" x1="${tx(t)}" x2="${tx(t)}" y1="${P.y0}"
      y2="${P.y1}"/><circle class="c-dot" cx="${tx(t)}" cy="${wy(w)}"
      r="4"/>`;
  scan.innerHTML = h;
  scan.dataset.lo = lo;
  scan.dataset.hi = hi;

  document.getElementById("scan-note").innerHTML = best
    ? `Every threshold, against the ${impName()} before (dashed). Green
      ring: the lowest. Drag here or on the plot.`
    : `Every sample here has the same ${xName(f, bD())}: no threshold
      separates them.`;
}

// -------------------------------------------------------------- controls

const bnInput = document.getElementById("b-n");
const bnoiseInput = document.getElementById("b-noise");

function bUpdateControls() {
  const s = bcur(), d = bdat();
  for (const b of document.querySelectorAll("#b-sets [data-set]")) {
    b.setAttribute("aria-checked",
      b.dataset.mode === bmode && b.dataset.set === s.set);
  }
  [bnInput.min, bnInput.max] = B_N_RANGE[bmode];
  bnInput.value = s.n;
  document.getElementById("b-nval").textContent = bmode === "class"
    ? `${s.n} per class (${d.y.length} in all)` : `${d.y.length}`;
  bnoiseInput.value = s.noise;
  document.getElementById("b-noiseval").textContent = s.noise.toFixed(2);

  // Training error of the whole tree.
  let leaves = 0, wrong = 0, sse = 0, depth = 0;
  for (const nd of bnodes) {
    depth = Math.max(depth, nd.depth);
    if (nd.feat !== null) continue;
    leaves++;
    if (bmode === "class") wrong += nd.n - (nd.n ? nd.counts[nd.pred] : 0);
    else sse += nd.n * nd.imp;
  }
  const N = d.y.length;
  const err = bmode === "class"
    ? `<b>${pct(wrong / N)}</b> (${wrong} of ${N})` : `<b>${fmtImp(sse / N)}</b>`;
  document.getElementById("tree-readout").innerHTML = `Training
    ${bmode === "class" ? "error" : "MSE"} ${err} &middot; ${leaves}
    ${leaves === 1 ? "leaf" : "leaves"}, depth ${depth}`;
}

function select(nd) {
  bdat().sel = nd;
  bInvalidate();
}

function setThr(v) {
  const sel = bdat().sel;
  if (sel.feat === null) return;
  const [lo, hi] = sel.box[sel.feat], e = (hi - lo) * 1e-4;
  sel.thr = clamp(v, lo + e, hi - e);
  bInvalidate();
}

for (const b of document.querySelectorAll("[data-split]")) {
  b.onclick = () => {
    const sel = bdat().sel, f = +b.dataset.split;
    setSplit(sel, f, medianThr(sel, f));
    bInvalidate();
  };
}
document.getElementById("best-split").onclick = () => {
  const sel = bdat().sel, b = bestHere(sel);
  if (b) setSplit(sel, b.feat, b.thr);
  bInvalidate();
};
for (const b of document.querySelectorAll("[data-feat]")) {
  b.onclick = () => {
    const sel = bdat().sel, f = +b.dataset.feat;
    if (sel.feat === f) return;
    setSplit(sel, f, medianThr(sel, f));
    bInvalidate();
  };
}
document.getElementById("prune").onclick = () => {
  Object.assign(bdat().sel, leaf());
  bInvalidate();
};
document.getElementById("grow").onclick = () => {
  growLevel();
  bInvalidate();
};
document.getElementById("clear").onclick = () => {
  const d = bdat();
  d.root = d.sel = leaf();
  bInvalidate();
};
bnInput.oninput = () => { bcur().n = +bnInput.value; bRegen(false); };
bnoiseInput.oninput = () => {
  bcur().noise = +bnoiseInput.value;
  bRegen(false);
};
document.getElementById("b-resample").onclick = () => {
  bcur().seed += 1;
  bRegen(false);
};

tsvg.addEventListener("click", e => {
  const g = e.target.closest(".tnode");
  if (g) select(bnodes[+g.dataset.id]);
});

// Dragging on the plot: a sample moves it; a split line (or the handle
// under it) selects that node and moves its threshold; a click anywhere
// else selects the leaf there. Capture on the SVG survives the redraw.
let bDrag = null;

function bDragTo(e) {
  const p = svgPoint(bsvg, e), v = VIEW[bmode], { ix, iy } = scales(v);
  const x = clamp(ix(p.x), ...v.xdom), y = clamp(iy(p.y), ...v.ydom);
  const d = bdat();
  if (bDrag.thr) {
    setThr(d.sel.feat === 0 ? x : y);
    return;
  }
  d.X[bDrag.i] = bmode === "class" ? [x, y] : [x];
  if (bmode === "reg") d.y[bDrag.i] = y;
  bInvalidate();
}

bsvg.addEventListener("pointerdown", e => {
  if (e.button !== 0) return;
  const s = e.target.closest(".s"), cut = e.target.closest("[data-id]");
  if (s) {
    bDrag = { i: +s.dataset.i };
  } else if (cut) {
    bdat().sel = bnodes[+cut.dataset.id];
    bDrag = { thr: true };
  } else {
    const p = svgPoint(bsvg, e), { ix, iy } = scales(VIEW[bmode]);
    select(leafOf(bdat().root, [ix(p.x), iy(p.y)]));
    return;
  }
  bsvg.setPointerCapture(e.pointerId);
  e.preventDefault();
  bDragTo(e);
});
bsvg.addEventListener("pointermove", e => { if (bDrag) bDragTo(e); });
bsvg.addEventListener("pointerup", () => { bDrag = null; });
bsvg.addEventListener("pointercancel", () => { bDrag = null; });

// The scan chart sets the threshold from the pointer's position.
let scanDrag = false;
function scanThr(e) {
  const p = svgPoint(scan, e), lo = +scan.dataset.lo, hi = +scan.dataset.hi;
  setThr(lo + (p.x - SC.x0) / (SC.x1 - SC.x0) * (hi - lo));
}
scan.addEventListener("pointerdown", e => {
  scanDrag = true;
  scan.setPointerCapture(e.pointerId);
  scanThr(e);
});
scan.addEventListener("pointermove", e => { if (scanDrag) scanThr(e); });
scan.addEventListener("pointerup", () => { scanDrag = false; });

/** Show the builder in mode "class" or "reg". */
function builderShow(mode) {
  bmode = mode;
  bBuild();
  if (!bdat().root) bRegen(true);
  else bInvalidate();
}
