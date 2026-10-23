// Random forest tab. Each tree is grown to full depth (or a max depth) on
// its own bootstrap bag of the training samples, and each split searches
// one feature drawn at random; the forest estimates by majority vote
// (Breiman 2001). The plot shades the vote, fading where the trees
// disagree. Thumbnails show the first trees with their bags; the chart
// tracks the testing error as trees join the vote. All T_MAX trees are
// grown at once, so the Trees slider only changes how many vote.

const F = { set: "moons", n: 40, noise: 0.6, seed: 1, T: 100, depth: 0,
  bootstrap: true, randFeat: true };
const T_MAX = 200, F_TEST = 1200, N_THUMB = 12, DEPTH_MAX = 12;
const F_GRID = 120, F_GRID_FAST = 50;
const F_N_RANGE = [10, 120];
// Chart plot box, in its 340 x 176 viewBox.
const FC = { x0: 46, x1: 330, y0: 10, y1: 140 };

// Training samples (plotted, draggable) and a large fresh test draw from
// the same source (not plotted), so the testing error is not noise.
let ftrain, ftest;
// trees[j]: {root, bag (n_train,) times each sample was drawn, grid
// (G*G,) estimate per plot cell, test (n_test,) estimate per test sample}.
let trees = [], fG = F_GRID;
// errTest, errTrain (T_MAX+1,): forest error with the first T trees;
// errOne: testing error of one tree grown on all the training samples.
let errTest, errTrain, errOne;
// Index of the tree shown alone, or null for the forest; pointer position.
let focus = null, hover = null;

const fsvg = document.getElementById("f-stage");
const fchart = document.getElementById("f-chart");
const fels = {};
const fcanvas = document.createElement("canvas");

function fC() { return CLASS_SETS[F.set].C; }

// ------------------------------------------------------------------ data

function fRegen() {
  ftrain = makeClass(F.set, F.n, F.noise, F.seed);
  ftest = makeClass(F.set, F_TEST / fC(), F.noise, F.seed + 5003);
  fInvalidate({ fit: true });
}

/** Grow every tree and score it on the grid and the samples. */
function fFit(fast) {
  const C = fC(), { X, y } = ftrain, n = y.length, p = VIEW.class;
  const G = fG = fast ? F_GRID_FAST : F_GRID;
  const all = y.map((_, i) => i), q = [0, 0];
  trees = [];
  for (let j = 0; j < T_MAX; j++) {
    // Tree j's own stream, so toggles and the slider never reshuffle it.
    const r = rng(F.seed * 100003 + j * 7919 + 17);
    const bag = new Uint16Array(n);
    let idx = all;
    if (F.bootstrap) {
      idx = [];
      for (let k = 0; k < n; k++) {
        const i = Math.floor(r() * n);
        idx.push(i);
        bag[i]++;
      }
    } else {
      bag.fill(1);
    }
    const root = fitTree(idx, X, y, C, F.depth || Infinity,
      F.randFeat ? r : null);
    const grid = new Uint8Array(G * G);
    for (let gy = 0; gy < G; gy++) {
      q[1] = p.ydom[1] - (gy + 0.5) / G * (p.ydom[1] - p.ydom[0]);
      for (let gx = 0; gx < G; gx++) {
        q[0] = p.xdom[0] + (gx + 0.5) / G * (p.xdom[1] - p.xdom[0]);
        grid[gy * G + gx] = leafOf(root, q).pred;
      }
    }
    const test = Uint8Array.from(ftest.X, x => leafOf(root, x).pred);
    const train = Uint8Array.from(X, x => leafOf(root, x).pred);
    trees.push({ root, bag, grid, test, train });
  }
  errTest = voteErrors(ftest.y, "test");
  errTrain = voteErrors(y, "train");
  // The tree one would grow without a forest: every sample, every feature.
  const solo = fitTree(all, X, y, C, F.depth || Infinity, null);
  errOne = 0;
  for (let m = 0; m < ftest.y.length; m++) {
    errOne += (leafOf(solo, ftest.X[m]).pred !== ftest.y[m]) / ftest.y.length;
  }
}

/**
 * Error of the majority vote of the first T trees, for every T.
 *
 * Args:
 *   key (string): "test" or "train", which stored estimates to count
 *
 * Returns:
 *   err (Float64Array): (T_MAX+1,) err[T] error rate; err[0] unused
 */
function voteErrors(truth, key) {
  const m = truth.length, C = fC(), err = new Float64Array(T_MAX + 1);
  const counts = Array.from({ length: m }, () => new Array(C).fill(0));
  for (let T = 1; T <= T_MAX; T++) {
    const est = trees[T - 1][key];
    let w = 0;
    for (let i = 0; i < m; i++) {
      counts[i][est[i]]++;
      w += argmax(counts[i]) !== truth[i];
    }
    err[T] = w / m;
  }
  return err;
}

/** Votes of the first T trees at point x: (C,) counts. */
function votesAt(x) {
  const counts = new Array(fC()).fill(0);
  for (let j = 0; j < F.T; j++) counts[leafOf(trees[j].root, x).pred]++;
  return counts;
}

// ------------------------------------------------------------- scheduling

const ftodo = { fit: false, fast: false };
let fQueued = false;

function fInvalidate({ fit = false, fast = false } = {}) {
  ftodo.fast = fQueued ? ftodo.fast && fast : fast;
  ftodo.fit = ftodo.fit || fit;
  if (fQueued) return;
  fQueued = true;
  requestAnimationFrame(() => {
    fQueued = false;
    fDraw();
  });
}

function fDraw() {
  if (state.mode !== "forest") return;
  if (ftodo.fit) fFit(ftodo.fast);
  ftodo.fit = false;
  drawForestRegions();
  drawForestSamples();
  drawThumbs();
  drawVote();
  drawFChart();
  drawImportance();
  fUpdateControls();
}

// ------------------------------------------------------------------ build

function fBuild() {
  const p = VIEW.class, w = p.x1 - p.x0, h = p.y1 - p.y0;
  fsvg.setAttribute("viewBox", `0 0 ${p.w} ${p.h}`);
  fsvg.innerHTML = "";
  const axes = node("g", { class: "axes" });
  axes.innerHTML = axesSvg(p, "class");
  fels.region = node("image", { x: p.x0, y: p.y0, width: w, height: h,
    preserveAspectRatio: "none" });
  fels.train = node("g", { class: "train" });
  fels.hover = node("g", { class: "hover" });
  const frame = node("rect", { class: "frame", x: p.x0, y: p.y0, width: w,
    height: h });
  fsvg.append(fels.region, axes, frame, fels.train, fels.hover);

  document.getElementById("f-sets").innerHTML = Object.entries(CLASS_SETS)
    .map(([key, s]) => `<button data-set="${key}" role="radio">${s.name}
      </button>`).join("");
  for (const b of document.querySelectorAll("#f-sets [data-set]")) {
    b.onclick = () => {
      F.set = b.dataset.set;
      focus = null;
      fRegen();
    };
  }

  const th = document.getElementById("thumbs");
  th.innerHTML = "";
  for (let j = 0; j < N_THUMB; j++) {
    const b = document.createElement("button");
    b.className = "thumb";
    b.innerHTML = `<canvas width="${F_GRID}" height="${F_GRID}"></canvas>
      <span></span>`;
    b.onclick = () => {
      focus = focus === j ? null : j;
      fInvalidate();
    };
    th.append(b);
  }
}

// ------------------------------------------------------------------ draw

/** Vote shading: the winner's shade, washed toward white by disagreement. */
function drawForestRegions() {
  const G = fG, C = fC(), cells = G * G;
  fcanvas.width = fcanvas.height = G;
  const ctx = fcanvas.getContext("2d"), img = ctx.createImageData(G, G);
  if (focus !== null) {
    const g = trees[focus].grid;
    for (let k = 0; k < cells; k++) img.data.set([...PALE[g[k]], 255], 4 * k);
  } else {
    const counts = new Uint16Array(cells * C);
    for (let j = 0; j < F.T; j++) {
      const g = trees[j].grid;
      for (let k = 0; k < cells; k++) counts[k * C + g[k]]++;
    }
    for (let k = 0; k < cells; k++) {
      let w = 0, top = -1, second = 0;
      for (let c = 0; c < C; c++) {
        const v = counts[k * C + c];
        if (v > top) {
          second = top;
          top = v;
          w = c;
        } else if (v > second) {
          second = v;
        }
      }
      // Margin 0 (a tie) keeps a fifth of the shade, a sweep keeps it all.
      const a = 0.2 + 0.8 * (top - Math.max(second, 0)) / F.T;
      const px = PALE[w].map(v => Math.round(255 - (255 - v) * a));
      img.data.set([...px, 255], 4 * k);
    }
  }
  ctx.putImageData(img, 0, 0);
  fels.region.setAttribute("href", fcanvas.toDataURL());
}

function drawForestSamples() {
  const { sx, sy } = scales(VIEW.class);
  const bag = focus !== null ? trees[focus].bag : null;
  // Shown alone, a tree's samples grow with the times its bag drew them;
  // those it never drew (out of bag) fade.
  fels.train.innerHTML = ftrain.X.map((x, i) => {
    const k = bag ? bag[i] : 1;
    const r = k ? 5 + 2.2 * (k - 1) : 4;
    const label = k > 1 ? `<text class="bag-n" x="${sx(x[0])}"
      y="${sy(x[1]) + 3.5}" text-anchor="middle">${k}</text>` : "";
    return `<circle class="s${k ? "" : " oob"}" data-i="${i}"
      cx="${sx(x[0]).toFixed(1)}" cy="${sy(x[1]).toFixed(1)}" r="${r}"
      fill="${COLORS[ftrain.y[i]]}"/>${label}`;
  }).join("");
  fels.hover.innerHTML = hover ? `<g transform="translate(${sx(hover[0])}
    ${sy(hover[1])})"><circle class="q-mark" r="7"/><circle r="2"
    class="q-dot"/></g>` : "";

  let h = [...Array(fC()).keys()].map(c => `<span class="key">${dot(c)}
    <span><i>y</i> = ${c}</span></span>`).join("")
    + `<span class="key"><span class="dot fill"></span>training</span>`;
  if (focus !== null) {
    h += `<span class="focus-note">Tree ${focus + 1} alone: size = times
      in its bag, faded = out of bag <button id="unfocus">Show the
      forest</button></span>`;
  } else {
    h += `<span class="key"><span class="fade-key"></span>the trees
      disagree</span>`;
  }
  document.getElementById("f-legend").innerHTML = h;
  const u = document.getElementById("unfocus");
  if (u) u.onclick = () => { focus = null; fInvalidate(); };
}

function drawThumbs() {
  const G = fG, { X, y } = ftrain, p = VIEW.class, k = G / F_GRID;
  const tx = v => (v - p.xdom[0]) / (p.xdom[1] - p.xdom[0]) * G;
  const ty = v => (p.ydom[1] - v) / (p.ydom[1] - p.ydom[0]) * G;
  const hv = hover ? votesAtEach(hover) : null;
  document.querySelectorAll(".thumb").forEach((b, j) => {
    const t = trees[j], cv = b.querySelector("canvas");
    cv.width = cv.height = G;
    const ctx = cv.getContext("2d"), img = ctx.createImageData(G, G);
    for (let c = 0; c < G * G; c++) {
      img.data.set([...PALE[t.grid[c]], 255], 4 * c);
    }
    ctx.putImageData(img, 0, 0);
    for (let i = 0; i < X.length; i++) {
      if (!t.bag[i]) continue;
      ctx.beginPath();
      ctx.arc(tx(X[i][0]), ty(X[i][1]),
        k * (2.2 + 1.3 * (t.bag[i] - 1)), 0, 2 * Math.PI);
      ctx.fillStyle = COLORS[y[i]];
      ctx.fill();
    }
    let w = 0;
    for (let m = 0; m < ftest.y.length; m++) w += t.test[m] !== ftest.y[m];
    b.querySelector("span").innerHTML = `Tree ${j + 1} <span
      class="muted">test ${pct(w / ftest.y.length)}</span>`;
    b.classList.toggle("off", j >= F.T);
    b.classList.toggle("focused", j === focus);
    b.style.borderColor = hv && j < F.T ? COLORS[hv[j]] : "";
  });
}

/** Estimate of each of the first N_THUMB trees at point x. */
function votesAtEach(x) {
  return trees.slice(0, N_THUMB).map(t => leafOf(t.root, x).pred);
}

function drawVote() {
  const el = document.getElementById("f-vote");
  if (!hover) {
    el.innerHTML = `<h3>The vote</h3><p class="muted">Point at the plot to
      see how the ${F.T} trees vote there.</p>`;
    return;
  }
  const counts = votesAt(hover), w = argmax(counts);
  const bar = counts.map((k, c) => k ? `<span style="flex:${k};background:${
    COLORS[c]}">${k}</span>` : "").join("");
  el.innerHTML = `<h3>The vote at (${fmt(hover[0])}, ${fmt(hover[1])})</h3>
    <div class="votebar">${bar}</div>
    <div class="row"><span>Forest estimate <i>ŷ</i></span><span
      class="val">${dot(w)}<i>y</i> = ${w} <span class="muted">(${counts[w]}
      of ${F.T} trees)</span></span></div>`;
}

function drawImportance() {
  const g = [0, 0];
  for (let j = 0; j < F.T; j++) {
    importance(trees[j].root, 2).forEach((v, f) => { g[f] += v / F.T; });
  }
  document.getElementById("f-imp").innerHTML = `<h3>Feature
    importance</h3>${g.map((v, f) => `<div class="imp-row"><span>${
    xName(f, 2)}</span><span class="imp-bar"><span style="width:${
    100 * v}%"></span></span><span class="val">${pct(v)}</span></div>`)
    .join("")}<p class="muted">Each tree's share of its Gini reduction made
    by splits on that feature, averaged over the ${F.T} trees.</p>`;
}

function drawFChart() {
  const P = FC, T = F.T;
  const kx = j => P.x0 + Math.log(j) / Math.log(T_MAX) * (P.x1 - P.x0);
  let top = errOne;
  for (let j = 1; j <= T_MAX; j++) top = Math.max(top, errTest[j]);
  top = niceCeil(1.05 * top || 0.05);
  const ey = v => P.y1 - Math.min(v, top) / top * (P.y1 - P.y0);
  const path = err => {
    let d = "";
    for (let j = 1; j <= T_MAX; j++) {
      d += `${j > 1 ? "L" : "M"}${kx(j).toFixed(1)} ${ey(err[j]).toFixed(1)}`;
    }
    return d;
  };
  let h = "";
  for (const t of [1, 2, 5, 10, 20, 50, 100, 200]) {
    h += `<line class="grid" x1="${kx(t)}" x2="${kx(t)}" y1="${P.y0}"
      y2="${P.y1}"/><text class="tick" x="${kx(t)}" y="${P.y1 + 14}"
      text-anchor="middle">${t}</text>`;
  }
  for (const v of [0, top / 2, top]) {
    h += `<line class="grid" x1="${P.x0}" x2="${P.x1}" y1="${ey(v)}"
      y2="${ey(v)}"/><text class="tick" x="${P.x0 - 5}" y="${ey(v) + 4}"
      text-anchor="end">${+(100 * v).toFixed(1)}%</text>`;
  }
  h += `<rect class="frame" x="${P.x0}" y="${P.y0}" width="${P.x1 - P.x0}"
      height="${P.y1 - P.y0}"/>
    <text class="axis-label" x="${(P.x0 + P.x1) / 2}" y="${P.y1 + 32}"
      text-anchor="middle">trees in the vote (log scale)</text>
    <line class="c-one" x1="${P.x0}" x2="${P.x1}" y1="${ey(errOne)}"
      y2="${ey(errOne)}"/>
    <path class="c-train" d="${path(errTrain)}"/>
    <path class="c-test" d="${path(errTest)}"/>
    <line class="c-k" x1="${kx(T)}" x2="${kx(T)}" y1="${P.y0}"
      y2="${P.y1}"/>
    <circle class="c-dot train" cx="${kx(T)}" cy="${ey(errTrain[T])}"
      r="3.5"/>
    <circle class="c-dot" cx="${kx(T)}" cy="${ey(errTest[T])}" r="3.5"/>`;
  fchart.innerHTML = h;
  document.getElementById("f-chart-title").innerHTML = `Error against
    trees<br><span class="lg test"></span>testing
    <span class="lg train"></span>training
    <span class="lg one"></span>one tree (testing)`;
  document.getElementById("f-errs").innerHTML = `<div class="row"><span>
      Forest of ${T}: testing error</span>
      <span class="val">${pct(errTest[T])}</span></div>
    <div class="row"><span>Forest of ${T}: training error</span>
      <span class="val">${pct(errTrain[T])}</span></div>
    <div class="row"><span>One tree on all samples: testing error</span>
      <span class="val">${pct(errOne)}</span></div>`;
}

// -------------------------------------------------------------- controls

const ftInput = document.getElementById("f-trees");
const fdInput = document.getElementById("f-depth");
const fnInput = document.getElementById("f-n");
const fnoiseInput = document.getElementById("f-noise");
// The Trees slider runs 0..T_RES on a log scale: position t is
// T = T_MAX^(t/T_RES).
const T_RES = 1000;

function fUpdateControls() {
  for (const b of document.querySelectorAll("#f-sets [data-set]")) {
    b.setAttribute("aria-checked", b.dataset.set === F.set);
  }
  ftInput.max = T_RES;
  ftInput.value = Math.round(T_RES * Math.log(F.T) / Math.log(T_MAX));
  document.getElementById("f-treesval").textContent = F.T === 1
    ? "1 tree" : `${F.T} trees`;
  fdInput.max = DEPTH_MAX + 1;
  fdInput.value = F.depth || DEPTH_MAX + 1;
  document.getElementById("f-depthval").textContent = F.depth
    ? `${F.depth}` : "full (until pure)";
  fnInput.min = F_N_RANGE[0];
  fnInput.max = F_N_RANGE[1];
  fnInput.value = F.n;
  document.getElementById("f-nval").textContent =
    `${F.n} per class (${ftrain.y.length} in all)`;
  fnoiseInput.value = F.noise;
  document.getElementById("f-noiseval").textContent = F.noise.toFixed(2);
  document.getElementById("f-boot").checked = F.bootstrap;
  document.getElementById("f-rand").checked = F.randFeat;
}

function setT(T) {
  F.T = clamp(Math.round(T), 1, T_MAX);
  fInvalidate();
}

ftInput.oninput = () => setT(T_MAX ** (+ftInput.value / T_RES));
fdInput.oninput = () => {
  F.depth = +fdInput.value > DEPTH_MAX ? 0 : +fdInput.value;
  fInvalidate({ fit: true });
};
fnInput.oninput = () => { F.n = +fnInput.value; fRegen(); };
fnoiseInput.oninput = () => { F.noise = +fnoiseInput.value; fRegen(); };
document.getElementById("f-boot").onchange = e => {
  F.bootstrap = e.target.checked;
  fInvalidate({ fit: true });
};
document.getElementById("f-rand").onchange = e => {
  F.randFeat = e.target.checked;
  fInvalidate({ fit: true });
};
document.getElementById("f-resample").onclick = () => {
  F.seed += 1;
  fRegen();
};
document.getElementById("f-reset").onclick = fRegen;

// Dragging a training sample moves it and regrows the forest (coarsely
// until release); moving the pointer anywhere shows the vote there.
let fDrag = null;

function fPoint(e) {
  const p = svgPoint(fsvg, e), v = VIEW.class, { ix, iy } = scales(v);
  return [clamp(ix(p.x), ...v.xdom), clamp(iy(p.y), ...v.ydom)];
}

fsvg.addEventListener("pointerdown", e => {
  const s = e.target.closest(".s");
  if (e.button !== 0 || !s) return;
  fDrag = { i: +s.dataset.i };
  fsvg.setPointerCapture(e.pointerId);
  e.preventDefault();
});
fsvg.addEventListener("pointermove", e => {
  const x = fPoint(e);
  if (fDrag) {
    const i = fDrag.i;
    ftrain.X[i] = x;
    fInvalidate({ fit: true, fast: true });
  } else {
    hover = x;
    fInvalidate();
  }
});
function fEndDrag() {
  if (fDrag) fInvalidate({ fit: true });
  fDrag = null;
}
fsvg.addEventListener("pointerup", fEndDrag);
fsvg.addEventListener("pointercancel", fEndDrag);

let fChartDrag = false;
function chartT(e) {
  const p = svgPoint(fchart, e);
  const t = clamp((p.x - FC.x0) / (FC.x1 - FC.x0), 0, 1);
  setT(Math.exp(t * Math.log(T_MAX)));
}
fchart.addEventListener("pointerdown", e => {
  fChartDrag = true;
  fchart.setPointerCapture(e.pointerId);
  chartT(e);
});
fchart.addEventListener("pointermove", e => { if (fChartDrag) chartT(e); });
fchart.addEventListener("pointerup", () => { fChartDrag = false; });

function forestShow() {
  fBuild();
  if (!ftrain) fRegen();
  else fInvalidate();
}
