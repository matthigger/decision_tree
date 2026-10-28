// Shared drawing helpers: class colors, plane geometry and axes, SVG
// element building, pointer position and number formatting.

const SVGNS = "http://www.w3.org/2000/svg";
const COLORS = ["#2563eb", "#ea580c", "#9333ea"];
// Region shade per class (RGB).
const PALE = [[191, 219, 254], [254, 215, 170], [233, 213, 255]];
const CURVE = "#ea580c";

// Plot boxes in viewBox units, and the data domain each one shows.
const VIEW = {
  class: { w: 700, h: 700, x0: 46, x1: 686, y0: 14, y1: 654,
    xdom: [-1.15, 1.15], ydom: [-1.15, 1.15] },
  reg: { w: 700, h: 470, x0: 52, x1: 686, y0: 12, y1: 410,
    xdom: [-0.1, 1.1], ydom: [-2.2, 2.2] },
};

/** Scales of plot box p: data to viewBox (sx, sy) and back (ix, iy). */
function scales(p) {
  const kx = (p.x1 - p.x0) / (p.xdom[1] - p.xdom[0]);
  const ky = (p.y1 - p.y0) / (p.ydom[1] - p.ydom[0]);
  return {
    sx: v => p.x0 + (v - p.xdom[0]) * kx,
    sy: v => p.y1 - (v - p.ydom[0]) * ky,
    ix: px => p.xdom[0] + (px - p.x0) / kx,
    iy: py => p.ydom[0] + (p.y1 - py) / ky,
  };
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function node(tag, attrs = {}) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

/** Pointer position in the viewBox units of an SVG. */
function svgPoint(el, e) {
  const pt = el.createSVGPoint();
  pt.x = e.clientX;
  pt.y = e.clientY;
  return pt.matrixTransform(el.getScreenCTM().inverse());
}

/** Number with a true minus sign. */
function fmt(x, digits = 2) {
  const a = Math.abs(x).toFixed(digits);
  return (x < 0 && a !== (0).toFixed(digits) ? "−" : "") + a;
}

/** Percentage, one decimal only when it is under 10% and not whole. */
function pct(v) {
  const p = 100 * v;
  return p.toFixed(p < 10 && Math.round(10 * p) % 10 ? 1 : 0) + "%";
}

function dot(c) {
  return `<span class="dot" style="background:${COLORS[c]}"></span>`;
}

/** Feature name as HTML: x with subscript f + 1, or bare x in 1-d. */
function xName(f, d) {
  return d === 1 ? "<i>x</i>" : `<i>x</i><sub>${f + 1}</sub>`;
}

/** Feature name as SVG text: italic x with a Unicode subscript. */
function xTspan(f, d) {
  return `<tspan class="sym">x</tspan>${d === 1 ? "" : "₁₂"[f]}`;
}

/**
 * Grid lines, ticks and axis labels of plot box p.
 *
 * Args:
 *   kind (string): "class" (x1 against x2) or "reg" (y against x)
 */
function axesSvg(p, kind) {
  const { sx, sy } = scales(p);
  let a = "";
  if (kind === "class") {
    for (const v of [-1, 0, 1]) {
      a += `<line class="grid" x1="${sx(v)}" x2="${sx(v)}" y1="${p.y0}"
        y2="${p.y1}"/><line class="grid" x1="${p.x0}" x2="${p.x1}"
        y1="${sy(v)}" y2="${sy(v)}"/>
        <text class="tick" x="${sx(v)}" y="${p.y1 + 16}"
        text-anchor="middle">${fmt(v, 0)}</text>
        <text class="tick" x="${p.x0 - 7}" y="${sy(v) + 4}"
        text-anchor="end">${fmt(v, 0)}</text>`;
    }
    a += `<text class="axis-label" x="${(p.x0 + p.x1) / 2}" y="${p.y1 + 36}"
      text-anchor="middle">${xTspan(0, 2)}</text>
      <text class="axis-label" text-anchor="middle"
      transform="translate(16 ${(p.y0 + p.y1) / 2}) rotate(-90)"
      >${xTspan(1, 2)}</text>`;
  } else {
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      a += `<line class="grid" x1="${sx(v)}" x2="${sx(v)}" y1="${p.y0}"
        y2="${p.y1}"/><text class="tick" x="${sx(v)}" y="${p.y1 + 18}"
        text-anchor="middle">${v}</text>`;
    }
    for (const v of [-2, -1, 0, 1, 2]) {
      a += `<line class="grid" x1="${p.x0}" x2="${p.x1}" y1="${sy(v)}"
        y2="${sy(v)}"/><text class="tick" x="${p.x0 - 8}" y="${sy(v) + 4}"
        text-anchor="end">${fmt(v, 0)}</text>`;
    }
    a += `<text class="axis-label" x="${(p.x0 + p.x1) / 2}" y="${p.y1 + 44}"
      text-anchor="middle"><tspan class="sym">x</tspan></text>
      <text class="axis-label" x="22" y="${(p.y0 + p.y1) / 2 + 5}"
      text-anchor="middle"><tspan class="sym">y</tspan></text>`;
  }
  return a;
}

/** Round v up to 1, 2, 2.5 or 5 times a power of ten. */
function niceCeil(v) {
  if (!(v > 0)) return 1;
  const e = 10 ** Math.floor(Math.log10(v));
  for (const m of [1, 2, 2.5, 5, 10]) if (v <= m * e + 1e-12) return m * e;
}

// Footer build stamp: the deployed commit (linked) and build time.
(function () {
  const el = document.getElementById("build");
  const repo = "https://github.com/matthigger/decision_tree";
  if (!BUILD) { el.textContent = "local copy"; return; }
  const when = new Date(BUILD.time).toLocaleString("en-US", {
    dateStyle: "medium", timeStyle: "short" });
  el.innerHTML = `build <a href="${repo}/commit/${BUILD.sha}">${
    BUILD.sha.slice(0, 7)}</a>, ${when}`;
})();
