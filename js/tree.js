// Decision trees: Gini impurity (classification, labels 0..C-1) or MSE
// (regression, C = 0), the sweep over every threshold of a feature, greedy
// growing, prediction and feature importance. X (n, d) holds the features
// and y (n,) the labels; idx lists the samples reaching a node and may
// repeat one (a bootstrap bag). A split asks "x_f <= thr?": yes goes left.

// Impurity at or below this counts as pure (guards float round-off).
const EPS = 1e-12;

/** Index of the largest entry, the first one on a tie. */
function argmax(a) {
  let w = 0;
  for (let c = 1; c < a.length; c++) if (a[c] > a[w]) w = c;
  return w;
}

/**
 * Impurity and estimate of the samples idx.
 *
 * Returns:
 *   {n, counts, imp, pred}: imp the Gini impurity 1 - sum_c p_c^2 or the
 *     MSE about the mean; pred the majority class (first on a tie) or the
 *     mean label; counts (C,) per class, null for regression. An empty
 *     node has imp 0 and pred null.
 */
function nodeStats(idx, y, C) {
  const n = idx.length;
  if (C) {
    const counts = new Array(C).fill(0);
    for (const i of idx) counts[y[i]]++;
    let imp = n ? 1 : 0;
    for (const k of counts) if (n) imp -= (k / n) ** 2;
    return { n, counts, imp, pred: n ? argmax(counts) : null };
  }
  let s = 0;
  for (const i of idx) s += y[i];
  const mean = s / n;
  let ss = 0;
  for (const i of idx) ss += (y[i] - mean) ** 2;
  return { n, counts: null, imp: n ? ss / n : 0, pred: n ? mean : null };
}

/**
 * Weighted impurity after splitting idx on feature f, at every threshold.
 *
 * Weighted impurity is (n_yes imp_yes + n_no imp_no) / n. Between two
 * neighboring distinct values a < b, every thr in [a, b) splits alike.
 *
 * Returns:
 *   {steps, best}: steps [{lo, hi, w}], w for thr in [lo, hi), in
 *     increasing order; best {feat, thr, w} the lowest w (leftmost on a
 *     tie), thr the midpoint (a + b) / 2, or null if every value ties
 */
function scanFeature(idx, X, y, C, f) {
  const o = idx.slice().sort((a, b) => X[a][f] - X[b][f]);
  const n = o.length, steps = [];
  let best = null;
  // Running per-side sums: class counts and their sum of squares
  // (classification), or label sum and sum of squares (regression).
  const L = new Array(C).fill(0), R = new Array(C).fill(0);
  let sqL = 0, sqR = 0, sL = 0, sR = 0, ssL = 0, ssR = 0;
  for (const i of o) {
    if (C) R[y[i]]++;
    sR += y[i];
    ssR += y[i] * y[i];
  }
  for (const k of R) sqR += k * k;
  for (let k = 0; k < n - 1; k++) {
    const i = o[k], nL = k + 1, nR = n - nL;
    let w;
    if (C) {
      const c = y[i];
      sqL += 2 * L[c] + 1;
      sqR -= 2 * R[c] - 1;
      L[c]++;
      R[c]--;
      w = (nL - sqL / nL + nR - sqR / nR) / n;
    } else {
      sL += y[i];
      sR -= y[i];
      ssL += y[i] * y[i];
      ssR -= y[i] * y[i];
      w = Math.max(0, ssL - sL * sL / nL + ssR - sR * sR / nR) / n;
    }
    const a = X[i][f], b = X[o[k + 1]][f];
    if (a === b) continue;
    steps.push({ lo: a, hi: b, w });
    if (!best || w < best.w - EPS) best = { feat: f, thr: (a + b) / 2, w };
  }
  return { steps, best };
}

/** Lowest weighted impurity over the features feats (first on a tie). */
function bestSplit(idx, X, y, C, feats) {
  let best = null;
  for (const f of feats) {
    const b = scanFeature(idx, X, y, C, f).best;
    if (b && (!best || b.w < best.w - EPS)) best = b;
  }
  return best;
}

/**
 * Grow a tree greedily on the samples idx (repeats allowed).
 *
 * Each node takes its best split until it is pure, its samples share one
 * point, or it reaches maxDepth. Given r, a split may search only one
 * feature drawn at random (another if that one cannot split): the random
 * forest rule max_features = sqrt(d), which is 1 for d = 2 (Breiman 2001).
 *
 * Returns:
 *   node: nodeStats() fields plus feat (null for a leaf), thr, left,
 *     right (yes / no children) and gain = n imp - n_yes imp_yes
 *     - n_no imp_no, the impurity removed (for feature importance)
 */
function fitTree(idx, X, y, C, maxDepth, r, depth = 0) {
  const nd = nodeStats(idx, y, C);
  nd.feat = null;
  if (depth >= maxDepth || nd.imp <= EPS) return nd;
  const d = X[0].length;
  let b = null;
  if (r) {
    for (const f of shuffled([...Array(d).keys()], r)) {
      b = bestSplit(idx, X, y, C, [f]);
      if (b) break;
    }
  } else {
    b = bestSplit(idx, X, y, C, [...Array(d).keys()]);
  }
  if (!b) return nd;
  const yes = [], no = [];
  for (const i of idx) (X[i][b.feat] <= b.thr ? yes : no).push(i);
  nd.feat = b.feat;
  nd.thr = b.thr;
  nd.left = fitTree(yes, X, y, C, maxDepth, r, depth + 1);
  nd.right = fitTree(no, X, y, C, maxDepth, r, depth + 1);
  nd.gain = nd.n * nd.imp - nd.left.n * nd.left.imp
    - nd.right.n * nd.right.imp;
  return nd;
}

/** Leaf that the point x reaches. */
function leafOf(nd, x) {
  while (nd.feat !== null) nd = x[nd.feat] <= nd.thr ? nd.left : nd.right;
  return nd;
}

/**
 * Share of a tree's impurity reduction made by each feature.
 *
 * Gini importance: the sum of gain over the splits on a feature, over the
 * sum over all splits (Breiman 2001).
 *
 * Returns:
 *   imp (number[]): (d,) shares summing to 1, all 0 for a lone leaf
 */
function importance(root, d) {
  const g = new Array(d).fill(0);
  (function walk(nd) {
    if (nd.feat === null) return;
    g[nd.feat] += nd.gain;
    walk(nd.left);
    walk(nd.right);
  })(root);
  const tot = g.reduce((a, b) => a + b, 0);
  return g.map(v => (tot > 0 ? v / tot : 0));
}
