import { useEffect, useState } from "react";

// ---------- ids and labels mirroring pipeline/models.py ----------
export type ModelId = "baseline" | "logistic" | "tree" | "rf" | "gbm" | "tabicl";
export const FITTED_MODELS: ModelId[] = ["logistic", "tree", "rf", "gbm", "tabicl"];
export const MODEL_LABEL: Record<ModelId, string> = {
  baseline: "Baseline (prevalence)",
  logistic: "Logistic regression",
  tree: "Decision tree",
  rf: "Random forest",
  gbm: "Gradient boosting",
  tabicl: "Foundation (TabICL)",
};
export const MODEL_SHORT: Record<ModelId, string> = {
  baseline: "Baseline",
  logistic: "Logistic",
  tree: "Tree",
  rf: "Forest",
  gbm: "Boosting",
  tabicl: "TabICL",
};
export const MODEL_COLOR: Record<ModelId, string> = {
  baseline: "var(--m-baseline)",
  logistic: "var(--m-linear)",
  tree: "var(--m-tree)",
  rf: "var(--m-rf)",
  gbm: "var(--m-gbm)",
  tabicl: "var(--m-tabicl)",
};

export type DatasetId = "diabetes" | "readmission" | "heart";
export const DATASET_IDS: DatasetId[] = ["diabetes", "readmission", "heart"];
export const DATASET_SHORT: Record<DatasetId, string> = {
  diabetes: "Diabetes screening",
  readmission: "30-day readmission",
  heart: "Heart disease",
};

// ---------- JSON types mirroring pipeline/run_all.py ----------
export interface Counts { tp: number; fn: number; fp: number; tn: number }

export interface FoldScore { auroc: number; train_auroc: number; auprc: number; brier: number }
export interface ModelCV {
  folds: FoldScore[];
  mean_auroc: number; sd_auroc: number; mean_train_auroc: number;
  mean_auprc: number; mean_brier: number; mean_pred: number;
  at_050: Counts;
}
export interface TestView {
  n_dev: number; n_test: number; dev_pos: number; test_pos: number;
  models: Record<string, { fold_auroc: number[]; mean: number; sd: number; test_auroc: number; test_ci: [number, number] }>;
  best_by_cv: ModelId; simplest_within_1sd: ModelId;
  test_y: number[]; test_pred: Record<string, number[] | null>;
}
export interface DatasetData {
  id: DatasetId; name: string; story: string; source: string; citation: string; url: string;
  labels: { pos: string; neg: string; unit: string; units: string };
  n_rows: number; n_pos: number; prevalence: number;
  cohort: { step: string; n: number }[];
  features: { id: string; label: string }[];
  split: string;
  y: number[];
  cv: {
    n_folds: number;
    models: Record<string, ModelCV | null>;
    /** Out-of-fold probabilities as integers in 1/10,000; null when the model was not run. */
    oof: Record<string, number[] | null>;
    /** AUROC of the pooled out-of-fold integers, for the dev-only browser cross-check. */
    pooled_auroc: Record<string, number>;
  };
  test_view?: TestView;
}

export interface Grid { prob: number[]; train_auroc: number; cv_auroc: number; leaf?: number[]; leaves?: [number, number][] }
export interface PlaygroundPair {
  label: string; x_label: string; x_units: string; y_label: string; y_units: string;
  pos_label: string; neg_label: string; n: number; n_pos: number; sample: string;
  x_axis: number[]; y_axis: number[];
  points: { x: number[]; y: number[]; label: number[] };
  logistic: Grid & { intercept: number; coef: [number, number] };
  tree: Record<string, Grid>;
  rf: Record<string, Grid>;
  gbm: Record<string, Grid>;
  tabicl: Grid | null;
}
export interface Logistic1D {
  n: number; n_pos: number; intercept: number; slope: number; p50_waist: number; flagged_at_050: number;
  curve: { x: number[]; p: number[] };
  points: { x: number[]; y: number[] };
  all: { x: number[]; y: number[] };
}
export interface PlaygroundData {
  pairs: Record<"heart" | "nhanes", PlaygroundPair>;
  tree_depths: number[]; rf_trees: number[]; rf_leaves: number[]; gb_rounds: number[]; gb_lrs: number[];
  logistic_1d: Logistic1D;
}

export interface CalBin { lo: number; hi: number; meanPred: number; observed: number; n: number }
export interface LeakRun {
  model: "rf" | "gbm"; resample: "none" | "inside" | "before"; leaky?: boolean; split?: "row" | "patient";
  test_auroc: number; train_auroc: number; folds: number[];
}
export interface PeekSplit {
  rep: number; fold: number; n_train_pos: number; n_test_pos: number;
  p_train: number[]; y_train: number[]; p_test: number[]; y_test: number[];
}
export interface LabData {
  hook: {
    n: number; n_pos: number; prevalence: number; always_no_acc: number;
    model: ModelId; threshold: number; counts: Counts; acc: number;
  };
  calibration: Record<string, CalBin[]>;
  train_vs_unseen: {
    dataset: string; features: string[]; n: number;
    tree: { depth: number[]; train: number[]; unseen: number[] };
    logistic: { train: number; unseen: number };
  };
  unfair: {
    id: string; title: string; detail: string; unfair: number; fair: number; slide_unfair: number; slide_fair: number;
  }[];
  peeking: {
    n: number; n_pos: number; repeats: number; folds: number;
    targets: number[]; peek: number[]; fair: number[];
    per_split: Record<string, { peek: number[]; fair: number[] }>;
    examples: PeekSplit[];
  };
  row_vs_patient: { model: string; row: number; patient: number; n_rows: number; n_patients: number };
  parkinsons_ref: { random_r2: number; group_r2: number; source: string };
  leakage: {
    diabetes: { n: number; n_pos: number; fair_features: string[]; leaky_features: string[]; runs: LeakRun[] };
    readmission: { n_rows: number; n_patients: number; n_pos: number; runs: LeakRun[] };
  };
}

export interface MetaData {
  generated: string; seed: number; n_folds: number; python: string; device: string; tabicl_ran: boolean;
  packages: Record<string, string>; model_details: Record<string, string>; splits: Record<string, string>;
  funnels: Record<string, { step: string; n: number }[]>;
  runtime_seconds: number;
}

// ---------- loader with a module-level cache ----------
const cache = new Map<string, Promise<unknown>>();

export function load<T>(name: string): Promise<T> {
  if (!cache.has(name)) {
    cache.set(
      name,
      fetch(`${import.meta.env.BASE_URL}data/${name}.json`).then((r) => {
        if (!r.ok) throw new Error(`Could not load ${name}.json (${r.status})`);
        return r.json().then((v: unknown) => {
          if (import.meta.env.DEV) devCheck(name, v);
          return v;
        });
      }),
    );
  }
  return cache.get(name) as Promise<T>;
}

export function useData<T>(name: string | null): T | null {
  const [state, setState] = useState<{ name: string; value: T } | null>(null);
  useEffect(() => {
    if (!name) return;
    let live = true;
    load<T>(name).then((v) => live && setState({ name, value: v }));
    return () => {
      live = false;
    };
  }, [name]);
  return state && state.name === name ? state.value : null;
}

/**
 * Dev-only: recompute every AUROC and every 0.50 matrix from the stored out-of-fold integers.
 * The pipeline computed `pooled_auroc` and `at_050` from the same integers, so they must agree within 0.001 and exactly.
 */
function devCheck(name: string, v: unknown) {
  const d = v as DatasetData;
  if (!d || !d.cv || !Array.isArray(d.y)) return;
  let checked = 0;
  for (const [m, ints] of Object.entries(d.cv.oof)) {
    const mc = d.cv.models[m];
    if (!ints || !mc) continue;
    const p = decodeP(ints);
    const ref = d.cv.pooled_auroc[m];
    const got = auc(p, d.y);
    if (m !== "baseline" && Math.abs(got - ref) > 0.001) console.error(`[dev] ${name}/${m}: browser AUROC ${got.toFixed(4)} ≠ pipeline ${ref}`);
    const c = confusion(p, d.y, 0.5);
    const s = mc.at_050;
    if (c.tp !== s.tp || c.fp !== s.fp || c.fn !== s.fn || c.tn !== s.tn) console.error(`[dev] ${name}/${m}: counts at 0.50 differ`, c, s);
    checked++;
  }
  console.info(`[dev] ${name}.json: ${checked} out-of-fold AUROCs and 0.50 matrices recomputed in the browser`);
}

// ---------- stats helpers ----------
export const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
export const sd = (a: number[]) => {
  const m = mean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1));
};

/** Inverse standard normal CDF (Acklam's approximation). */
export function qnorm(p: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const pl = 0.02425;
  if (p < pl) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - pl) return -qnorm(1 - p);
  const q = p - 0.5, r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Standard normal CDF (Abramowitz & Stegun 7.1.26 via erf), accurate to ~1e-7. */
export function pnorm(x: number): number {
  const z = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * z);
  const erf = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z);
  return x >= 0 ? 0.5 * (1 + erf) : 0.5 * (1 - erf);
}

export const fmt = (v: number | null | undefined, d = 2) =>
  v == null || !isFinite(v) ? "–" : (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }).replace(/^-/, "−");
export const fmtSigned = (v: number, d = 2) => (v > 0 ? "+" : v < 0 ? "−" : "") + fmt(Math.abs(v), d);
export const fmtPct = (v: number | null | undefined, d = 0) => (v == null || !isFinite(v) ? "–" : `${fmt(v * 100, d)}%`);
/** A metric that can be 0 ÷ 0 (precision with no one flagged) prints as undefined, never as 0. */
export const fmtMetric = (v: number | null, d = 0) => (v == null ? "undefined (0 ÷ 0)" : fmtPct(v, d));
export const fmtInt = (v: number) => v.toLocaleString("en-US");

// ---------- classification maths ----------
/** Thresholds come from sliders in 0.01 steps; a tiny tolerance keeps p = t exactly on the "flag" side. */
const EPS = 1e-9;

/** Predict positive when p ≥ t. */
export function confusion(p: ArrayLike<number>, y: ArrayLike<number>, t: number): Counts {
  let tp = 0, fn = 0, fp = 0, tn = 0;
  for (let i = 0; i < p.length; i++) {
    const flag = p[i] >= t - EPS;
    if (y[i] === 1) flag ? tp++ : fn++;
    else flag ? fp++ : tn++;
  }
  return { tp, fn, fp, tn };
}

export interface Metrics {
  acc: number; sens: number; spec: number; fpr: number;
  prec: number | null; npv: number | null; f1: number;
  flagged: number; prevalence: number; n: number;
}

export function metrics(c: Counts): Metrics {
  const n = c.tp + c.fn + c.fp + c.tn;
  const P = c.tp + c.fn, N = c.fp + c.tn;
  const flagged = c.tp + c.fp;
  const prec = flagged === 0 ? null : c.tp / flagged;
  const npv = c.tn + c.fn === 0 ? null : c.tn / (c.tn + c.fn);
  const sens = P === 0 ? 0 : c.tp / P;
  const spec = N === 0 ? 0 : c.tn / N;
  const f1 = 2 * c.tp + c.fp + c.fn === 0 ? 0 : (2 * c.tp) / (2 * c.tp + c.fp + c.fn);
  return { acc: (c.tp + c.tn) / n, sens, spec, fpr: 1 - spec, prec, npv, f1, flagged, prevalence: P / n, n };
}

/** Descending order of probability, ties grouped (stable on index for determinism). */
function descOrder(p: ArrayLike<number>) {
  const idx = Array.from({ length: p.length }, (_, i) => i);
  idx.sort((a, b) => p[b] - p[a] || a - b);
  return idx;
}

export interface Roc { fpr: number[]; tpr: number[]; thr: number[] }

/**
 * The ROC staircase. Point k is what you get when you flag everyone with p ≥ thr[k].
 * Point 0 is (0, 0) with thr = Infinity (flag no one); the last point is (1, 1).
 */
export function rocPoints(p: ArrayLike<number>, y: ArrayLike<number>): Roc {
  const order = descOrder(p);
  let P = 0;
  for (let i = 0; i < y.length; i++) P += y[i] === 1 ? 1 : 0;
  const N = y.length - P;
  const fpr = [0], tpr = [0], thr = [Infinity];
  let tp = 0, fp = 0;
  for (let k = 0; k < order.length; ) {
    const v = p[order[k]];
    while (k < order.length && p[order[k]] === v) {
      if (y[order[k]] === 1) tp++;
      else fp++;
      k++;
    }
    fpr.push(N ? fp / N : 0);
    tpr.push(P ? tp / P : 0);
    thr.push(v);
  }
  return { fpr, tpr, thr };
}

/** AUROC via Mann–Whitney with mid-ranks: ties between a positive and a negative count 0.5. */
export function auc(p: ArrayLike<number>, y: ArrayLike<number>): number {
  const n = p.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => p[a] - p[b]);
  let nPos = 0, rankSum = 0;
  for (let i = 0; i < n; ) {
    let j = i;
    while (j + 1 < n && p[idx[j + 1]] === p[idx[i]]) j++;
    const mid = (i + j) / 2 + 1; // ranks are 1-based
    for (let k = i; k <= j; k++) if (y[idx[k]] === 1) {
      nPos++;
      rankSum += mid;
    }
    i = j + 1;
  }
  const nNeg = n - nPos;
  if (!nPos || !nNeg) return NaN;
  return (rankSum - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

/** Precision–recall points at every distinct threshold (precision undefined before the first flag is skipped). */
export function prCurve(p: ArrayLike<number>, y: ArrayLike<number>) {
  const { fpr, tpr, thr } = rocPoints(p, y);
  let P = 0;
  for (let i = 0; i < y.length; i++) P += y[i] === 1 ? 1 : 0;
  const N = y.length - P;
  const recall: number[] = [], precision: number[] = [], t: number[] = [];
  for (let k = 1; k < thr.length; k++) {
    const tp = tpr[k] * P, fp = fpr[k] * N;
    recall.push(tpr[k]);
    precision.push(tp / (tp + fp));
    t.push(thr[k]);
  }
  return { recall, precision, thr: t };
}

/** Reliability table: equal-count (quantile) or equal-width bins of predicted probability. */
export function calibrationBins(
  p: ArrayLike<number>, y: ArrayLike<number>, opts: { bins?: number; strategy?: "quantile" | "uniform" } = {},
): CalBin[] {
  const bins = opts.bins ?? 10;
  const n = p.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => p[a] - p[b] || a - b);
  const groups: number[][] = Array.from({ length: bins }, () => []);
  if ((opts.strategy ?? "quantile") === "quantile") {
    idx.forEach((i, r) => groups[Math.min(bins - 1, Math.floor((r * bins) / n))].push(i));
  } else {
    idx.forEach((i) => groups[Math.min(bins - 1, Math.floor(p[i] * bins))].push(i));
  }
  return groups.filter((g) => g.length).map((g) => {
    let sp = 0, sy = 0, lo = Infinity, hi = -Infinity;
    for (const i of g) {
      sp += p[i];
      sy += y[i];
      lo = Math.min(lo, p[i]);
      hi = Math.max(hi, p[i]);
    }
    return { lo, hi, meanPred: sp / g.length, observed: sy / g.length, n: g.length };
  });
}

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
export const logit = (p: number) => Math.log(p / (1 - p));

/** Mean log-loss of a 1-feature logistic model with raw-unit coefficients. */
export function logLoss(x: ArrayLike<number>, y: ArrayLike<number>, b0: number, b1: number) {
  let s = 0;
  for (let i = 0; i < x.length; i++) {
    const z = b0 + b1 * x[i];
    // log(1 + e^z) computed stably
    const softplus = z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z));
    s += softplus - (y[i] === 1 ? z : 0);
  }
  return s / x.length;
}

/** Unpenalised 1-feature logistic regression by Newton–Raphson (IRLS); x is standardised internally. */
export function logisticIRLS(x: ArrayLike<number>, y: ArrayLike<number>, iters = 50) {
  const n = x.length;
  let mx = 0;
  for (let i = 0; i < n; i++) mx += x[i];
  mx /= n;
  let vx = 0;
  for (let i = 0; i < n; i++) vx += (x[i] - mx) ** 2;
  const sx = Math.sqrt(vx / n) || 1;
  let a = 0, b = 0;
  for (let k = 0; k < iters; k++) {
    let g0 = 0, g1 = 0, h00 = 0, h01 = 0, h11 = 0;
    for (let i = 0; i < n; i++) {
      const u = (x[i] - mx) / sx;
      const p = sigmoid(a + b * u);
      const w = p * (1 - p);
      g0 += y[i] - p;
      g1 += (y[i] - p) * u;
      h00 += w;
      h01 += w * u;
      h11 += w * u * u;
    }
    const det = h00 * h11 - h01 * h01;
    const da = (h11 * g0 - h01 * g1) / det;
    const db = (h00 * g1 - h01 * g0) / det;
    a += da;
    b += db;
    if (Math.abs(da) + Math.abs(db) < 1e-12) break;
  }
  return { b0: a - (b * mx) / sx, b1: b / sx };
}

/** Positive predictive value from sensitivity, specificity and prevalence (Bayes). */
export const ppv = (sens: number, spec: number, prev: number) => (sens * prev) / (sens * prev + (1 - spec) * (1 - prev));

/** A threshold window: every t in (below, top] gives the same predictions. */
export interface ThresholdRange { top: number; below: number }

/** Thresholds that flag exactly the k highest-probability patients (or as close as ties allow). */
export function thresholdForCapacity(p: ArrayLike<number>, k: number): ThresholdRange {
  const s = Array.from(p).sort((a, b) => b - a);
  if (k <= 0) return { top: Infinity, below: s[0] };
  if (k >= s.length) return { top: s[s.length - 1], below: -Infinity };
  return { top: s[k - 1], below: s[k] };
}

/** The highest threshold that still catches at least `target` of the positives. */
export function thresholdForSensitivity(p: ArrayLike<number>, y: ArrayLike<number>, target: number): number {
  const pos: number[] = [];
  for (let i = 0; i < p.length; i++) if (y[i] === 1) pos.push(p[i]);
  pos.sort((a, b) => b - a);
  const k = Math.max(1, Math.ceil(target * pos.length - 1e-9));
  return pos[Math.min(k, pos.length) - 1];
}

/** Threshold window minimising cMiss·FN + cAlarm·FP over every distinct cut-off. */
export function minCostThreshold(p: ArrayLike<number>, y: ArrayLike<number>, cMiss: number, cAlarm: number) {
  const { fpr, tpr, thr } = rocPoints(p, y);
  let P = 0;
  for (let i = 0; i < y.length; i++) P += y[i] === 1 ? 1 : 0;
  const N = y.length - P;
  let best = 0, bestCost = Infinity;
  const cost = fpr.map((f, k) => cMiss * Math.round((1 - tpr[k]) * P) + cAlarm * Math.round(f * N));
  cost.forEach((c, k) => {
    if (c < bestCost - 1e-9) {
      bestCost = c;
      best = k;
    }
  });
  return { k: best, top: thr[best], below: best + 1 < thr.length ? thr[best + 1] : -Infinity, cost: bestCost, costs: cost, thr };
}

/** Round a threshold window to the slider grid: the 0.01 steps that give these exact predictions. */
export function sliderWindow(r: ThresholdRange, step = 0.01): [number, number] | null {
  const lo = Math.max(0, Math.floor(r.below / step + 1e-9) * step + step);
  const hi = Math.min(1, Math.floor(r.top / step + 1e-9) * step);
  return lo <= hi + 1e-9 ? [round2(lo), round2(hi)] : null;
}

export const round2 = (v: number) => Math.round(v * 100) / 100;

/** Deterministic walk through every (positive, negative) pair: j = stride·i mod (nPos·nNeg). */
export function pairOrder(nPos: number, nNeg: number, stride = 433): [number, number][] {
  const total = nPos * nNeg;
  const out: [number, number][] = [];
  for (let i = 0; i < total; i++) {
    const j = (stride * i) % total;
    out.push([Math.floor(j / nNeg), j % nNeg]);
  }
  return out;
}

/** The pipeline stores out-of-fold probabilities as integers in 1/10,000. */
export const decodeP = (ints: ArrayLike<number>) => Float64Array.from(ints as ArrayLike<number>, (v) => v / 10000);

// ---------- Model A: the hypothetical running example from slides 7–23 ----------
/** [lo, hi, nReadmitted, nNot] with probabilities in thousandths. */
const BANDS: [number, number, number, number][] = [
  [850, 1000, 2, 1], [700, 850, 3, 5], [500, 700, 3, 12], [300, 500, 1, 12], [150, 300, 1, 24], [0, 150, 0, 36],
];
/** Keeps each slide matrix stable for thresholds ±0.02 around 0.30 / 0.50 / 0.70. */
const GAP = 20;

export interface ModelA { p: number[]; y: number[] }

/** 100 discharged patients, 10 readmitted, sorted by ascending probability. */
export const MODEL_A: ModelA = (() => {
  const rows: { p: number; y: number }[] = [];
  for (const [lo, hi, nPos, nNeg] of BANDS) {
    const n = nPos + nNeg;
    const posAt = new Set(Array.from({ length: nPos }, (_, j) => Math.floor(((j + 0.5) * n) / nPos)));
    for (let k = 0; k < n; k++) {
      const milli = lo + GAP + ((hi - lo - 2 * GAP) * (k + 0.5)) / n;
      rows.push({ p: milli / 1000, y: posAt.has(k) ? 1 : 0 });
    }
  }
  rows.sort((a, b) => a.p - b.p);
  return { p: rows.map((r) => r.p), y: rows.map((r) => r.y) };
})();

// ---------- sandbox: a calibrated synthetic set with a chosen separation ----------
export function sandbox({ n = 1000, prevalence = 0.1, dprime = 1.6 }: { n?: number; prevalence?: number; dprime?: number } = {}) {
  const nPos = Math.max(1, Math.round(n * prevalence));
  const nNeg = n - nPos;
  const d = dprime;
  const shift = logit(nPos / n);
  const p: number[] = [], y: number[] = [];
  // p = sigmoid(d·s − d²/2 + logit(prevalence)) is the exact posterior for N(0,1) vs N(d,1) scores: calibrated by construction
  for (let i = 0; i < nNeg; i++) {
    const s = qnorm((i + 0.5) / nNeg);
    p.push(sigmoid(d * s - (d * d) / 2 + shift));
    y.push(0);
  }
  for (let j = 0; j < nPos; j++) {
    const s = d + qnorm((j + 0.5) / nPos);
    p.push(sigmoid(d * s - (d * d) / 2 + shift));
    y.push(1);
  }
  return { p, y, nPos, nNeg };
}
