import { ReactNode, useEffect, useId, useMemo, useRef, useState } from "react";
import { scaleLinear } from "d3-scale";
import { line } from "d3-shape";
import { contours } from "d3-contour";
import { Plot, PlotCtx, TipRow, Tooltip, useTip, useWidth } from "./kit";
import { PauseThink } from "../components/ui";
import { CalBin, Counts, fmt, fmtInt, Grid, Metrics, metrics, PlaygroundPair, round2 } from "../data";

// ======================================================================
// shared helpers
// ======================================================================
export type CellKey = "tp" | "fn" | "fp" | "tn";
export const CELL_ABBR: Record<CellKey, string> = { tp: "TP", fn: "FN", fp: "FP", tn: "TN" };
export const CELL_NAME: Record<CellKey, string> = { tp: "True positive", fn: "False negative", fp: "False positive", tn: "True negative" };
export const CELL_MARK: Record<CellKey, string> = { tp: "✓ correct", fn: "✗ missed case", fp: "✗ false alarm", tn: "✓ correct" };
export const CELL_GLYPH: Record<CellKey, string> = { tp: "✓", fn: "✗", fp: "✗", tn: "✓" };
export const CELL_COLOR: Record<CellKey, string> = { tp: "var(--tp)", fn: "var(--fn)", fp: "var(--fp)", tn: "var(--tn)" };

export const cellOf = (y: number, flagged: boolean): CellKey => (y === 1 ? (flagged ? "tp" : "fn") : flagged ? "fp" : "tn");

/** Thresholds on the 0.01 slider grid print with 2 decimals; corners picked on the ROC can need 3. */
export const fmtT = (t: number) => (Math.abs(t * 100 - Math.round(t * 100)) < 1e-6 ? t.toFixed(2) : t.toFixed(3));

/** Percent with 1 decimal only when needed: 80%, 30.8%, 44.4%. */
export const pctAuto = (v: number | null, undef = "undefined") => {
  if (v == null || !isFinite(v)) return undef;
  const x = Math.round(v * 1000) / 10;
  return `${Number.isInteger(x) ? x.toFixed(0) : x.toFixed(1)}%`;
};

export function useReducedMotion() {
  const query = "(prefers-reduced-motion: reduce)";
  const [reduce, setReduce] = useState(() => typeof window !== "undefined" && !!window.matchMedia?.(query).matches);
  useEffect(() => {
    const mq = window.matchMedia?.(query);
    if (!mq) return;
    const f = () => setReduce(mq.matches);
    mq.addEventListener("change", f);
    return () => mq.removeEventListener("change", f);
  }, []);
  return reduce;
}

/** Animate an integer count over `ms` (instant when the viewer prefers reduced motion). */
function useTween(v: number, ms = 200) {
  const reduce = useReducedMotion();
  const [shown, setShown] = useState(v);
  const cur = useRef(v);
  useEffect(() => {
    if (reduce || cur.current === v) {
      cur.current = v;
      setShown(v);
      return;
    }
    const from = cur.current;
    const start = performance.now();
    let raf = 0;
    const step = (now: number) => {
      const k = Math.min(1, (now - start) / ms);
      const val = Math.round(from + (v - from) * (1 - (1 - k) ** 3));
      cur.current = val;
      setShown(val);
      if (k < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [v, ms, reduce]);
  return shown;
}

/** A polite live region whose text settles 300 ms after the last change (for drag announcements). */
export function LiveRegion({ text, delay = 300 }: { text?: string; delay?: number }) {
  const [shown, setShown] = useState("");
  useEffect(() => {
    if (!text) return;
    const t = setTimeout(() => setShown(text), delay);
    return () => clearTimeout(t);
  }, [text, delay]);
  return <div className="sr-only" aria-live="polite">{shown}</div>;
}

/** Short matrix summary for screen readers. */
export const announceCounts = (t: number, c: Counts, pos = "readmissions") =>
  `Threshold ${fmtT(t)}: ${c.tp} ${pos} found, ${c.fn} missed, ${c.fp} false alarms, ${c.tn} correctly cleared.`;

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

/** Pointer position in the local coordinates of an SVG element (works through any parent transforms). */
function localPoint(el: SVGGraphicsElement, clientX: number, clientY: number) {
  const svg = el.ownerSVGElement;
  const m = el.getScreenCTM();
  if (!svg || !m) return { x: 0, y: 0 };
  const pt = svg.createSVGPoint();
  pt.x = clientX;
  pt.y = clientY;
  const l = pt.matrixTransform(m.inverse());
  return { x: l.x, y: l.y };
}

// ======================================================================
// Draggable vertical line (threshold handle, or a labelled marker)
// ======================================================================
/** An SVG slider: role="slider", arrows ±0.01, Shift ±0.05, PgUp/PgDn ±0.10, Home/End; pointer drag with capture. */
export function DragLine({
  x, value, onChange, top, bottom, min = 0, max = 1, step = 0.01, label, valueText, color = "var(--ink)", text, dashed, knob = true,
}: {
  x: ((v: number) => number) & { invert?: (px: number) => number }; value: number; onChange: (v: number) => void; top: number; bottom: number;
  min?: number; max?: number; step?: number; label: string; valueText: string; color?: string; text?: string; dashed?: boolean; knob?: boolean;
}) {
  const ref = useRef<SVGGElement>(null);
  const drag = useRef(false);
  const px = x(value);
  const snap = (v: number) => clamp(Math.round(v / step) * step, min, max);
  const fromPointer = (cx: number, cy: number) => {
    if (!ref.current || !x.invert) return value;
    return round2(snap(x.invert(localPoint(ref.current, cx, cy).x)));
  };
  const onKey = (e: React.KeyboardEvent) => {
    const big = e.shiftKey ? 0.05 : step;
    let v: number | null = null;
    if (e.key === "ArrowRight" || e.key === "ArrowUp") v = value + big;
    else if (e.key === "ArrowLeft" || e.key === "ArrowDown") v = value - big;
    else if (e.key === "PageUp") v = value + 0.1;
    else if (e.key === "PageDown") v = value - 0.1;
    else if (e.key === "Home") v = min;
    else if (e.key === "End") v = max;
    if (v == null) return;
    e.preventDefault();
    e.stopPropagation();
    onChange(round2(snap(v)));
  };
  return (
    <g
      ref={ref} className="thr-handle" role="slider" tabIndex={0} aria-label={label} aria-orientation="horizontal"
      aria-valuemin={min} aria-valuemax={max} aria-valuenow={round2(value)} aria-valuetext={valueText}
      onKeyDown={onKey}
      onPointerDown={(e) => {
        (e.currentTarget as SVGGElement).setPointerCapture(e.pointerId);
        drag.current = true;
        onChange(fromPointer(e.clientX, e.clientY));
      }}
      onPointerMove={(e) => drag.current && onChange(fromPointer(e.clientX, e.clientY))}
      onPointerUp={(e) => {
        drag.current = false;
        (e.currentTarget as SVGGElement).releasePointerCapture(e.pointerId);
      }}
      onPointerCancel={() => (drag.current = false)}
      style={{ touchAction: "none" }}
    >
      <rect x={px - 14} y={top - 16} width={28} height={bottom - top + 16} fill="transparent" />
      <line x1={px} x2={px} y1={top} y2={bottom} stroke={color} strokeWidth={2.2} strokeDasharray={dashed ? "5 4" : undefined} />
      {knob && <circle className="thr-knob" cx={px} cy={top - 6} r={8} style={{ stroke: color }} />}
      {text && (
        <text x={px} y={top - 18} textAnchor="middle" fontSize={12} fontWeight={700} fill={color} className="tnum">{text}</text>
      )}
    </g>
  );
}

// ======================================================================
// Threshold slider (the primary, native control)
// ======================================================================
export function ThresholdSlider({
  value, onChange, min = 0, max = 1, step = 0.01, presets, label = "Threshold", valueText, width,
}: {
  value: number; onChange: (t: number) => void; min?: number; max?: number; step?: number;
  presets?: number[]; label?: ReactNode; valueText?: (t: number) => string; width?: number;
}) {
  const id = useId();
  return (
    <div className="slider" style={width ? { maxWidth: width } : undefined}>
      <div className="slider-head">
        <label htmlFor={id}>{label}</label>
        <b>{fmtT(value)}</b>
      </div>
      <input
        id={id} type="range" min={min} max={max} step={step} value={value}
        aria-valuetext={valueText ? valueText(value) : fmtT(value)}
        onChange={(e) => onChange(round2(Number(e.target.value)))}
      />
      {presets && (
        <div className="presets" role="group" aria-label="Preset thresholds">
          {presets.map((p) => (
            <button key={p} aria-pressed={Math.abs(p - value) < 1e-9} onClick={() => onChange(p)}>{p.toFixed(2)}</button>
          ))}
        </div>
      )}
    </div>
  );
}

// ======================================================================
// Confusion matrix
// ======================================================================
export type MetricKey = "sens" | "tpr" | "spec" | "fpr" | "prec" | "acc" | "f1";
type Highlight = { num: CellKey[]; den: CellKey[] };
export const HIGHLIGHTS: Record<MetricKey, Highlight> = {
  sens: { num: ["tp"], den: ["tp", "fn"] },
  tpr: { num: ["tp"], den: ["tp", "fn"] },
  spec: { num: ["tn"], den: ["tn", "fp"] },
  fpr: { num: ["fp"], den: ["fp", "tn"] },
  prec: { num: ["tp"], den: ["tp", "fp"] },
  acc: { num: ["tp", "tn"], den: ["tp", "fn", "fp", "tn"] },
  f1: { num: ["tp"], den: ["tp", "fn", "fp"] },
};

function CountText({ v }: { v: number }) {
  const shown = useTween(v);
  return <>{fmtInt(shown)}</>;
}

export function ConfusionMatrix({
  counts, layout = "course", highlight, showTotals = true, showNames = true, ghost, onCellHover, hoverCell,
  posLabel = "readmitted", negLabel = "not readmitted", compact, caption,
}: {
  counts: Counts; layout?: "course" | "sklearn"; highlight?: MetricKey | Highlight | null; showTotals?: boolean; showNames?: boolean;
  ghost?: Counts | null; onCellHover?: (c: CellKey | null) => void; hoverCell?: CellKey | null;
  posLabel?: string; negLabel?: string; compact?: boolean; caption?: string;
}) {
  const hl = highlight ? (typeof highlight === "string" ? HIGHLIGHTS[highlight] : highlight) : null;
  const course = layout === "course";
  const rows: CellKey[][] = course ? [["tp", "fn"], ["fp", "tn"]] : [["tn", "fp"], ["fn", "tp"]];
  const rowLab = course ? [`Actually ${posLabel}`, `Actually ${negLabel}`] : [`Actually ${negLabel}`, `Actually ${posLabel}`];
  const colLab = course ? [`Predicted: ${posLabel}`, `Predicted: ${negLabel}`] : [`Predicted: ${negLabel}`, `Predicted: ${posLabel}`];
  const rowTot = rows.map((r) => counts[r[0]] + counts[r[1]]);
  const colTot = [0, 1].map((c) => counts[rows[0][c]] + counts[rows[1][c]]);
  return (
    <div>
      {caption && <div className="sr-only">{caption}</div>}
      <div className={`cmatrix ${compact ? "compact" : ""}`} role="table" aria-label={caption ?? "Confusion matrix"}>
        <div className="cm-corner" role="presentation" />
        {colLab.map((l) => <div key={l} className="cm-colhead" role="columnheader">{l}</div>)}
        <div role="presentation" />
        {rows.map((r, i) => (
          <div key={i} style={{ display: "contents" }} role="row">
            <div className="cm-rowhead" role="rowheader">{rowLab[i]}</div>
            {r.map((k) => {
              const cls = [
                "cm-cell", k,
                hl?.num.includes(k) ? "num" : "",
                hl?.den.includes(k) ? "den" : "",
                hl && !hl.num.includes(k) && !hl.den.includes(k) ? "dim" : "",
                hoverCell === k ? "hover" : "",
              ].join(" ");
              const d = ghost ? counts[k] - ghost[k] : 0;
              return (
                <div
                  key={k} className={cls} role="cell" tabIndex={onCellHover ? 0 : undefined}
                  aria-label={`${CELL_NAME[k]} (${CELL_ABBR[k]}): ${counts[k]}, ${CELL_MARK[k].slice(2)}`}
                  onMouseEnter={onCellHover && (() => onCellHover(k))} onMouseLeave={onCellHover && (() => onCellHover(null))}
                  onFocus={onCellHover && (() => onCellHover(k))} onBlur={onCellHover && (() => onCellHover(null))}
                >
                  <div className="cm-name">{showNames ? `${CELL_NAME[k]} (${CELL_ABBR[k]})` : CELL_ABBR[k]}</div>
                  <div className="cm-count"><CountText v={counts[k]} /></div>
                  <div className="cm-mark">{showNames ? CELL_MARK[k] : CELL_GLYPH[k]}</div>
                  {ghost && d !== 0 && <span className="cm-delta">{d > 0 ? `+${d}` : `−${-d}`}</span>}
                </div>
              );
            })}
            <div className="cm-total row" aria-hidden={!showTotals}>{showTotals ? `total ${fmtInt(rowTot[i])}` : ""}</div>
          </div>
        ))}
        {showTotals && (
          <>
            <div />
            {colTot.map((t, i) => <div key={i} className="cm-total">total {fmtInt(t)}</div>)}
            <div />
          </>
        )}
      </div>
    </div>
  );
}

// ======================================================================
// Formula with tokens in their cell colours
// ======================================================================
const METRIC_NAME: Record<MetricKey, string> = {
  sens: "Sensitivity", tpr: "TPR", spec: "Specificity", fpr: "FPR", prec: "Precision", acc: "Accuracy", f1: "F1",
};

function Tok({ k, children }: { k: CellKey; children: ReactNode }) {
  return <span className={`tok ${k}`}>{children}</span>;
}

export function FormulaLine({ metric, counts, name }: { metric: MetricKey; counts: Counts; name?: string }) {
  const c = counts;
  const m = metrics(c);
  const label = name ?? METRIC_NAME[metric];
  const frac = (num: CellKey, den: CellKey[]) => (
    <>
      <Tok k={num}>{CELL_ABBR[num]}</Tok> ÷ ({den.map((k, i) => <span key={k}>{i > 0 && " + "}<Tok k={k}>{CELL_ABBR[k]}</Tok></span>)})
      {" = "}
      <Tok k={num}>{c[num]}</Tok> ÷ ({den.map((k, i) => <span key={k}>{i > 0 && " + "}<Tok k={k}>{c[k]}</Tok></span>)})
    </>
  );
  let body: ReactNode, res: string;
  switch (metric) {
    case "sens":
    case "tpr":
      body = frac("tp", ["tp", "fn"]);
      res = metric === "tpr" ? fmt(m.sens, 2) : pctAuto(m.sens);
      break;
    case "spec":
      body = frac("tn", ["tn", "fp"]);
      res = pctAuto(m.spec);
      break;
    case "fpr":
      body = frac("fp", ["fp", "tn"]);
      res = fmt(m.fpr, 2);
      break;
    case "prec":
      body = frac("tp", ["tp", "fp"]);
      res = m.prec == null ? "undefined (0 ÷ 0)" : pctAuto(m.prec);
      break;
    case "acc":
      body = (
        <>
          (<Tok k="tp">TP</Tok> + <Tok k="tn">TN</Tok>) ÷ total = (<Tok k="tp">{c.tp}</Tok> + <Tok k="tn">{c.tn}</Tok>) ÷ {fmtInt(m.n)}
        </>
      );
      res = pctAuto(m.acc);
      break;
    case "f1":
      body = (
        <>
          2<Tok k="tp">TP</Tok> ÷ (2<Tok k="tp">TP</Tok> + <Tok k="fp">FP</Tok> + <Tok k="fn">FN</Tok>) = <Tok k="tp">{2 * c.tp}</Tok> ÷ (
          <Tok k="tp">{2 * c.tp}</Tok> + <Tok k="fp">{c.fp}</Tok> + <Tok k="fn">{c.fn}</Tok>)
        </>
      );
      res = pctAuto(m.f1);
      break;
  }
  return (
    <div className="formula">
      <b>{label}</b> = {body} = <span className="res">{res}</span>
    </div>
  );
}

// ======================================================================
// Metric table (columns = models or rules)
// ======================================================================
const TABLE_ROWS: { key: keyof Metrics; label: string; lowerBetter?: boolean }[] = [
  { key: "acc", label: "Accuracy" },
  { key: "sens", label: "Sensitivity (recall, TPR)" },
  { key: "spec", label: "Specificity" },
  { key: "fpr", label: "False positive rate (FPR)", lowerBetter: true },
  { key: "prec", label: "Precision (PPV)" },
  { key: "f1", label: "F1" },
];

export function MetricTable({ columns, star }: { columns: { id: string; label: ReactNode; counts: Counts }[]; star?: boolean }) {
  const ms = columns.map((c) => metrics(c.counts));
  return (
    <div className="table-scroll">
      <table className="data">
        <thead>
          <tr>
            <th>Metric</th>
            {columns.map((c) => <th key={c.id}>{c.label}</th>)}
          </tr>
        </thead>
        <tbody>
          {TABLE_ROWS.map((r) => {
            const vals = ms.map((m) => m[r.key] as number | null);
            const ok = vals.filter((v): v is number => v != null);
            const best = ok.length ? (r.lowerBetter ? Math.min(...ok) : Math.max(...ok)) : null;
            const tie = ok.filter((v) => Math.abs(v - (best ?? -1)) < 1e-12).length === ok.length;
            return (
              <tr key={r.key}>
                <td>{r.label}</td>
                {vals.map((v, i) => (
                  <td key={columns[i].id} style={{ fontWeight: star && !tie && v != null && Math.abs(v - (best ?? -1)) < 1e-12 ? 750 : undefined }}>
                    {star && !tie && v != null && Math.abs(v - (best ?? -1)) < 1e-12 && "★ "}
                    {v == null ? <span className="muted">undefined</span> : pctAuto(v)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ======================================================================
// Probability strip: patients as dots on a 0–1 axis, with a draggable threshold
// ======================================================================
/** Deterministic beeswarm: offsets (in px) from the lane centre so that dots of radius r don't overlap. */
function beeswarm(xs: number[], r: number, maxOff: number): number[] {
  const order = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b] || a - b);
  const off = new Array(xs.length).fill(0);
  const placed: { x: number; y: number }[] = [];
  const d = 2 * r * 0.92;
  for (const i of order) {
    const near = placed.filter((p) => xs[i] - p.x < d);
    let y = 0;
    for (let k = 0; k < 200; k++) {
      const cand = k === 0 ? 0 : (k % 2 ? 1 : -1) * Math.ceil(k / 2) * d * 0.9;
      if (Math.abs(cand) > maxOff) break;
      if (!near.some((p) => (xs[i] - p.x) ** 2 + (cand - p.y) ** 2 < d * d)) {
        y = cand;
        break;
      }
      y = cand;
    }
    off[i] = y;
    placed.push({ x: xs[i], y });
    while (placed.length && xs[i] - placed[0].x >= d) placed.shift();
  }
  return off;
}

export interface StripMarker { p: number; label: string; onChange?: (p: number) => void; color?: string }
export interface StripLink { i: number; j: number; win: boolean; label?: string }

export function ProbabilityStrip({
  p, labels, lanes = "split", colorBy = "actual", threshold, onThreshold, ghostThreshold, markers, mode = "dots",
  posLabel = "Readmitted", negLabel = "Not readmitted", xLabel = "Estimated probability of readmission",
  links, emphasis, announce, laneHeight: laneH, r: radius, predLabels = ["Predicted: not readmitted", "Predicted: readmitted"],
}: {
  p: ArrayLike<number>; labels: ArrayLike<number> | null; lanes?: "split" | "single"; colorBy?: "actual" | "result" | "none";
  threshold?: number; onThreshold?: (t: number) => void; ghostThreshold?: number | null; markers?: StripMarker[];
  mode?: "dots" | "hist"; posLabel?: string; negLabel?: string; xLabel?: string; links?: StripLink[]; emphasis?: Set<number>;
  announce?: string; laneHeight?: number; r?: number; predLabels?: [string, string];
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const { tip, show, hide } = useTip();
  // room above the lanes for the handle label, plus a separate row for the region labels when markers are present too
  const m = { top: threshold != null && markers ? 56 : threshold != null || markers ? 40 : 14, right: 18, bottom: 42, left: 18 };
  const regionY = -(m.top - 14);
  const w = Math.max(50, width - m.left - m.right);
  const x = scaleLinear().domain([0, 1]).range([0, w]);
  const n = p.length;
  const split = lanes === "split" && labels != null;
  const r = radius ?? (n > 300 ? 3 : w < 420 ? 4 : 5.2);

  const layout = useMemo(() => {
    if (mode === "hist") return null;
    const idxPos: number[] = [], idxNeg: number[] = [];
    for (let i = 0; i < n; i++) (split && labels![i] !== 1 ? idxNeg : idxPos).push(i);
    const maxOff = 60;
    const offs = new Array(n).fill(0);
    for (const group of split ? [idxPos, idxNeg] : [idxPos]) {
      const o = beeswarm(group.map((i) => x(p[i])), r, maxOff);
      group.forEach((i, k) => (offs[i] = o[k]));
    }
    const span = (g: number[]) => (g.length ? Math.max(...g.map((i) => Math.abs(offs[i]))) : 0);
    return { offs, spanPos: span(idxPos), spanNeg: span(idxNeg) };
    // x depends only on w
  }, [p, labels, n, split, r, mode, w]); // eslint-disable-line react-hooks/exhaustive-deps

  const minLane = laneH ?? (mode === "hist" ? 64 : 34);
  const laneA = layout ? Math.max(minLane, 2 * layout.spanPos + 2 * r + 10) : minLane;
  const laneB = split ? (layout ? Math.max(minLane, 2 * layout.spanNeg + 2 * r + 10) : minLane) : 0;
  const gap = split ? 14 : 0;
  const h = laneA + gap + laneB;
  const cyA = laneA / 2, cyB = laneA + gap + laneB / 2;
  const laneY = (i: number) => (split && labels![i] !== 1 ? cyB : cyA);

  const colorOf = (i: number) => {
    const isPos = labels ? labels[i] === 1 : false;
    if (colorBy === "none" || labels == null) return "var(--muted)";
    if (colorBy === "actual" || threshold == null) return isPos ? "var(--cls-pos)" : "var(--cls-neg)";
    return CELL_COLOR[cellOf(isPos ? 1 : 0, p[i] >= threshold - 1e-9)];
  };

  // histogram mode: per-lane share of patients in 0.01-wide bins
  const hist = useMemo(() => {
    if (mode !== "hist") return null;
    const bins = 100;
    const a = new Array(bins).fill(0), b = new Array(bins).fill(0);
    let na = 0, nb = 0;
    for (let i = 0; i < n; i++) {
      const k = Math.min(bins - 1, Math.floor(p[i] * bins + 1e-9));
      if (split && labels![i] !== 1) {
        b[k]++;
        nb++;
      } else {
        a[k]++;
        na++;
      }
    }
    return { bins, a: a.map((v) => v / Math.max(1, na)), b: b.map((v) => v / Math.max(1, nb)), na, nb };
  }, [mode, p, labels, n, split]);

  const height = m.top + h + m.bottom;
  const ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
  const xt = x as typeof x & { invert: (px: number) => number };

  return (
    <div className="chart" ref={ref} onMouseLeave={hide}>
      <svg width={width} height={height} role="img" aria-label={`Patients placed by their estimated probability${threshold != null ? `, threshold ${fmtT(threshold)}` : ""}`}>
        <g transform={`translate(${m.left},${m.top})`}>
          {threshold != null && (
            <>
              <rect x={x(threshold)} y={0} width={Math.max(0, w - x(threshold))} height={h} fill="var(--accent-soft)" opacity={0.65} />
              <text x={Math.max(0, x(threshold) - 26)} y={regionY} textAnchor="end" fontSize={11.5} fill="var(--ink-2)">← {predLabels[0]}</text>
              <text x={Math.min(w, x(threshold) + 26)} y={regionY} fontSize={11.5} fill="var(--ink-2)">{predLabels[1]} →</text>
            </>
          )}
          {ticks.map((t) => (
            <g key={t} className="tick" transform={`translate(${x(t)},${h})`}>
              <line y1={-h} y2={0} opacity={0.6} />
              <line y1={0} y2={5} />
              <text y={18} textAnchor="middle">{t === 0 || t === 1 ? t : t.toFixed(1)}</text>
            </g>
          ))}
          <line className="axis-line" x1={0} x2={w} y1={h} y2={h} />
          <text className="axis-label" x={w / 2} y={h + 36} textAnchor="middle">{xLabel}</text>
          {split && (
            <>
              <line x1={0} x2={w} y1={laneA + gap / 2} y2={laneA + gap / 2} stroke="var(--grid)" strokeDasharray="2 3" />
              <text x={4} y={12} fontSize={11.5} fontWeight={650} fill="var(--ink-2)">● {posLabel} ({fmtInt(hist ? hist.na : countOf(labels!, 1))})</text>
              <text x={w - 4} y={laneA + gap + 12} fontSize={11.5} fontWeight={650} fill="var(--ink-2)" textAnchor="end">○ {negLabel} ({fmtInt(hist ? hist.nb : n - countOf(labels!, 1))})</text>
            </>
          )}

          {hist && (
            <g>
              {[0, 1].map((lane) => {
                const arr = lane === 0 ? hist.a : hist.b;
                if (lane === 1 && !split) return null;
                const mx = Math.max(...arr, 1e-9);
                const base = lane === 0 ? laneA : laneA + gap + laneB;
                const lh = lane === 0 ? laneA - 16 : laneB - 16;
                return arr.map((v, k) => {
                  if (!v) return null;
                  const flagged = threshold != null && k / hist.bins >= threshold - 1e-9;
                  const cell = cellOf(lane === 0 && (split || labels == null) ? 1 : 0, flagged);
                  const fill = colorBy === "result" && threshold != null ? CELL_COLOR[cell] : lane === 0 ? "var(--cls-pos)" : "var(--cls-neg)";
                  const bh = (v / mx) * lh;
                  return <rect key={`${lane}-${k}`} x={x(k / hist.bins) + 0.5} width={Math.max(0.5, x(1 / hist.bins) - 1)} y={base - bh} height={bh}
                    fill={fill} opacity={lane === 0 ? 0.9 : 0.75} />;
                });
              })}
            </g>
          )}

          {layout && links?.map((lk, i) => (
            <line key={`l${i}`} x1={x(p[lk.i])} y1={laneY(lk.i) + layout.offs[lk.i]} x2={x(p[lk.j])} y2={laneY(lk.j) + layout.offs[lk.j]}
              stroke={lk.win ? "var(--tp)" : "var(--fn)"} strokeWidth={2.5} />
          ))}

          {layout && Array.from({ length: n }, (_, i) => {
            const isPos = labels ? labels[i] === 1 : false;
            const c = colorOf(i);
            const em = emphasis?.has(i);
            const rr = em ? r * 1.7 : r;
            return (
              <circle key={i} cx={x(p[i])} cy={laneY(i) + layout.offs[i]} r={rr}
                fill={labels == null ? c : isPos ? c : "var(--surface)"} stroke={c} strokeWidth={labels == null || isPos ? 0.8 : 1.8}
                fillOpacity={labels == null ? 0.6 : 1} opacity={emphasis && emphasis.size && !em ? 0.45 : 1}
                onMouseEnter={() => show(x(p[i]) + m.left, laneY(i) + layout.offs[i] + m.top - 6, (
                  <>
                    <TipRow label="Estimated probability" value={p[i].toFixed(3)} />
                    {labels && <TipRow label="Actual" value={isPos ? posLabel : negLabel} />}
                    {labels && threshold != null && <TipRow label="Result" value={CELL_NAME[cellOf(isPos ? 1 : 0, p[i] >= threshold - 1e-9)]} />}
                  </>
                ))}
                onMouseLeave={hide} />
            );
          })}

          {layout && links?.map((lk, i) => (
            <text key={`t${i}`} x={(x(p[lk.i]) + x(p[lk.j])) / 2 + 8} y={(laneY(lk.i) + laneY(lk.j)) / 2 + 4} fontSize={13} fontWeight={800}
              fill={lk.win ? "var(--tp)" : "var(--fn)"}>{lk.win ? "✓" : "✗"}{lk.label ? ` ${lk.label}` : ""}</text>
          ))}

          {ghostThreshold != null && threshold != null && Math.abs(ghostThreshold - threshold) > 1e-9 && (
            <g>
              <line x1={x(ghostThreshold)} x2={x(ghostThreshold)} y1={-4} y2={h} stroke="var(--muted)" strokeWidth={2} strokeDasharray="5 4" />
              <text x={x(ghostThreshold)} y={h - 4} dx={4} fontSize={11} fill="var(--muted)">{fmtT(ghostThreshold)}</text>
            </g>
          )}
          {markers?.map((mk) => (
            mk.onChange ? (
              <DragLine key={mk.label} x={xt} value={mk.p} onChange={mk.onChange} top={0} bottom={h} min={0} max={1} step={0.01}
                label={`Patient ${mk.label}'s estimated probability`} valueText={`${mk.label}: ${fmtT(mk.p)}`} color={mk.color ?? "var(--accent)"}
                text={`${mk.label}: ${fmtT(mk.p)}`} dashed knob={false} />
            ) : (
              <g key={mk.label}>
                <line x1={x(mk.p)} x2={x(mk.p)} y1={0} y2={h} stroke={mk.color ?? "var(--accent)"} strokeWidth={2} strokeDasharray="5 4" />
                <text x={x(mk.p)} y={-8} textAnchor="middle" fontSize={12} fontWeight={700} fill={mk.color ?? "var(--accent)"}>{mk.label}</text>
              </g>
            )
          ))}
          {threshold != null && (
            onThreshold ? (
              <DragLine x={xt} value={threshold} onChange={onThreshold} top={-2} bottom={h} min={0} max={1} step={0.01}
                label="Threshold" valueText={announce ?? `Threshold ${fmtT(threshold)}`} color="var(--ink)" text={fmtT(threshold)} knob />
            ) : (
              <line x1={x(threshold)} x2={x(threshold)} y1={-4} y2={h} stroke="var(--ink)" strokeWidth={2.2} />
            )
          )}
        </g>
      </svg>
      <Tooltip tip={tip} />
      <LiveRegion text={announce} />
    </div>
  );
}

function countOf(a: ArrayLike<number>, v: number) {
  let c = 0;
  for (let i = 0; i < a.length; i++) if (a[i] === v) c++;
  return c;
}

// ======================================================================
// Mirrored class histograms with a draggable threshold
// ======================================================================
export function ProbHistogram({
  p, y, threshold, onThreshold, bins = 50, height = 260, posLabel = "Readmitted", negLabel = "Not readmitted", xMax = 1,
  xLabel = "Estimated probability", announce,
}: {
  p: ArrayLike<number>; y: ArrayLike<number>; threshold?: number; onThreshold?: (t: number) => void; bins?: number; height?: number;
  posLabel?: string; negLabel?: string; xMax?: number; xLabel?: string; announce?: string;
}) {
  const h = useMemo(() => {
    const a = new Array(bins).fill(0), b = new Array(bins).fill(0);
    let na = 0, nb = 0;
    for (let i = 0; i < p.length; i++) {
      const k = Math.min(bins - 1, Math.floor((p[i] / xMax) * bins + 1e-9));
      if (y[i] === 1) {
        a[k]++;
        na++;
      } else {
        b[k]++;
        nb++;
      }
    }
    return { a: a.map((v) => v / Math.max(1, na)), b: b.map((v) => v / Math.max(1, nb)), na, nb };
  }, [p, y, bins, xMax]);
  const mx = Math.max(...h.a, ...h.b, 0.01) * 1.1;
  const bw = xMax / bins;
  return (
    <div>
      <Plot height={height} xDomain={[0, xMax]} yDomain={[-mx, mx]} nice={false} yTicks={[-mx * 0.8, 0, mx * 0.8]}
        yFormat={(v) => (v === 0 ? "0" : `${Math.round(Math.abs(v) * 100)}%`)} xLabel={xLabel} yLabel="Share of class in bin"
        margin={{ top: threshold != null ? 30 : 12 }} ariaLabel="Mirrored histograms of predicted probability by actual class">
        {(ctx) => (
          <g>
            {h.a.map((v, k) => {
              const lo = k * bw;
              const flagged = threshold != null && lo >= threshold - 1e-9;
              return v > 0 && <rect key={`a${k}`} x={ctx.x(lo) + 0.5} width={Math.max(0.5, ctx.x(bw) - ctx.x(0) - 1)} y={ctx.y(v)} height={ctx.y(0) - ctx.y(v)}
                fill={threshold == null ? "var(--cls-pos)" : flagged ? "var(--tp)" : "var(--fn)"} />;
            })}
            {h.b.map((v, k) => {
              const lo = k * bw;
              const flagged = threshold != null && lo >= threshold - 1e-9;
              return v > 0 && <rect key={`b${k}`} x={ctx.x(lo) + 0.5} width={Math.max(0.5, ctx.x(bw) - ctx.x(0) - 1)} y={ctx.y(0)} height={ctx.y(-v) - ctx.y(0)}
                fill={threshold == null ? "var(--cls-neg)" : flagged ? "var(--fp)" : "var(--tn)"} opacity={0.85} />;
            })}
            <line x1={0} x2={ctx.w} y1={ctx.y(0)} y2={ctx.y(0)} stroke="var(--ink-2)" />
            <text x={ctx.w - 4} y={ctx.y(mx) + 14} textAnchor="end" fontSize={11.5} fontWeight={650} fill="var(--ink-2)">▲ {posLabel} ({fmtInt(h.na)})</text>
            <text x={ctx.w - 4} y={ctx.y(-mx) - 6} textAnchor="end" fontSize={11.5} fontWeight={650} fill="var(--ink-2)">▼ {negLabel} ({fmtInt(h.nb)})</text>
            {threshold != null && (onThreshold ? (
              <DragLine x={ctx.x as typeof ctx.x & { invert: (v: number) => number }} value={threshold} onChange={onThreshold} top={0} bottom={ctx.h}
                min={0} max={xMax} step={0.01} label="Threshold" valueText={announce ?? `Threshold ${fmtT(threshold)}`} color="var(--ink)" text={fmtT(threshold)} knob />
            ) : (
              <line x1={ctx.x(threshold)} x2={ctx.x(threshold)} y1={0} y2={ctx.h} stroke="var(--ink)" strokeWidth={2.2} />
            ))}
          </g>
        )}
      </Plot>
      <LiveRegion text={announce} />
    </div>
  );
}

// ======================================================================
// ROC plot
// ======================================================================
export interface RocExtra { fpr: number[]; tpr: number[]; color: string; label: string; dash?: string }

export function RocPlot({
  fpr, tpr, k, onPick, marks, shadeArea, polyline, extra, color = "var(--cls-pos)", maxSize = 440, faint, pointText, label = "ROC curve",
}: {
  fpr: number[]; tpr: number[]; k?: number; onPick?: (k: number) => void; marks?: { fpr: number; tpr: number; label: string }[];
  shadeArea?: boolean; polyline?: { fpr: number; tpr: number }[] | null; extra?: RocExtra[]; color?: string; maxSize?: number;
  faint?: boolean; pointText?: string; label?: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const size = Math.max(220, Math.min(width, maxSize));
  const margin = { top: 14, right: 16, bottom: 46, left: 62 };
  const height = size - margin.left - margin.right + margin.top + margin.bottom;
  return (
    <div ref={ref} style={{ width: "100%" }}>
      <div style={{ width: size, maxWidth: "100%" }}>
        <Plot height={height} xDomain={[0, 1]} yDomain={[0, 1]} nice={false} margin={margin} xTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]} yTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
          xLabel="False positive rate (FPR = 1 − specificity)" yLabel="True positive rate (TPR = sensitivity)" ariaLabel={label}>
          {(ctx) => <RocBody ctx={ctx} fpr={fpr} tpr={tpr} k={k} onPick={onPick} marks={marks} shadeArea={shadeArea} polyline={polyline}
            extra={extra} color={color} faint={faint} pointText={pointText} />}
        </Plot>
      </div>
    </div>
  );
}

function RocBody({
  ctx, fpr, tpr, k, onPick, marks, shadeArea, polyline, extra, color, faint, pointText,
}: {
  ctx: PlotCtx; fpr: number[]; tpr: number[]; k?: number; onPick?: (k: number) => void; marks?: { fpr: number; tpr: number; label: string }[];
  shadeArea?: boolean; polyline?: { fpr: number; tpr: number }[] | null; extra?: RocExtra[]; color: string; faint?: boolean; pointText?: string;
}) {
  const { x, y } = ctx;
  const ref = useRef<SVGGElement>(null);
  const drag = useRef(false);
  const path = line<number>().x((i) => x(fpr[i])).y((i) => y(tpr[i]))(fpr.map((_, i) => i)) ?? "";
  const areaPath = `${path} L${x(1)},${y(0)} L${x(0)},${y(0)} Z`;
  const nearest = (cx: number, cy: number) => {
    if (!ref.current) return k ?? 0;
    const l = localPoint(ref.current, cx, cy);
    let best = 0, bd = Infinity;
    for (let i = 0; i < fpr.length; i++) {
      const d = (x(fpr[i]) - l.x) ** 2 + (y(tpr[i]) - l.y) ** 2;
      if (d < bd) {
        bd = d;
        best = i;
      }
    }
    return best;
  };
  return (
    <g ref={ref}>
      {shadeArea && <path d={areaPath} fill={color} opacity={0.14} />}
      <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="var(--ink-2)" strokeDasharray="5 5" strokeWidth={1.4} />
      {ctx.w > 300 && (
        <text x={x(0.62)} y={y(0.58)} fontSize={11.5} fill="var(--muted)" transform={`rotate(${-Math.atan2(y(0) - y(1), x(1) - x(0)) * 180 / Math.PI} ${x(0.62)} ${y(0.58)})`}>
          random flagging (AUROC 0.5)
        </text>
      )}
      {extra?.map((e) => (
        <path key={e.label} d={line<number>().x((i) => x(e.fpr[i])).y((i) => y(e.tpr[i]))(e.fpr.map((_, i) => i)) ?? ""}
          fill="none" stroke={e.color} strokeWidth={2.2} strokeDasharray={e.dash} />
      ))}
      <path d={path} fill="none" stroke={color} strokeWidth={faint ? 1.5 : 2.6} opacity={faint ? 0.35 : 1} strokeLinejoin="round" />
      {polyline && (
        <g>
          <path d={line<{ fpr: number; tpr: number }>().x((d) => x(d.fpr)).y((d) => y(d.tpr))(polyline) ?? ""} fill="none" stroke={color} strokeWidth={2.6} />
          {polyline.map((d, i) => <circle key={i} cx={x(d.fpr)} cy={y(d.tpr)} r={4} fill={color} />)}
        </g>
      )}
      {marks?.map((mk) => (
        <g key={mk.label}>
          <circle cx={x(mk.fpr)} cy={y(mk.tpr)} r={5} fill="var(--surface)" stroke={color} strokeWidth={2} />
          <text x={x(mk.fpr) + 9} y={y(mk.tpr) + 14} fontSize={12} fontWeight={650} fill="var(--ink-2)">{mk.label}</text>
        </g>
      ))}
      {k != null && (
        <g
          className="roc-point" role={onPick ? "slider" : undefined} tabIndex={onPick ? 0 : undefined}
          aria-label="Operating point on the ROC curve" aria-valuemin={0} aria-valuemax={fpr.length - 1} aria-valuenow={k}
          aria-valuetext={pointText ?? `FPR ${fmt(fpr[k], 2)}, TPR ${fmt(tpr[k], 2)}`}
          onKeyDown={(e) => {
            if (!onPick) return;
            let nk: number | null = null;
            if (e.key === "ArrowRight" || e.key === "ArrowUp") nk = Math.min(fpr.length - 1, k + 1);
            else if (e.key === "ArrowLeft" || e.key === "ArrowDown") nk = Math.max(0, k - 1);
            else if (e.key === "Home") nk = 0;
            else if (e.key === "End") nk = fpr.length - 1;
            if (nk == null) return;
            e.preventDefault();
            e.stopPropagation();
            onPick(nk);
          }}
          onPointerDown={(e) => {
            if (!onPick) return;
            (e.currentTarget as SVGGElement).setPointerCapture(e.pointerId);
            drag.current = true;
          }}
          onPointerMove={(e) => drag.current && onPick && onPick(nearest(e.clientX, e.clientY))}
          onPointerUp={(e) => {
            drag.current = false;
            (e.currentTarget as SVGGElement).releasePointerCapture(e.pointerId);
          }}
          style={{ touchAction: "none" }}
        >
          <line x1={x(fpr[k])} x2={x(fpr[k])} y1={y(tpr[k])} y2={y(0)} stroke="var(--ink-2)" strokeDasharray="2 3" />
          <line x1={x(0)} x2={x(fpr[k])} y1={y(tpr[k])} y2={y(tpr[k])} stroke="var(--ink-2)" strokeDasharray="2 3" />
          <circle cx={x(fpr[k])} cy={y(tpr[k])} r={14} fill="transparent" />
          <circle cx={x(fpr[k])} cy={y(tpr[k])} r={8} fill={color} stroke="var(--surface)" strokeWidth={2.5} />
        </g>
      )}
    </g>
  );
}

// ======================================================================
// Patient waffle: 100 patients that rearrange into the matrix
// ======================================================================
export type WaffleArrangement = "grid" | "byActual" | "byCell";

export function PatientWaffle({
  y, pred, arrangement, showOutcome = true, highlightCell, posLabel = "readmitted", negLabel = "not readmitted", maxWidth = 460,
}: {
  y: number[]; pred: number[] | null; arrangement: WaffleArrangement; showOutcome?: boolean; highlightCell?: CellKey | null;
  posLabel?: string; negLabel?: string; maxWidth?: number;
}) {
  const n = y.length;
  const cell = 22, r = 7.5;
  const cols = 9;
  // deterministic scatter for the "at discharge" grid: patient i sits at position (37·i) mod n
  const scatter = (i: number) => (37 * i + 11) % n;
  const cellOfI = (i: number): CellKey => cellOf(y[i], pred ? pred[i] === 1 : false);

  const pos: { x: number; y: number }[] = new Array(n);
  let W = 10 * cell, H = 10 * cell;
  const labels: { x: number; y: number; text: string; k?: CellKey }[] = [];
  if (arrangement === "grid") {
    for (let i = 0; i < n; i++) {
      const s = scatter(i);
      pos[i] = { x: (s % 10) * cell + cell / 2, y: Math.floor(s / 10) * cell + cell / 2 };
    }
  } else if (arrangement === "byActual") {
    const order = [...Array(n).keys()].sort((a, b) => y[b] - y[a] || a - b);
    order.forEach((i, k) => (pos[i] = { x: (k % 10) * cell + cell / 2, y: Math.floor(k / 10) * cell + cell / 2 + 18 }));
    const nPos = y.filter((v) => v === 1).length;
    labels.push({ x: 0, y: 12, text: `${nPos} ${posLabel} first, then ${n - nPos} ${negLabel}` });
    H = Math.ceil(n / 10) * cell + 18;
  } else {
    const groups: Record<CellKey, number[]> = { tp: [], fn: [], fp: [], tn: [] };
    for (let i = 0; i < n; i++) groups[cellOfI(i)].push(i);
    const rowsOf = (k: CellKey) => Math.max(1, Math.ceil(groups[k].length / cols));
    const gapX = 26, head = 20, gapY = 14;
    const topH = Math.max(rowsOf("tp"), rowsOf("fn")) * cell;
    const botH = Math.max(rowsOf("fp"), rowsOf("tn")) * cell;
    const origin: Record<CellKey, [number, number]> = {
      tp: [0, head], fn: [cols * cell + gapX, head], fp: [0, head + topH + gapY + head], tn: [cols * cell + gapX, head + topH + gapY + head],
    };
    (Object.keys(groups) as CellKey[]).forEach((k) => {
      const [ox, oy] = origin[k];
      groups[k].forEach((i, j) => (pos[i] = { x: ox + (j % cols) * cell + cell / 2, y: oy + Math.floor(j / cols) * cell + cell / 2 }));
      labels.push({ x: ox, y: oy - 6, text: `${CELL_ABBR[k]} ${CELL_GLYPH[k]} ${groups[k].length}`, k });
    });
    W = 2 * cols * cell + gapX;
    H = head + topH + gapY + head + botH;
  }

  return (
    <div className="waffle" style={{ maxWidth }}>
      <svg viewBox={`-4 -4 ${Math.max(W, 10 * cell) + 8} ${H + 8}`} width="100%" role="img"
        aria-label={`${n} patients, ${arrangement === "grid" ? (showOutcome ? "outcomes shown" : "at discharge, outcomes not yet known") : arrangement === "byActual" ? "grouped by actual outcome" : "sorted into the four cells of the confusion matrix"}`}>
        {labels.map((l) => (
          <text key={l.text} x={l.x} y={l.y} className="waffle-label" style={l.k ? { fill: `color-mix(in srgb, ${CELL_COLOR[l.k]} 72%, var(--ink))` } : undefined}>{l.text}</text>
        ))}
        {y.map((v, i) => {
          const known = showOutcome || arrangement !== "grid";
          const k = cellOfI(i);
          const c = !known ? "var(--muted)" : arrangement === "byCell" ? CELL_COLOR[k] : v === 1 ? "var(--cls-pos)" : "var(--cls-neg)";
          const dim = highlightCell && arrangement === "byCell" ? k !== highlightCell : highlightCell ? k !== highlightCell : false;
          return (
            <g key={i} className="pt" style={{ transform: `translate(${pos[i].x}px, ${pos[i].y}px)`, opacity: dim ? 0.18 : 1 }}>
              <circle r={known && v === 0 ? r - 1 : r} fill={!known ? c : v === 1 ? c : "var(--surface)"} stroke={c} strokeWidth={known && v === 0 ? 2 : 1}
                fillOpacity={known ? 1 : 0.55} />
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ======================================================================
// Calibration (reliability) plot
// ======================================================================
export function CalibrationPlot({
  series, histogram, max = 1, height = 320,
}: {
  series: { id: string; label: string; color: string; bins: CalBin[]; dash?: string }[];
  histogram?: { p: ArrayLike<number>; color: string; label: string }[];
  max?: number; height?: number;
}) {
  const hist = useMemo(() => {
    if (!histogram) return null;
    const nb = 40;
    return histogram.map((hh) => {
      const c = new Array(nb).fill(0);
      for (let i = 0; i < hh.p.length; i++) c[Math.min(nb - 1, Math.floor((hh.p[i] / max) * nb))]++;
      return { ...hh, share: c.map((v) => v / hh.p.length), nb };
    });
  }, [histogram, max]);
  const hmax = hist ? Math.max(...hist.flatMap((hh) => hh.share)) : 1;
  return (
    <div>
      <Plot height={height} xDomain={[0, max]} yDomain={[0, max]} nice={false} xLabel="Mean predicted probability (each dot = one tenth of patients)"
        yLabel="Observed rate" ariaLabel="Calibration plot: predicted vs observed rate by decile">
        {({ x, y, show, hide }) => (
          <g>
            <line x1={x(0)} y1={y(0)} x2={x(max)} y2={y(max)} stroke="var(--ink-2)" strokeDasharray="5 5" />
            <text x={x(max * 0.55)} y={y(max * 0.55) - 8} fontSize={11.5} fill="var(--muted)"
              transform={`rotate(${(-Math.atan2(y(0) - y(max), x(max) - x(0)) * 180) / Math.PI} ${x(max * 0.55)} ${y(max * 0.55) - 8})`}>perfectly calibrated</text>
            {series.map((s) => (
              <g key={s.id}>
                <path d={line<CalBin>().x((b) => x(b.meanPred)).y((b) => y(b.observed))(s.bins) ?? ""} fill="none" stroke={s.color} strokeWidth={2.2} strokeDasharray={s.dash} />
                {s.bins.map((b, i) => (
                  <circle key={i} cx={x(b.meanPred)} cy={y(b.observed)} r={4.5} fill={s.color} stroke="var(--surface)" strokeWidth={1.2}
                    onMouseEnter={() => show(x(b.meanPred), y(b.observed), (
                      <>
                        <TipRow color={s.color} label={<b>{s.label}</b>} value="" />
                        <TipRow label="Predicted (mean)" value={`${(b.meanPred * 100).toFixed(1)}%`} />
                        <TipRow label="Observed" value={`${(b.observed * 100).toFixed(1)}%`} />
                        <TipRow label="Patients" value={fmtInt(b.n)} />
                      </>
                    ))} onMouseLeave={hide} />
                ))}
              </g>
            ))}
          </g>
        )}
      </Plot>
      {hist && (
        <Plot height={110} xDomain={[0, max]} yDomain={[0, hmax]} nice={false} yTicks={[]} margin={{ top: 4 }}
          xLabel="Predicted probability: where the patients are" ariaLabel="Histogram of predicted probabilities">
          {({ x, y }) => (
            <g>
              {hist.map((hh, s) => hh.share.map((v, k) => v > 0 && (
                <rect key={`${s}-${k}`} x={x((k / hh.nb) * max) + 0.5 + s * 1.5} width={Math.max(0.5, x(max / hh.nb) - x(0) - 2)} y={y(v)} height={y(0) - y(v)}
                  fill={hh.color} opacity={0.55} />
              )))}
            </g>
          )}
        </Plot>
      )}
    </div>
  );
}

// ======================================================================
// Probability map: a 2-feature model's probability surface with the decision boundary
// ======================================================================
/** Hue runs from --cls-neg to --cls-pos; strength over the surface also grows with p, so the gradient reads in both themes. */
const mapColor = (pct: number) => `color-mix(in srgb, color-mix(in srgb, var(--cls-pos) ${pct}%, var(--cls-neg)) ${10 + pct * 0.62}%, var(--surface))`;

export function ProbabilityMap({
  pair, grid, threshold, showPoints = true, height = 420, tip,
}: {
  pair: PlaygroundPair; grid: Grid; threshold: number; showPoints?: boolean; height?: number;
  tip?: (ix: number, iy: number) => ReactNode;
}) {
  const xs = pair.x_axis, ys = pair.y_axis;
  const nx = xs.length, ny = ys.length;
  const dx = xs[1] - xs[0], dy = ys[1] - ys[0];
  const [hover, setHover] = useState<[number, number] | null>(null);
  const boundary = useMemo(() => {
    const t = Math.round(threshold * 100);
    if (t <= 0) return null;
    return contours().size([nx, ny]).thresholds([t])(grid.prob)[0];
  }, [grid, threshold, nx, ny]);
  return (
    <Plot height={height} xDomain={[xs[0] - dx / 2, xs[nx - 1] + dx / 2]} yDomain={[ys[0] - dy / 2, ys[ny - 1] + dy / 2]} nice={false}
      xLabel={`${pair.x_label} (${pair.x_units})`} yLabel={`${pair.y_label} (${pair.y_units})`}
      ariaLabel={`Probability map for ${pair.label}, with the decision boundary at ${fmtT(threshold)}`}>
      {(ctx) => {
        const { x, y } = ctx;
        // contour coordinates are in grid-cell units: c = 0 is the left edge of cell 0
        const gx = (c: number) => x(xs[0] - dx / 2 + c * dx);
        const gy = (c: number) => y(ys[0] - dy / 2 + c * dy);
        const bpath = boundary?.coordinates.map((poly) => poly.map((ring) => `M${ring.map(([a, b]) => `${gx(a)},${gy(b)}`).join("L")}Z`).join("")).join("") ?? "";
        const cw = Math.abs(x(dx) - x(0)) + 0.6, ch = Math.abs(y(0) - y(dy)) + 0.6;
        return (
          <g>
            {grid.prob.map((v, idx) => {
              const ix = idx % nx, iy = Math.floor(idx / nx);
              return <rect key={idx} x={x(xs[ix] - dx / 2)} y={y(ys[iy] + dy / 2)} width={cw} height={ch}
                style={{ fill: mapColor(v) }} />;
            })}
            {showPoints && pair.points.x.map((px, i) => {
              const pos = pair.points.label[i] === 1;
              return <circle key={i} cx={x(px)} cy={y(pair.points.y[i])} r={pos ? 3.4 : 3} fill={pos ? "var(--cls-pos)" : "var(--surface)"}
                stroke={pos ? "var(--surface)" : "var(--cls-neg)"} strokeWidth={pos ? 0.8 : 1.4} opacity={0.9} />;
            })}
            <path d={bpath} fill="none" stroke="var(--ink)" strokeWidth={2.6} />
            <path d={bpath} fill="none" stroke="var(--surface)" strokeWidth={1} strokeDasharray="4 4" />
            {hover && (
              <rect x={x(xs[hover[0]] - dx / 2)} y={y(ys[hover[1]] + dy / 2)} width={cw} height={ch} fill="none" stroke="var(--ink)" strokeWidth={1.5} />
            )}
            <rect width={ctx.w} height={ctx.h} fill="transparent"
              onMouseMove={(e) => {
                const box = (e.currentTarget as SVGRectElement).getBoundingClientRect();
                const vx = x.invert(e.clientX - box.left), vy = y.invert(e.clientY - box.top);
                const ix = clamp(Math.round((vx - xs[0]) / dx), 0, nx - 1), iy = clamp(Math.round((vy - ys[0]) / dy), 0, ny - 1);
                setHover([ix, iy]);
                const p = grid.prob[iy * nx + ix];
                ctx.show(x(xs[ix]), y(ys[iy]) - 6, (
                  <>
                    <TipRow label={<b>{pair.x_label} {fmt(xs[ix], 0)} · {pair.y_label} {fmt(ys[iy], 0)}</b>} value="" />
                    <TipRow label="Estimated probability" value={`${p}%`} />
                    <TipRow label="Predicted" value={p >= threshold * 100 - 1e-9 ? pair.pos_label : pair.neg_label} />
                    {tip?.(ix, iy)}
                  </>
                ));
              }}
              onMouseLeave={() => {
                setHover(null);
                ctx.hide();
              }} />
          </g>
        );
      }}
    </Plot>
  );
}

export function ProbabilityLegend({ pos, neg, threshold }: { pos: string; neg: string; threshold: number }) {
  return (
    <div className="legend">
      <span className="legend-item">
        <span style={{ width: 90, height: 10, borderRadius: 3, display: "inline-block", border: "1px solid var(--border)", background: `linear-gradient(90deg, ${mapColor(0)}, ${mapColor(50)}, ${mapColor(100)})` }} />
        probability 0 → 1
      </span>
      <span className="legend-item"><span className="dot" style={{ background: "var(--cls-pos)" }} /> {pos}</span>
      <span className="legend-item"><span className="dot" style={{ background: "var(--surface)", border: "1.5px solid var(--cls-neg)" }} /> {neg}</span>
      <span className="legend-item"><span className="legend-line" style={{ background: "var(--ink)" }} /> boundary: p = {fmtT(threshold)}</span>
    </div>
  );
}

// ======================================================================
// Pair grid: every (readmitted, not readmitted) pair, won or lost
// ======================================================================
export function PairGrid({
  pos, neg, drawn, showAll, current,
}: { pos: number[]; neg: number[]; drawn: Set<number>; showAll: boolean; current?: [number, number] | null }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const nP = pos.length, nN = neg.length;
  const left = 46, top = 8;
  const cw = Math.max(2, (width - left - 4) / nN);
  const ch = Math.max(10, Math.min(18, cw * 2.2));
  const height = top + nP * ch + 30;
  const pid = useId().replace(/:/g, "");
  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img" aria-label={`${nP} by ${nN} grid of patient pairs: ${drawn.size} drawn`}>
        <defs>
          <pattern id={`loss-${pid}`} width={4} height={4} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width={4} height={4} fill="var(--fn)" />
            <line x1={0} y1={0} x2={0} y2={4} stroke="var(--surface)" strokeWidth={1.4} />
          </pattern>
        </defs>
        {pos.map((pp, i) => (
          <text key={i} x={left - 6} y={top + i * ch + ch / 2} dy="0.32em" textAnchor="end" fontSize={10.5} fill="var(--ink-2)" className="tnum">{pp.toFixed(3)}</text>
        ))}
        {pos.map((pp, i) => neg.map((nn, j) => {
          const idx = i * nN + j;
          const on = showAll || drawn.has(idx);
          const win = pp > nn;
          return (
            <rect key={idx} x={left + j * cw} y={top + i * ch} width={Math.max(1, cw - (cw > 4 ? 1 : 0.3))} height={ch - 1.5}
              fill={!on ? "var(--surface-2)" : win ? "var(--tp)" : `url(#loss-${pid})`} opacity={on ? (drawn.has(idx) || !showAll ? 1 : 0.55) : 1} />
          );
        }))}
        {current && (
          <rect x={left + current[1] * cw - 1.5} y={top + current[0] * ch - 1.5} width={cw + 2} height={ch + 1.5} fill="none" stroke="var(--ink)" strokeWidth={2} />
        )}
        <text x={left} y={top + nP * ch + 18} fontSize={11} fill="var(--muted)">← not readmitted patients, highest probability first ({nN}) →</text>
      </svg>
    </div>
  );
}

// ======================================================================
// Homework and quizzes
// ======================================================================
export function TryIt({ tasks, answers, title = "Try it · homework" }: { tasks: ReactNode[]; answers: ReactNode[]; title?: string }) {
  return (
    <div className="tryit">
      <div className="tryit-tag">✎ {title}</div>
      <ol>{tasks.map((t, i) => <li key={i}>{t}</li>)}</ol>
      <PauseThink label="Check your answers" prompt="Work through the tasks first, then check.">
        <ol style={{ margin: 0, paddingLeft: 22 }}>{answers.map((a, i) => <li key={i} style={{ marginBottom: 4 }}>{a}</li>)}</ol>
      </PauseThink>
    </div>
  );
}

export function Quiz({ n, of = 3, q, children }: { n: number; of?: number; q: ReactNode; children: ReactNode }) {
  return (
    <PauseThink label={`Check your understanding · ${n} of ${of}`} prompt={q}>
      <div>{children}</div>
    </PauseThink>
  );
}
