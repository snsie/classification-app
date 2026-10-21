import { useMemo, useState } from "react";
import { curveStepAfter, line } from "d3-shape";
import { announceCounts, ConfusionMatrix, ProbabilityStrip, ProbHistogram } from "../charts/classify";
import { Plot } from "../charts/kit";
import { Callout, Card, DataChip, Loading, PauseThink, Segmented, Slider, SourceNote, Stat } from "../components/ui";
import {
  confusion, DatasetData, decodeP, fmt, fmtInt, metrics, minCostThreshold, MODEL_A, sliderWindow, thresholdForCapacity,
  thresholdForSensitivity, useData,
} from "../data";

type Src = "modelA" | "real";
type By = "capacity" | "cost" | "target";
const GRID = Array.from({ length: 101 }, (_, i) => i / 100);

export default function Job() {
  const [src, setSrc] = useState<Src>("modelA");
  const [by, setBy] = useState<By>("capacity");
  const [cap, setCap] = useState(26);
  const [cMiss, setCMiss] = useState(5);
  const [target, setTarget] = useState(0.9);
  const ds = useData<DatasetData>(src === "real" ? "readmission" : null);

  const data = useMemo(() => {
    if (src === "modelA") return { p: MODEL_A.p as ArrayLike<number>, y: MODEL_A.y as ArrayLike<number>, n: 100 };
    if (!ds || !ds.cv.oof.gbm) return null;
    return { p: decodeP(ds.cv.oof.gbm), y: ds.y, n: ds.n_rows };
  }, [src, ds]);

  const grid = useMemo(() => (data ? GRID.map((g) => confusion(data.p, data.y, g)) : null), [data]);

  if (!data || !grid) return <Loading />;
  const { p, y, n } = data;
  const P = src === "modelA" ? 10 : (ds?.n_pos ?? 0);

  // the chosen threshold window: every t in (below, top] gives the same predictions
  let top: number, below: number;
  if (by === "capacity") {
    const r = thresholdForCapacity(p, Math.round((cap / 100) * n));
    top = r.top;
    below = r.below;
  } else if (by === "cost") {
    const r = minCostThreshold(p, y, cMiss, 1);
    top = r.top;
    below = r.below;
  } else {
    top = thresholdForSensitivity(p, y, target);
    below = -Infinity;
    for (let i = 0; i < p.length; i++) if (p[i] < top && p[i] > below) below = p[i];
  }
  const t = top;
  const c = confusion(p, y, t);
  const m = metrics(c);
  const win = sliderWindow({ top, below });
  const tText = !isFinite(top) ? "above every patient (no one flagged)" : win ? (win[0] === win[1] ? win[0].toFixed(2) : `${win[0].toFixed(2)}–${win[1].toFixed(2)}`) : `≈ ${top.toFixed(3)}`;
  const per100 = (v: number) => (v / n) * 100;
  const cost = cMiss * c.fn + c.fp;

  const yOf = (g: (typeof grid)[number]) =>
    by === "capacity" ? per100(g.tp + g.fp) : by === "cost" ? per100(cMiss * g.fn + g.fp) : g.tp / (g.tp + g.fn);
  const yMax = by === "capacity" ? 100 : by === "cost" ? Math.max(...grid.map(yOf)) * 1.05 : 1;
  const cur = by === "capacity" ? per100(m.flagged) : by === "cost" ? per100(cost) : m.sens;
  const reach = (k: number) => confusion(p, y, thresholdForCapacity(p, Math.round((k / 100) * n)).top).tp;

  return (
    <div className="stack">
      <div className="controls-bar">
        <Segmented label="Data" value={src} onChange={setSrc}
          options={[{ value: "modelA", label: "Model A (100 patients)" }, { value: "real", label: "Real readmissions (GBM out-of-fold)" }]} />
        <Segmented label="Choose by" value={by} onChange={setBy}
          options={[{ value: "capacity", label: "Call capacity" }, { value: "cost", label: "Cost of a miss" }, { value: "target", label: "Target sensitivity" }]} />
        {by === "capacity" && <Slider label="Calls per 100 discharges" value={cap} min={0} max={100} onChange={setCap} width={300} />}
        {by === "cost" && (
          <Slider label="A missed readmission costs as much as N unneeded calls" value={cMiss} min={1} max={30} onChange={setCMiss} format={(v) => `N = ${v}`} width={360} />
        )}
        {by === "target" && (
          <Segmented label="Target sensitivity" value={target} onChange={setTarget}
            options={[0.5, 0.8, 0.9, 1].map((v) => ({ value: v, label: `${v * 100}%` }))} />
        )}
      </div>

      <div className="split-wide">
        <Card title={by === "capacity" ? "Calls per 100 discharges at every threshold" : by === "cost" ? "Total cost per 100 discharges (in units of one call)" : "Sensitivity at every threshold"}
          sub={by === "capacity" ? "Pick the highest threshold whose calls fit the team's capacity." : by === "cost" ? `Cost = ${cMiss} × missed readmissions + 1 × false alarms. The dot is the cheapest threshold.` : "Pick the highest threshold that still reaches the target."}
          chip={src === "modelA" ? <DataChip /> : <DataChip n={n} unit="discharges" />}>
          <Plot height={250} xDomain={[0, 1]} yDomain={[0, yMax]} nice={false} xLabel="Threshold" xTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
            yLabel={by === "capacity" ? "Calls per 100" : by === "cost" ? "Cost per 100" : "Sensitivity"} yFormat={by === "target" ? (v) => `${Math.round(v * 100)}%` : undefined}
            ariaLabel="Decision criterion against threshold">
            {({ x, y: ys }) => (
              <g>
                <path d={line<number>().curve(curveStepAfter).x((i) => x(GRID[i])).y((i) => ys(yOf(grid[i])))(GRID.map((_, i) => i)) ?? ""}
                  fill="none" stroke="var(--accent)" strokeWidth={2.4} />
                {by === "capacity" && <line x1={0} x2={x(1)} y1={ys(cap)} y2={ys(cap)} stroke="var(--ink-2)" strokeDasharray="5 4" />}
                {by === "target" && <line x1={0} x2={x(1)} y1={ys(target)} y2={ys(target)} stroke="var(--ink-2)" strokeDasharray="5 4" />}
                {isFinite(t) && (
                  <>
                    <line x1={x(Math.min(1, t))} x2={x(Math.min(1, t))} y1={ys(0)} y2={ys(cur)} stroke="var(--ink)" strokeDasharray="2 3" />
                    <circle cx={x(Math.min(1, t))} cy={ys(cur)} r={7} fill="var(--ink)" stroke="var(--surface)" strokeWidth={2} />
                  </>
                )}
              </g>
            )}
          </Plot>
          {src === "modelA" ? (
            <ProbabilityStrip p={p} labels={y} threshold={isFinite(t) ? Math.min(1, t) : 1} colorBy="result" announce={announceCounts(t, c)} />
          ) : (
            <ProbHistogram p={p} y={y} threshold={isFinite(t) ? Math.min(1, t) : 1} bins={100} height={230} xLabel="Estimated probability of readmission (GBM, out-of-fold)" />
          )}
        </Card>
        <div className="stack">
          <Card>
            <div className="stat-label">Threshold</div>
            <div className="stat-value tnum" style={{ fontSize: "1.6rem" }}>{tText}</div>
            <div className="stat-note">{win && win[0] !== win[1] ? "Any threshold in this range gives the same calls." : ""}</div>
          </Card>
          <div className="grid-2" style={{ gap: 12 }}>
            <Stat label="Calls" value={fmtInt(m.flagged)} note={src === "modelA" ? "per 100 patients" : `${fmt(per100(m.flagged), 1)} per 100`} />
            <Stat label="Readmissions reached" value={`${fmtInt(c.tp)} of ${fmtInt(P)}`} color="var(--tp)" note={`sensitivity ${Math.round(m.sens * 100)}%`} />
            <Stat label="Wasted calls" value={fmtInt(c.fp)} color="var(--fp)" note="false alarms" />
            <Stat label="Hit rate" value={c.tp ? `1 in ${fmt(m.flagged / c.tp, 1)}` : "–"} note="calls reach a future readmission" />
          </div>
          {by === "cost" && <Stat label="Total cost" value={fmtInt(cost)} note={`${cMiss} × ${fmtInt(c.fn)} missed + ${fmtInt(c.fp)} false alarms (in calls)`} />}
          <Card title="Confusion matrix">
            <ConfusionMatrix counts={c} compact showNames={false} />
          </Card>
        </div>
      </div>

      <PauseThink prompt="The team doubles its capacity from 26 to 52 calls per 100 discharges. How many more readmissions does it reach?">
        {src === "modelA" ? (
          <>+2: from {reach(26)} to {reach(52)} of 10. The extra 26 calls reach only 2 more readmitted patients (1 in 13). The ranking puts most readmissions near the top, so each extra call is worth less.</>
        ) : (
          <>From {fmtInt(reach(26))} to {fmtInt(reach(52))} of {fmtInt(P)} (+{fmtInt(reach(52) - reach(26))}). With a weak model the extra calls still reach readmissions, but at a low hit rate.</>
        )}
      </PauseThink>

      <div className="grid-2">
        <Callout tone="good"><b>Reach for…</b> the people who act on the flag. Choose the threshold with them: their capacity, and what a missed case costs compared with a wasted call.</Callout>
        <Callout tone="warn"><b>Watch out for…</b> accuracy as the yardstick: Model A's most accurate thresholds (91%) catch at most 4 of 10 readmissions.</Callout>
      </div>
      {src === "real" && ds && (
        <SourceNote>
          {fmtInt(ds.n_rows)} discharges, {fmtInt(ds.n_pos)} readmitted. Probabilities are LightGBM's out-of-fold predictions from 5-fold cross-validation, so each
          discharge was scored by a model that never saw it. Bars show the share of each class in each 0.01-wide bin.
        </SourceNote>
      )}
    </div>
  );
}
