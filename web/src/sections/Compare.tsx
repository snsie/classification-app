import { useEffect, useState } from "react";
import { scaleLinear } from "d3-scale";
import { TryIt } from "../charts/classify";
import { Plot, TipRow, useWidth } from "../charts/kit";
import { Callout, Card, DataChip, DatasetPicker, Loading, ModelPicker, SourceNote, Toggle } from "../components/ui";
import { DATASET_SHORT, DatasetData, DatasetId, fmt, FITTED_MODELS, load, MODEL_COLOR, MODEL_LABEL, ModelId } from "../data";

const HEAT_ROWS: DatasetId[] = ["diabetes", "readmission", "heart"];

/** A tint capped at 45% accent, so ink text keeps ≥ 4.5:1 contrast in both themes. */
function heat(v: number) {
  const k = Math.max(0, Math.min(1, (v - 0.5) / 0.45));
  return `color-mix(in srgb, var(--accent) ${Math.round(k * 45)}%, var(--surface))`;
}

/** Mean ± SD bars with the chance line at 0.5. */
function CvBars({ ds, models, from }: { ds: DatasetData; models: ModelId[]; from: number }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const lw = width < 520 ? 104 : 170, rh = 42;
  const x = scaleLinear().domain([from, 1]).range([lw, Math.max(lw + 40, width - 70)]);
  const ticks = from === 0 ? [0, 0.25, 0.5, 0.75, 1] : [0.5, 0.6, 0.7, 0.8, 0.9, 1];
  const h = models.length * rh + 30;
  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={h} role="img" aria-label={`Cross-validated AUROC by model, ${ds.name}`}>
        {ticks.map((t) => (
          <g key={t} className="tick">
            <line x1={x(t)} x2={x(t)} y1={0} y2={models.length * rh} />
            <text x={x(t)} y={models.length * rh + 18} textAnchor="middle">{t}</text>
          </g>
        ))}
        <line x1={x(0.5)} x2={x(0.5)} y1={0} y2={models.length * rh} stroke="var(--ink-2)" strokeWidth={1.6} strokeDasharray="5 4" />
        {models.map((m, i) => {
          const mm = ds.cv.models[m]!;
          const cy = i * rh + rh / 2;
          return (
            <g key={m}>
              <text x={lw - 10} y={cy} dy="0.32em" textAnchor="end" fontSize={13} fill="var(--ink-2)">{MODEL_LABEL[m].replace("Foundation (TabICL)", "TabICL")}</text>
              <rect x={x(from)} y={cy - 11} width={Math.max(1, x(mm.mean_auroc) - x(from))} height={22} rx={4} fill={MODEL_COLOR[m]} />
              <line x1={x(mm.mean_auroc - mm.sd_auroc)} x2={x(Math.min(1, mm.mean_auroc + mm.sd_auroc))} y1={cy} y2={cy} stroke="var(--ink)" strokeWidth={1.6} />
              <text x={x(Math.min(1, mm.mean_auroc + mm.sd_auroc)) + 6} y={cy} dy="0.32em" fontSize={13} fontWeight={650} fill="var(--ink)" className="tnum">{fmt(mm.mean_auroc, 3)}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export default function Compare() {
  const [all, setAll] = useState<Record<DatasetId, DatasetData> | null>(null);
  const [dsId, setDsId] = useState<DatasetId>("diabetes");
  const [zero, setZero] = useState(false);
  const [a, setA] = useState<ModelId>("logistic");
  const [b, setB] = useState<ModelId>("tabicl");
  useEffect(() => {
    Promise.all(HEAT_ROWS.map((id) => load<DatasetData>(id))).then((arr) => setAll(Object.fromEntries(arr.map((d) => [d.id, d])) as Record<DatasetId, DatasetData>));
  }, []);
  if (!all) return <Loading />;
  const ds = all[dsId];
  const models = FITTED_MODELS.filter((m) => ds.cv.models[m]);
  const B = models.includes(b) ? b : "rf";
  const fa = ds.cv.models[a]!.folds.map((f) => f.auroc), fb = ds.cv.models[B]!.folds.map((f) => f.auroc);
  const diff = ds.cv.models[B]!.mean_auroc - ds.cv.models[a]!.mean_auroc;
  const sdA = ds.cv.models[a]!.sd_auroc;
  const wins = fb.filter((v, i) => v > fa[i]).length;
  const ext = [Math.min(...fa, ...fb), Math.max(...fa, ...fb)];
  const pad = (ext[1] - ext[0]) * 0.25 + 0.005;
  const dom: [number, number] = [ext[0] - pad, ext[1] + pad];
  const best = (d: DatasetData) => FITTED_MODELS.filter((m) => d.cv.models[m]).reduce((x, y) => (d.cv.models[y]!.mean_auroc > d.cv.models[x]!.mean_auroc ? y : x));

  return (
    <div className="stack">
      <div className="controls-bar">
        <DatasetPicker value={dsId} onChange={setDsId} ids={["diabetes", "readmission"]} />
        <Toggle checked={zero} onChange={setZero}>Start the axis at 0</Toggle>
      </div>

      <div className="split-wide">
        <Card title={`${ds.name}: AUROC on unseen patients`} sub={`Mean ± SD of the same 5 cross-validation folds for every model. ${zero ? "Axis from 0." : "Axis from 0.5 (chance), as on the slide."}`}
          chip={<DataChip n={ds.n_rows} unit={ds.labels.units} />}>
          <CvBars ds={ds} models={models} from={zero ? 0 : 0.5} />
          <p className="small ink2" style={{ margin: 0 }}>
            The baseline (everyone gets the training prevalence) has AUROC 0.5: it is the dashed chance line, not a bar.
          </p>
        </Card>
        <Card title="Is the gap bigger than the spread?" sub="Paired: the same 5 folds for both models">
          <div className="row">
            <ModelPicker value={a} onChange={setA} models={models} label="Model A (x-axis)" short />
            <ModelPicker value={B} onChange={setB} models={models} label="Model B (y-axis)" short />
          </div>
          <Plot height={260} xDomain={dom} yDomain={dom} nice={false} xLabel={`${MODEL_LABEL[a]} AUROC`} yLabel={`${MODEL_LABEL[B]} AUROC`} xTicks={3} yTicks={3}
            ariaLabel="Per-fold AUROC of model B against model A">
            {({ x, y, show, hide }) => (
              <g>
                <line x1={x(dom[0])} y1={y(dom[0])} x2={x(dom[1])} y2={y(dom[1])} stroke="var(--ink-2)" strokeDasharray="5 4" />
                <text x={x(dom[1]) - 6} y={y(dom[0]) - 8} textAnchor="end" fontSize={12} fill="var(--ink-2)">A better</text>
                <text x={x(dom[0]) + 6} y={y(dom[1]) + 16} fontSize={12} fill="var(--ink-2)">B better</text>
                {fa.map((v, i) => (
                  <circle key={i} cx={x(v)} cy={y(fb[i])} r={6} fill={fb[i] > v ? MODEL_COLOR[B] : MODEL_COLOR[a]} stroke="var(--surface)" strokeWidth={1.5}
                    onMouseEnter={() => show(x(v), y(fb[i]), <><TipRow label={<b>Fold {i + 1}</b>} value="" /><TipRow label="A" value={fmt(v, 3)} /><TipRow label="B" value={fmt(fb[i], 3)} /></>)}
                    onMouseLeave={hide} />
                ))}
              </g>
            )}
          </Plot>
          <div className="small ink2">
            {a === B ? "Pick two different models." : (
              <>
                <b>B − A = {diff >= 0 ? "+" : "−"}{fmt(Math.abs(diff), 3)}</b> ≈ {fmt(Math.abs(diff) / sdA, 1)} fold SDs of model A ({fmt(sdA, 3)}).{" "}
                <b>B wins {wins} of 5 folds.</b> {Math.abs(diff) <= sdA ? "Within the noise: weak evidence that one is better." : "Bigger than the fold-to-fold spread."}
              </>
            )}
          </div>
        </Card>
      </div>

      <Card title="Every dataset, every model" sub="Mean cross-validated AUROC ± SD across 5 folds. ★ = best in the row. Stronger tint = higher." chip={<span className="data-chip real">real data · 3 datasets</span>}>
        <div className="table-scroll">
          <table className="data heat" style={{ minWidth: 640 }}>
            <thead>
              <tr>
                <th>Dataset</th>
                {FITTED_MODELS.map((m) => <th key={m} style={{ textAlign: "center" }}><span className="swatch" style={{ background: MODEL_COLOR[m], marginRight: 6 }} />{MODEL_LABEL[m].replace("Foundation (TabICL)", "TabICL")}</th>)}
              </tr>
            </thead>
            <tbody>
              {HEAT_ROWS.map((id) => {
                const d = all[id];
                const bm = best(d);
                return (
                  <tr key={id}>
                    <td style={{ fontWeight: 600 }}>{DATASET_SHORT[id]} <span className="small muted">({d.n_rows.toLocaleString("en-US")})</span></td>
                    {FITTED_MODELS.map((m) => {
                      const mm = d.cv.models[m];
                      if (!mm) return <td key={m} className="cell muted">not run</td>;
                      return (
                        <td key={m} className="cell" style={{ background: heat(mm.mean_auroc), color: "var(--ink)", fontWeight: m === bm ? 750 : 450 }}>
                          {m === bm && "★ "}{fmt(mm.mean_auroc, 3)}<span style={{ opacity: 0.8, fontSize: "0.8em" }}> ±{fmt(mm.sd_auroc, 3)}</span>
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <TryIt title="Try it"
        tasks={[<>Name the best model on each dataset, and say whether its lead over logistic regression is bigger than one fold-to-fold SD of logistic regression.</>]}
        answers={HEAT_ROWS.map((id) => {
          const d = all[id];
          const bm = best(d);
          const lr = d.cv.models.logistic!;
          const gap = d.cv.models[bm]!.mean_auroc - lr.mean_auroc;
          return (
            <>{DATASET_SHORT[id]}: <b>{MODEL_LABEL[bm]}</b> ({fmt(d.cv.models[bm]!.mean_auroc, 3)}){bm === "logistic" ? " itself." : <>, ahead of logistic regression ({fmt(lr.mean_auroc, 3)}) by {fmt(gap, 3)} = {fmt(gap / lr.sd_auroc, 1)} SD: {gap > lr.sd_auroc ? "more than one SD, though still small in absolute terms" : "within one SD, so weak evidence"}.</>}</>
          );
        })}
      />

      <div className="grid-2">
        <Callout tone="good"><b>Reach for the simpler model</b> when results are close: one you can explain and check often deserves to win. A fancier model rarely rescues weak features.</Callout>
        <Callout tone="warn"><b>Watch out for…</b> axes that start at 0.5 (as on the slide): they make small differences look large. Toggle “Start the axis at 0” to see them at full scale.</Callout>
      </div>
      <SourceNote>
        Every model was scored on identical StratifiedKFold(5, shuffle=True, random_state=0) folds, with imputation and scaling inside each model's pipeline.
        Models: logistic regression, decision tree (depth 4), random forest (300 trees, min 3 per leaf), LightGBM (defaults), TabICL (pretrained, no tuning).
      </SourceNote>
    </div>
  );
}
