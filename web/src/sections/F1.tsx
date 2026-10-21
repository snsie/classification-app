import { useMemo, useState } from "react";
import { line } from "d3-shape";
import { FormulaLine, MetricTable, pctAuto, ThresholdSlider, fmtT } from "../charts/classify";
import { Plot } from "../charts/kit";
import { Callout, Card, DataChip, PauseThink, Slider, Toggle } from "../components/ui";
import { confusion, metrics, MODEL_A, prCurve } from "../data";

const ISO = [0.2, 0.4, 0.6, 0.8];

export default function F1() {
  const [t, setT] = useState(0.5);
  const [extra, setExtra] = useState(0);
  const [avgLines, setAvgLines] = useState(false);

  const base = confusion(MODEL_A.p, MODEL_A.y, t);
  const c = { ...base, tn: base.tn + extra };
  const never = { tp: 0, fn: 10, fp: 0, tn: 90 + extra };
  const m = metrics(c);
  const P = m.prec, R = m.sens;
  const avg = P == null ? null : (P + R) / 2;
  const pr = useMemo(() => prCurve(MODEL_A.p, MODEL_A.y), []);

  const bars = [
    { label: "Precision", v: P, color: "var(--fp)" },
    { label: "Recall", v: R, color: "var(--tp)" },
    { label: "Simple average", v: avg, color: "var(--muted)" },
    { label: "F1", v: m.f1, color: "var(--accent)" },
  ];

  return (
    <div className="stack">
      <div className="controls-bar">
        <ThresholdSlider value={t} onChange={setT} presets={[0.3, 0.5, 0.7]} width={420}
          valueText={(v) => `${fmtT(v)}: precision ${pctAuto(metrics(confusion(MODEL_A.p, MODEL_A.y, v)).prec)}, recall ${pctAuto(metrics(confusion(MODEL_A.p, MODEL_A.y, v)).sens)}`} />
        <Slider label="Extra correctly cleared patients (true negatives)" value={extra} min={0} max={900} step={10} onChange={setExtra} format={(v) => `+${v}`} width={360} />
        <Toggle checked={avgLines} onChange={setAvgLines}>Show simple-average lines</Toggle>
      </div>

      <div className="split-wide">
        <Card title="F1 is pulled toward the smaller value" sub="F1 = 2 × P × R ÷ (P + R), the harmonic mean of precision (P) and recall (R)" chip={<DataChip />}>
          <div className="stack" style={{ gap: 10 }}>
            {bars.map((b) => (
              <div key={b.label} className="row" style={{ flexWrap: "nowrap", gap: 12 }}>
                <div style={{ width: 120, flex: "none", fontWeight: b.label === "F1" ? 700 : 500 }}>{b.label}</div>
                <div style={{ flex: 1, height: 26, background: "var(--surface-2)", borderRadius: 6, overflow: "hidden" }}>
                  <div style={{ width: `${(b.v ?? 0) * 100}%`, height: "100%", background: b.color, opacity: b.label === "Simple average" ? 0.55 : 1, transition: "width .2s" }} />
                </div>
                <div className="tnum" style={{ width: 70, textAlign: "right", fontWeight: 700 }}>{b.v == null ? "undefined" : `${(b.v * 100).toFixed(1)}%`}</div>
              </div>
            ))}
          </div>
          <div className="spacer" />
          <FormulaLine metric="f1" counts={c} />
          <div className="small ink2" style={{ marginTop: 6 }}>
            Check: 2 × {P == null ? "P" : P.toFixed(4)} × {R.toFixed(2)} ÷ ({P == null ? "P" : P.toFixed(4)} + {R.toFixed(2)}) ≈ {m.f1.toFixed(3)}. Accuracy here is {pctAuto(m.acc)}
            {extra > 0 && <> (with {extra} extra true negatives)</>}.
          </div>
        </Card>
        <Card title="Precision–recall plane" sub="Grey curves: every point with the same F1. Dot: Model A at this threshold.">
          <Plot height={320} xDomain={[0, 1]} yDomain={[0, 1]} nice={false} xLabel="Recall (sensitivity)" yLabel="Precision" xTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]} yTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
            ariaLabel="Precision against recall with F1 contour lines">
            {({ x, y }) => (
              <g>
                {ISO.map((f) => {
                  const pts = Array.from({ length: 80 }, (_, i) => {
                    const r = f / (2 - f) + ((1 - f / (2 - f)) * i) / 79;
                    return { r, p: (f * r) / (2 * r - f) };
                  });
                  const last = pts[pts.length - 1];
                  return (
                    <g key={f}>
                      <path d={line<{ r: number; p: number }>().x((d) => x(d.r)).y((d) => y(d.p))(pts) ?? ""} fill="none" stroke="var(--axis)" strokeWidth={1.4} />
                      <text x={x(last.r) - 4} y={y(last.p) - 5} textAnchor="end" fontSize={11} fill="var(--muted)">F1 {f}</text>
                    </g>
                  );
                })}
                {avgLines && ISO.map((a) => (
                  <g key={`a${a}`}>
                    <line x1={x(Math.max(0, 2 * a - 1))} y1={y(Math.min(1, 2 * a))} x2={x(Math.min(1, 2 * a))} y2={y(Math.max(0, 2 * a - 1))}
                      stroke="var(--muted)" strokeDasharray="4 4" />
                    <text x={x(Math.min(1, 2 * a)) - 2} y={y(Math.max(0, 2 * a - 1)) - 4} textAnchor="end" fontSize={10.5} fill="var(--muted)">avg {a}</text>
                  </g>
                ))}
                <path d={line<number>().x((i) => x(pr.recall[i])).y((i) => y(pr.precision[i]))(pr.recall.map((_, i) => i)) ?? ""}
                  fill="none" stroke="var(--cls-pos)" strokeWidth={1.6} opacity={0.4} />
                {P != null && <circle cx={x(R)} cy={y(P)} r={7} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} />}
              </g>
            )}
          </Plot>
          <p className="small ink2" style={{ margin: 0 }}>Faint line: Model A at every threshold. Extra true negatives don't move the dot: neither precision nor recall uses TN.</p>
        </Card>
      </div>

      <Card title={`Model A against the “always no” rule, threshold ${fmtT(t)}${extra ? `, +${extra} true negatives` : ""}`} sub="★ = better in that row. The “always no” rule wins on accuracy, specificity and FPR, and loses on everything about finding readmissions." chip={<DataChip />}>
        <MetricTable star columns={[{ id: "a", label: "Model A", counts: c }, { id: "n", label: "“Always no”", counts: never }]} />
      </Card>

      <div className="grid-2">
        <PauseThink prompt="Add 900 true negatives with the slider. What happens to F1, and to accuracy?">
          F1 stays at 44.4%: it never looks at true negatives. Accuracy rises from 80% to 98.0% ((8 + 972) ÷ 1,000). Same model, same mistakes.
        </PauseThink>
        <PauseThink prompt="Can F1 ever be higher than the simple average of precision and recall?">
          No. The harmonic mean is never above the ordinary mean; they are equal only when precision equals recall. Turn on the simple-average lines to see it.
        </PauseThink>
      </div>

      <div className="grid-2">
        <Callout tone="good"><b>Reach for F1 when…</b> finding cases and avoiding false alarms both matter, and you want one number that is high only when both are.</Callout>
        <Callout tone="warn">
          <b>Watch out for…</b> what F1 leaves out: (1) it ignores the {c.tn} true negatives; (2) it treats a missed readmission and a false alarm as equally costly;
          (3) it isn't automatically better than accuracy. Choose the metric from the question, and report more than one.
        </Callout>
      </div>
    </div>
  );
}
