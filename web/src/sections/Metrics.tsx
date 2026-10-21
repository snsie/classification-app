import { useState } from "react";
import { line } from "d3-shape";
import { ConfusionMatrix, FormulaLine, MetricKey, pctAuto, Quiz } from "../charts/classify";
import { Plot } from "../charts/kit";
import { Callout, Card, DataChip, Segmented, Slider, Toggle } from "../components/ui";
import { confusion, Counts, MODEL_A, ppv } from "../data";

type M = "sens" | "spec" | "prec" | "fpr";
type Rule = "modelA" | "never" | "all";

const QUESTION: Record<M, string> = {
  sens: "Of the patients who were actually readmitted, how many did the model identify?",
  spec: "Of the patients who were not readmitted, how many did the model correctly clear?",
  prec: "Of the patients the model flagged, how many were actually readmitted?",
  fpr: "Of the patients who were not readmitted, what share got a false alarm?",
};
const GROUP: Record<M, (c: Counts) => string> = {
  sens: (c) => `TP + FN = ${c.tp} + ${c.fn} = ${c.tp + c.fn} patients actually readmitted (the top row)`,
  spec: (c) => `TN + FP = ${c.tn} + ${c.fp} = ${c.tn + c.fp} patients not readmitted (the bottom row)`,
  prec: (c) => `TP + FP = ${c.tp} + ${c.fp} = ${c.tp + c.fp} patients predicted readmitted (the left column)`,
  fpr: (c) => `FP + TN = ${c.fp} + ${c.tn} = ${c.fp + c.tn} patients not readmitted (the bottom row)`,
};
const SAY: { m: M; q: string }[] = [
  { m: "sens", q: "“How many of the patients who will come back do we reach?”" },
  { m: "spec", q: "“How many patients who won't come back do we leave alone?”" },
  { m: "prec", q: "“When we spend a nurse's call, how often is it on a patient who will come back?”" },
];

export default function Metrics() {
  const [metric, setMetric] = useState<M>("sens");
  const [rule, setRule] = useState<Rule>("modelA");
  const [whatIf, setWhatIf] = useState(false);
  const [prev, setPrev] = useState(10);

  const c: Counts = rule === "modelA" ? confusion(MODEL_A.p, MODEL_A.y, 0.5) : rule === "never" ? { tp: 0, fn: 10, fp: 0, tn: 90 } : { tp: 10, fn: 0, fp: 90, tn: 0 };
  const curve = Array.from({ length: 50 }, (_, i) => ({ prev: (i + 1) / 100, ppv: ppv(0.8, 0.8, (i + 1) / 100) }));
  const cur = ppv(0.8, 0.8, prev / 100);
  const per1000 = { pos: prev * 10, tp: prev * 10 * 0.8, fp: (1000 - prev * 10) * 0.2 };

  return (
    <div className="stack">
      <div className="grid-3">
        {SAY.map((s) => (
          <button key={s.m} className="card card-btn" aria-pressed={metric === s.m} onClick={() => setMetric(s.m)}
            style={{ textAlign: "left", cursor: "pointer", borderColor: metric === s.m ? "var(--accent)" : undefined, borderWidth: metric === s.m ? 2 : 1 }}>
            <div className="stat-label">Say it out loud</div>
            <div style={{ fontWeight: 550, margin: "4px 0" }}>{s.q}</div>
            <div className="small muted">→ {s.m === "sens" ? "Sensitivity" : s.m === "spec" ? "Specificity" : "Precision"}</div>
          </button>
        ))}
      </div>

      <div className="controls-bar">
        <Segmented label="Metric" value={metric} onChange={setMetric}
          options={[{ value: "sens", label: "Sensitivity (recall, TPR)" }, { value: "spec", label: "Specificity" }, { value: "prec", label: "Precision (PPV)" }, { value: "fpr", label: "FPR" }]} />
        <Segmented label="Model" value={rule} onChange={setRule}
          options={[{ value: "modelA", label: "Model A (0.50)" }, { value: "never", label: "Always “no”" }, { value: "all", label: "Flag everyone" }]} />
      </div>

      <div className="split-wide">
        <Card title="Each metric describes a different group of patients" sub="Solid = the numerator. Dashed outline = the denominator: the group the metric is about." chip={<DataChip />}>
          <ConfusionMatrix counts={c} highlight={metric as MetricKey} />
        </Card>
        <div className="stack">
          <Card>
            <div className="stat-label">The question</div>
            <div style={{ fontWeight: 600, margin: "4px 0 10px" }}>{QUESTION[metric]}</div>
            <FormulaLine metric={metric} counts={c} name={metric === "sens" ? "Sensitivity" : metric === "spec" ? "Specificity" : metric === "prec" ? "Precision" : "FPR"} />
            <div className="small ink2" style={{ marginTop: 8 }}>Denominator: {GROUP[metric](c)}.</div>
            {metric === "fpr" && <div className="small ink2" style={{ marginTop: 6 }}>FPR = 1 − specificity. Lower is better.</div>}
            {metric === "sens" && <div className="small ink2" style={{ marginTop: 6 }}>Sensitivity, recall and TPR: three names, one number.</div>}
          </Card>
          <Card title="All four for this rule">
            {(["sens", "spec", "prec", "fpr"] as M[]).map((k) => {
              const v = k === "sens" ? c.tp / (c.tp + c.fn) : k === "spec" ? c.tn / (c.tn + c.fp) : k === "fpr" ? c.fp / (c.fp + c.tn) : c.tp + c.fp ? c.tp / (c.tp + c.fp) : null;
              return (
                <div key={k} className="row" style={{ justifyContent: "space-between", fontWeight: k === metric ? 700 : 400 }}>
                  <span>{k === "sens" ? "Sensitivity" : k === "spec" ? "Specificity" : k === "prec" ? "Precision" : "FPR"}</span>
                  <span className="tnum">{v == null ? "undefined (0 ÷ 0)" : pctAuto(v)}</span>
                </div>
              );
            })}
          </Card>
        </div>
      </div>

      <Card title="TPR and FPR: the two rates behind the ROC curve" sub="Both are shares of a row of the confusion matrix. We want TPR high and FPR low." chip={<DataChip />}>
        <div className="grid-2">
          <div>
            <div className="stat-label">True positive rate (TPR)</div>
            <div className="small ink2" style={{ margin: "4px 0" }}>Of the patients actually readmitted, what share did the model flag?</div>
            <FormulaLine metric="tpr" counts={c} />
            <div className="small ink2" style={{ marginTop: 6 }}>Same number as sensitivity and recall: three names, one metric.</div>
          </div>
          <div>
            <div className="stat-label">False positive rate (FPR)</div>
            <div className="small ink2" style={{ margin: "4px 0" }}>Of the patients not readmitted, what share got a false alarm?</div>
            <FormulaLine metric="fpr" counts={c} />
            <div className="small ink2" style={{ marginTop: 6 }}>FPR = 1 − specificity. Lower is better.</div>
          </div>
        </div>
        <p className="small ink2" style={{ marginBottom: 0 }}>
          Why two more names? The ROC curve (§8) plots TPR against FPR, so these are the terms you'll see on its axes. Careful: FPR is <i>not</i> “the share of
          positive predictions that were wrong”; that is 1 − precision, with a different denominator.
        </p>
      </Card>

      <Card title="Precision depends on how common the outcome is" sub="Sensitivity and specificity held at 80%, as for Model A. Only the readmission rate changes." chip={<DataChip />}>
        <Toggle checked={whatIf} onChange={setWhatIf}>What if readmission were rarer?</Toggle>
        {whatIf && (
          <div className="split-wide" style={{ marginTop: 12 }}>
            <Plot height={240} xDomain={[0.01, 0.5]} yDomain={[0, 1]} nice={false} xLabel="Readmission rate" yLabel="Precision (PPV)"
              xTicks={[0.02, 0.1, 0.2, 0.3, 0.4, 0.5]} yTicks={[0, 0.25, 0.5, 0.75, 1]} xFormat={(v) => `${Math.round(v * 100)}%`} yFormat={(v) => `${Math.round(v * 100)}%`}
              ariaLabel="Precision against readmission rate at 80% sensitivity and specificity">
              {({ x, y }) => (
                <g>
                  <path d={line<{ prev: number; ppv: number }>().x((d) => x(d.prev)).y((d) => y(d.ppv))(curve) ?? ""} fill="none" stroke="var(--accent)" strokeWidth={2.5} />
                  <line x1={x(prev / 100)} x2={x(prev / 100)} y1={y(0)} y2={y(cur)} stroke="var(--ink-2)" strokeDasharray="3 3" />
                  <circle cx={x(prev / 100)} cy={y(cur)} r={6} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} />
                </g>
              )}
            </Plot>
            <div className="stack">
              <Slider label="Readmission rate" value={prev} min={1} max={50} onChange={setPrev} format={(v) => `${v}%`} />
              <div>
                <div className="stat-label">Precision</div>
                <div className="stat-value tnum">{(cur * 100).toFixed(1)}%</div>
                <div className="stat-note">
                  Per 1,000 discharges: {per1000.pos} readmitted, {Math.round(per1000.tp)} of them flagged, plus {Math.round(per1000.fp)} false alarms.
                  So {Math.round(per1000.tp)} of {Math.round(per1000.tp + per1000.fp)} flags are real.
                </div>
              </div>
              <div className="small ink2">At 10%: 30.8% · at 5%: 17.4% · at 2%: 7.5%. Same model quality, very different experience for the nurse making the calls.</div>
            </div>
          </div>
        )}
      </Card>

      <div className="grid-3">
        <Quiz n={1} q="The transitional-care team can make only a few calls a day and wants few wasted calls. Which metric matters most to them?">
          Precision: of the patients the model flags, how many will actually be readmitted.
        </Quiz>
        <Quiz n={2} q="The hospital's priority is to miss as few readmissions as possible. Which metric should it watch?">
          Sensitivity (recall): of the patients who will be readmitted, how many the model finds.
        </Quiz>
        <Quiz n={3} q="A model flags all 100 patients. What are its recall and precision?">
          Recall = 10 ÷ 10 = 100%. Precision = 10 ÷ 100 = 10%. Check it: choose <b>Flag everyone</b> above. Perfect recall can come with many false alarms.
        </Quiz>
      </div>

      <div className="grid-2">
        <Callout tone="good"><b>Reach for a metric by…</b> saying the question out loud first, then finding its group of patients in the matrix.</Callout>
        <Callout tone="warn"><b>Watch out for…</b> mixing up sensitivity and precision. They share TP but not the denominator: everyone actually readmitted vs everyone flagged.</Callout>
      </div>
    </div>
  );
}
