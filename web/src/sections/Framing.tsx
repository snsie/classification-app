import { useState } from "react";
import { fmtT, ProbabilityStrip, ThresholdSlider } from "../charts/classify";
import { Callout, Card, DataChip, PauseThink, Segmented, Toggle } from "../components/ui";
import { MODEL_A } from "../data";

type Hl = "features" | "target" | "classes";

const ROWS = [
  { id: 1, age: 78, adm: 3, los: 9, meds: 14, bin: "Readmitted", multi: "Nursing facility" },
  { id: 2, age: 45, adm: 0, los: 2, meds: 3, bin: "Not readmitted", multi: "Home" },
  { id: 3, age: 66, adm: 1, los: 5, meds: 8, bin: "Not readmitted", multi: "Home" },
  { id: 4, age: 59, adm: 2, los: 7, meds: 11, bin: "Readmitted", multi: "Rehabilitation" },
];
const CLASS_STYLE: Record<string, React.CSSProperties> = {
  Readmitted: { background: "var(--cls-pos)", color: "var(--surface)" },
  "Not readmitted": { background: "var(--surface)", color: "var(--ink)", border: "1.5px solid var(--cls-neg)" },
  Home: { background: "color-mix(in srgb, var(--m-gbm) 18%, var(--surface))", color: "var(--ink)", border: "1.5px solid var(--m-gbm)" },
  Rehabilitation: { background: "color-mix(in srgb, var(--m-rf) 18%, var(--surface))", color: "var(--ink)", border: "1.5px dashed var(--m-rf)" },
  "Nursing facility": { background: "color-mix(in srgb, var(--m-tabicl) 18%, var(--surface))", color: "var(--ink)", border: "1.5px dotted var(--m-tabicl)" },
};

// slide 2: the four objectives, with the sections that cover each
const OBJECTIVES = [
  { t: "Connect regression to classification", d: "Features, targets and classes; probabilities and thresholds.", link: "framing", secs: "§1" },
  { t: "Use accuracy, and know its limits", d: "Why a model that finds no one can still score 90%.", link: "accuracy", secs: "§2" },
  { t: "Read a confusion matrix", d: "Sensitivity (recall), specificity, precision, TPR, FPR, F1 and AUROC.", link: "matrix", secs: "§3–10" },
  { t: "Compare models fairly", d: "Same splits, unseen data, no leaks; when one model really is better.", link: "classifiers", secs: "§11–17" },
];

const SAME = [
  ["Input features", "Information about each patient"],
  ["Examples with known outcomes", "Past patients teach the model a pattern"],
  ["A prediction for a new patient", "Judged on patients the model has not seen"],
];

export default function Framing() {
  const [hl, setHl] = useState<Hl>("features");
  const [multi, setMulti] = useState(false);
  const [t, setT] = useState(0.5);
  const [a, setA] = useState(0.7);
  const [b, setB] = useState(0.3);

  const featBg = hl === "features" ? "var(--accent-soft)" : undefined;
  const targetBg = hl !== "features" ? "var(--accent-soft)" : undefined;
  const classes = multi ? ["Home", "Rehabilitation", "Nursing facility"] : ["Readmitted", "Not readmitted"];
  const verdict = (p: number) => (p >= t - 1e-9 ? "readmitted" : "not readmitted");
  const ineq = (p: number) => (p >= t - 1e-9 ? "≥" : "<");

  return (
    <div className="stack">
      <div>
        <div className="stat-label" style={{ marginBottom: 8 }}>By the end of this lesson, you can…</div>
        <div className="grid-4">
          {OBJECTIVES.map((o, i) => (
            <a key={o.link} href={`#/${o.link}`} className="card" style={{ textDecoration: "none", color: "inherit" }}>
              <div className="row" style={{ gap: 10, marginBottom: 6, flexWrap: "nowrap" }}>
                <span className="badge-num">{i + 1}</span>
                <b>{o.t}</b>
              </div>
              <div className="small ink2">{o.d}</div>
              <div className="small muted" style={{ marginTop: 6 }}>{o.secs}</div>
            </a>
          ))}
        </div>
        <p className="small muted" style={{ margin: "8px 0 0" }}>One running example throughout: predicting whether a patient will be readmitted within 30 days.</p>
      </div>

      <Card title="Regression predicts a number; classification a category" sub="Last week the outcome was a number. This week it is a category.">
        <div className="grid-2">
          <div className="card" style={{ background: "var(--surface-2)" }}>
            <div className="stat-label">Last week · regression</div>
            <div style={{ fontWeight: 550, margin: "4px 0" }}>“How many days will this patient stay in hospital?”</div>
            <div className="stat-value tnum">5.2 days</div>
            <div className="stat-note">A number on a continuous scale</div>
          </div>
          <div className="card" style={{ background: "var(--surface-2)" }}>
            <div className="stat-label">This week · classification</div>
            <div style={{ fontWeight: 550, margin: "4px 0" }}>“Will this patient be readmitted within 30 days?”</div>
            <div className="row" style={{ gap: 8, margin: "6px 0" }}>
              <span className="chip" style={CLASS_STYLE.Readmitted}>Readmitted</span>
              <span className="muted">or</span>
              <span className="chip" style={CLASS_STYLE["Not readmitted"]}>Not readmitted</span>
            </div>
            <div className="stat-note">One of a fixed set of categories</div>
          </div>
        </div>
        <div className="stat-label" style={{ margin: "14px 0 6px" }}>What stays the same</div>
        <div className="grid-3">
          {SAME.map(([t, d]) => (
            <div key={t} className="callout"><b>{t}</b>: {d.charAt(0).toLowerCase() + d.slice(1)}.</div>
          ))}
        </div>
      </Card>

      <Card title="Three words to know: feature, target and class" sub="A few past patients: the kind of table a classifier learns from (hypothetical values)" chip={<DataChip />}>
        <div className="controls-bar">
          <Segmented label="Highlight" value={hl} onChange={setHl}
            options={[{ value: "features", label: "Features" }, { value: "target", label: "Target" }, { value: "classes", label: "Classes" }]} />
          <Toggle checked={multi} onChange={setMulti}>Multiclass example</Toggle>
        </div>
        <div className="table-scroll">
          <table className="data">
            <thead>
              <tr>
                <th>Patient</th>
                <th style={{ background: featBg }}>Age</th>
                <th style={{ background: featBg }}>Admissions in past year</th>
                <th style={{ background: featBg }}>Length of stay (days)</th>
                <th style={{ background: featBg }}>Medications at discharge</th>
                <th style={{ background: targetBg, textAlign: "left" }}>{multi ? "Discharge destination" : "Readmitted within 30 days?"}</th>
              </tr>
            </thead>
            <tbody>
              {ROWS.map((r) => {
                const cls = multi ? r.multi : r.bin;
                return (
                  <tr key={r.id}>
                    <td>{r.id}</td>
                    <td style={{ background: featBg }}>{r.age}</td>
                    <td style={{ background: featBg }}>{r.adm}</td>
                    <td style={{ background: featBg }}>{r.los}</td>
                    <td style={{ background: featBg }}>{r.meds}</td>
                    <td style={{ background: targetBg, textAlign: "left" }}>
                      {hl === "classes" ? <span className="chip" style={CLASS_STYLE[cls]}>{cls}</span> : cls}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="grid-3" style={{ marginTop: 14 }}>
          <div className="callout" style={{ opacity: hl === "features" ? 1 : 0.55 }}>
            <b>Feature</b>: a piece of information about a patient that the model uses as an input, such as age or admissions in the past year.
          </div>
          <div className="callout" style={{ opacity: hl === "target" ? 1 : 0.55 }}>
            <b>Target</b>: the outcome we want to predict. Here: {multi ? "where does the patient go after discharge?" : "was the patient readmitted within 30 days?"}
          </div>
          <div className="callout" style={{ opacity: hl === "classes" ? 1 : 0.55 }}>
            <b>Class</b>: one possible value of the target. Here there are {classes.length}: {classes.join(", ")}.{" "}
            {multi ? "Three or more classes is multiclass classification." : "Two classes is binary classification, today's focus."}
          </div>
        </div>
      </Card>

      <div className="grid-3">
        <Card>
          <div className="stat-label">1 · Predict a number</div>
          <div className="stat-value tnum">5.2 days</div>
          <div className="stat-note">Regression (last week): any value on a scale</div>
        </Card>
        <Card>
          <div className="stat-label">2 · Estimate a probability</div>
          <div className="stat-value tnum" style={{ color: "var(--accent)" }}>{a.toFixed(2)}</div>
          <div className="stat-note">Patient A's chance of readmission: always between 0 and 1. What many classifiers produce first.</div>
        </Card>
        <Card>
          <div className="stat-label">3 · Assign a class</div>
          <div className="stat-value" style={{ fontSize: "1.6rem" }}>{a >= t - 1e-9 ? "Readmitted" : "Not readmitted"}</div>
          <div className="stat-note">After applying the threshold ({fmtT(t)}): one label from a fixed set</div>
        </Card>
      </div>

      <div className="split-wide">
        <Card title="A threshold turns a probability into a class" sub="100 patients at discharge (grey: outcomes not known yet). Drag A, B or the threshold." chip={<DataChip />}>
          <ProbabilityStrip p={MODEL_A.p} labels={null} lanes="single" threshold={t} onThreshold={(v) => setT(Math.min(0.95, Math.max(0.05, v)))}
            markers={[
              { p: a, label: "A", onChange: setA, color: "var(--accent)" },
              { p: b, label: "B", onChange: setB, color: "var(--m-rf)" },
            ]}
            announce={`Threshold ${fmtT(t)}. Patient A ${a.toFixed(2)}: ${verdict(a)}. Patient B ${b.toFixed(2)}: ${verdict(b)}.`} />
          <div className="controls-bar" style={{ marginTop: 8 }}>
            <ThresholdSlider value={t} onChange={setT} min={0.05} max={0.95} presets={[0.3, 0.5, 0.7]}
              valueText={(v) => `${fmtT(v)}: A is ${verdict(a)}, B is ${verdict(b)}`} />
          </div>
        </Card>
        <div className="stack">
          <Card>
            <div className="stat-label">Patient A</div>
            <div className="stat-value tnum" style={{ fontSize: "1.5rem" }}>{a.toFixed(2)} {ineq(a)} {fmtT(t)}</div>
            <div className="stat-note">→ predicted <b>{verdict(a)}</b></div>
          </Card>
          <Card>
            <div className="stat-label">Patient B</div>
            <div className="stat-value tnum" style={{ fontSize: "1.5rem" }}>{b.toFixed(2)} {ineq(b)} {fmtT(t)}</div>
            <div className="stat-note">→ predicted <b>{verdict(b)}</b></div>
          </Card>
          <p className="small ink2" style={{ margin: 0 }}>Rule: predict “readmitted” when the estimated probability is <b>at or above</b> the threshold.</p>
        </div>
      </div>

      <div className="grid-2">
        <PauseThink prompt="Is “will the length of stay be more than 7 days?” a regression or a classification task?">
          Binary classification: the target has two classes (more than 7 days, or not). Predicting the number of days itself would be regression.
        </PauseThink>
        <PauseThink prompt={`The model says ${a.toFixed(2)} for patient A. Is it ${Math.round(a * 100)}% sure A will be readmitted?`}>
          Not necessarily. The {a.toFixed(2)} is an estimate. Whether a predicted {Math.round(a * 100)}% really means about {Math.round(a * 100)} in 100 such patients
          is called calibration, and it needs its own check (see §10).
        </PauseThink>
      </div>

      <div className="grid-2">
        <Callout tone="good"><b>Reach for 0.50 when…</b> you need a starting point. It is a choice people make, not something the model learns; §6–7 move it.</Callout>
        <Callout tone="warn"><b>Watch out for…</b> models whose outputs are scores, not probabilities (for example an SVM's decision value). Higher still means “more likely”, but don't read it as a chance.</Callout>
      </div>
    </div>
  );
}
