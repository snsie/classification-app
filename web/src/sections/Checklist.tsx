import { useEffect, useState } from "react";
import { Callout, Card, Segmented, Slider } from "../components/ui";
import { DATASET_IDS, DatasetData, DatasetId, fmt, fmtInt, fmtPct, LabData, load, MetaData, useData } from "../data";

type Worry = "miss" | "waste" | "both" | "rank" | "equal";
type ThrFrom = "capacity" | "cost" | "target";

const WORRY: Record<Worry, { label: string; metric: string; why: string }> = {
  miss: { label: "Missing a case", metric: "sensitivity (recall)", why: "of the people who have the outcome, how many we find" },
  waste: { label: "Wasted follow-up", metric: "precision (PPV)", why: "of the people we flag, how many have the outcome" },
  both: { label: "Both", metric: "F1, with sensitivity and precision alongside", why: "high only when both are high" },
  rank: { label: "Ranking a worklist", metric: "AUROC, with a calibration plot", why: "whether cases sit near the top of the list" },
  equal: { label: "Both errors cost the same", metric: "accuracy, next to the most-common-class baseline", why: "fair only when classes are balanced" },
};

const TASK: Record<DatasetId, { what: string; pos: string }> = {
  readmission: { what: "readmission within 30 days of discharge", pos: "readmitted" },
  diabetes: { what: "diabetes (told by a doctor, or HbA1c ≥ 6.5%)", pos: "has diabetes" },
  heart: { what: "angiographic heart disease", pos: "heart disease" },
};

const ITEMS: { text: string; sec: number }[] = [
  { text: "Positive class named, and the “most common class for everyone” baseline computed", sec: 1 },
  { text: "Confusion matrix read with its labels (rows actual, columns predicted)", sec: 3 },
  { text: "Metric chosen from the question: sensitivity, specificity, precision or F1", sec: 4 },
  { text: "Threshold chosen for the job (capacity, cost or target), not 0.50 by default", sec: 7 },
  { text: "AUROC reported with a calibration plot, never as “accuracy”", sec: 10 },
  { text: "Every model scored on unseen patients, on the same folds", sec: 13 },
  { text: "Test set locked, then used once at the end", sec: 13 },
  { text: "Imputation, scaling and resampling inside a pipeline", sec: 14 },
  { text: "No leaked columns: nothing derived from, or recorded after, the outcome", sec: 14 },
  { text: "Each patient's records kept together across splits", sec: 14 },
  { text: "Threshold set on training data, never on the test set", sec: 15 },
  { text: "Differences compared with the fold-to-fold SD; simpler model preferred when close", sec: 16 },
];

function readChecks(): boolean[] {
  try {
    const v = JSON.parse(localStorage.getItem("cls-checklist") || "[]");
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
}

export default function Checklist() {
  const [checks, setChecks] = useState<boolean[]>(readChecks);
  const [task, setTask] = useState<DatasetId>("readmission");
  const [worry, setWorry] = useState<Worry>("miss");
  const [thr, setThr] = useState<ThrFrom>("capacity");
  const [cap, setCap] = useState(26);
  const [cost, setCost] = useState(5);
  const [target, setTarget] = useState(90);
  const [copied, setCopied] = useState(false);
  const lab = useData<LabData>("lab");
  const meta = useData<MetaData>("meta");
  const [sources, setSources] = useState<DatasetData[]>([]);
  useEffect(() => {
    Promise.all(DATASET_IDS.map((id) => load<DatasetData>(id))).then(setSources);
  }, []);
  useEffect(() => {
    try {
      localStorage.setItem("cls-checklist", JSON.stringify(checks));
    } catch {
      /* storage unavailable: checks still work for this visit */
    }
  }, [checks]);
  const done = ITEMS.filter((_, i) => checks[i]).length;
  const ds = sources.find((s) => s.id === task);
  const readmit = sources.find((s) => s.id === "readmission");

  const thrText =
    thr === "capacity" ? `the team's capacity: ${cap} follow-ups per 100 patients` : thr === "cost" ? `costs: one missed case costs as much as ${cost} unneeded follow-ups` : `a target sensitivity of ${target}%`;
  const prompt = `CONTEXT    Predict ${TASK[task].what}. Positive class: ${TASK[task].pos}.${ds ? ` ${fmtInt(ds.n_rows)} ${ds.labels.units}, ${fmtPct(ds.prevalence, 1)} positive.` : ""}
TASK       Compare a baseline (prevalence), logistic regression, a tree ensemble and a tabular foundation model.
METRIC     Main worry: ${WORRY[worry].label.toLowerCase()}. Report ${WORRY[worry].metric}; also AUROC and a calibration plot.
THRESHOLD  Choose it from ${thrText}, using training data only. Report the confusion matrix,
           sensitivity, specificity and precision at that threshold.
RULES      • Same StratifiedKFold(5, shuffle=True, random_state=0) folds for every model;
             StratifiedGroupKFold by patient if any patient appears more than once.
           • Impute, scale and resample (if at all) inside a Pipeline, refit in every fold.
           • Drop any column derived from, or recorded after, the outcome.
           • Lock a 20% test set at the start; use it once, at the end.
           • Show training and cross-validated scores side by side (mean ± SD);
             prefer the simpler model when the difference is within one SD.
OUTPUT     Code + results table + confusion matrix at the chosen threshold. List every assumption.`;

  return (
    <div className="stack">
      <Card title="Build your request in four steps" sub="Task → metric → threshold → model. The prompt below updates as you choose.">
        <div className="grid-2">
          <div className="stack" style={{ gap: 14 }}>
            <div>
              <div className="row" style={{ gap: 8, marginBottom: 4 }}><span className="badge-num">1</span><b>The task: which class is positive?</b></div>
              <Segmented value={task} onChange={setTask} ariaLabel="Task"
                options={[{ value: "readmission", label: "30-day readmission" }, { value: "diabetes", label: "Diabetes screening" }, { value: "heart", label: "Heart disease" }]} />
              <div className="small ink2" style={{ marginTop: 4 }}>Positive class: <b>{TASK[task].pos}</b>, the outcome you want the model to find.</div>
            </div>
            <div>
              <div className="row" style={{ gap: 8, marginBottom: 4 }}><span className="badge-num">2</span><b>The metric: what's the main worry?</b></div>
              <Segmented value={worry} onChange={setWorry} ariaLabel="Main worry"
                options={(Object.keys(WORRY) as Worry[]).map((w) => ({ value: w, label: WORRY[w].label }))} />
              <div className="small ink2" style={{ marginTop: 4 }}>→ <b>{WORRY[worry].metric}</b>: {WORRY[worry].why}.</div>
            </div>
          </div>
          <div className="stack" style={{ gap: 14 }}>
            <div>
              <div className="row" style={{ gap: 8, marginBottom: 4 }}><span className="badge-num">3</span><b>The threshold: chosen from…</b></div>
              <Segmented value={thr} onChange={setThr} ariaLabel="Threshold from"
                options={[{ value: "capacity", label: "Capacity" }, { value: "cost", label: "Cost" }, { value: "target", label: "Target sensitivity" }]} />
              <div style={{ marginTop: 6 }}>
                {thr === "capacity" && <Slider label="Follow-ups per 100 patients" value={cap} min={1} max={100} onChange={setCap} />}
                {thr === "cost" && <Slider label="A missed case costs as much as N unneeded follow-ups" value={cost} min={1} max={30} onChange={setCost} format={(v) => `N = ${v}`} />}
                {thr === "target" && <Slider label="Target sensitivity" value={target} min={50} max={100} step={5} onChange={setTarget} format={(v) => `${v}%`} />}
              </div>
            </div>
            <div>
              <div className="row" style={{ gap: 8, marginBottom: 4 }}><span className="badge-num">4</span><b>The model: compare fairly</b></div>
              <div className="small ink2">Same splits, unseen data, no leaks. Prefer the simpler model when the results are close.</div>
            </div>
          </div>
        </div>
        <pre className="code" style={{ marginTop: 14 }}>{prompt}</pre>
        <button className="btn" style={{ marginTop: 10 }} onClick={() => {
          navigator.clipboard?.writeText(prompt).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); }, () => {});
        }}>{copied ? "Copied ✓" : "Copy prompt"}</button>
      </Card>

      <div className="split-wide">
        <Card title={`Checklist · ${done} of ${ITEMS.length}`} sub="Ticks are saved in this browser. Attach the completed list to your report.">
          {ITEMS.map((it, i) => (
            <label key={i} className={`check-item ${checks[i] ? "done" : ""}`}>
              <input type="checkbox" checked={!!checks[i]} onChange={(e) => { const c = [...checks]; c[i] = e.target.checked; setChecks(c); }} />
              <span className="check-text">{it.text} <a href={`#/${it.sec}`} className="small" onClick={(e) => e.stopPropagation()}>see §{it.sec} →</a></span>
            </label>
          ))}
        </Card>
        <div className="stack">
          {lab && (
            <Card title={`Back to the assistant's ${fmtPct(lab.hook.acc, 0)}`}>
              <div className="small ink2">
                “Accuracy {fmtPct(lab.hook.acc, 0)} (excellent)”: saying “no” to all {fmtInt(lab.hook.n)} patients scores {fmtPct(lab.hook.always_no_acc, 1)}, and at 0.50 the model caught{" "}
                <b>{fmtInt(lab.hook.counts.tp)} of {fmtInt(lab.hook.n_pos)}</b> readmissions.
                {readmit && readmit.cv.models.gbm && (() => {
                  const ms = Object.entries(readmit.cv.models).filter(([k, v]) => v && k !== "baseline" && k !== "logistic_balanced");
                  const best = Math.max(...ms.map(([, v]) => v!.mean_auroc));
                  return <> Its AUROC on unseen patients is {fmt(readmit.cv.models.gbm.mean_auroc, 2)}, and the best model in §16 reaches only {fmt(best, 2)}: a weak ranking either way.</>;
                })()}
              </div>
              <div className="small ink2" style={{ marginTop: 8 }}>
                You can now say why the number told the nurses almost nothing, and what to report instead: the matrix at a threshold chosen for the team's capacity.
              </div>
            </Card>
          )}
          <Card title="Before next session">
            <div className="small">
              The <b>Try it</b> prompts in <a href="#/1">§1</a>, <a href="#/6">§6</a> and <a href="#/8">§8</a> (about 15 min) · Problem Set 14 · Quiz 14 on Canvas
            </div>
          </Card>
          <Card title="Four questions, in order">
            <ol style={{ margin: 0, paddingLeft: 20 }} className="small">
              <li><b>The task.</b> What category, and which class is positive?</li>
              <li><b>The metric.</b> Which mistakes matter?</li>
              <li><b>The threshold.</b> Chosen for the job, not by default.</li>
              <li><b>The model.</b> Compared fairly; simpler when close.</li>
            </ol>
          </Card>
        </div>
      </div>

      <Card title="Data sources and reproducibility">
        <ul className="small" style={{ paddingLeft: 18, margin: 0 }}>
          {sources.map((s) => (
            <li key={s.id} style={{ marginBottom: 6 }}>
              <b>{s.name}</b>: {fmtInt(s.n_rows)} {s.labels.units}, {fmtInt(s.n_pos)} positive. {s.source}. {s.citation}{" "}
              <a href={s.url} target="_blank" rel="noreferrer">source</a>
              {s.cohort.length > 1 && <span className="muted"> · cohort: {s.cohort.map((c) => fmtInt(c.n)).join(" → ")}</span>}
            </li>
          ))}
          {meta?.funnels.waist && (
            <li>
              <b>Waist → diabetes (§12)</b>: NHANES 2017–2018 adults with a waist measurement, {meta.funnels.waist.map((c) => fmtInt(c.n)).join(" → ")}.
            </li>
          )}
          <li>Model A (§1–10) is hypothetical: 100 patients built in your browser to match the lecture slides exactly.</li>
        </ul>
        {meta && (
          <p className="small muted" style={{ marginTop: 10, marginBottom: 0 }}>
            Results generated {meta.generated.slice(0, 10)} by <code>pipeline/run_all.py</code> · seed {meta.seed} · {meta.n_folds}-fold CV · Python {meta.python} ·
            TabICL {meta.tabicl_ran ? `ran on ${meta.device}` : "not run"} · {Object.entries(meta.packages).map(([k, v]) => `${k} ${v}`).join(" · ")} · runtime {Math.round(meta.runtime_seconds / 60)} min.
          </p>
        )}
      </Card>
      <Callout>A classifier always gives you a probability. Turning it into a decision, and judging whether that decision deserves trust, is your job.</Callout>
    </div>
  );
}
