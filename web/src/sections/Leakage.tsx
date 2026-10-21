import { ReactNode, useState } from "react";
import { Callout, Card, DataChip, Loading, PauseThink, Segmented, Toggle } from "../components/ui";
import { fmt, fmtInt, LabData, LeakRun, useData } from "../data";

type Pipe = "diabetes" | "readmission";
type Resample = "before" | "inside" | "none";
interface State { pipe: Pipe; model: "rf" | "gbm"; unseen: boolean; dropLeaky: boolean; resample: Resample; byPatient: boolean }

const ASSISTANT = { unseen: false, dropLeaky: false, resample: "before" as Resample, byPatient: false };
const FIXED = { unseen: true, dropLeaky: true, resample: "inside" as Resample, byPatient: true };

const PRESETS: { label: string; s: State }[] = [
  { label: "Forest scored on its training patients", s: { pipe: "diabetes", model: "rf", unseen: false, dropLeaky: true, resample: "none", byPatient: false } },
  { label: "…the same forest, scored fairly", s: { pipe: "diabetes", model: "rf", unseen: true, dropLeaky: true, resample: "none", byPatient: false } },
  { label: "Leaked columns + SMOTE before splitting", s: { pipe: "diabetes", model: "gbm", unseen: true, dropLeaky: false, resample: "before", byPatient: false } },
  { label: "…without them, SMOTE inside training", s: { pipe: "diabetes", model: "gbm", unseen: true, dropLeaky: true, resample: "inside", byPatient: false } },
  { label: "Readmission: SMOTE before splitting", s: { pipe: "readmission", model: "gbm", unseen: true, dropLeaky: true, resample: "before", byPatient: true } },
  { label: "…SMOTE inside each training fold", s: { pipe: "readmission", model: "gbm", unseen: true, dropLeaky: true, resample: "inside", byPatient: true } },
];

// slide 28, with the section that shows each rule
const RULES: [string, string, string][] = [
  ["Same splits, same metrics.", "Every model is trained and scored on the same folds, with the same metrics.", "compare"],
  ["Preprocessing and choices inside training.", "Fill in missing values, scale features and tune settings using training data only, inside each cross-validation round.", "leakage"],
  ["Test set last, and once.", "Keep the test set locked until the final evaluation.", "cv"],
  ["Keep each patient together.", "If a patient has several admissions, all of them go in the same split. Otherwise the model is tested on people it has already seen.", "leakage"],
  ["Choose the metric and threshold for the task.", "Decide before looking at results, based on what each kind of error costs.", "peeking"],
];
const SECTION_NUM: Record<string, number> = { cv: 13, peeking: 15, compare: 16 };

const RESAMPLE_LABEL: Record<Resample, string> = { before: "SMOTE before the split", inside: "SMOTE inside each fold", none: "No resampling" };

export default function Leakage() {
  const lab = useData<LabData>("lab");
  const [s, setS] = useState<State>({ pipe: "diabetes", model: "gbm", ...ASSISTANT });
  if (!lab) return <Loading />;
  const L = lab.leakage;
  const set = <K extends keyof State>(k: K) => (v: State[K]) => setS({ ...s, [k]: v });
  const diab = s.pipe === "diabetes";
  const runs: LeakRun[] = diab ? L.diabetes.runs : L.readmission.runs;
  const match = (r: LeakRun) => r.model === s.model && r.resample === s.resample && (diab ? r.leaky === !s.dropLeaky : r.split === (s.byPatient ? "patient" : "row"));
  const run = runs.find(match)!;
  const score = s.unseen ? run.test_auroc : run.train_auroc;
  const isAssistant = !s.unseen && s.resample === "before" && (diab ? !s.dropLeaky : !s.byPatient);
  const allFixed = s.unseen && s.resample === "inside" && (diab ? s.dropLeaky : s.byPatient);
  const est = s.model === "rf" ? "RandomForestClassifier(min_samples_leaf=3)" : "LGBMClassifier()";

  const cls = (ok: boolean) => (ok ? "line fixed" : "line flag");
  const lines: ReactNode[] = [];
  if (diab) {
    lines.push(s.dropLeaky
      ? <span key="x" className={cls(true)}>X = df[FAIR_COLUMNS]<span className="cm">{"            # ② no HbA1c, no diabetes medicines"}</span></span>
      : <span key="x" className={cls(false)}>X = df.drop(columns=['diabetes'])<span className="cm">{"  # ② keeps HbA1c, insulin, pills"}</span></span>);
    lines.push(<span key="y" className="line">y = df['diabetes']<span className="cm">{"                 # told by a doctor, or HbA1c ≥ 6.5%"}</span></span>);
  } else {
    lines.push(<span key="x" className="line">X, y = df[FEATURES], df['readmitted_30d']<span className="cm">{`   # ${fmtInt(L.readmission.n_rows)} encounters, ${fmtInt(L.readmission.n_patients)} patients`}</span></span>);
  }
  if (s.resample === "before") {
    lines.push(<span key="imp" className={cls(false)}>X = SimpleImputer().fit_transform(X)<span className="cm">{"   # ③ fitted on every patient"}</span></span>);
    lines.push(<span key="sm" className={cls(false)}>X, y = SMOTE().fit_resample(X, y)<span className="cm">{"      # ③ synthetic patients before the split"}</span></span>);
    lines.push(<span key="m" className="line">model = {est}</span>);
  } else if (s.resample === "inside") {
    lines.push(<span key="m" className={cls(true)}>model = make_pipeline(SimpleImputer(), SMOTE(), {est})<span className="cm">{"  # ③ inside each training fold"}</span></span>);
  } else {
    lines.push(<span key="m" className="line">model = {s.model === "rf" ? `make_pipeline(SimpleImputer(), ${est})` : est}<span className="cm">{"  # ③ no resampling"}</span></span>);
  }
  if (diab) {
    lines.push(<span key="cv" className="line">cv = StratifiedKFold(5, shuffle=True, random_state=0)</span>);
  } else if (s.byPatient) {
    lines.push(<span key="cv" className={cls(true)}>cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)<span className="cm">{"  # ④ by patient"}</span></span>);
  } else {
    lines.push(<span key="cv" className={cls(false)}>cv = StratifiedKFold(5, shuffle=True, random_state=0)<span className="cm">{"  # ④ random rows"}</span></span>);
  }
  lines.push(<span key="r" className="line">r = cross_validate(model, X, y, cv=cv{!diab && s.byPatient ? ", groups=df['patient_nbr']" : ""}, scoring='roc_auc',</span>);
  lines.push(<span key="r2" className="line">                   return_train_score=True)</span>);
  lines.push(s.unseen
    ? <span key="p" className={cls(true)}>print('AUROC:', r['test_score'].mean())<span className="cm">{"   # ① unseen patients"}</span></span>
    : <span key="p" className={cls(false)}>print('AUROC:', r['train_score'].mean())<span className="cm">{"  # ① its own training patients"}</span></span>);

  const verdict = score > 0.95 ? "“Near-perfect discrimination.”" : allFixed ? "The honest number." : "";

  return (
    <div className="stack">
      <Card title="Five rules for a fair model comparison">
        <div className="stack" style={{ gap: 8 }}>
          {RULES.map(([t, d, link], i) => (
            <div key={t} className="row" style={{ gap: 10, flexWrap: "nowrap", alignItems: "flex-start" }}>
              <span className="badge-num">{i + 1}</span>
              <div><b>{t}</b> <span className="ink2">{d}</span> {link && <a href={`#/${link}`} className="small" style={{ whiteSpace: "nowrap" }}>{link === "leakage" ? "below ↓" : `see §${SECTION_NUM[link]} →`}</a>}</div>
            </div>
          ))}
        </div>
      </Card>

      <div className="stat-label" style={{ marginTop: 6 }}>What breaks without them: the assistant's pipeline</div>
      <div className="controls-bar">
        <Segmented label="Pipeline" value={s.pipe} onChange={set("pipe")}
          options={[{ value: "diabetes", label: "Diabetes screening" }, { value: "readmission", label: "30-day readmission (every encounter)" }]} />
        <Segmented label="Model" value={s.model} onChange={set("model")} options={[{ value: "rf", label: "Random forest" }, { value: "gbm", label: "Gradient boosting" }]} />
      </div>
      <div className="row small" style={{ gap: 6 }}>
        <span className="muted">Slide 29's pairs:</span>
        <div className="presets">
          {PRESETS.map((p) => (
            <button key={p.label} aria-pressed={JSON.stringify(p.s) === JSON.stringify(s)} onClick={() => setS(p.s)}>{p.label}</button>
          ))}
        </div>
      </div>

      <div className="split-wide">
        <Card title="What the assistant wrote" sub="Flip the fixes on the right; flagged lines turn green as you fix them" chip={<DataChip n={diab ? L.diabetes.n : L.readmission.n_rows} unit={diab ? "adults" : "encounters"} />}>
          <pre className="code" aria-label="Assistant-written Python code">{lines}</pre>
          <div className="spacer" />
          <div className="grid-2">
            <div className="card" style={{ background: "var(--surface-2)" }}>
              <div className="stat-label">Reported AUROC ({s.unseen ? "unseen patients" : "training patients"})</div>
              <div className="stat-value huge tnum" style={{ color: score > 0.95 ? "var(--ink)" : "var(--accent)" }}>{fmt(score, 3)}</div>
              <div className="stat-note">{verdict}</div>
            </div>
            <div className="card" style={{ background: "var(--surface-2)" }}>
              <div className="stat-label">Same model, every fix applied</div>
              <div className="stat-value huge tnum" style={{ color: "var(--m-linear)" }}>
                {fmt(runs.find((r) => r.model === s.model && r.resample === "inside" && (diab ? r.leaky === false : r.split === "patient"))!.test_auroc, 3)}
              </div>
              <div className="stat-note">AUROC on unseen patients</div>
            </div>
          </div>
        </Card>

        <div className="stack">
          <Card title="Apply the fixes">
            <div className="stack" style={{ gap: 14 }}>
              <div>
                <Toggle checked={s.unseen} onChange={set("unseen")}><b>① Score on unseen patients</b></Toggle>
                <div className="small ink2" style={{ marginLeft: 42 }}>The original code printed the training score.</div>
              </div>
              {diab ? (
                <div>
                  <Toggle checked={s.dropLeaky} onChange={set("dropLeaky")}><b>② Drop leaky columns</b></Toggle>
                  <div className="small ink2" style={{ marginLeft: 42 }}>HbA1c defines the label; “takes insulin” and “takes diabetes pills” give it away.</div>
                </div>
              ) : (
                <div className="small muted">② Leaky columns: not an issue in this pipeline.</div>
              )}
              <div>
                <Segmented label="③ Resampling" value={s.resample} onChange={set("resample")}
                  options={[{ value: "before", label: "Before the split" }, { value: "inside", label: "Inside each training fold" }, { value: "none", label: "None" }]} />
                <div className="small ink2" style={{ marginTop: 4 }}>SMOTE invents minority-class patients by interpolating between real ones.</div>
              </div>
              {diab ? (
                <div className="small muted">④ Repeated patients: one row per adult here.</div>
              ) : (
                <div>
                  <Toggle checked={s.byPatient} onChange={set("byPatient")}><b>④ Keep each patient's admissions together</b></Toggle>
                  <div className="small ink2" style={{ marginLeft: 42 }}>{fmtInt(L.readmission.n_rows)} encounters from {fmtInt(L.readmission.n_patients)} patients.</div>
                </div>
              )}
            </div>
          </Card>
          <div className="row">
            <button className="btn" disabled={isAssistant} onClick={() => setS({ ...s, ...ASSISTANT })}>Reset to the assistant's code</button>
            <button className="btn primary" disabled={allFixed} onClick={() => setS({ ...s, ...FIXED })}>Apply all fixes</button>
          </div>
        </div>
      </div>

      <PauseThink prompt="Why does SMOTE before the split inflate the score so much, even when every patient is scored on a fold it wasn't trained on?">
        SMOTE builds each synthetic patient from a real one and its near neighbours. Done before splitting, a validation patient's near-copies land in the training folds,
        and synthetic patients land in the validation folds next to their parents. The model is graded on near-duplicates of what it learned. Inside a pipeline, SMOTE only ever
        sees the training part of each fold.
      </PauseThink>

      <div className="split-wide">
        <Card title="Unfair comparisons make weak models look strong" sub="Slide 29's pairs. Grey: scored unfairly. Navy: the same model scored fairly. Bars start at 0.5 (chance)."
          chip={<span className="data-chip real">real data · 3 pipelines</span>}>
          <div className="stack" style={{ gap: 14 }}>
            {lab.unfair.map((u) => (
              <div key={u.id}>
                <div style={{ fontWeight: 600 }}>{u.title}</div>
                <div className="small muted" style={{ marginBottom: 4 }}>{u.detail}</div>
                {[["unfair", u.unfair, "var(--muted)"], ["fair", u.fair, "var(--cls-pos)"]].map(([k, v, c]) => (
                  <div key={k as string} className="row" style={{ flexWrap: "nowrap", gap: 8, marginBottom: 3 }}>
                    <span className="small" style={{ width: 44 }}>{k as string}</span>
                    <div style={{ flex: 1, height: 18, background: "var(--surface-2)", borderRadius: 4 }}>
                      <div style={{ width: `${Math.max(0, ((v as number) - 0.5) / 0.5) * 100}%`, height: "100%", background: c as string, borderRadius: 4 }} />
                    </div>
                    <b className="tnum small" style={{ width: 44, textAlign: "right" }}>{fmt(v as number, 3)}</b>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </Card>
        <div className="stack">
          <Card title="Keep each patient together" chip={<DataChip n={lab.row_vs_patient.n_rows} unit="encounters" />}>
            <div className="small ink2">
              Readmission data, every encounter ({fmtInt(lab.row_vs_patient.n_patients)} patients), logistic regression:
            </div>
            <div className="stat-value tnum" style={{ fontSize: "1.5rem" }}>{fmt(lab.row_vs_patient.row, 3)} vs {fmt(lab.row_vs_patient.patient, 3)}</div>
            <div className="stat-note">split by row vs split by patient: almost no change here.</div>
            <div className="spacer" />
            <div className="small ink2">Last week's Parkinson's telemonitoring data, same mistake:</div>
            <div className="stat-value tnum" style={{ fontSize: "1.5rem" }}>R² {fmt(lab.parkinsons_ref.random_r2, 2)} → {fmt(lab.parkinsons_ref.group_r2, 2)}</div>
            <div className="stat-note">random rows vs new patients (see last week's app). You only find out by grouping the records.</div>
          </Card>
          <Callout tone="warn"><b>Watch out for…</b> silence. None of these mistakes raises an error or a warning; a fair comparison has to be designed in from the start.</Callout>
        </div>
      </div>

      <Card title={`Every combination · ${diab ? "diabetes screening" : "readmission, every encounter"}`} sub="5-fold cross-validation. Your current pipeline is highlighted; the score it reports is in bold.">
        <div className="table-scroll">
          <table className="data">
            <thead>
              <tr>
                <th>Model</th><th style={{ textAlign: "left" }}>③ Resampling</th><th style={{ textAlign: "left" }}>{diab ? "② Leaky columns" : "④ Split"}</th>
                <th>Training AUROC</th><th>Unseen AUROC</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r, i) => {
                const cur = match(r);
                return (
                  <tr key={i} style={cur ? { background: "var(--accent-soft)" } : undefined}>
                    <td>{r.model === "rf" ? "Random forest" : "Gradient boosting"}</td>
                    <td style={{ textAlign: "left" }}>{RESAMPLE_LABEL[r.resample]}</td>
                    <td style={{ textAlign: "left" }}>{diab ? (r.leaky ? "kept (leak)" : "dropped") : r.split === "row" ? "random rows" : "by patient"}</td>
                    <td style={{ fontWeight: cur && !s.unseen ? 750 : 400 }}>{fmt(r.train_auroc, 3)}</td>
                    <td style={{ fontWeight: cur && s.unseen ? 750 : 400 }}>{fmt(r.test_auroc, 3)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <Callout tone="good"><b>Reach for a pipeline</b> (<code>make_pipeline</code> / imbalanced-learn's <code>Pipeline</code>) whenever anything is learned from data: imputation, scaling, resampling, feature selection. Then cross-validation refits it inside every training fold.</Callout>
    </div>
  );
}
