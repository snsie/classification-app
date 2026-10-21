import { useEffect, useState } from "react";
import { TrainTestBars, TrainTestLegend } from "../charts/bars";
import { TryIt } from "../charts/classify";
import { Callout, Card, DataChip, DatasetPicker, Loading, ModelPicker, Segmented, StepSlider } from "../components/ui";
import { DatasetData, DatasetId, fmt, fmtInt, FITTED_MODELS, LabData, MODEL_COLOR, MODEL_LABEL, ModelId, useData } from "../data";

export default function CrossValidation() {
  const lab = useData<LabData>("lab");
  const [depth, setDepth] = useState(20);
  const [dsId, setDsId] = useState<DatasetId>("diabetes");
  const [round, setRound] = useState(0);
  const [play, setPlay] = useState(false);
  const [pick, setPick] = useState<ModelId>("logistic");
  const [unlocked, setUnlocked] = useState<{ ds: DatasetId; model: ModelId } | null>(null);
  const [peeks, setPeeks] = useState(0);
  const ds = useData<DatasetData>(dsId);

  useEffect(() => {
    if (!play) return;
    const timer = setInterval(() => setRound((r) => (r + 1) % 5), 1300);
    return () => clearInterval(timer);
  }, [play]);

  const tv = ds?.test_view;
  const models = FITTED_MODELS.filter((m) => tv?.models[m]);
  const choose = (m: ModelId) => {
    setPick(m);
    // changing the model after looking at the test set turns the test set into one more validation set
    if (unlocked && unlocked.ds === dsId && m !== unlocked.model) {
      setPeeks((n) => n + 1);
      setUnlocked({ ds: dsId, model: m });
    }
  };
  const shown = unlocked && unlocked.ds === dsId;

  return (
    <div className="stack">
      {!lab ? <Loading /> : (() => {
        const tvu = lab.train_vs_unseen;
        const i = tvu.tree.depth.indexOf(depth);
        return (
          <div className="split-wide">
            <Card title="A · Training vs unseen" sub={`Heart disease, ${tvu.n} patients, two features (age, maximum heart rate). AUROC: mean of 5 cross-validation folds.`}
              chip={<DataChip n={tvu.n} unit="patients" />}>
              <div className="controls-bar"><StepSlider label="Tree depth" values={tvu.tree.depth} value={depth} onChange={setDepth} /></div>
              <TrainTestLegend valueLabel="AUROC" sd={false} />
              <TrainTestBars domain={[0.5, 1]} valueLabel="AUROC" labelWidth={190}
                rows={[
                  { id: "tree", label: `Decision tree, depth ${depth}`, color: MODEL_COLOR.tree, train: tvu.tree.train[i], test: tvu.tree.unseen[i] },
                  { id: "logistic", label: "Logistic regression (2 features)", color: MODEL_COLOR.logistic, train: tvu.logistic.train, test: tvu.logistic.unseen },
                ]} />
              <p className="small muted" style={{ margin: 0 }}>Bars start at 0.5, chance-level AUROC. Dashed outline = training patients; solid = unseen patients.</p>
            </Card>
            <div className="stack">
              <Card>
                <div className="stat-label">Depth-{depth} tree</div>
                <div className="stat-value tnum" style={{ color: tvu.tree.train[i] - tvu.tree.unseen[i] > 0.1 ? "var(--fn)" : undefined }}>
                  {fmt(tvu.tree.train[i], 2)} vs {fmt(tvu.tree.unseen[i], 2)}
                </div>
                <div className="stat-note">training vs unseen patients</div>
              </Card>
              <Card>
                <div className="stat-label">Logistic regression</div>
                <div className="stat-value tnum">{fmt(tvu.logistic.train, 2)} vs {fmt(tvu.logistic.unseen, 2)}</div>
                <div className="stat-note">no gap: it can't memorise</div>
              </Card>
              <Callout tone="warn"><b>The gap is overfitting.</b> A deep tree memorises its training patients: near-perfect on them, little better than chance on new ones.</Callout>
            </div>
          </div>
        );
      })()}

      <Card title="B · Cross-validation to choose; the test set to confirm" sub="Set aside a test set first and lock it. Inside the development data, 5 rounds: every patient is validated exactly once."
        chip={ds ? <DataChip n={ds.n_rows} unit={ds.labels.units} /> : undefined}>
        <div className="controls-bar">
          <DatasetPicker value={dsId} onChange={(d) => { setDsId(d); setPeeks(0); }} ids={["diabetes", "readmission"]} />
          <Segmented label="Round" value={round} onChange={(r) => { setPlay(false); setRound(r); }} options={[0, 1, 2, 3, 4].map((r) => ({ value: r, label: `${r + 1}` }))} />
          <button className="btn" style={{ alignSelf: "flex-end" }} onClick={() => setPlay(!play)}>{play ? "❚❚ Pause" : "▶ Animate"}</button>
        </div>
        <div className="stack" style={{ gap: 4 }} role="img" aria-label={`Round ${round + 1}: fold ${round + 1} of the development data is validation; the test set stays locked`}>
          <div className="row small muted" style={{ flexWrap: "nowrap" }}>
            <span style={{ width: 74 }} />
            <span style={{ flex: 80 }}>Development data (80%)</span>
            <span style={{ flex: 20, textAlign: "center" }}>Test set (20%)</span>
          </div>
          {[0, 1, 2, 3, 4].map((r) => (
            <div key={r} style={{ display: "grid", gridTemplateColumns: "74px repeat(5, 16fr) 20fr", gap: 3, alignItems: "center" }}>
              <div className="small" style={{ fontWeight: r === round ? 700 : 400, color: r === round ? "var(--ink)" : "var(--muted)" }}>Round {r + 1}</div>
              {[0, 1, 2, 3, 4].map((f) => (
                <div key={f} className="fold-cell" style={{ background: f === r ? "var(--m-rf)" : "var(--m-linear)", opacity: r === round ? (f === r ? 1 : 0.75) : 0.18 }}
                  title={f === r ? "validate" : "train"} />
              ))}
              <div className="fold-cell test" style={{ opacity: r === round ? 1 : 0.35, display: "grid", placeItems: "center", fontSize: 12 }}>🔒</div>
            </div>
          ))}
        </div>
        <div className="legend" style={{ marginTop: 8 }}>
          <span className="legend-item"><span className="swatch" style={{ background: "var(--m-linear)" }} /> train</span>
          <span className="legend-item"><span className="swatch" style={{ background: "var(--m-rf)" }} /> validate</span>
          <span className="legend-item"><span className="swatch" style={{ background: "repeating-linear-gradient(135deg, var(--axis) 0 3px, var(--surface-2) 3px 6px)" }} /> test set: locked until the end</span>
        </div>

        {!ds || !tv ? <Loading /> : (
          <>
            <p className="small ink2" style={{ marginTop: 10 }}>
              {ds.name}: {fmtInt(tv.n_dev)} development {ds.labels.units} ({fmtInt(tv.dev_pos)} positive) and {fmtInt(tv.n_test)} locked test {ds.labels.units} ({fmtInt(tv.test_pos)} positive).
            </p>
            <div className="table-scroll">
              <table className="data">
                <thead>
                  <tr>
                    <th>Validation AUROC</th>
                    {[0, 1, 2, 3, 4].map((r) => <th key={r} style={{ color: r === round ? "var(--ink)" : undefined }}>Round {r + 1}</th>)}
                    <th>Mean ± SD</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((m) => {
                    const mm = tv.models[m];
                    return (
                      <tr key={m} style={m === pick ? { background: "var(--accent-soft)" } : undefined}>
                        <td><span className="swatch" style={{ background: MODEL_COLOR[m], marginRight: 8 }} />{MODEL_LABEL[m]}</td>
                        {mm.fold_auroc.map((v, r) => <td key={r} style={{ fontWeight: r === round ? 700 : 400 }}>{fmt(v, 3)}</td>)}
                        <td><b>{fmt(mm.mean, 3)}</b> ± {fmt(mm.sd, 3)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="small ink2">
              Highest mean: <b>{MODEL_LABEL[tv.best_by_cv]}</b>. Simplest model within one SD of it: <b>{MODEL_LABEL[tv.simplest_within_1sd]}</b>.
            </p>
            <div className="split-wide" style={{ marginTop: 6 }}>
              <div className="stack">
                <ModelPicker value={pick} onChange={choose} models={models} label="Your final model (choose from CV only)" short />
                <div className="row">
                  <button className="btn primary" disabled={!!shown} onClick={() => setUnlocked({ ds: dsId, model: pick })}>🔓 Unlock the test set (once)</button>
                  {shown && <span className="small muted">peeks: {peeks}</span>}
                </div>
                {shown && peeks > 0 && (
                  <Callout tone="warn">
                    You changed the model after seeing the test result. The test set is now one more validation set: its score is optimistic, and you need a new,
                    untouched test set to confirm anything. (peeks: {peeks})
                  </Callout>
                )}
              </div>
              {shown && (() => {
                const m = unlocked!.model;
                const mm = tv.models[m];
                const within = Math.abs(mm.test_auroc - mm.mean) <= mm.sd;
                return (
                  <Card>
                    <div className="stat-label">Test AUROC · {MODEL_LABEL[m]} {peeks === 0 ? "(chosen before unlocking)" : "(chosen after peeking)"}</div>
                    <div className="stat-value tnum">{fmt(mm.test_auroc, 3)}</div>
                    <div className="stat-note">
                      95% bootstrap interval {fmt(mm.test_ci[0], 3)}–{fmt(mm.test_ci[1], 3)}. Cross-validation said {fmt(mm.mean, 3)} ± {fmt(mm.sd, 3)}:
                      the test result is {within ? "within" : "outside"} one SD of it.
                    </div>
                  </Card>
                );
              })()}
            </div>
          </>
        )}
      </Card>

      <TryIt title="Try it"
        tasks={[
          <>Step through rounds 1–5. Is any development patient validated twice? Is any test patient used?</>,
          <>Pick your final model from the cross-validation table only, then unlock the test set. Is the test AUROC within about one SD of the CV mean?</>,
        ]}
        answers={[
          <>No: each fold takes exactly one turn as validation, so every development patient is checked once. The test set is never touched during the rounds.</>,
          <>{tv ? <>For {ds!.name.toLowerCase()}, e.g. logistic regression: CV {fmt(tv.models.logistic.mean, 3)} ± {fmt(tv.models.logistic.sd, 3)}, test {fmt(tv.models.logistic.test_auroc, 3)}. </> : null}
            Test results usually land within about one SD, and the bootstrap interval shows how uncertain a single test set is.</>,
        ]}
      />

      <div className="grid-2">
        <Callout tone="good"><b>Reach for cross-validation to choose</b> models, settings and the threshold, and for the test set to confirm, once, at the end.</Callout>
        <Callout tone="warn"><b>Watch out for…</b> using the test set for choices. Once you've looked at it and changed something, it is no longer unseen.</Callout>
      </div>
    </div>
  );
}
