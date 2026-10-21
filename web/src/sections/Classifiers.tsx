import { useState } from "react";
import { TrainTestBars, TrainTestLegend } from "../charts/bars";
import { fmtT, ProbabilityLegend, ProbabilityMap, ThresholdSlider } from "../charts/classify";
import { TipRow } from "../charts/kit";
import { Callout, Card, DataChip, Loading, PauseThink, Segmented, StepSlider, Toggle } from "../components/ui";
import { fmt, Grid, MODEL_COLOR, MODEL_LABEL, ModelId, PlaygroundData, useData } from "../data";

type Fam = Exclude<ModelId, "baseline">;
type PairId = "heart" | "nhanes";

const FAMILIES: { id: Fam; last: string; how: string; watch: string }[] = [
  { id: "logistic", last: "Linear regression", how: "Weighted sum of features, squeezed into 0–1 by an S-shaped curve", watch: "Misses curved or combined effects unless added" },
  { id: "tree", last: "Regression tree", how: "Share of training patients with the outcome in the patient's leaf", watch: "Unstable; deep trees memorize" },
  { id: "rf", last: "Random forest", how: "Average of many trees' probabilities", watch: "Hard to explain one prediction" },
  { id: "gbm", last: "Gradient boosting", how: "Trees added in sequence, each fixing remaining errors", watch: "Several settings; can overfit" },
  { id: "tabicl", last: "—", how: "Pretrained network reads your training examples; no tuning", watch: "Black box; needs a GPU; size limits" },
];

export default function Classifiers() {
  const pg = useData<PlaygroundData>("playground");
  const [fam, setFam] = useState<Fam>("logistic");
  const [pair, setPair] = useState<PairId>("heart");
  const [depth, setDepth] = useState(3);
  const [trees, setTrees] = useState(100);
  const [leaf, setLeaf] = useState(5);
  const [rounds, setRounds] = useState(100);
  const [lr, setLr] = useState(0.1);
  const [t, setT] = useState(0.5);
  const [showPts, setShowPts] = useState(true);
  if (!pg) return <Loading />;
  const P = pg.pairs[pair];
  const grid: Grid | null =
    fam === "logistic" ? P.logistic : fam === "tree" ? P.tree[String(depth)] : fam === "rf" ? P.rf[`${trees}_${leaf}`] : fam === "gbm" ? P.gbm[`${lr}_${rounds}`] : P.tabicl;
  const setting =
    fam === "tree" ? `depth ${depth}` : fam === "rf" ? `${trees} trees, min ${leaf} per leaf` : fam === "gbm" ? `${rounds} rounds, learning rate ${lr}` : fam === "tabicl" ? "no tuning" : "fitted by maximum likelihood (L2, C = 1)";

  const rows = [
    ...(fam !== "logistic" ? [{ id: "logistic", label: "Logistic (ref.)", color: MODEL_COLOR.logistic, train: P.logistic.train_auroc, test: P.logistic.cv_auroc }] : []),
    ...(grid ? [{ id: fam, label: MODEL_LABEL[fam].replace("Foundation (TabICL)", "TabICL"), color: MODEL_COLOR[fam], train: grid.train_auroc, test: grid.cv_auroc }] : []),
  ];

  return (
    <div className="stack">
      <div className="grid-4" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))" }}>
        {FAMILIES.map((f) => (
          <button key={f.id} className="card card-btn model-card" aria-pressed={fam === f.id} onClick={() => setFam(f.id)}
            style={{ borderColor: fam === f.id ? MODEL_COLOR[f.id] : undefined, opacity: fam === f.id ? 1 : 0.78 }}>
            <div className="row" style={{ gap: 8 }}>
              <span className="swatch" style={{ background: MODEL_COLOR[f.id], width: 12, height: 12 }} />
              <b>{MODEL_LABEL[f.id].replace("Foundation (TabICL)", "Tabular foundation model (TabICL)")}</b>
            </div>
            <div className="small muted" style={{ marginTop: 4 }}>Last week: {f.last}</div>
            <div className="small" style={{ marginTop: 8 }}><b className="muted">PROBABILITY</b> {f.how}</div>
            <div className="small" style={{ marginTop: 4 }}><b className="muted">WATCH OUT</b> {f.watch}</div>
          </button>
        ))}
      </div>

      <div className="controls-bar">
        <Segmented label="Data" value={pair} onChange={setPair}
          options={[{ value: "heart", label: "Heart disease (age × max heart rate)" }, { value: "nhanes", label: "NHANES (age × waist)" }]} />
        {fam === "tree" && <StepSlider label="Tree depth" values={pg.tree_depths} value={depth} onChange={setDepth} />}
        {fam === "rf" && (
          <>
            <StepSlider label="Number of trees" values={pg.rf_trees} value={trees} onChange={setTrees} />
            <Segmented label="Min. patients per leaf" value={leaf} onChange={setLeaf} options={pg.rf_leaves.map((v) => ({ value: v, label: String(v) }))} />
          </>
        )}
        {fam === "gbm" && (
          <>
            <StepSlider label="Boosting rounds" values={pg.gb_rounds} value={rounds} onChange={setRounds} />
            <Segmented label="Learning rate" value={lr} onChange={setLr} options={pg.gb_lrs.map((v) => ({ value: v, label: String(v) }))} />
          </>
        )}
        {fam === "tabicl" && <div className="small muted" style={{ alignSelf: "flex-end" }}>No settings to tune: the pretrained model reads the training patients as context.</div>}
      </div>

      <div className="split-wide">
        <Card title={`${MODEL_LABEL[fam]} · ${setting}`} sub={`${P.label}. Colour = estimated probability; black line = decision boundary at ${fmtT(t)}. Hover for a patient profile's probability.`}
          chip={<DataChip n={P.n} unit={pair === "heart" ? "patients" : "adults"} />}>
          <ProbabilityLegend pos={P.pos_label} neg={P.neg_label} threshold={t} />
          {grid ? (
            <ProbabilityMap pair={P} grid={grid} threshold={t} showPoints={showPts}
              tip={fam === "tree" && grid.leaf && grid.leaves ? (ix, iy) => {
                const [k, n] = grid.leaves![grid.leaf![iy * P.x_axis.length + ix]];
                return <TipRow label="Leaf" value={`${k} of ${n} training patients: ${P.pos_label.toLowerCase()}`} />;
              } : undefined} />
          ) : (
            <div className="loading">TabICL was not run for this build (the pipeline was run with --no-tabicl).</div>
          )}
          <div className="controls-bar" style={{ marginTop: 8 }}>
            <ThresholdSlider value={t} onChange={setT} min={0.05} max={0.95} presets={[0.3, 0.5, 0.7]} width={380} />
            <Toggle checked={showPts} onChange={setShowPts}>Show patients</Toggle>
          </div>
          <p className="small muted" style={{ margin: 0 }}>{P.sample}.</p>
        </Card>
        <div className="stack">
          <Card title="AUROC: training vs unseen" sub="Mean of 5 cross-validation folds, for the current setting. The axis starts at 0.5 (chance).">
            <TrainTestLegend valueLabel="AUROC" sd={false} />
            <TrainTestBars rows={rows} domain={[0.5, 1]} valueLabel="AUROC" rowHeight={44} labelWidth={120} />
          </Card>
          {grid && (
            <Card>
              <div className="stat-label">The gap</div>
              <div className="stat-value tnum" style={{ color: grid.train_auroc - grid.cv_auroc > 0.1 ? "var(--fn)" : undefined }}>
                {fmt(grid.train_auroc, 2)} → {fmt(grid.cv_auroc, 2)}
              </div>
              <div className="stat-note">Training AUROC vs AUROC on patients the model hasn't seen. A big gap means the model memorised its training patients.</div>
            </Card>
          )}
          <PauseThink prompt="Choose the decision tree and set the depth to 20. Why is the training AUROC 1.00?">
            {pair === "heart" ? (
              <>At depth 20 almost every leaf holds a handful of training patients, often just one, so each leaf's share is 0 or 1: the tree has memorised them.
                Training AUROC {fmt(P.tree["20"].train_auroc, 2)}; on unseen patients {fmt(P.tree["20"].cv_auroc, 2)}. That gap is §13's starting point.</>
            ) : (
              <>At depth 20 the leaves are tiny, so each leaf's share is close to 0 or 1 for its own training patients. Training AUROC {fmt(P.tree["20"].train_auroc, 2)};
                unseen {fmt(P.tree["20"].cv_auroc, 2)}. That gap is §13's starting point.</>
            )}
          </PauseThink>
        </div>
      </div>

      <div className="grid-2">
        <Callout tone="good"><b>Reach for…</b> the simplest family that captures the pattern. Logistic regression draws a straight boundary; trees draw boxes; ensembles smooth the boxes.</Callout>
        <Callout tone="warn"><b>Watch out for…</b> the probabilities themselves. No model family guarantees well-calibrated probabilities; check them (§10).</Callout>
      </div>
    </div>
  );
}
