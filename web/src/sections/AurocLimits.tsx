import { useMemo, useState } from "react";
import { curveStepAfter, line } from "d3-shape";
import { CalibrationPlot, fmtT, Quiz, ThresholdSlider } from "../charts/classify";
import { Legend, Plot } from "../charts/kit";
import { Callout, Card, DataChip, Loading, PauseThink, Segmented, SourceNote, Stat } from "../components/ui";
import { auc, calibrationBins, confusion, DatasetData, decodeP, fmt, fmtInt, fmtPct, mean, metrics, MODEL_A, useData } from "../data";

type Version = "fitted" | "halved" | "balanced";
const GRID = Array.from({ length: 101 }, (_, i) => i / 100);

export default function AurocLimits() {
  const [t, setT] = useState(0.5);
  const [ver, setVer] = useState<Version>("fitted");
  const ds = useData<DatasetData>("readmission");
  const areaA = useMemo(() => auc(MODEL_A.p, MODEL_A.y), []);
  const accCurve = useMemo(() => GRID.map((g) => metrics(confusion(MODEL_A.p, MODEL_A.y, g)).acc), []);
  const mA = metrics(confusion(MODEL_A.p, MODEL_A.y, t));
  const meanA = mean(MODEL_A.p);

  const real = useMemo(() => {
    if (!ds || !ds.cv.oof.logistic || !ds.cv.oof.logistic_balanced) return null;
    const fitted = decodeP(ds.cv.oof.logistic);
    const halved = fitted.map((v) => v / 2);
    const balanced = decodeP(ds.cv.oof.logistic_balanced);
    return { fitted, halved, balanced };
  }, [ds]);

  return (
    <div className="stack">
      <Card title="1 · It is not accuracy, and it doesn't choose the threshold" sub="Model A: accuracy changes with every threshold; AUROC is one number for the whole curve." chip={<DataChip />}>
        <div className="split-wide">
          <div>
            <Legend items={[{ label: "Accuracy at each threshold", color: "var(--accent)" }, { label: `AUROC ${fmt(areaA, 2)} (no threshold)`, color: "var(--ink-2)", kind: "dash" }]} />
            <Plot height={240} xDomain={[0, 1]} yDomain={[0, 1]} nice={false} xLabel="Threshold" xTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]} yTicks={[0, 0.25, 0.5, 0.75, 1]}
              yFormat={(v) => `${Math.round(v * 100)}%`} ariaLabel="Accuracy against threshold, with AUROC as a flat line">
              {({ x, y }) => (
                <g>
                  <path d={line<number>().curve(curveStepAfter).x((i) => x(GRID[i])).y((i) => y(accCurve[i]))(GRID.map((_, i) => i)) ?? ""} fill="none" stroke="var(--accent)" strokeWidth={2.4} />
                  <line x1={0} x2={x(1)} y1={y(areaA)} y2={y(areaA)} stroke="var(--ink-2)" strokeWidth={2} strokeDasharray="6 4" />
                  <line x1={x(t)} x2={x(t)} y1={y(0)} y2={y(mA.acc)} stroke="var(--ink)" strokeDasharray="2 3" />
                  <circle cx={x(t)} cy={y(mA.acc)} r={6} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} />
                </g>
              )}
            </Plot>
            <ThresholdSlider value={t} onChange={setT} presets={[0.3, 0.5, 0.7]} valueText={(v) => `${fmtT(v)}: accuracy ${Math.round(metrics(confusion(MODEL_A.p, MODEL_A.y, v)).acc * 100)}%`} />
          </div>
          <div className="stack">
            <Stat label={`Accuracy at ${fmtT(t)}`} value={fmtPct(mA.acc, 0)} color="var(--accent)" note="depends on the threshold" />
            <Stat label="AUROC" value={fmt(areaA, 2)} note="the same at 0.30, 0.50 and 0.70: it is not “87% correct”" />
          </div>
        </div>
      </Card>

      {!ds || !real ? <Loading /> : (() => {
        const p = ver === "fitted" ? real.fitted : ver === "halved" ? real.halved : real.balanced;
        const fittedM = ds.cv.models.logistic!, balM = ds.cv.models.logistic_balanced!;
        const aur = ver === "balanced" ? balM.mean_auroc : fittedM.mean_auroc;
        const meanPred = ver === "fitted" ? fittedM.mean_pred : ver === "halved" ? fittedM.mean_pred / 2 : balM.mean_pred;
        const bins = calibrationBins(p, ds.y, { bins: 10, strategy: "quantile" });
        const top = bins[bins.length - 1];
        return (
          <Card title="2 · It doesn't show whether the probabilities are right" sub="Real readmission data: logistic regression, out-of-fold probabilities. Each dot is one tenth of the patients, sorted by predicted risk."
            chip={<DataChip n={ds.n_rows} unit="discharges" />}>
            <div className="controls-bar">
              <Segmented label="Probabilities" value={ver} onChange={setVer}
                options={[{ value: "fitted", label: "As fitted" }, { value: "halved", label: "Every probability halved" }, { value: "balanced", label: "Class weights balanced (refit)" }]} />
            </div>
            <div className="split-wide">
              <CalibrationPlot max={1} series={[{ id: ver, label: ver === "fitted" ? "As fitted" : ver === "halved" ? "Halved" : "Balanced weights", color: ver === "balanced" ? "var(--fp)" : "var(--m-linear)", bins }]}
                histogram={[{ p, color: ver === "balanced" ? "var(--fp)" : "var(--m-linear)", label: "patients" }]} />
              <div className="stack">
                <Stat label="AUROC" value={fmt(aur, 3)} note={ver === "halved" ? "unchanged: halving keeps every patient's rank" : "mean of 5 folds"} />
                <Stat label="Mean predicted risk" value={fmtPct(meanPred, 1)} color={ver === "fitted" ? undefined : "var(--fn)"} />
                <Stat label="Actual readmission rate" value={fmtPct(ds.prevalence, 1)} note={`${fmtInt(ds.n_pos)} of ${fmtInt(ds.n_rows)}`} />
                <div className="small ink2">
                  {ver === "fitted" && <>Dots near the diagonal: the riskiest tenth is predicted {fmtPct(top.meanPred, 0)} on average and {fmtPct(top.observed, 0)} of them came back. Well calibrated, but weak (AUROC {fmt(fittedM.mean_auroc, 2)}).</>}
                  {ver === "halved" && <>Same ranking, so the same AUROC, but every dot falls left of the diagonal: the riskiest tenth is now predicted {fmtPct(top.meanPred, 0)}, and {fmtPct(top.observed, 0)} of them came back.</>}
                  {ver === "balanced" && <><code>class_weight="balanced"</code> left AUROC at {fmt(balM.mean_auroc, 3)} (from {fmt(fittedM.mean_auroc, 3)}) but raised the average predicted risk from {fmtPct(fittedM.mean_pred, 0)} to {fmtPct(balM.mean_pred, 0)}, when the true rate is {fmtPct(ds.prevalence, 0)}.</>}
                </div>
              </div>
            </div>
          </Card>
        );
      })()}

      <div className="grid-2">
        <Quiz n={1} q="Model X has an AUROC of 0.80. Does that mean it classifies 80% of patients correctly?">
          No. AUROC measures how well the model ranks patients across all thresholds. Accuracy depends on the threshold you choose (and on how imbalanced the data are).
        </Quiz>
        <Quiz n={2} q="You lower the threshold from 0.50 to 0.30. What happens to sensitivity, specificity and AUROC?">
          Sensitivity usually rises and specificity falls. AUROC does not change: the model and its ranking are the same; you have only moved along the curve.
        </Quiz>
      </div>

      <PauseThink prompt="Model A flags 26 patients at ≥ 0.50, but only 10 were readmitted. Could its probabilities be right?">
        No. If they were calibrated, 26 patients with a predicted risk of at least 50% would include at least 13 readmissions on average, and yet there are only 10
        readmissions in total. Its mean predicted risk is {fmtPct(meanA, 1)} against 10% observed. That's why the calibration demo above uses real data, not Model A.
      </PauseThink>

      <div className="grid-2">
        <Callout tone="good"><b>Reach for…</b> three things together: AUROC for ranking, a calibration plot for the probabilities, and sensitivity and specificity at the threshold you chose.</Callout>
        <Callout tone="warn"><b>Watch out for…</b> “fixes” for imbalance. Reweighting the classes or SMOTE leaves the ranking alone and breaks the probabilities.</Callout>
      </div>
      {ds && (
        <SourceNote>
          {fmtInt(ds.n_rows)} discharges from the UCI Diabetes 130-US hospitals data. Logistic regression (median imputation, standardised features, scikit-learn defaults),
          with and without <code>class_weight="balanced"</code>, same 5 folds. “Halved” divides the fitted out-of-fold probabilities by 2 in your browser.
        </SourceNote>
      )}
    </div>
  );
}
