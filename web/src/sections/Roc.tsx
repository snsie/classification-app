import { useMemo, useState } from "react";
import { announceCounts, ConfusionMatrix, fmtT, ProbabilityStrip, RocPlot, ThresholdSlider, TryIt } from "../charts/classify";
import { Callout, Card, DataChip, Stat, Toggle } from "../components/ui";
import { auc, confusion, fmt, metrics, MODEL_A, rocPoints } from "../data";

/** Slide 23 joins only these seven thresholds; the true staircase reaches TPR 1.0 at FPR 0.48. */
const SLIDE_POINTS = [
  { fpr: 0, tpr: 0 }, { fpr: 1 / 90, tpr: 0.2 }, { fpr: 6 / 90, tpr: 0.5 }, { fpr: 18 / 90, tpr: 0.8 },
  { fpr: 30 / 90, tpr: 0.9 }, { fpr: 54 / 90, tpr: 1 }, { fpr: 1, tpr: 1 },
];

export default function Roc() {
  const [t, setT] = useState(0.5);
  const [labels, setLabels] = useState(true);
  const [slideOnly, setSlideOnly] = useState(false);
  const [shade, setShade] = useState(false);
  const roc = useMemo(() => rocPoints(MODEL_A.p, MODEL_A.y), []);
  const area = useMemo(() => auc(MODEL_A.p, MODEL_A.y), []);
  const c = confusion(MODEL_A.p, MODEL_A.y, t);
  const m = metrics(c);
  // the ROC corner for this threshold: every probability group at or above t is flagged
  let k = 0;
  for (let i = 1; i < roc.thr.length; i++) if (roc.thr[i] >= t - 1e-9) k = i;
  const pick = (i: number) => setT(i === 0 ? 1 : Math.floor(roc.thr[i] * 1000) / 1000);
  const mark = (v: number) => {
    const mm = metrics(confusion(MODEL_A.p, MODEL_A.y, v));
    return { fpr: mm.fpr, tpr: mm.sens, label: v.toFixed(2) };
  };

  return (
    <div className="stack">
      <div className="controls-bar">
        <ThresholdSlider value={t} onChange={setT} presets={[0.3, 0.5, 0.7]} width={420}
          valueText={(v) => {
            const mm = metrics(confusion(MODEL_A.p, MODEL_A.y, v));
            return `${fmtT(v)}: FPR ${fmt(mm.fpr, 2)}, TPR ${fmt(mm.sens, 2)}`;
          }} />
        <Toggle checked={labels} onChange={setLabels}>Label 0.30 / 0.50 / 0.70</Toggle>
        <Toggle checked={slideOnly} onChange={setSlideOnly}>Join only the slide's thresholds</Toggle>
        <Toggle checked={shade} onChange={setShade}>Shade the area</Toggle>
      </div>

      <div className="split-wide">
        <Card title="Model A's ROC curve" sub="Each threshold is one point: up the side TPR (sensitivity), along the bottom FPR (false alarms). Drag the dot, or focus it and use the arrow keys."
          chip={<DataChip />}>
          <RocPlot fpr={roc.fpr} tpr={roc.tpr} k={k} onPick={pick} shadeArea={shade}
            marks={labels ? [mark(0.3), mark(0.5), mark(0.7)] : undefined}
            polyline={slideOnly ? SLIDE_POINTS : null} faint={slideOnly}
            pointText={`Threshold ${fmtT(t)}: FPR ${fmt(m.fpr, 2)}, TPR ${fmt(m.sens, 2)}`} label="ROC curve of Model A" />
          {slideOnly && (
            <p className="small ink2" style={{ margin: "6px 0 0" }}>
              The slide joins seven thresholds with straight lines. The true curve (faint) is a staircase through every threshold, and it reaches TPR 1.0 earlier, at FPR 0.48.
            </p>
          )}
        </Card>
        <div className="stack">
          <div className="grid-2" style={{ gap: 12 }}>
            <Stat label="TPR (up)" value={fmt(m.sens, 2)} color="var(--tp)" note={`${c.tp} ÷ ${c.tp + c.fn} readmitted flagged`} />
            <Stat label="FPR (across)" value={fmt(m.fpr, 2)} color="var(--fp)" note={`${c.fp} ÷ ${c.fp + c.tn} = 1 − specificity (${fmt(m.spec, 2)})`} />
          </div>
          <Stat label="AUROC" value={fmt(area, 2)} note={<>Area under the curve: {Math.round(area * 900)}/900 = {area.toFixed(3)}. It <b>doesn't change</b> when you drag: it describes every threshold at once.</>} />
          <Card title="The two rows behind the two axes" sub="TPR comes from the top row, FPR from the bottom row">
            <ConfusionMatrix counts={c} compact showNames={false} highlight={{ num: ["tp", "fp"], den: ["tp", "fn", "fp", "tn"] }} />
          </Card>
        </div>
      </div>

      <Card title="The same threshold on the patients" sub="As you lower the threshold past a filled dot (readmitted), the ROC point steps up by 0.1. Past a ring (not readmitted), it steps right by 1/90 ≈ 0.011.">
        <ProbabilityStrip p={MODEL_A.p} labels={MODEL_A.y} threshold={t} onThreshold={setT} colorBy="result" announce={announceCounts(t, c)} />
      </Card>

      <TryIt
        tasks={[
          <>Drag the threshold from 0.95 down to 0.05. Which way does the point travel on the ROC plot?</>,
          <>Read off (FPR, TPR) at 0.70, 0.50 and 0.30.</>,
          <>Find the highest threshold that gives TPR = 1.0. What is the FPR there?</>,
          <>Move slowly from 0.60 to 0.40. What does each kind of dot do to the ROC point as the line passes it?</>,
          <>Did AUROC change while you dragged? Why not?</>,
        ]}
        answers={[
          <>Up and to the right: from (0, 0), flagging no one, toward (1, 1), flagging everyone.</>,
          <>(0.07, 0.50), (0.20, 0.80) and (0.33, 0.90).</>,
          <>0.22 (the last readmitted patient is at 0.225), with FPR 0.48: 43 of the 90 non-readmitted patients get a false alarm.</>,
          <>Each filled dot (readmitted) moves the point up by 0.1; each ring (not readmitted) moves it right by 1/90 ≈ 0.011.</>,
          <>No: {fmt(area, 2)} throughout. AUROC summarises all thresholds at once. Moving the threshold only moves you along the same curve.</>,
        ]}
      />

      <div className="grid-2">
        <Callout tone="good"><b>Reach for the ROC curve when…</b> you want to see every trade-off at once. A useful model bows toward the top-left: high TPR with few false alarms.</Callout>
        <Callout tone="warn"><b>Watch out for…</b> the dashed diagonal: random flagging. It catches readmitted and non-readmitted patients at the same rate.</Callout>
      </div>
    </div>
  );
}
