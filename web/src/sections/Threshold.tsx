import { useMemo, useState } from "react";
import { curveStepAfter, line } from "d3-shape";
import { announceCounts, ConfusionMatrix, fmtT, pctAuto, ProbabilityStrip, ThresholdSlider, TryIt } from "../charts/classify";
import { Legend, Plot } from "../charts/kit";
import { Callout, Card, DataChip, Segmented, Stat, Toggle } from "../components/ui";
import { confusion, metrics, MODEL_A } from "../data";

const GRID = Array.from({ length: 101 }, (_, i) => i / 100);

export default function Threshold() {
  const [t, setT] = useState(0.5);
  const [ghost, setGhost] = useState(false);
  const [colorBy, setColorBy] = useState<"result" | "actual">("result");
  const c = confusion(MODEL_A.p, MODEL_A.y, t);
  const m = metrics(c);
  const c50 = confusion(MODEL_A.p, MODEL_A.y, 0.5);
  const curves = useMemo(() => GRID.map((g) => metrics(confusion(MODEL_A.p, MODEL_A.y, g))), []);
  const valueText = (v: number) => {
    const mm = metrics(confusion(MODEL_A.p, MODEL_A.y, v));
    return `${fmtT(v)}, flags ${mm.flagged} patients`;
  };

  return (
    <div className="stack">
      <div className="controls-bar">
        <ThresholdSlider value={t} onChange={setT} presets={[0.3, 0.5, 0.7]} valueText={valueText} width={460} />
        <Toggle checked={ghost} onChange={setGhost}>Compare with 0.50</Toggle>
        <Segmented label="Colour dots by" value={colorBy} onChange={setColorBy} options={[{ value: "result", label: "Result" }, { value: "actual", label: "Actual class" }]} />
      </div>

      <Card title={`Model A: the same 100 probabilities, threshold ${fmtT(t)}`} sub="Top lane: the 10 patients who were readmitted (filled). Bottom lane: the 90 who were not (rings). Drag the black handle, or focus it and use the arrow keys."
        chip={<DataChip />}>
        <ProbabilityStrip p={MODEL_A.p} labels={MODEL_A.y} threshold={t} onThreshold={setT} ghostThreshold={ghost ? 0.5 : null} colorBy={colorBy}
          announce={announceCounts(t, c)} />
        <Legend items={colorBy === "result" ? [
          { label: "TP ✓ found", color: "var(--tp)", kind: "dot" },
          { label: "FN ✗ missed", color: "var(--fn)", kind: "dot" },
          { label: "FP ✗ false alarm", color: "var(--fp)", kind: "dot" },
          { label: "TN ✓ cleared", color: "var(--tn)", kind: "dot" },
        ] : [
          { label: "Readmitted", color: "var(--cls-pos)", kind: "dot" },
          { label: "Not readmitted", color: "var(--cls-neg)", kind: "dot" },
        ]} />
      </Card>

      <div className="split-wide">
        <Card title="Confusion matrix at this threshold" sub={ghost ? "Chips show the change from 0.50" : undefined}>
          <ConfusionMatrix counts={c} ghost={ghost ? c50 : null} />
        </Card>
        <div className="grid-2" style={{ gap: 12 }}>
          <Stat label="Flagged" value={m.flagged} note="patients predicted readmitted" />
          <Stat label="Sensitivity" value={pctAuto(Math.round(m.sens * 100) / 100)} note={`${c.tp} of 10 readmissions found`} color="var(--tp)" />
          <Stat label="Specificity" value={`${Math.round(m.spec * 100)}%`} note={`${c.tn} of 90 correctly cleared`} />
          <Stat label="Precision" value={m.prec == null ? "undefined" : `${(m.prec * 100).toFixed(1)}%`} note={m.prec == null ? "no one flagged (0 ÷ 0)" : `${c.tp} of ${m.flagged} flags are real`} />
          <Stat label="Accuracy" value={`${Math.round(m.acc * 100)}%`} note={`${c.tp + c.tn} of 100 correct`} />
          <Stat label="False alarms" value={c.fp} note="flagged, not readmitted" color="var(--fp)" />
        </div>
      </div>

      <Card title="Sensitivity and specificity at every threshold" sub="The vertical line is the current threshold. Lower it: sensitivity rises, specificity falls.">
        <Legend items={[{ label: "Sensitivity", color: "var(--tp)" }, { label: "Specificity", color: "var(--tn)" }, { label: "Accuracy", color: "var(--muted)", kind: "dash" }]} />
        <Plot height={260} xDomain={[0, 1]} yDomain={[0, 1]} nice={false} xLabel="Threshold" yLabel="Rate" xTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]} yTicks={[0, 0.25, 0.5, 0.75, 1]}
          yFormat={(v) => `${Math.round(v * 100)}%`} ariaLabel="Sensitivity and specificity against threshold">
          {({ x, y }) => {
            const mk = (f: (i: number) => number) => line<number>().curve(curveStepAfter).x((i) => x(GRID[i])).y((i) => y(f(i)))(GRID.map((_, i) => i)) ?? "";
            return (
              <g>
                <path d={mk((i) => curves[i].acc)} fill="none" stroke="var(--muted)" strokeWidth={1.6} strokeDasharray="5 4" />
                <path d={mk((i) => curves[i].sens)} fill="none" stroke="var(--tp)" strokeWidth={2.6} />
                <path d={mk((i) => curves[i].spec)} fill="none" stroke="var(--tn)" strokeWidth={2.6} />
                <line x1={x(t)} x2={x(t)} y1={0} y2={y(0)} stroke="var(--ink)" strokeWidth={1.5} />
                <circle cx={x(t)} cy={y(m.sens)} r={5} fill="var(--tp)" />
                <circle cx={x(t)} cy={y(m.spec)} r={5} fill="var(--tn)" />
              </g>
            );
          }}
        </Plot>
      </Card>

      <TryIt
        tasks={[
          <>Move the threshold from 0.50 to 0.30. How many more readmissions are found, and how many more false alarms?</>,
          <>Find the highest threshold that still catches 9 of the 10 readmitted patients. How many false alarms?</>,
          <>At 0.70, accuracy is 89%. Why is that higher than at 0.50?</>,
          <>Find the threshold with the highest accuracy. How many readmissions does it catch?</>,
          <>While you dragged, did any patient's probability change?</>,
        ]}
        answers={[
          <>+1 found (8 → 9) and +12 false alarms (18 → 30). Turn on <b>Compare with 0.50</b> to see the change chips.</>,
          <>0.40 (the ninth readmitted patient sits at exactly 0.400), with 24 false alarms. At 0.41 you're back to 8.</>,
          <>Raising it to 0.70 removes 12 false alarms but adds only 3 misses: 12 more correct, 3 fewer, net +9 on 100. With 90 non-readmitted patients, flagging fewer usually raises accuracy.</>,
          <>91%, anywhere between about 0.77 and 0.96. It catches at most 4 of the 10, so the most accurate threshold is a poor one for finding readmissions.</>,
          <>No. The model and its probabilities are fixed; only the cut-off moved, so there's no retraining.</>,
        ]}
      />

      <div className="grid-2">
        <Callout tone="good"><b>Reach for a lower threshold when…</b> missing a case is costly: sensitivity ↑, specificity ↓. Raise it when false alarms are the bigger problem.</Callout>
        <Callout tone="warn"><b>Watch out for…</b> choosing the threshold by accuracy. With 90% not readmitted, the most accurate cut-offs flag almost no one (see §7).</Callout>
      </div>
    </div>
  );
}
