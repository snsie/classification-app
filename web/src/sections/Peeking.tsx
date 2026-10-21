import { useState } from "react";
import { ProbabilityStrip } from "../charts/classify";
import { Legend, Plot } from "../charts/kit";
import { Callout, Card, DataChip, Loading, PauseThink, Slider, StepSlider } from "../components/ui";
import { fmtPct, LabData, mean, thresholdForSensitivity, useData } from "../data";

export default function Peeking() {
  const lab = useData<LabData>("lab");
  const [target, setTarget] = useState(90);
  const [ex, setEx] = useState(0);
  if (!lab) return <Loading />;
  const K = lab.peeking;
  const j = K.targets.findIndex((v) => Math.abs(v - target / 100) < 1e-9);
  const peek = K.peek[j], fair = K.fair[j];
  const nSplits = K.repeats * K.folds;

  const e = K.examples[ex];
  const pTr = e.p_train.map((v) => v / 1000), pTe = e.p_test.map((v) => v / 1000);
  const tFair = thresholdForSensitivity(pTr, e.y_train, target / 100);
  const tPeek = thresholdForSensitivity(pTe, e.y_test, target / 100);
  const caught = (thr: number) => e.y_test.reduce((s, v, i) => s + (v === 1 && pTe[i] >= thr - 1e-9 ? 1 : 0), 0);
  const P = e.n_test_pos;

  const keys = Object.keys(K.per_split);
  const bars = [
    { label: "Target", v: target / 100, color: "var(--ink-2)", note: "what you promise the clinical team" },
    { label: "Peeking", v: peek, color: "var(--muted)", note: "threshold set on the test patients themselves" },
    { label: "Fair", v: fair, color: "var(--cls-pos)", note: "threshold set on the training patients" },
  ];
  const lo = 0.6;

  return (
    <div className="stack">
      <div className="controls-bar">
        <Slider label="Target sensitivity (catch this share of cases)" value={target} min={70} max={95} onChange={setTarget} format={(v) => `${v}%`} width={380} />
      </div>

      <div className="split-wide">
        <Card title={`Promise ${target}%: what do new patients actually get?`} sub={`Heart disease, logistic regression on all 13 features, ${nSplits} train/test splits (${K.folds}-fold × ${K.repeats} repeats). Average sensitivity achieved on the test patients.`}
          chip={<DataChip n={K.n} unit="patients" />}>
          <div className="stack" style={{ gap: 12 }}>
            {bars.map((b) => (
              <div key={b.label}>
                <div className="row" style={{ flexWrap: "nowrap", gap: 10 }}>
                  <div style={{ width: 80, fontWeight: 650 }}>{b.label}</div>
                  <div style={{ flex: 1, height: 26, background: "var(--surface-2)", borderRadius: 6, position: "relative" }}>
                    <div style={{ width: `${((b.v - lo) / (1 - lo)) * 100}%`, height: "100%", background: b.color, borderRadius: 6 }} />
                    <div style={{ position: "absolute", left: `${((target / 100 - lo) / (1 - lo)) * 100}%`, top: -4, bottom: -4, borderLeft: "2px dashed var(--ink)" }} />
                  </div>
                  <div className="tnum" style={{ width: 64, textAlign: "right", fontWeight: 750 }}>{fmtPct(b.v, 1)}</div>
                </div>
                <div className="small muted" style={{ marginLeft: 90 }}>{b.note}</div>
              </div>
            ))}
          </div>
          <p className="small muted" style={{ marginBottom: 0 }}>Bars start at 60%. Dashed line: the target.</p>
        </Card>
        <div className="stack">
          <div className="card">
            <div className="stat-label">Peeking promises</div>
            <div className="stat-value tnum">{fmtPct(peek, 1)}</div>
            <div className="stat-note">Set on the test fold, the threshold always reaches the target there, by construction.</div>
          </div>
          <div className="card">
            <div className="stat-label">New patients get</div>
            <div className="stat-value tnum" style={{ color: "var(--cls-pos)" }}>{fmtPct(fair, 1)}</div>
            <div className="stat-note">Set fairly on the training patients (inner cross-validation), then applied to the test fold.</div>
          </div>
        </div>
      </div>

      <Card title="Every split, at four targets" sub={`Each dot is one of the ${nSplits} test folds: sensitivity actually achieved. Left column of each pair: peeking; right: fair.`}>
        <Legend items={[{ label: "Peeking", color: "var(--muted)", kind: "dot" }, { label: "Fair", color: "var(--cls-pos)", kind: "dot" }, { label: "Target", color: "var(--ink)", kind: "dash" }]} />
        <Plot height={280} xDomain={[0.5, keys.length + 0.5]} yDomain={[0.55, 1.02]} nice={false} xTicks={keys.map((_, i) => i + 1)}
          xFormat={(v) => `${Math.round(Number(keys[v - 1]) * 100)}% target`} yTicks={[0.6, 0.7, 0.8, 0.9, 1]} yFormat={(v) => `${Math.round(v * 100)}%`}
          yLabel="Sensitivity on test patients" ariaLabel="Achieved sensitivity per split, peeking vs fair, at four targets">
          {({ x, y }) => (
            <g>
              {keys.map((k, i) => {
                const tv = Number(k);
                const d = K.per_split[k];
                const cur = Math.abs(tv - target / 100) < 1e-9;
                return (
                  <g key={k} opacity={cur ? 1 : 0.55}>
                    <line x1={x(i + 1) - 40} x2={x(i + 1) + 40} y1={y(tv)} y2={y(tv)} stroke="var(--ink)" strokeWidth={2} strokeDasharray="5 3" />
                    {d.peek.map((v, s) => (
                      <circle key={`p${s}`} cx={x(i + 1) - 16 + ((((s * 0.618) % 1) - 0.5) * 18)} cy={y(v)} r={2.6} fill="var(--muted)" opacity={0.7} />
                    ))}
                    {d.fair.map((v, s) => (
                      <circle key={`f${s}`} cx={x(i + 1) + 16 + ((((s * 0.618) % 1) - 0.5) * 18)} cy={y(v)} r={2.6} fill="var(--cls-pos)" opacity={0.7} />
                    ))}
                    <line x1={x(i + 1) - 26} x2={x(i + 1) - 6} y1={y(mean(d.peek))} y2={y(mean(d.peek))} stroke="var(--ink)" strokeWidth={3} />
                    <line x1={x(i + 1) + 6} x2={x(i + 1) + 26} y1={y(mean(d.fair))} y2={y(mean(d.fair))} stroke="var(--ink)" strokeWidth={3} />
                  </g>
                );
              })}
            </g>
          )}
        </Plot>
        <p className="small ink2" style={{ margin: 0 }}>Short black bars are the averages. Peeking never falls below the target; fair thresholds scatter around it, as they will on real new patients.</p>
      </Card>

      <Card title="One split up close" sub="Top: training patients (scored by inner cross-validation), where the fair threshold is set. Bottom: the test patients. Black line: fair threshold; dashed: the peeking threshold."
        chip={<DataChip n={K.n} unit="patients" />}>
        <div className="controls-bar">
          <StepSlider label="Example split" values={K.examples.map((_, i) => i)} value={ex} onChange={setEx}
            format={(i) => `repeat ${K.examples[i].rep + 1}, fold ${K.examples[i].fold + 1}`} />
        </div>
        <div className="small ink2" style={{ fontWeight: 600 }}>Training part ({e.y_train.length} patients, {e.n_train_pos} with heart disease)</div>
        <ProbabilityStrip p={pTr} labels={e.y_train} threshold={tFair} colorBy="result" r={3.6} posLabel="Heart disease" negLabel="No heart disease"
          xLabel="Estimated probability of heart disease" markers={[{ p: tPeek, label: "peek", color: "var(--muted)" }]}
          predLabels={["below", "flagged"]} />
        <div className="small ink2" style={{ fontWeight: 600, marginTop: 8 }}>Test fold ({e.y_test.length} patients, {P} with heart disease)</div>
        <ProbabilityStrip p={pTe} labels={e.y_test} threshold={tFair} colorBy="result" r={4} posLabel="Heart disease" negLabel="No heart disease"
          xLabel="Estimated probability of heart disease" markers={[{ p: tPeek, label: "peek", color: "var(--muted)" }]}
          predLabels={["below", "flagged"]} />
        <div className="grid-2" style={{ marginTop: 8 }}>
          <div className="small"><b>Fair threshold {tFair.toFixed(3)}</b> (from training): catches {caught(tFair)} of {P} test cases = <b>{fmtPct(caught(tFair) / P, 1)}</b>.</div>
          <div className="small"><b>Peeking threshold {tPeek.toFixed(3)}</b> (from these test patients): catches {caught(tPeek)} of {P} = <b>{fmtPct(caught(tPeek) / P, 1)}</b>, guaranteed ≥ {target}%.</div>
        </div>
      </Card>

      <PauseThink prompt={`Why does the fair threshold fall just under ${target}% on average, while peeking lands above it?`}>
        Peeking picks the threshold that catches at least {target}% of <i>these</i> test patients, so it can't miss: rounding up to whole patients pushes it a little over.
        The fair threshold is fitted to the training patients. On new patients, some cases fall just below it by chance, so on average it catches slightly fewer than promised,
        and in individual folds much fewer. That is what a deployed model will do too; peeking hides it.
      </PauseThink>

      <div className="grid-2">
        <Callout tone="good"><b>Rule 5:</b> choose the metric and the threshold from the task, before you see the results, and set the threshold using training data only.</Callout>
        <Callout tone="warn"><b>Watch out for…</b> any number tuned on the test set: a threshold, a model, a feature. It promises a result you won't get on new patients.</Callout>
      </div>
    </div>
  );
}
