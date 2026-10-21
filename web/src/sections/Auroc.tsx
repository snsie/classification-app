import { useMemo, useState } from "react";
import { PairGrid, ProbabilityStrip, ProbHistogram, Quiz, RocPlot } from "../charts/classify";
import { Callout, Card, DataChip, PauseThink, Segmented, Slider, Stat, Toggle } from "../components/ui";
import { auc, confusion, fmt, metrics, MODEL_A, pairOrder, pnorm, rocPoints, sandbox } from "../data";

type Src = "modelA" | "sandbox";

export default function Auroc() {
  const [src, setSrc] = useState<Src>("modelA");
  const [drawn, setDrawn] = useState(0);
  const [showAll, setShowAll] = useState(false);
  const [d, setD] = useState(1.6);
  const [prev, setPrev] = useState(10);

  // Model A: positives and negatives in ascending probability; pairOrder indexes into these
  const A = useMemo(() => {
    const posIdx: number[] = [], negIdx: number[] = [];
    MODEL_A.y.forEach((v, i) => (v === 1 ? posIdx : negIdx).push(i));
    const order = pairOrder(posIdx.length, negIdx.length);
    return { posIdx, negIdx, order, roc: rocPoints(MODEL_A.p, MODEL_A.y), auc: auc(MODEL_A.p, MODEL_A.y) };
  }, []);
  const S = useMemo(() => {
    const s = sandbox({ n: 1000, prevalence: prev / 100, dprime: d });
    return { ...s, roc: rocPoints(s.p, s.y), auc: auc(s.p, s.y), prec: metrics(confusion(s.p, s.y, 0.5)) };
  }, [d, prev]);

  const nP = A.posIdx.length, nN = A.negIdx.length;
  const wins = A.order.slice(0, drawn).filter(([i, j]) => MODEL_A.p[A.posIdx[i]] > MODEL_A.p[A.negIdx[j]]).length;
  const last = drawn > 0 ? A.order[drawn - 1] : null;
  // the grid shows the highest probabilities first
  const gridSet = useMemo(() => new Set(A.order.slice(0, drawn).map(([i, j]) => (nP - 1 - i) * nN + (nN - 1 - j))), [A, drawn, nP, nN]);
  const posDesc = A.posIdx.map((i) => MODEL_A.p[i]).reverse();
  const negDesc = A.negIdx.map((i) => MODEL_A.p[i]).reverse();
  const lastWin = last ? MODEL_A.p[A.posIdx[last[0]]] > MODEL_A.p[A.negIdx[last[1]]] : false;

  return (
    <div className="stack">
      <div className="controls-bar">
        <Segmented label="Data" value={src} onChange={setSrc} options={[{ value: "modelA", label: "Model A" }, { value: "sandbox", label: "Sandbox (1,000 patients)" }]} />
        {src === "modelA" ? (
          <>
            <button className="btn primary" style={{ alignSelf: "flex-end" }} onClick={() => setDrawn(Math.min(nP * nN, drawn + 1))}>Draw a pair</button>
            <button className="btn" style={{ alignSelf: "flex-end" }} onClick={() => setDrawn(Math.min(nP * nN, drawn + 100))}>Draw 100 pairs</button>
            <button className="btn" style={{ alignSelf: "flex-end" }} onClick={() => setDrawn(0)}>Reset</button>
            <Toggle checked={showAll} onChange={setShowAll}>Show all {nP * nN} pairs</Toggle>
          </>
        ) : (
          <>
            <div className="stack" style={{ gap: 4, minWidth: 260 }}>
              <Slider label="Separation (d′)" value={d} min={0} max={3} step={0.1} onChange={setD} format={(v) => v.toFixed(1)} />
              <div className="presets">
                {[[0, "Coin flip 0"], [1.6, "Model A-like 1.6"], [3, "Near perfect 3"]].map(([v, l]) => (
                  <button key={l} aria-pressed={Math.abs(d - (v as number)) < 1e-9} onClick={() => setD(v as number)}>{l}</button>
                ))}
              </div>
            </div>
            <Slider label="Readmitted %" value={prev} min={2} max={50} onChange={setPrev} format={(v) => `${v}%`} width={260} />
          </>
        )}
      </div>

      {src === "modelA" ? (
        <>
          <div className="split-wide">
            <Card title="Pick one readmitted and one not-readmitted patient" sub="✓ if the readmitted patient got the higher probability, ✗ if not. AUROC is the share of ✓ over every possible pair." chip={<DataChip />}>
              <ProbabilityStrip p={MODEL_A.p} labels={MODEL_A.y} colorBy="actual"
                links={last ? [{ i: A.posIdx[last[0]], j: A.negIdx[last[1]], win: lastWin }] : undefined}
                emphasis={last ? new Set([A.posIdx[last[0]], A.negIdx[last[1]]]) : undefined}
                announce={last ? `Pair ${drawn}: readmitted ${MODEL_A.p[A.posIdx[last[0]]].toFixed(3)} vs not readmitted ${MODEL_A.p[A.negIdx[last[1]]].toFixed(3)}: ${lastWin ? "won" : "lost"}` : undefined} />
              <div className="spacer" />
              <PairGrid pos={posDesc} neg={negDesc} drawn={gridSet} showAll={showAll}
                current={last ? [nP - 1 - last[0], nN - 1 - last[1]] : null} />
              <div className="legend">
                <span className="legend-item"><span className="swatch" style={{ background: "var(--tp)" }} /> ✓ readmitted patient ranked higher</span>
                <span className="legend-item"><span className="swatch" style={{ background: "repeating-linear-gradient(45deg, var(--fn) 0 3px, var(--surface) 3px 4px)" }} /> ✗ ranked lower</span>
                <span className="legend-item">rows: the 10 readmitted patients (probability shown)</span>
              </div>
            </Card>
            <div className="stack">
              <Stat label="Pairs won" value={drawn ? `${wins} of ${drawn}` : "–"} note={drawn ? `${fmt((wins / drawn) * 100, 1)}% so far` : "Draw a pair to start"} color="var(--tp)" />
              <Stat label="All pairs: exact AUROC" value={`${Math.round(A.auc * nP * nN)} / ${nP * nN}`} note={`= ${A.auc.toFixed(3)}, shown as ${fmt(A.auc, 2)}`} />
              <Card title="Same number, as an area">
                <RocPlot fpr={A.roc.fpr} tpr={A.roc.tpr} shadeArea maxSize={300} label="ROC curve of Model A with the area shaded" />
              </Card>
            </div>
          </div>
          <PauseThink prompt={`Is an AUROC of ${fmt(A.auc, 2)} good?`}>
            There is no universal cut-off. Whether {fmt(A.auc, 2)} is useful depends on the task and the alternatives: a modest AUROC can be useful for prioritising phone calls,
            and a high one may still be inadequate for a high-stakes diagnosis. Rules of thumb like “0.8 is good” ignore all of that.
          </PauseThink>
        </>
      ) : (
        <>
          <div className="split-wide">
            <Card title="Two overlapping groups" sub={`Scores ~ N(0, 1) for the not readmitted, N(${d.toFixed(1)}, 1) for the readmitted, turned into calibrated probabilities. Bars show each class's share.`}
              chip={<span className="data-chip hypo">synthetic · 1,000 patients</span>}>
              <ProbHistogram p={S.p} y={S.y} bins={50} height={260} xLabel="Estimated probability of readmission" />
            </Card>
            <div className="stack">
              <div className="grid-2" style={{ gap: 12 }}>
                <Stat label="AUROC (this sample)" value={S.auc.toFixed(3)} />
                <Stat label="Theory: Φ(d′/√2)" value={pnorm(d / Math.SQRT2).toFixed(3)} />
              </div>
              <Stat label="Precision at 0.50" value={S.prec.prec == null ? "undefined" : `${(S.prec.prec * 100).toFixed(1)}%`}
                note={`Change “Readmitted %”: precision moves, AUROC barely does. ${S.prec.flagged} flagged at 0.50.`} />
              <Card title="ROC curve">
                <RocPlot fpr={S.roc.fpr} tpr={S.roc.tpr} shadeArea maxSize={300} label="ROC curve of the sandbox" />
              </Card>
            </div>
          </div>
          <Quiz n={3} q="A model assigns probabilities at random. Roughly what AUROC would you expect? (Check it: set the separation to 0.)">
            About 0.5, the chance diagonal: a readmitted patient is ranked higher only half the time. With d′ = 0 the sandbox gives {auc(sandbox({ dprime: 0 }).p, sandbox({ dprime: 0 }).y).toFixed(3)}.
          </Quiz>
        </>
      )}

      <div className="grid-2">
        <Callout tone="good"><b>Reach for AUROC when…</b> the question is ranking: will the patients who come back be near the top of the list? It doesn't depend on any one threshold.</Callout>
        <Callout tone="warn"><b>Watch out for…</b> reading it as accuracy, or as a verdict. It says nothing about where to cut, or whether the probabilities are right (§10).</Callout>
      </div>
    </div>
  );
}
