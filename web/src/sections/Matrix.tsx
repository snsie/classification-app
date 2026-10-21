import { useEffect, useState } from "react";
import { CELL_ABBR, CELL_NAME, CellKey, ConfusionMatrix, FormulaLine, PatientWaffle, WaffleArrangement } from "../charts/classify";
import { Callout, Card, DataChip, PauseThink, Segmented } from "../components/ui";
import { confusion, MODEL_A } from "../data";

type View = "discharge" | "later" | "sorted";
const VIEWS: View[] = ["discharge", "later", "sorted"];

// slide 11: what each of the four results means
const MEANING: Record<CellKey, (pos: string) => string> = {
  tp: (pos) => `The model predicted ${pos}, and the patient was ${pos}.`,
  fn: (pos) => `The model predicted not ${pos}, but the patient was ${pos}. A missed case.`,
  fp: (pos) => `The model predicted ${pos}, but the patient was not ${pos}. A false alarm.`,
  tn: (pos) => `The model predicted not ${pos}, and the patient was not ${pos}.`,
};

export default function Matrix() {
  const [view, setView] = useState<View>("sorted");
  const [replay, setReplay] = useState(false);
  const [flip, setFlip] = useState(false);
  const [layout, setLayout] = useState<"course" | "sklearn">("course");
  const [hover, setHover] = useState<CellKey | null>(null);

  useEffect(() => {
    if (!replay) return;
    setView("discharge");
    let k = 0;
    const timer = setInterval(() => {
      k++;
      if (k >= VIEWS.length) {
        clearInterval(timer);
        setReplay(false);
        return;
      }
      setView(VIEWS[k]);
    }, 1200);
    return () => clearInterval(timer);
  }, [replay]);

  const pred = MODEL_A.p.map((v) => (v >= 0.5 ? 1 : 0));
  const y = flip ? MODEL_A.y.map((v) => 1 - v) : MODEL_A.y;
  const pr = flip ? pred.map((v) => 1 - v) : pred;
  const c = confusion(pr, y, 0.5);
  const pos = flip ? "not readmitted" : "readmitted";
  const neg = flip ? "readmitted" : "not readmitted";
  const arrangement: WaffleArrangement = view === "sorted" ? "byCell" : "grid";

  return (
    <div className="stack">
      <div className="split-wide">
        <Card title="“Positive” means the outcome we are looking for" sub="To see which mistakes a model makes, first name the two classes.">
          <div className="grid-2">
            <div className="callout"><b>Positive class</b>: {flip ? "Not readmitted" : "Readmitted within 30 days"}. The outcome of interest: the event we want the model to find.</div>
            <div className="callout"><b>Negative class</b>: {flip ? "Readmitted" : "Not readmitted"}. Every other patient.</div>
          </div>
          <p className="small ink2" style={{ marginBottom: 0 }}>
            “Positive” does not mean good. Like a positive test result, it means the thing we were looking for was found, which is often bad news for the patient.
          </p>
        </Card>
        <Card title="Every patient has two outcomes">
          <div className="stack" style={{ gap: 8 }}>
            <div><b>Actual outcome</b> <span className="small ink2">What really happened. Known only after the 30 days have passed.</span></div>
            <div><b>Predicted outcome</b> <span className="small ink2">What the model said at discharge, after applying the threshold.</span></div>
          </div>
          <p className="small ink2" style={{ marginBottom: 0 }}>Comparing the two, patient by patient, gives only four possible results. Step through them with <b>Show patients</b> below.</p>
        </Card>
      </div>

      <div className="controls-bar">
        <Segmented label="Show patients" value={view} onChange={(v) => { setReplay(false); setView(v); }}
          options={[{ value: "discharge", label: "At discharge" }, { value: "later", label: "30 days later" }, { value: "sorted", label: "Sorted into the grid" }]} />
        <button className="btn" style={{ alignSelf: "flex-end" }} onClick={() => setReplay(true)} disabled={replay}>▶ Replay</button>
        <Segmented label="Positive class" value={flip ? "neg" : "pos"} onChange={(v) => setFlip(v === "neg")}
          options={[{ value: "pos", label: "Readmitted" }, { value: "neg", label: "Not readmitted" }]} />
        <Segmented label="Layout" value={layout} onChange={setLayout}
          options={[{ value: "course", label: "This course" }, { value: "sklearn", label: "scikit-learn" }]} />
      </div>

      <div className="split-wide">
        <Card title="Model A on 100 discharged patients, threshold 0.50"
          sub={view === "discharge" ? "At discharge the model has made its call; nobody knows the outcome yet." : view === "later" ? "30 days later: filled dots were readmitted, rings were not." : "Each patient lands in one of four groups. Hover a cell in the matrix to find its patients."}
          chip={<DataChip />}>
          <PatientWaffle y={y} pred={pr} arrangement={arrangement} showOutcome={view !== "discharge"} highlightCell={view === "sorted" ? hover : null}
            posLabel={pos} negLabel={neg} />
          <div className="legend" style={{ marginTop: 8 }}>
            <span className="legend-item"><span className="dot" style={{ background: "var(--ink-2)" }} /> filled = actually {pos}</span>
            <span className="legend-item"><span className="dot" style={{ border: "2px solid var(--ink-2)" }} /> ring = actually {neg}</span>
            {view === "discharge" && <span className="legend-item"><span className="dot" style={{ background: "var(--muted)", opacity: 0.55 }} /> outcome not known yet</span>}
          </div>
        </Card>
        <div className="stack">
          <Card title="The confusion matrix" sub={layout === "course" ? "Rows: what actually happened. Columns: what the model predicted." : "scikit-learn's order: negatives first, so TN is top-left."}>
            <ConfusionMatrix counts={c} layout={layout} posLabel={pos} negLabel={neg} onCellHover={setHover} hoverCell={hover} />
            <div className="small ink2" style={{ marginTop: 8, minHeight: "2.6em" }} aria-live="polite">
              {hover ? <><b>{CELL_NAME[hover]} ({CELL_ABBR[hover]})</b>: {MEANING[hover](pos)}</> : "Hover or focus a cell to read what it means. Correct results (TP, TN) sit on the diagonal; the two errors usually have different costs."}
            </div>
          </Card>
          <Card>
            <div className="stat-label">Accuracy from the matrix</div>
            <FormulaLine metric="acc" counts={c} />
            <div className="small ink2" style={{ marginTop: 6 }}>
              {c.tp + c.fn} patients were actually {pos}: {c.tp} flagged (TP) and {c.fn} missed (FN). {c.fp + c.tn} were not: {c.tn} correctly cleared (TN) and {c.fp} false alarms (FP).
            </div>
          </Card>
        </div>
      </div>

      <PauseThink prompt="Flip the positive class to “Not readmitted”. What happens to the four cells, and to accuracy?">
        TP and TN swap (8 ↔ 72), and so do FN and FP (2 ↔ 18). Accuracy stays 80%, because it counts both diagonal cells. Every other metric changes meaning,
        which is why you name the positive class first: the outcome you want the model to find.
      </PauseThink>

      <div className="grid-2">
        <Callout tone="good"><b>Reach for the matrix when…</b> you need to know which mistakes a model makes. Every metric that follows comes from 8, 2, 18, 72.</Callout>
        <Callout tone="warn">
          <b>Watch out for…</b> other layouts. scikit-learn's <code>confusion_matrix(y, pred)</code> returns <code>[[TN, FP], [FN, TP]]</code>; pass{" "}
          <code>labels=[1, 0]</code> to get this course's layout. Always read the labels before the numbers.
        </Callout>
      </div>
    </div>
  );
}
