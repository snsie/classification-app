import { useState } from "react";
import { PatientWaffle, Quiz, TryIt } from "../charts/classify";
import { Callout, Card, DataChip, Loading, PauseThink, Segmented, Slider, SourceNote, Stat } from "../components/ui";
import { confusion, fmt, fmtInt, fmtPct, LabData, metrics, MODEL_A, useData } from "../data";

type Rule = "always" | "modelA";

export default function Hook() {
  const lab = useData<LabData>("lab");
  const [guess, setGuess] = useState("");
  const [asked, setAsked] = useState(false);
  const [verified, setVerified] = useState(false);
  const [rule, setRule] = useState<Rule>("always");
  const [nPos, setNPos] = useState(10);

  // the 100-patient version: "always no" on a chosen mix, or Model A at 0.50
  const y = rule === "always" ? Array.from({ length: 100 }, (_, i) => (i < nPos ? 1 : 0)) : MODEL_A.y;
  const pred = rule === "always" ? y.map(() => 0) : MODEL_A.p.map((v) => (v >= 0.5 ? 1 : 0));
  const c = rule === "always" ? { tp: 0, fn: nPos, fp: 0, tn: 100 - nPos } : confusion(MODEL_A.p, MODEL_A.y, 0.5);
  const m = metrics(c);

  if (!lab) return <Loading />;
  const h = lab.hook;
  const g = Number(guess);
  const hasGuess = guess.trim() !== "" && isFinite(g);

  return (
    <div className="stack">
      <div className="grid-2">
        <Card title="Before you ask" chip={<DataChip n={h.n} unit="discharges" />}>
          <p className="ink2" style={{ marginTop: 0 }}>
            {fmtInt(h.n)} patients discharged from US hospitals. Some were back within 30 days. Imagine a “model” that ignores every patient and
            predicts <b>not readmitted</b> for all of them.
          </p>
          <label className="stat-label" htmlFor="hook-guess">Your guess: accuracy of “not readmitted” for everyone (%)</label>
          <div className="row" style={{ marginTop: 6 }}>
            <input id="hook-guess" type="number" inputMode="decimal" min={0} max={100} placeholder="e.g. 50" value={guess}
              onChange={(e) => setGuess(e.target.value)} />
            <span className="muted">%</span>
          </div>
          <p className="small muted" style={{ marginTop: 10, marginBottom: 0 }}>Write it down before the assistant answers.</p>
        </Card>

        <Card title="AI assistant" chip={<span className="chip">illustrative chat · real numbers</span>}>
          <div className="bubble">
            Here are {fmtInt(h.n)} discharges. Build a model that predicts readmission within 30 days, and tell me how good it is.
          </div>
          {!asked ? (
            <button className="btn primary" style={{ marginTop: 14 }} onClick={() => setAsked(true)}>Ask the assistant</button>
          ) : (
            <div className="bubble reply pause-body" style={{ marginTop: 10 }}>
              I trained a gradient-boosting classifier (LightGBM) and checked it with 5-fold cross-validation.
              <div style={{ fontSize: "1.6rem", fontWeight: 700, margin: "8px 0 0" }}>
                Accuracy {fmtPct(h.acc, 0)} <span className="muted" style={{ fontSize: "1rem", fontWeight: 500 }}>(excellent)</span>
              </div>
              {hasGuess && <div className="small muted" style={{ marginTop: 6 }}>Your guess for “no one is readmitted”: {guess}%</div>}
            </div>
          )}
        </Card>
      </div>

      {asked && (
        <PauseThink
          prompt={
            <div className="stack" style={{ gap: 10 }}>
              <span>Should you believe {fmtPct(h.acc, 0)}? What would you compare it with?</span>
              <div>
                <button className="btn primary" onClick={() => setVerified(true)} disabled={verified}>Show the verification</button>
              </div>
            </div>
          }
        >
          <ul style={{ margin: 0, paddingLeft: 20 }}>
            <li>Compare with the simplest rule there is: predict the most common class, “not readmitted”, for everyone.</li>
            <li>Ask how many of the readmitted patients the model actually found. Accuracy can't tell you.</li>
          </ul>
        </PauseThink>
      )}

      {verified && (
        <Card title="Verified on the same discharges" sub="Out-of-fold predictions: every patient scored by a model that never saw them" chip={<DataChip n={h.n} unit="discharges" />}>
          <div className="grid-4">
            <Stat label="Readmitted within 30 days" value={fmtInt(h.n_pos)} note={`of ${fmtInt(h.n)} (${fmtPct(h.prevalence, 1)})`} />
            <Stat label="“Always not readmitted”" value={fmtPct(h.always_no_acc, 1)} note="accurate · finds 0 readmissions" />
            <Stat label="The assistant's model at 0.50" value={fmtPct(h.acc, 1)}
              note={<>accurate · caught <b>{fmtInt(h.counts.tp)} of {fmtInt(h.n_pos)}</b> readmissions</>} />
            <Stat label="Your guess" value={hasGuess ? `${fmt(g, 0)}%` : "–"}
              note={hasGuess ? `${fmt(Math.abs(g - h.always_no_acc * 100), 1)} points from the truth` : "enter a guess above"} />
          </div>
          <div className="spacer" />
          <div className="small ink2" style={{ marginBottom: 4 }}>All {fmtInt(h.n)} discharges</div>
          <div className="propbar" role="img" aria-label={`${fmtPct(h.prevalence, 1)} readmitted`}>
            <div style={{ width: `${(1 - h.prevalence) * 100}%`, background: "color-mix(in srgb, var(--tn) 30%, var(--surface))" }} />
            <div style={{ width: `${h.prevalence * 100}%`, background: "var(--cls-pos)" }} />
          </div>
          <div className="legend" style={{ marginTop: 6 }}>
            <span className="legend-item"><span className="swatch" style={{ background: "color-mix(in srgb, var(--tn) 30%, var(--surface))", border: "1px solid var(--border)" }} /> not readmitted {fmtPct(1 - h.prevalence, 1)}</span>
            <span className="legend-item"><span className="swatch" style={{ background: "var(--cls-pos)" }} /> readmitted {fmtPct(h.prevalence, 1)}</span>
          </div>
          <div className="small ink2" style={{ margin: "10px 0 4px" }}>The {fmtInt(h.n_pos)} readmitted patients: who did the model flag at 0.50?</div>
          <div className="propbar" role="img" aria-label={`${h.counts.tp} found, ${h.counts.fn} missed`}>
            <div style={{ width: `${Math.max(0.6, (h.counts.tp / h.n_pos) * 100)}%`, background: "var(--tp)" }} />
            <div style={{ flex: 1, background: "repeating-linear-gradient(135deg, color-mix(in srgb, var(--fn) 55%, var(--surface)) 0 6px, color-mix(in srgb, var(--fn) 30%, var(--surface)) 6px 12px)" }} />
          </div>
          <div className="legend" style={{ marginTop: 6 }}>
            <span className="legend-item"><span className="swatch" style={{ background: "var(--tp)" }} /> ✓ found {fmtInt(h.counts.tp)}</span>
            <span className="legend-item"><span className="swatch" style={{ background: "color-mix(in srgb, var(--fn) 45%, var(--surface))" }} /> ✗ missed {fmtInt(h.counts.fn)}</span>
          </div>
          <div className="spacer" />
          <Callout tone="warn">
            Saying “no” to everyone already scores <b>{fmtPct(h.always_no_acc, 1)}</b>, because only {fmtPct(h.prevalence, 0)} of patients were readmitted.
            The assistant's model scores {fmtPct(h.acc, 1)}: no better than doing nothing, while catching just {fmtInt(h.counts.tp)} of {fmtInt(h.n_pos)} readmissions
            ({fmtPct(h.counts.tp / h.n_pos, 1)}). For a team trying to call patients at risk, that number said almost nothing.
          </Callout>
        </Card>
      )}

      <div className="split-wide">
        <Card title="The same trap, 100 patients" chip={<DataChip />}>
          <div className="controls-bar">
            <Segmented label="Rule" value={rule} onChange={setRule}
              options={[{ value: "always", label: "Always “not readmitted”" }, { value: "modelA", label: "Model A (0.50)" }]} />
            <Slider label={<>Readmitted, out of 100 <span className="muted">(“always” rule only)</span></>} value={rule === "always" ? nPos : 10}
              min={1} max={50} onChange={(v) => { setRule("always"); setNPos(v); }} width={320} />
          </div>
          <PatientWaffle y={y} pred={pred} arrangement="byCell" maxWidth={520} />
          <div className="legend" style={{ marginTop: 8 }}>
            <span className="legend-item"><span className="dot" style={{ background: "var(--ink-2)" }} /> filled = readmitted</span>
            <span className="legend-item"><span className="dot" style={{ border: "2px solid var(--ink-2)" }} /> ring = not readmitted</span>
            <span className="legend-item">TP / FN / FP / TN groups: see §3</span>
          </div>
        </Card>
        <div className="stack">
          <Stat label="Accuracy" value={fmtPct(m.acc, 0)} note={`${c.tp + c.tn} of 100 predictions correct`} />
          <Stat label="Readmissions found" value={`${c.tp} of ${c.tp + c.fn}`} color={c.tp ? "var(--tp)" : "var(--fn)"} note={c.tp ? "flagged and readmitted" : "every readmission missed"} />
          <Stat label="False alarms" value={c.fp} note="flagged, but not readmitted" />
        </div>
      </div>

      <div className="grid-3">
        <Quiz n={1} q="In a clinic, 5% of patients are readmitted. A model is 95% accurate. What simple rule scores the same?">
          Predicting “not readmitted” for everyone: 95 of 100 correct, and it finds no one. Always compare with that baseline.
        </Quiz>
        <Quiz n={2} q="Model A is 80% accurate; the “always no” rule is 90% accurate. Which finds more readmitted patients?">
          Model A: 8 of 10, versus 0 of 10. Higher accuracy did not mean more useful.
        </Quiz>
        <Quiz n={3} q="When is accuracy a reasonable summary on its own?">
          When the classes are roughly balanced and both kinds of mistake matter about equally. Even then, check which mistakes were made.
        </Quiz>
      </div>

      <TryIt
        tasks={[
          <>Enter your guess above, ask the assistant, then press <b>Show the verification</b>.</>,
          <>In the 100-patient card, switch the rule to <b>Model A (0.50)</b>. Which rule is more accurate, and which finds more readmissions?</>,
          <>Switch back and set <b>Readmitted, out of 100</b> to 5. What does “always no” score? And at 50?</>,
          <>Write one sentence explaining why the assistant's {fmtPct(h.acc, 0)} said almost nothing.</>,
        ]}
        answers={[
          <>“Not readmitted” for everyone scores {fmtPct(h.always_no_acc, 1)} (1 − {fmtInt(h.n_pos)} ÷ {fmtInt(h.n)}). The assistant's model scores {fmtPct(h.acc, 1)} and
            catches {fmtInt(h.counts.tp)} of {fmtInt(h.n_pos)}.</>,
          <>“Always no” is more accurate (90% vs 80%), but Model A finds more (8 vs 0 of 10).</>,
          <>95% with 5 readmitted; 50% with 50. Always-no accuracy is simply the share of patients who were not readmitted.</>,
          <>For example: “Because {fmtPct(1 - h.prevalence, 0)} of patients weren't readmitted, answering ‘no’ for everyone already scores {fmtPct(h.always_no_acc, 0)}; the model matched that
            while finding almost none of the readmissions.”</>,
        ]}
      />

      <div className="grid-2">
        <Callout tone="good"><b>Reach for accuracy when…</b> the classes are roughly balanced and both kinds of mistake matter about equally.</Callout>
        <Callout tone="warn"><b>Watch out for…</b> class imbalance. Always compute “most common class for everyone” first; with {fmtPct(h.prevalence, 0)} readmitted it already scores {fmtPct(h.always_no_acc, 0)}.</Callout>
      </div>

      <div className="grid-4">
        {[
          ["1", "Read the matrix", "TP, FN, FP, TN, and the metric each group of patients gives you.", "matrix"],
          ["2", "Thresholds & ROC", "Move the cut-off, pick it for the job, and read ROC and AUROC.", "threshold"],
          ["3", "Model families", "Last week's models as classifiers; logistic regression up close.", "classifiers"],
          ["4", "Compare fairly", "Unseen data, no leaks, no peeking, and when a model really is better.", "cv"],
        ].map(([n, t, d, link]) => (
          <a key={n} href={`#/${link}`} className="card" style={{ textDecoration: "none", color: "inherit" }}>
            <div className="row" style={{ gap: 10, marginBottom: 6, flexWrap: "nowrap" }}>
              <span className="badge-num">{n}</span>
              <b>{t}</b>
            </div>
            <div className="small ink2">{d}</div>
          </a>
        ))}
      </div>
      <SourceNote>
        Real data: UCI Diabetes 130-US hospitals (Strack et al. 2014), one discharge per patient, {fmtInt(h.n)} sampled. “The assistant's model” is LightGBM with default
        settings; each patient's prediction comes from 5-fold cross-validation, so no patient was scored by a model trained on them. The chat is illustrative; the
        numbers are real. Model A is hypothetical, built to match the lecture slides.
      </SourceNote>
    </div>
  );
}
