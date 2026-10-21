import { useMemo, useState } from "react";
import { line } from "d3-shape";
import { fmtT, ThresholdSlider } from "../charts/classify";
import { Legend, Plot } from "../charts/kit";
import { Callout, Card, DataChip, Loading, PauseThink, Slider, SourceNote, Toggle } from "../components/ui";
import { fmt, fmtInt, fmtPct, fmtSigned, logisticIRLS, logit, logLoss, mean, PlaygroundData, sigmoid, useData } from "../data";

type Snap = "best" | "flat" | null;

export default function Logistic() {
  const pg = useData<PlaygroundData>("playground");
  const L = pg?.logistic_1d;
  const base = useMemo(() => {
    if (!L) return null;
    const xbar = mean(L.all.x);
    const irls = logisticIRLS(L.all.x, L.all.y);
    return { xbar, irls, bestLoss: logLoss(L.all.x, L.all.y, irls.b0, irls.b1), rate: L.n_pos / L.n };
  }, [L]);
  // the line is parameterised as (weight per cm, log-odds at the average waist) so the sliders stay intuitive
  const [b1, setB1] = useState<number | null>(null);
  const [level, setLevel] = useState<number | null>(null);
  const [snapped, setSnapped] = useState<Snap>("best");
  const [showZ, setShowZ] = useState(false);
  const [t, setT] = useState(0.5);
  const [ex, setEx] = useState(100);
  if (!L || !base) return <Loading />;

  const slope = b1 ?? L.slope;
  const lev = level ?? L.intercept + L.slope * base.xbar;
  const b0 = lev - slope * base.xbar;
  const loss = logLoss(L.all.x, L.all.y, b0, slope);
  const ratio = loss / base.bestLoss;
  const cross = slope > 0 ? (logit(t) - b0) / slope : Infinity;
  let flagged = 0, caught = 0;
  L.all.x.forEach((w, i) => {
    if (sigmoid(b0 + slope * w) >= t - 1e-9) {
      flagged++;
      caught += L.all.y[i];
    }
  });
  const zEx = b0 + slope * ex, pEx = sigmoid(zEx);
  // the fitted model's threshold that flags the top 10%
  const pAll = L.all.x.map((w) => sigmoid(L.intercept + L.slope * w)).sort((a, b) => b - a);
  const k10 = Math.ceil(0.1 * L.n);
  const t10 = pAll[k10 - 1];
  const w10 = (logit(t10) - L.intercept) / L.slope;

  const snap = (which: Snap) => {
    setSnapped(which);
    if (which === "best") {
      setB1(L.slope);
      setLevel(L.intercept + L.slope * base.xbar);
    } else {
      setB1(0);
      setLevel(logit(base.rate));
    }
  };
  const xd: [number, number] = [L.curve.x[0], L.curve.x[L.curve.x.length - 1]];
  const xs = L.curve.x;
  const btn = (w: Snap) => (snapped === w ? "btn primary" : "btn");

  return (
    <div className="stack">
      <div className="split-wide">
        <Card title="Waist circumference → diabetes" sub={`Dots at the top have diabetes, at the bottom don't (600 of ${fmtInt(L.n)} adults drawn; the fit uses all of them). The curve is your model's probability.`}
          chip={<DataChip n={L.n} unit="adults" />}>
          <div className="controls-bar">
            <Slider label="Weight per cm of waist (b₁)" value={slope} min={0} max={3 * L.slope} step={L.slope / 100}
              onChange={(v) => { setB1(v); setLevel(lev); setSnapped(null); }} format={(v) => v.toFixed(4)} />
            <Slider label="Log-odds at the average waist" value={lev} min={-4} max={1} step={0.01}
              onChange={(v) => { setLevel(v); setB1(slope); setSnapped(null); }} format={(v) => fmtSigned(v, 2)} />
          </div>
          <div className="row" style={{ marginBottom: 8 }}>
            <button className={btn("best")} onClick={() => snap("best")}>Snap to best fit (maximum likelihood)</button>
            <button className={btn("flat")} onClick={() => snap("flat")}>Flat: everyone gets the average risk</button>
          </div>
          <Legend items={[
            { label: "Estimated probability", color: "var(--cls-pos)" },
            { label: `Threshold ${fmtT(t)}`, color: "var(--ink-2)", kind: "dash" },
            { label: "Has diabetes (top) / no diabetes (bottom)", color: "var(--cls-neg)", kind: "dot" },
          ]} />
          <Plot height={360} xDomain={xd} yDomain={[-0.12, 1.12]} nice={false} xLabel="Waist circumference (cm)" yLabel="Probability of diabetes"
            yTicks={[0, 0.25, 0.5, 0.75, 1]} ariaLabel="Logistic curve of diabetes probability against waist">
            {({ x, y }) => (
              <g>
                {L.points.x.map((w, i) => {
                  const jit = (((i * 0.6180339887) % 1) - 0.5) * 0.11;
                  const pos = L.points.y[i] === 1;
                  return <circle key={i} cx={x(w)} cy={y(L.points.y[i] + (pos ? -1 : 1) * Math.abs(jit))} r={3}
                    fill={pos ? "var(--cls-pos)" : "var(--surface)"} stroke={pos ? "none" : "var(--cls-neg)"} strokeWidth={1.2} opacity={0.75} />;
                })}
                <line x1={x(xd[0])} x2={x(xd[1])} y1={y(t)} y2={y(t)} stroke="var(--ink-2)" strokeDasharray="5 4" />
                {isFinite(cross) && cross >= xd[0] && cross <= xd[1] && (
                  <g>
                    <line x1={x(cross)} x2={x(cross)} y1={y(t)} y2={y(-0.12)} stroke="var(--ink)" strokeWidth={1.5} />
                    <text x={x(cross) + 6} y={y(t) + 16} fontSize={12} fontWeight={700} fill="var(--ink)">{fmt(cross, 0)} cm</text>
                  </g>
                )}
                <path d={line<number>().x((w) => x(w)).y((w) => y(sigmoid(b0 + slope * w)))(xs) ?? ""} fill="none" stroke="var(--cls-pos)" strokeWidth={3} />
                <line x1={x(ex)} x2={x(ex)} y1={y(0)} y2={y(pEx)} stroke="var(--accent)" strokeDasharray="2 3" />
                <circle cx={x(ex)} cy={y(pEx)} r={6} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} />
              </g>
            )}
          </Plot>
          {showZ && (
            <Plot height={170} xDomain={xd} yDomain={[-6, 2]} nice={false} xLabel="Waist circumference (cm)" yLabel="Log-odds z" yTicks={[-6, -4, -2, 0, 2]}
              ariaLabel="The weighted sum (log-odds) is a straight line">
              {({ x, y }) => (
                <g>
                  <line x1={x(xd[0])} x2={x(xd[1])} y1={y(logit(t))} y2={y(logit(t))} stroke="var(--ink-2)" strokeDasharray="5 4" />
                  <text x={x(xd[0]) + 4} y={y(logit(t)) - 6} fontSize={11} fill="var(--muted)">z = logit({fmtT(t)}) = {fmt(logit(t), 2)}</text>
                  <line x1={x(xd[0])} x2={x(xd[1])} y1={y(b0 + slope * xd[0])} y2={y(b0 + slope * xd[1])} stroke="var(--cls-pos)" strokeWidth={2.5} />
                </g>
              )}
            </Plot>
          )}
          <div className="controls-bar" style={{ marginTop: 8 }}>
            <Toggle checked={showZ} onChange={setShowZ}>Show the weighted sum (log-odds)</Toggle>
            <ThresholdSlider value={t} onChange={setT} min={0.05} max={0.95} presets={[0.2, 0.3, 0.5]} width={300} />
            <Slider label="Example waist" value={ex} min={60} max={170} onChange={setEx} format={(v) => `${v} cm`} width={260} />
          </div>
        </Card>

        <div className="stack">
          <Card>
            <div className="stat-label">Your model</div>
            <div style={{ fontSize: "1.05rem", fontWeight: 600, margin: "6px 0" }} className="tnum">
              p = 1 ÷ (1 + e<sup>−z</sup>), &nbsp; z = {fmt(b0, 3)} {slope >= 0 ? "+" : "−"} {Math.abs(slope).toFixed(4)} × waist
            </div>
            <div className="small ink2">
              <b>Worked example</b>, waist {ex} cm: z = {fmt(b0, 3)} + {slope.toFixed(4)} × {ex} = {fmt(zEx, 3)} → p = {fmt(pEx, 3)} →{" "}
              {pEx >= t - 1e-9 ? <b>predicted: has diabetes</b> : <b>predicted: no diabetes</b>} (threshold {fmtT(t)})
            </div>
          </Card>
          <Card>
            <div className="stat-label">Where p crosses {fmtT(t)}</div>
            <div className="stat-value tnum">{isFinite(cross) ? `≈ ${fmt(cross, 0)} cm` : "never"}</div>
            <div className="stat-note">Flags <b>{fmtInt(flagged)} of {fmtInt(L.n)}</b> adults ({fmtPct(flagged / L.n, 1)}), {fmtInt(caught)} of the {fmtInt(L.n_pos)} with diabetes.</div>
          </Card>
          <Card>
            <div className="stat-label">Log-loss over all {fmtInt(L.n)} adults</div>
            <div className="stat-value tnum">{loss.toFixed(4)}</div>
            <div style={{ height: 10, background: "var(--grid)", borderRadius: 6, overflow: "hidden" }} aria-hidden>
              <div style={{ height: "100%", width: `${Math.min(100, 100 / ratio)}%`, background: ratio < 1.0005 ? "var(--good)" : "var(--accent)", transition: "width .2s" }} />
            </div>
            <div className="small ink2" style={{ marginTop: 6 }}>
              {ratio < 1.0005 ? <>✓ This is the minimum: the maximum-likelihood fit.</> : <>{fmt((ratio - 1) * 100, 1)}% above the smallest possible value ({base.bestLoss.toFixed(4)}).</>}
              {" "}Logistic regression chooses the weights that make this as small as possible.
            </div>
          </Card>
        </div>
      </div>

      <div className="grid-2">
        <PauseThink prompt="Two adults differ by 10 cm of waist. How do their odds of diabetes compare under the fitted model?">
          Odds × e<sup>10·b₁</sup> = e<sup>10 × {L.slope.toFixed(4)}</sup> = <b>{Math.exp(10 * L.slope).toFixed(2)}</b>: about {Math.round((Math.exp(10 * L.slope) - 1) * 100)}% higher odds,
          wherever you are on the curve. The probability gap, though, depends on where you start: that's the S-shape.
        </PauseThink>
        <PauseThink prompt="With the fitted model, which threshold flags the 10% of adults with the largest waists?">
          About <b>{t10.toFixed(2)}</b>: p ≥ {t10.toFixed(3)} means a waist of at least {fmt(w10, 0)} cm, which flags {fmtInt(k10)} adults. The probability only
          ranks people by waist here, so any cut-off is a cut-off on waist.
        </PauseThink>
      </div>

      <div className="grid-2">
        <Callout tone="good">
          <b>Reach for logistic regression when…</b> you want something simple and checkable: each weight can be read. In code:{" "}
          <code>p = LogisticRegression().fit(X, y).predict_proba(X_new)[:, 1]</code>, then <code>p &gt;= threshold</code>. Note that scikit-learn adds L2 regularisation
          by default (C = 1); this fit used C = 10⁶, effectively none.
        </Callout>
        <Callout tone="warn">
          <b>Watch out for…</b> the default threshold. At 0.50 this model flags only {fmtInt(L.flagged_at_050)} of {fmtInt(L.n)} adults ({fmtPct(L.flagged_at_050 / L.n, 1)}),
          because p crosses 0.50 at {fmt(L.p50_waist, 0)} cm, above almost everyone in the data.
        </Callout>
      </div>
      <SourceNote>
        NHANES 2017–2018 adults with HbA1c, a diabetes answer and a waist measurement (n = {fmtInt(L.n)}; {fmtInt(L.n_pos)} with diabetes: told by a doctor, or HbA1c ≥ 6.5%).
        Pipeline fit: b₀ = {L.intercept.toFixed(3)}, b₁ = {L.slope.toFixed(4)} per cm. The log-loss meter recomputes the maximum-likelihood fit in your browser
        (b₀ = {base.irls.b0.toFixed(3)}, b₁ = {base.irls.b1.toFixed(4)}).
      </SourceNote>
    </div>
  );
}
