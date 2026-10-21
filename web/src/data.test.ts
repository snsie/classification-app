import { describe, expect, it } from "vitest";
import {
  auc, calibrationBins, confusion, Counts, fmtMetric, logisticIRLS, logLoss, metrics, minCostThreshold, MODEL_A, pairOrder,
  pnorm, ppv, prCurve, rocPoints, round2, sandbox, sigmoid, sliderWindow, thresholdForCapacity, thresholdForSensitivity,
} from "./data";

const { p, y } = MODEL_A;
const range = (lo: number, hi: number) => {
  const out: number[] = [];
  for (let k = Math.round(lo * 100); k <= Math.round(hi * 100); k++) out.push(k / 100);
  return out;
};
const counts = (tp: number, fn: number, fp: number, tn: number): Counts => ({ tp, fn, fp, tn });

describe("Model A reproduces every slide number", () => {
  it("has 100 patients, 10 readmitted, all probabilities distinct", () => {
    expect(p.length).toBe(100);
    expect(y.reduce((s, v) => s + v, 0)).toBe(10);
    expect(new Set(p).size).toBe(100);
  });

  it("puts the readmitted patients at the expected probabilities", () => {
    const pos = p.filter((_, i) => y[i] === 1).sort((a, b) => b - a).map((v) => v.toFixed(3));
    expect(pos).toEqual([".962", ".888", ".809", ".782", ".741", ".653", ".600", ".547", ".400", ".225"].map((s) => "0" + s));
  });

  it.each([
    [0.28, 0.32, counts(9, 1, 30, 60)],
    [0.48, 0.52, counts(8, 2, 18, 72)],
    [0.68, 0.72, counts(5, 5, 6, 84)],
    [0.83, 0.87, counts(2, 8, 1, 89)],
    [0.13, 0.17, counts(10, 0, 54, 36)],
  ])("thresholds %s–%s give the slide matrix", (lo, hi, want) => {
    for (const t of range(lo, hi)) expect(confusion(p, y, t)).toEqual(want);
  });

  it("slides 7, 15 and 18: accuracy 80%, precision 30.8%, F1 0.444", () => {
    const m = metrics(confusion(p, y, 0.5));
    expect(m.acc).toBeCloseTo(0.8, 10);
    expect(m.sens).toBeCloseTo(0.8, 10);
    expect(m.spec).toBeCloseTo(0.8, 10);
    expect(m.fpr).toBeCloseTo(0.2, 10);
    expect((m.prec! * 100).toFixed(1)).toBe("30.8");
    expect(m.f1.toFixed(3)).toBe("0.444");
    expect(m.f1).toBeCloseTo(16 / 36, 12);
    expect(m.flagged).toBe(26);
  });

  it("slide 20: 0.70 has 89% accuracy; flag counts 39 / 26 / 11", () => {
    expect(metrics(confusion(p, y, 0.7)).acc).toBeCloseTo(0.89, 10);
    expect(metrics(confusion(p, y, 0.3)).flagged).toBe(39);
    expect(metrics(confusion(p, y, 0.5)).flagged).toBe(26);
    expect(metrics(confusion(p, y, 0.7)).flagged).toBe(11);
    expect(Math.round(metrics(confusion(p, y, 0.3)).spec * 100)).toBe(67);
    expect(Math.round(metrics(confusion(p, y, 0.7)).spec * 100)).toBe(93);
  });

  it("the ROC staircase passes through all 7 slide points", () => {
    const r = rocPoints(p, y);
    const pts = r.fpr.map((f, i) => [f, r.tpr[i]]);
    const want = [[0, 0], [1 / 90, 0.2], [6 / 90, 0.5], [18 / 90, 0.8], [30 / 90, 0.9], [54 / 90, 1], [1, 1]];
    for (const [f, t] of want) expect(pts.some(([a, b]) => Math.abs(a - f) < 1e-12 && Math.abs(b - t) < 1e-12)).toBe(true);
  });

  it("AUROC is exactly 787/900 = 0.874 (shows as 0.87)", () => {
    expect(auc(p, y)).toBeCloseTo(787 / 900, 12);
    expect(auc(p, y).toFixed(2)).toBe("0.87");
  });

  it("the trapezoid area under the staircase equals the Mann–Whitney AUROC", () => {
    const r = rocPoints(p, y);
    let area = 0;
    for (let i = 1; i < r.fpr.length; i++) area += ((r.fpr[i] - r.fpr[i - 1]) * (r.tpr[i] + r.tpr[i - 1])) / 2;
    expect(area).toBeCloseTo(auc(p, y), 12);
  });

  it("§8 Try-it 3: on the slider grid, 0.22 is the highest threshold with TPR 1.0, at FPR 0.48", () => {
    const r = rocPoints(p, y);
    const k = r.tpr.findIndex((t) => t === 1);
    expect(r.thr[k].toFixed(3)).toBe("0.225"); // the last readmitted patient
    const grid = range(0, 1).filter((t) => metrics(confusion(p, y, t)).sens === 1);
    const top = Math.max(...grid);
    expect(top).toBe(0.22);
    expect(metrics(confusion(p, y, top)).fpr.toFixed(2)).toBe("0.48");
  });

  it("§8 Try-it 2: (FPR, TPR) at 0.70, 0.50, 0.30", () => {
    const at = (t: number) => {
      const m = metrics(confusion(p, y, t));
      return [m.fpr.toFixed(2), m.sens.toFixed(2)];
    };
    expect(at(0.7)).toEqual(["0.07", "0.50"]);
    expect(at(0.5)).toEqual(["0.20", "0.80"]);
    expect(at(0.3)).toEqual(["0.33", "0.90"]);
  });
});

describe("metric edge cases", () => {
  it("precision is undefined (not 0) when no one is flagged", () => {
    const m = metrics(counts(0, 10, 0, 90));
    expect(m.prec).toBeNull();
    expect(fmtMetric(m.prec)).toBe("undefined (0 ÷ 0)");
    expect(m.acc).toBeCloseTo(0.9, 12);
    expect(m.f1).toBe(0);
  });

  it("flag everyone: recall 100%, precision 10% (§4 Quiz 3)", () => {
    const m = metrics(confusion(p, y, 0));
    expect(m.sens).toBe(1);
    expect(m.prec).toBeCloseTo(0.1, 12);
  });

  it("§2: always-no accuracy equals 1 − prevalence", () => {
    expect(metrics(counts(0, 5, 0, 95)).acc).toBeCloseTo(0.95, 12);
    expect(metrics(counts(0, 50, 0, 50)).acc).toBeCloseTo(0.5, 12);
  });

  it("§5: 900 extra true negatives leave F1 at 44.4% and lift accuracy to 98.0%", () => {
    const c = confusion(p, y, 0.5);
    const m = metrics({ ...c, tn: c.tn + 900 });
    expect(m.f1.toFixed(3)).toBe("0.444");
    expect(m.acc.toFixed(3)).toBe("0.980");
  });

  it("§4 what-if PPV at sensitivity = specificity = 80%", () => {
    expect((ppv(0.8, 0.8, 0.1) * 100).toFixed(1)).toBe("30.8");
    expect((ppv(0.8, 0.8, 0.05) * 100).toFixed(1)).toBe("17.4");
    expect((ppv(0.8, 0.8, 0.02) * 100).toFixed(1)).toBe("7.5");
  });

  it("auc counts ties as 0.5", () => {
    expect(auc([0.5, 0.5], [1, 0])).toBe(0.5);
    expect(auc([0.9, 0.5, 0.5, 0.1], [1, 1, 0, 0])).toBeCloseTo((1 + 1 + 0.5 + 1) / 4, 12);
    expect(auc([0.2, 0.2, 0.2, 0.2], [1, 0, 1, 0])).toBe(0.5);
  });

  it("rocPoints groups tied probabilities into one step", () => {
    const r = rocPoints([0.9, 0.5, 0.5, 0.1], [1, 1, 0, 0]);
    expect(r.fpr).toEqual([0, 0, 0.5, 1]);
    expect(r.tpr).toEqual([0, 0.5, 1, 1]);
  });

  it("prCurve ends at recall 1 with precision = prevalence", () => {
    const pr = prCurve(p, y);
    expect(pr.recall[pr.recall.length - 1]).toBe(1);
    expect(pr.precision[pr.precision.length - 1]).toBeCloseTo(0.1, 12);
  });
});

describe("§6 Try-it answers", () => {
  it("0.50 → 0.30 finds 1 more and adds 12 false alarms", () => {
    const a = confusion(p, y, 0.5), b = confusion(p, y, 0.3);
    expect(b.tp - a.tp).toBe(1);
    expect(b.fp - a.fp).toBe(12);
  });

  it("0.40 is the highest threshold that still catches 9 of 10, with 24 false alarms", () => {
    expect(confusion(p, y, 0.4).tp).toBe(9);
    expect(confusion(p, y, 0.41).tp).toBe(8);
    expect(confusion(p, y, 0.4).fp).toBe(24);
  });

  it("the best accuracy on the slider grid is 91%, between about 0.77 and 0.96, catching at most 4", () => {
    const grid = range(0, 1);
    const accs = grid.map((t) => metrics(confusion(p, y, t)).acc);
    const best = Math.max(...accs);
    expect(best).toBeCloseTo(0.91, 12);
    const at = grid.filter((_, i) => Math.abs(accs[i] - best) < 1e-12);
    expect(Math.min(...at)).toBe(0.77);
    expect(Math.max(...at)).toBe(0.96);
    expect(Math.max(...at.map((t) => confusion(p, y, t).tp))).toBe(4);
  });

  it("at 0.70, 12 fewer false alarms outweigh 3 extra misses", () => {
    const a = confusion(p, y, 0.5), b = confusion(p, y, 0.7);
    expect(a.fp - b.fp).toBe(12);
    expect(b.fn - a.fn).toBe(3);
  });
});

describe("§7 thresholds for the job", () => {
  it("capacity 26 calls → threshold window 0.48–0.52", () => {
    expect(sliderWindow(thresholdForCapacity(p, 26))).toEqual([0.48, 0.52]);
  });
  it("capacity 10 reaches 5 readmissions; 52 reaches all 10 (+2 over 26)", () => {
    const reach = (k: number) => confusion(p, y, thresholdForCapacity(p, k).top).tp;
    expect(reach(10)).toBe(5);
    expect(reach(26)).toBe(8);
    expect(reach(52)).toBe(10);
  });
  it("target sensitivity 90% → threshold 0.40", () => {
    expect(round2(thresholdForSensitivity(p, y, 0.9))).toBe(0.4);
    expect(thresholdForSensitivity(p, y, 1)).toBeCloseTo(0.225, 3);
  });
  it("minimum-cost threshold is never beaten by any other cut-off", () => {
    for (const cMiss of [1, 5, 10, 30]) {
      const r = minCostThreshold(p, y, cMiss, 1);
      for (const t of range(0, 1)) {
        const c = confusion(p, y, t);
        expect(cMiss * c.fn + c.fp).toBeGreaterThanOrEqual(r.cost);
      }
    }
  });
});

describe("§9 ranking", () => {
  it("pairOrder visits all 900 pairs once and the first draw is a win", () => {
    const order = pairOrder(10, 90);
    expect(new Set(order.map(([a, b]) => a * 90 + b)).size).toBe(900);
    const pos = p.filter((_, i) => y[i] === 1);
    const neg = p.filter((_, i) => y[i] === 0);
    const [i, j] = order[0];
    expect(pos[i]).toBeGreaterThan(neg[j]);
    const wins = order.filter(([a, b]) => pos[a] > neg[b]).length;
    expect(wins).toBe(787);
  });

  it("sandbox d′ = 1.6 gives AUROC ≈ Φ(d′/√2) = 0.871", () => {
    const s = sandbox({ dprime: 1.6 });
    expect(pnorm(1.6 / Math.SQRT2)).toBeCloseTo(0.871, 3);
    expect(Math.abs(auc(s.p, s.y) - pnorm(1.6 / Math.SQRT2))).toBeLessThan(0.004);
  });

  it("sandbox d′ = 0 gives exactly 0.5", () => {
    const s = sandbox({ dprime: 0 });
    expect(auc(s.p, s.y)).toBe(0.5);
  });

  it("sandbox AUROC does not depend on prevalence; precision does", () => {
    const a = sandbox({ dprime: 1.6, prevalence: 0.1 }), b = sandbox({ dprime: 1.6, prevalence: 0.4 });
    expect(Math.abs(auc(a.p, a.y) - auc(b.p, b.y))).toBeLessThan(0.004);
    const pa = metrics(confusion(a.p, a.y, 0.5)).prec!, pb = metrics(confusion(b.p, b.y, 0.5)).prec!;
    expect(pb).toBeGreaterThan(pa);
  });

  it("sandbox is calibrated: every decile's observed rate is near its mean prediction", () => {
    const s = sandbox({ dprime: 1.6, n: 1000 });
    const bins = calibrationBins(s.p, s.y, { bins: 10 });
    expect(bins.length).toBe(10);
    for (const b of bins) expect(Math.abs(b.observed - b.meanPred)).toBeLessThan(0.08);
    const meanP = s.p.reduce((a, v) => a + v, 0) / s.p.length;
    expect(meanP).toBeCloseTo(0.1, 1);
  });
});

describe("logistic regression maths", () => {
  it("IRLS finds the maximum-likelihood fit (score equations are zero)", () => {
    const x = [60, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 130, 140];
    const yy = [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1];
    const { b0, b1 } = logisticIRLS(x, yy);
    let g0 = 0, g1 = 0;
    x.forEach((v, i) => {
      const r = yy[i] - sigmoid(b0 + b1 * v);
      g0 += r;
      g1 += r * v;
    });
    expect(Math.abs(g0)).toBeLessThan(1e-8);
    expect(Math.abs(g1)).toBeLessThan(1e-6);
    const best = logLoss(x, yy, b0, b1);
    expect(logLoss(x, yy, b0 + 0.05, b1)).toBeGreaterThan(best);
    expect(logLoss(x, yy, b0, b1 * 1.02)).toBeGreaterThan(best);
  });

  it("log-loss of the flat model is the entropy of the base rate", () => {
    const yy = [1, 0, 0, 0];
    const b0 = Math.log(0.25 / 0.75);
    expect(logLoss([1, 2, 3, 4], yy, b0, 0)).toBeCloseTo(-(0.25 * Math.log(0.25) + 0.75 * Math.log(0.75)), 12);
  });
});
