"""Train every model, run every verification analysis, and write JSON for the web app.

Usage:  python run_all.py                     (full run, needs a CUDA GPU for TabICL)
        python run_all.py --quick             (fewer repeats and bootstrap draws, for development)
        python run_all.py --no-tabicl         (TabICL results are written as null)
        python run_all.py --only lab          (rerun one part: datasets [ids], playground, lab)

Output goes to ../web/public/data/.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict

import datasets as D
import models as M
from models import MODEL_DETAILS, MODELS, N_FOLDS, SEED, SPLITS, dev_test, folds, make_model, smote_before, smote_inside

sys.stdout.reconfigure(encoding="utf-8")

OUT = Path(__file__).parent.parent / "web" / "public" / "data"
FITTED = ["logistic", "tree", "rf", "gbm", "tabicl"]
SIMPLICITY = ["logistic", "tree", "rf", "gbm", "tabicl"]  # simplest first

# Hard-coded from last week's app (regression-app web/public/data/lab.json, Parkinson's leakage runs:
# random forest with leaks removed, random-row KFold vs GroupKFold by patient).
PARKINSONS_REF = {"random_r2": 0.9727, "group_r2": -1.1315,
                  "source": "regression-app lab.json: Parkinson's telemonitoring, random rows vs new patients"}


# ---------------------------------------------------------------- helpers
def rnd(a, d=4):
    """Round to a JSON-friendly list; NaN becomes null."""
    if np.isscalar(a):
        return None if a is None or not np.isfinite(a) else round(float(a), d)
    arr = np.asarray(a, dtype=float)
    return [None if not np.isfinite(v) else round(float(v), d) for v in arr.ravel()] if arr.ndim == 1 else [rnd(r, d) for r in arr]


def ints(p, scale=10000):
    """Probabilities as integers in 1/scale (the app divides them back)."""
    return np.rint(np.clip(np.asarray(p, float), 0, 1) * scale).astype(int).tolist()


def auroc(y, p):
    return float(roc_auc_score(y, p))


def counts_at(y, p, t=0.5):
    flag = np.asarray(p) >= t
    y = np.asarray(y)
    return {"tp": int((flag & (y == 1)).sum()), "fn": int((~flag & (y == 1)).sum()),
            "fp": int((flag & (y == 0)).sum()), "tn": int((~flag & (y == 0)).sum())}


def proba(est, X):
    return est.predict_proba(X)[:, 1]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Ctx:
    quick = False
    tabicl = True
    bootstrap = 1000
    peek_repeats = 20


C = Ctx()


def models_to_run(names):
    return [m for m in names if m != "tabicl" or C.tabicl]


# ---------------------------------------------------------------- per-dataset analyses
def cross_validation(ds: D.Dataset, extra: dict | None = None):
    """5-fold stratified CV, same folds for every model; per-fold scores and out-of-fold probabilities."""
    X, y = ds.X.to_numpy(float), ds.y.to_numpy(int)
    cv = folds("kfold", y)
    specs = {m: (lambda m=m: make_model(m)) for m in MODELS} | (extra or {})
    models, oof, pooled = {}, {}, {}
    for name, mk in specs.items():
        if name == "tabicl" and not C.tabicl:
            models[name] = oof[name] = None
            continue
        t0 = time.time()
        o = np.full(len(y), np.nan)
        fs = []
        for tr, te in cv:
            est = mk().fit(X[tr], y[tr])
            ptr, pte = proba(est, X[tr]), proba(est, X[te])
            o[te] = pte
            fs.append({"auroc": rnd(auroc(y[te], pte)), "train_auroc": rnd(auroc(y[tr], ptr)),
                       "auprc": rnd(average_precision_score(y[te], pte)), "brier": rnd(brier_score_loss(y[te], pte))})
        q = ints(o)
        pr = np.asarray(q) / 10000  # every number the app can recompute comes from the rounded integers
        au = [f["auroc"] for f in fs]
        models[name] = {
            "folds": fs,
            "mean_auroc": rnd(np.mean(au)), "sd_auroc": rnd(np.std(au, ddof=1)),
            "mean_train_auroc": rnd(np.mean([f["train_auroc"] for f in fs])),
            "mean_auprc": rnd(np.mean([f["auprc"] for f in fs])), "mean_brier": rnd(np.mean([f["brier"] for f in fs])),
            "mean_pred": rnd(pr.mean()),
            "at_050": counts_at(y, pr),
        }
        oof[name] = q
        pooled[name] = rnd(auroc(y, pr))
        log(f"  CV {ds.id}/{name}: {np.mean(au):.4f} ± {np.std(au, ddof=1):.4f}  ({time.time() - t0:.0f}s)")
    return {"n_folds": N_FOLDS, "models": models, "oof": oof, "pooled_auroc": pooled}


def bootstrap_ci(y, p, n, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(roc_auc_score(y[i], p[i]))
    return [rnd(np.percentile(vals, 2.5)), rnd(np.percentile(vals, 97.5))]


def test_view(ds: D.Dataset):
    """Lock a 20% test set, choose with 5-fold CV inside development, then score the test set once per model."""
    X, y = ds.X.to_numpy(float), ds.y.to_numpy(int)
    groups = None if ds.groups is None else ds.groups.to_numpy()
    dev, test = dev_test(y, groups)
    cv = folds("kfold", y[dev])
    out = {"n_dev": int(len(dev)), "n_test": int(len(test)), "dev_pos": int(y[dev].sum()), "test_pos": int(y[test].sum()),
           "models": {}, "test_y": y[test].tolist(), "test_pred": {}}
    for m in models_to_run(["baseline"] + FITTED):
        fa = []
        for tr, va in cv:
            est = make_model(m).fit(X[dev][tr], y[dev][tr])
            fa.append(auroc(y[dev][va], proba(est, X[dev][va])))
        est = make_model(m).fit(X[dev], y[dev])
        pt = proba(est, X[test])
        q = ints(pt)
        out["models"][m] = {"fold_auroc": rnd(fa), "mean": rnd(np.mean(fa)), "sd": rnd(np.std(fa, ddof=1)),
                            "test_auroc": rnd(auroc(y[test], np.asarray(q) / 10000)),
                            "test_ci": bootstrap_ci(y[test], np.asarray(q) / 10000, C.bootstrap)}
        out["test_pred"][m] = q
        log(f"  test view {ds.id}/{m}: CV {np.mean(fa):.4f}, test {out['models'][m]['test_auroc']}")
    if not C.tabicl:
        out["models"]["tabicl"] = None
        out["test_pred"]["tabicl"] = None
    ran = [m for m in FITTED if out["models"].get(m)]
    best = max(ran, key=lambda m: out["models"][m]["mean"])
    bar = out["models"][best]["mean"] - out["models"][best]["sd"]
    out["best_by_cv"] = best
    out["simplest_within_1sd"] = next(m for m in SIMPLICITY if m in ran and out["models"][m]["mean"] >= bar)
    return out


def dataset_json(ds: D.Dataset, extra=None, with_test=True):
    log(f"dataset {ds.id}: n={len(ds.y)}, positives={int(ds.y.sum())}")
    y = ds.y.to_numpy(int)
    out = {
        "id": ds.id, "name": ds.name, "story": ds.story, "source": ds.source, "citation": ds.citation, "url": ds.url,
        "labels": ds.labels,
        "n_rows": int(len(y)), "n_pos": int(y.sum()), "prevalence": rnd(y.mean()),
        "cohort": ds.cohort,
        "features": [{"id": c, "label": ds.feature_labels.get(c, c)} for c in ds.X.columns],
        "split": ds.split,
        "y": y.tolist(),
        "cv": cross_validation(ds, extra),
    }
    if with_test:
        out["test_view"] = test_view(ds)
    return out


# ---------------------------------------------------------------- playground (2 features → probability map)
GRID = 41
TREE_DEPTHS = list(range(1, 21))
RF_TREES = [1, 3, 10, 30, 100, 300]
RF_LEAVES = [1, 5, 20]
GB_ROUNDS = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300]
GB_LRS = [0.03, 0.1, 0.3]


def pct(p):
    return np.rint(np.clip(p, 0, 1) * 100).astype(int).tolist()


def _cv_staged(fit_stage, X, y, stages):
    """Mean train and CV AUROC for each stage, using one fit per fold. fit_stage(Xtr, ytr) → f(X, stage) → p."""
    tr_s, te_s = [], []
    for tr, te in folds("kfold", y):
        f = fit_stage(X[tr], y[tr])
        tr_s.append([auroc(y[tr], f(X[tr], s)) for s in stages])
        te_s.append([auroc(y[te], f(X[te], s)) for s in stages])
    return np.mean(tr_s, axis=0), np.mean(te_s, axis=0)


def _rf_stager(leaf):
    def fit(Xtr, ytr):
        m = make_model("rf", n_estimators=max(RF_TREES), min_samples_leaf=leaf, impute=False).fit(Xtr, ytr)
        return lambda X, n: np.mean([t.predict_proba(X)[:, 1] for t in m.estimators_[:n]], axis=0)
    return fit


def _gb_stager(lr):
    def fit(Xtr, ytr):
        m = LGBMClassifier(n_estimators=max(GB_ROUNDS), learning_rate=lr, num_leaves=4, min_child_samples=5,
                           random_state=SEED, verbose=-1, n_jobs=8).fit(Xtr, ytr)
        prev = ytr.mean()
        return lambda X, r: np.full(len(X), prev) if r == 0 else m.predict_proba(X, num_iteration=r)[:, 1]
    return fit


def playground_pair(key, X, y, xr, yr, meta, points_idx):
    log(f"playground {key}: n={len(y)}")
    xs, ys = np.linspace(*xr, GRID), np.linspace(*yr, GRID)
    G = np.array([[a, b] for b in ys for a in xs])  # row-major: rows are y values (ascending), columns are x values
    entry = dict(meta) | {
        "n": int(len(y)), "n_pos": int(y.sum()),
        "x_axis": rnd(xs, 3), "y_axis": rnd(ys, 3),
        "points": {"x": rnd(X[points_idx, 0], 1), "y": rnd(X[points_idx, 1], 1), "label": y[points_idx].astype(int).tolist()},
    }

    def scores(mk):
        tr_s, te_s = [], []
        for tr, te in folds("kfold", y):
            m = mk().fit(X[tr], y[tr])
            tr_s.append(auroc(y[tr], proba(m, X[tr])))
            te_s.append(auroc(y[te], proba(m, X[te])))
        return rnd(np.mean(tr_s)), rnd(np.mean(te_s))

    lg = make_model("logistic").fit(X, y)
    sc, lr = lg.named_steps["standardscaler"], lg.named_steps["logisticregression"]
    coef = lr.coef_[0] / sc.scale_
    intercept = lr.intercept_[0] - np.sum(lr.coef_[0] * sc.mean_ / sc.scale_)
    tr_a, cv_a = scores(lambda: make_model("logistic"))
    entry["logistic"] = {"prob": pct(proba(lg, G)), "train_auroc": tr_a, "cv_auroc": cv_a,
                         "intercept": rnd(intercept, 6), "coef": rnd(coef, 6)}

    trees = {}
    for d in TREE_DEPTHS:
        m = make_model("tree", max_depth=d, impute=False).fit(X, y)
        node_g = m.apply(G)
        node_t = m.apply(X)
        ids = sorted(set(node_t))
        remap = {n: i for i, n in enumerate(ids)}
        leaves = [[int(y[node_t == n].sum()), int((node_t == n).sum())] for n in ids]
        tr_a, cv_a = scores(lambda d=d: make_model("tree", max_depth=d, impute=False))
        trees[str(d)] = {"prob": pct(proba(m, G)), "leaf": [remap[n] for n in node_g], "leaves": leaves,
                         "train_auroc": tr_a, "cv_auroc": cv_a}
    entry["tree"] = trees

    rf = {}
    for leaf in RF_LEAVES:
        full = _rf_stager(leaf)(X, y)
        tr_s, te_s = _cv_staged(_rf_stager(leaf), X, y, RF_TREES)
        for i, n in enumerate(RF_TREES):
            rf[f"{n}_{leaf}"] = {"prob": pct(full(G, n)), "train_auroc": rnd(tr_s[i]), "cv_auroc": rnd(te_s[i])}
    entry["rf"] = rf

    gb = {}
    for lrate in GB_LRS:
        full = _gb_stager(lrate)(X, y)
        tr_s, te_s = _cv_staged(_gb_stager(lrate), X, y, GB_ROUNDS)
        for i, r in enumerate(GB_ROUNDS):
            gb[f"{lrate}_{r}"] = {"prob": pct(full(G, r)), "train_auroc": rnd(tr_s[i]), "cv_auroc": rnd(te_s[i])}
    entry["gbm"] = gb

    if C.tabicl:
        m = make_model("tabicl").fit(X, y)
        tr_a, cv_a = scores(lambda: make_model("tabicl"))
        entry["tabicl"] = {"prob": pct(proba(m, G)), "train_auroc": tr_a, "cv_auroc": cv_a}
    else:
        entry["tabicl"] = None
    return entry


def logistic_curve():
    """Slide 28: one feature (waist) → diabetes, fitted on every adult with a waist measurement."""
    w, funnel = D.waist_cohort()
    x, y = w.waist.to_numpy(float), w.y.to_numpy(int)
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(x[:, None], y)
    b0, b1 = float(lr.intercept_[0]), float(lr.coef_[0, 0])
    p = lr.predict_proba(x[:, None])[:, 1]
    grid = np.linspace(60, 180, 61)
    s = w.sample(n=600, random_state=SEED)
    log(f"logistic 1-D: n={len(y)}, b0={b0:.4f}, b1={b1:.5f}, p50={-b0 / b1:.1f} cm, flagged@0.5={int((p >= 0.5).sum())}")
    return {
        "n": int(len(y)), "n_pos": int(y.sum()), "intercept": rnd(b0, 6), "slope": rnd(b1, 6),
        "p50_waist": rnd(-b0 / b1, 2), "flagged_at_050": int((p >= 0.5).sum()),
        "curve": {"x": rnd(grid, 1), "p": rnd(lr.predict_proba(grid[:, None])[:, 1], 5)},
        "points": {"x": rnd(s.waist.to_numpy(float), 1), "y": s.y.astype(int).tolist()},
        "all": {"x": rnd(x, 1), "y": y.tolist()},
        "cohort": funnel,
    }


def playground():
    h = D.heart()
    Xh = h.X[["age", "thalach"]].to_numpy(float)
    yh = h.y.to_numpy(int)
    heart = playground_pair("heart", Xh, yh, (25, 80), (65, 210), {
        "label": "Heart disease: age × maximum heart rate", "x_label": "Age", "x_units": "years",
        "y_label": "Maximum heart rate", "y_units": "bpm", "pos_label": "Heart disease", "neg_label": "No heart disease",
        "sample": "all 303 Cleveland patients",
    }, np.arange(len(yh)))
    w, _ = D.waist_cohort()
    Xn = w[["age", "waist"]].to_numpy(float)
    yn = w.y.to_numpy(int)
    shown = w.sample(n=600, random_state=SEED).index.to_numpy()
    nhanes = playground_pair("nhanes", Xn, yn, (18, 80), (60, 170), {
        "label": "NHANES: age × waist → diabetes", "x_label": "Age", "x_units": "years",
        "y_label": "Waist", "y_units": "cm", "pos_label": "Has diabetes", "neg_label": "No diabetes",
        "sample": "600 of 4,977 NHANES adults drawn; models fitted on all",
    }, shown)
    return {"pairs": {"heart": heart, "nhanes": nhanes}, "tree_depths": TREE_DEPTHS, "rf_trees": RF_TREES,
            "rf_leaves": RF_LEAVES, "gb_rounds": GB_ROUNDS, "gb_lrs": GB_LRS, "logistic_1d": logistic_curve()}


# ---------------------------------------------------------------- lab analyses
def hook(readmit_json):
    """Slide 8: the assistant's GBM at 0.50 versus 'not readmitted' for everyone, on the out-of-fold predictions."""
    g = readmit_json["cv"]["models"]["gbm"]
    c = g["at_050"]
    n, n_pos = readmit_json["n_rows"], readmit_json["n_pos"]
    return {"n": n, "n_pos": n_pos, "prevalence": rnd(n_pos / n), "always_no_acc": rnd(1 - n_pos / n),
            "model": "gbm", "threshold": 0.5, "counts": c, "acc": rnd((c["tp"] + c["tn"]) / n)}


def calibration(readmit_json):
    y = np.asarray(readmit_json["y"])
    out = {}
    for name, p in [("logistic", readmit_json["cv"]["oof"]["logistic"]),
                    ("logistic_halved", (np.asarray(readmit_json["cv"]["oof"]["logistic"]) / 2).tolist()),
                    ("logistic_balanced", readmit_json["cv"]["oof"]["logistic_balanced"])]:
        p = np.asarray(p) / 10000
        order = np.argsort(p, kind="stable")
        bins = []
        for chunk in np.array_split(order, 10):
            bins.append({"lo": rnd(p[chunk].min()), "hi": rnd(p[chunk].max()), "meanPred": rnd(p[chunk].mean()),
                         "observed": rnd(y[chunk].mean()), "n": int(len(chunk))})
        out[name] = bins
    return out


def train_vs_unseen():
    """Slide 29: heart, age and max heart rate; mean over the 5 folds of training and unseen AUROC."""
    h = D.heart()
    X, y = h.X[["age", "thalach"]].to_numpy(float), h.y.to_numpy(int)
    cv = folds("kfold", y)

    def score(mk):
        a, b = [], []
        for tr, te in cv:
            m = mk().fit(X[tr], y[tr])
            a.append(auroc(y[tr], proba(m, X[tr])))
            b.append(auroc(y[te], proba(m, X[te])))
        return np.mean(a), np.mean(b)

    tree = [score(lambda d=d: make_model("tree", max_depth=d, impute=False)) for d in TREE_DEPTHS]
    lg = score(lambda: make_model("logistic"))
    log(f"train vs unseen: depth-20 tree {tree[-1][0]:.3f}/{tree[-1][1]:.3f}; logistic {lg[0]:.3f}/{lg[1]:.3f}")
    return {"dataset": "heart", "features": ["age", "thalach"], "n": int(len(y)),
            "tree": {"depth": TREE_DEPTHS, "train": rnd([t[0] for t in tree]), "unseen": rnd([t[1] for t in tree])},
            "logistic": {"train": rnd(lg[0]), "unseen": rnd(lg[1])}}


def _cv_scores(mk, X, y, split="kfold", groups=None):
    tr_s, te_s = [], []
    for tr, te in folds(split, y, groups):
        m = mk().fit(X[tr], y[tr])
        tr_s.append(auroc(y[tr], proba(m, X[tr])))
        te_s.append(auroc(y[te], proba(m, X[te])))
    return tr_s, te_s


def leak_run(model, resample, X, y, split="kfold", groups=None):
    """One leakage-lab configuration; returns train and test AUROC per fold."""
    if resample == "none":
        mk = lambda: make_model(model)
        return _cv_scores(mk, X, y, split, groups)
    if resample == "inside":
        mk = lambda: smote_inside(make_model(model, impute=False))
        return _cv_scores(mk, X, y, split, groups)
    Xr, yr = smote_before(X, y)
    n_syn = len(yr) - len(y)
    gr = None
    if groups is not None:
        # synthetic rows have no patient: each gets its own id, so "split by patient" still lets near-copies leak
        gr = np.concatenate([np.asarray(groups), np.asarray(groups).max() + 1 + np.arange(n_syn)])
    return _cv_scores(lambda: make_model(model, impute=False), Xr, yr, split, gr)


def leakage_lab():
    log("leakage lab: diabetes screening")
    fair = D.diabetes(leaky=False)
    leaky = D.diabetes(leaky=True)
    y = fair.y.to_numpy(int)
    runs_d = []
    for model in ("rf", "gbm"):
        for resample in ("none", "inside", "before"):
            for lk in (False, True):
                X = (leaky if lk else fair).X.to_numpy(float)
                tr_s, te_s = leak_run(model, resample, X, y)
                runs_d.append({"model": model, "resample": resample, "leaky": lk,
                               "train_auroc": rnd(np.mean(tr_s)), "test_auroc": rnd(np.mean(te_s)), "folds": rnd(te_s)})
                log(f"    diabetes {model} {resample:6s} leaky={lk!s:5s}: train {np.mean(tr_s):.3f} test {np.mean(te_s):.3f}")
    log("leakage lab: readmission encounters")
    e = D.readmission_encounters()
    X, y, g = e.X.to_numpy(float), e.y.to_numpy(int), e.groups.to_numpy()
    runs_r = []
    for model in ("rf", "gbm"):
        for resample in ("none", "inside", "before"):
            for split in ("row", "patient"):
                tr_s, te_s = leak_run(model, resample, X, y, split, g)
                runs_r.append({"model": model, "resample": resample, "split": split,
                               "train_auroc": rnd(np.mean(tr_s)), "test_auroc": rnd(np.mean(te_s)), "folds": rnd(te_s)})
                log(f"    readmission {model} {resample:6s} {split:7s}: train {np.mean(tr_s):.3f} test {np.mean(te_s):.3f}")
    return {
        "diabetes": {"n": int(len(fair.y)), "n_pos": int(fair.y.sum()), "fair_features": list(fair.X.columns),
                     "leaky_features": list(D.LEAKY), "runs": runs_d},
        "readmission": {"n_rows": int(len(y)), "n_patients": int(len(np.unique(g))), "n_pos": int(y.sum()), "runs": runs_r},
    }


def unfair(leak, readmit: D.Dataset):
    """Slide 32's bars. The diabetes pairs come from the leakage lab; SMOTE-before uses the main 20,000-patient cohort."""
    def run(model, resample, lk):
        return next(r for r in leak["diabetes"]["runs"] if r["model"] == model and r["resample"] == resample and r["leaky"] == lk)

    rf = run("rf", "none", False)
    X, y = readmit.X.to_numpy(float), readmit.y.to_numpy(int)
    _, before = leak_run("gbm", "before", X, y)
    _, inside = leak_run("gbm", "inside", X, y)
    log(f"unfair: SMOTE before {np.mean(before):.3f} vs inside {np.mean(inside):.3f} (GBM, readmission)")
    return [
        {"id": "train", "title": "Scored on its own training patients", "detail": "Random forest, diabetes screening",
         "unfair": rf["train_auroc"], "fair": rf["test_auroc"], "slide_unfair": 0.998, "slide_fair": 0.808},
        {"id": "smote", "title": "Resampling (SMOTE) before splitting the data", "detail": "Gradient boosting, 30-day readmission",
         "unfair": rnd(np.mean(before)), "fair": rnd(np.mean(inside)), "slide_unfair": 0.96, "slide_fair": 0.62},
        {"id": "leaky", "title": "Columns that give away the answer, plus SMOTE before splitting",
         "detail": "Gradient boosting, diabetes screening (HbA1c and diabetes medicines left in)",
         "unfair": run("gbm", "before", True)["test_auroc"], "fair": run("gbm", "inside", False)["test_auroc"],
         "slide_unfair": 0.997, "slide_fair": 0.796},
    ]


def _thr(p_pos_desc, t):
    """The threshold that catches ceil(t·P) of the P positives: the k-th highest positive probability."""
    k = max(1, math.ceil(round(t, 2) * len(p_pos_desc) - 1e-9))
    return p_pos_desc[k - 1]


def peeking():
    """Slide 32: a target sensitivity set on the test fold (peeking) versus on the training part (fair)."""
    h = D.heart()
    X, y = h.X.to_numpy(float), h.y.to_numpy(int)
    targets = [round(t, 2) for t in np.arange(0.70, 0.951, 0.01)]
    keep = [0.80, 0.85, 0.90, 0.95]
    rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=C.peek_repeats, random_state=SEED)
    peek = np.zeros((C.peek_repeats * N_FOLDS, len(targets)))
    fair = np.zeros_like(peek)
    examples = []
    for s, (tr, te) in enumerate(rskf.split(X, y)):
        m = make_model("logistic").fit(X[tr], y[tr])
        pte = proba(m, X[te])
        inner = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
        ptr = cross_val_predict(make_model("logistic"), X[tr], y[tr], cv=inner, method="predict_proba")[:, 1]
        pos_te = np.sort(pte[y[te] == 1])[::-1]
        pos_tr = np.sort(ptr[y[tr] == 1])[::-1]
        for j, t in enumerate(targets):
            peek[s, j] = np.mean(pos_te >= _thr(pos_te, t))
            fair[s, j] = np.mean(pos_te >= _thr(pos_tr, t))
        if s < 10:
            examples.append({"rep": s // N_FOLDS, "fold": s % N_FOLDS, "n_train_pos": int(y[tr].sum()), "n_test_pos": int(y[te].sum()),
                             "p_train": ints(ptr, 1000), "y_train": y[tr].tolist(), "p_test": ints(pte, 1000), "y_test": y[te].tolist()})
    j90 = targets.index(0.9)
    log(f"peeking at 90%: peek {peek[:, j90].mean():.4f}, fair {fair[:, j90].mean():.4f} over {len(peek)} splits")
    return {"n": int(len(y)), "n_pos": int(y.sum()), "repeats": C.peek_repeats, "folds": N_FOLDS, "targets": targets,
            "peek": rnd(peek.mean(axis=0)), "fair": rnd(fair.mean(axis=0)),
            "per_split": {f"{t:.2f}": {"peek": rnd(peek[:, targets.index(t)]), "fair": rnd(fair[:, targets.index(t)])} for t in keep},
            "examples": examples}


def row_vs_patient():
    """Slide 32: random rows versus keeping each patient's encounters together (readmission, every encounter)."""
    e = D.readmission_encounters()
    X, y, g = e.X.to_numpy(float), e.y.to_numpy(int), e.groups.to_numpy()
    _, row = _cv_scores(lambda: make_model("logistic"), X, y, "row")
    _, pat = _cv_scores(lambda: make_model("logistic"), X, y, "patient", g)
    log(f"row vs patient (logistic): {np.mean(row):.4f} vs {np.mean(pat):.4f}")
    return {"model": "logistic", "row": rnd(np.mean(row)), "patient": rnd(np.mean(pat)),
            "n_rows": int(len(y)), "n_patients": int(len(np.unique(g)))}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer bootstrap draws and peeking repeats")
    ap.add_argument("--only", nargs="*", help="subset: datasets [readmission diabetes heart] | playground | lab")
    ap.add_argument("--no-tabicl", action="store_true", help="skip TabICL and write null in its place")
    ap.add_argument("--device", default="cuda", help="device for TabICL (cuda or cpu)")
    args = ap.parse_args()
    C.quick, C.tabicl = args.quick, not args.no_tabicl
    C.bootstrap = 200 if args.quick else 1000
    C.peek_repeats = 4 if args.quick else 20
    M.set_device(args.device)
    parts = {"datasets", "playground", "lab"}
    only = set(args.only or parts)
    ids = only - parts
    if ids:
        only.add("datasets")
    t0 = time.time()
    D.download()
    OUT.mkdir(parents=True, exist_ok=True)

    def write(name, obj):
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        log(f"wrote {name}.json ({path.stat().st_size / 1e3:.0f} kB)")

    readmit = D.readmission()
    if "datasets" in only:
        balanced = {"logistic_balanced": lambda: make_model("logistic", class_weight="balanced")}
        for ds, extra, with_test in [(readmit, balanced, True), (D.diabetes(), None, True), (D.heart(), None, False)]:
            if ids and ds.id not in ids:
                continue
            write(ds.id, dataset_json(ds, extra, with_test))
    if "playground" in only:
        write("playground", playground())
    if "lab" in only:
        rj = json.loads((OUT / "readmission.json").read_text(encoding="utf-8"))
        leak = leakage_lab()
        write("lab", {
            "hook": hook(rj),
            "calibration": calibration(rj),
            "train_vs_unseen": train_vs_unseen(),
            "unfair": unfair(leak, readmit),
            "peeking": peeking(),
            "row_vs_patient": row_vs_patient(),
            "parkinsons_ref": PARKINSONS_REF,
            "leakage": leak,
        })

    pkgs = ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "lightgbm", "tabicl", "torch"]
    meta_path = OUT / "meta.json"
    prev = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    w, waist_funnel = D.waist_cohort()
    write("meta", {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "n_folds": N_FOLDS,
        "python": platform.python_version(),
        "device": args.device,
        "tabicl_ran": C.tabicl if only == parts else prev.get("tabicl_ran", C.tabicl) and C.tabicl,
        "quick": C.quick,
        "packages": {p: version(p) for p in pkgs},
        "models": MODELS,
        "model_details": MODEL_DETAILS,
        "splits": SPLITS,
        "funnels": {"readmission": readmit.cohort, "readmission_encounters": D.readmission_encounters().cohort,
                    "diabetes": D.diabetes().cohort, "waist": waist_funnel, "heart": D.heart().cohort},
        "readmission_features": len(readmit.X.columns),
        "runtime_seconds": round(time.time() - t0) if only == parts else prev.get("runtime_seconds", round(time.time() - t0)),
    })


if __name__ == "__main__":
    main()
