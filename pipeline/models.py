"""The model ladder, the splits and the resampling helpers, defined once and shared by every analysis."""

from __future__ import annotations

import warnings

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", message=".*X has feature names.*")

SEED = 0
N_FOLDS = 5
DEVICE = "cuda"

MODELS = {
    "baseline": "Baseline (prevalence)",
    "logistic": "Logistic regression",
    "tree": "Decision tree",
    "rf": "Random forest",
    "gbm": "Gradient boosting",
    "tabicl": "Foundation model (TabICL)",
}

MODEL_DETAILS = {
    "baseline": "DummyClassifier(strategy='prior'): everyone gets the training prevalence",
    "logistic": "make_pipeline(SimpleImputer(median), StandardScaler(), LogisticRegression(max_iter=5000)): L2-penalised, C=1",
    "tree": "make_pipeline(SimpleImputer(median), DecisionTreeClassifier(max_depth=4, random_state=0))",
    "rf": "make_pipeline(SimpleImputer(median), RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0)). "
          "Note min_samples_leaf=3: last week's regression forest used the default of 1",
    "gbm": "LGBMClassifier(random_state=0): defaults (100 trees, learning rate 0.1, 31 leaves), missing values handled natively",
    "tabicl": "TabICLClassifier(random_state=0, device='cuda'): pretrained tabular foundation model, Apache-2.0, no tuning",
}

SPLITS = {
    "kfold": "StratifiedKFold(5, shuffle=True, random_state=0): same folds for every model",
    "group": "StratifiedGroupKFold(5, shuffle=True, random_state=0): each patient's encounters stay in one fold",
    "dev_test": "First fold of StratifiedGroupKFold(5, shuffle=True, random_state=0) is the locked 20% test set",
    "repeated": "RepeatedStratifiedKFold(5 folds × 20 repeats, random_state=0)",
}


def set_device(device: str) -> None:
    global DEVICE
    DEVICE = device


def _impute(est, impute: bool):
    return make_pipeline(SimpleImputer(strategy="median"), est) if impute else est


def make_model(name: str, seed: int = SEED, impute: bool = True, **kw):
    if name == "baseline":
        return DummyClassifier(strategy="prior")
    if name == "logistic":
        lr = LogisticRegression(max_iter=5000, **kw)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), lr) if impute else make_pipeline(StandardScaler(), lr)
    if name == "tree":
        return _impute(DecisionTreeClassifier(max_depth=kw.pop("max_depth", 4), random_state=seed, **kw), impute)
    if name == "rf":
        kw = {"n_estimators": 300, "min_samples_leaf": 3} | kw
        return _impute(RandomForestClassifier(n_jobs=-1, random_state=seed, **kw), impute)
    if name == "gbm":
        return LGBMClassifier(random_state=seed, verbose=-1, n_jobs=8, **kw)
    if name == "tabicl":
        from tabicl import TabICLClassifier

        return TabICLClassifier(random_state=seed, device=DEVICE, verbose=False, **kw)
    raise ValueError(name)


def folds(split: str, y, groups=None, seed: int = SEED):
    """(train_idx, test_idx) pairs: StratifiedKFold, or StratifiedGroupKFold when patients repeat."""
    y = np.asarray(y)
    idx = np.zeros(len(y))
    if split in ("kfold", "row"):
        return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(idx, y))
    if split in ("group", "patient"):
        return list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(idx, y, groups=groups))
    raise ValueError(split)


def dev_test(y, groups=None, seed: int = SEED):
    """An 80/20 development/test split that is stratified and never puts one patient on both sides."""
    y = np.asarray(y)
    g = np.arange(len(y)) if groups is None else np.asarray(groups)
    dev, test = next(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups=g))
    return dev, test


def smote_inside(est, seed: int = SEED):
    """Impute and oversample inside each training fold only (imbalanced-learn pipeline)."""
    return ImbPipeline([("impute", SimpleImputer(strategy="median")), ("smote", SMOTE(random_state=seed)), ("model", est)])


def smote_before(X, y, seed: int = SEED):
    """The leaky version: impute and oversample the whole dataset before any split."""
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    return SMOTE(random_state=seed).fit_resample(Xi, np.asarray(y))
