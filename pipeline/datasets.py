"""Load the public datasets used by the app.

Every dataset comes from a public, citable source:
  * UCI Diabetes 130-US hospitals 1999-2008 (Strack et al. 2014): 30-day readmission
  * NHANES 2017-2018 (CDC National Center for Health Statistics): diabetes screening, waist → diabetes
  * UCI Heart Disease, Cleveland subset (Detrano et al. 1989)

Raw files are downloaded once into ``raw/`` by ``download()``.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from models import SEED

RAW = Path(__file__).parent / "raw"

DIABETES130_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
NHANES_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{}.xpt"
NHANES_FILES = ["DEMO_J", "BMX_J", "BPX_J", "GHB_J", "DIQ_J", "HDL_J", "TCHOL_J"]
CLEVELAND_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


def download() -> None:
    RAW.mkdir(exist_ok=True)
    targets = {f"{f}.xpt": NHANES_URL.format(f) for f in NHANES_FILES}
    targets["processed.cleveland.data"] = CLEVELAND_URL
    for name, url in targets.items():
        path = RAW / name
        if path.exists():
            continue
        print(f"downloading {url}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    if not (RAW / "diabetic_data.csv").exists():
        print(f"downloading {DIABETES130_URL}")
        r = requests.get(DIABETES130_URL, timeout=300)
        r.raise_for_status()
        _extract_csv(r.content, "diabetic_data.csv", RAW / "diabetic_data.csv")


def _extract_csv(blob: bytes, member: str, dest: Path) -> None:
    """The UCI archive has held the CSV both at the top level and inside a nested zip."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for n in z.namelist():
            if n.endswith(member):
                dest.write_bytes(z.read(n))
                return
        for n in z.namelist():
            if n.endswith(".zip"):
                try:
                    _extract_csv(z.read(n), member, dest)
                    return
                except FileNotFoundError:
                    continue
    raise FileNotFoundError(member)


@dataclass
class Dataset:
    id: str
    name: str
    story: str
    source: str
    citation: str
    url: str
    labels: dict[str, str]
    X: pd.DataFrame
    y: pd.Series
    feature_labels: dict[str, str]
    cohort: list[dict] = field(default_factory=list)
    split: str = "StratifiedKFold(5, shuffle=True, random_state=0): one row per patient"
    groups: pd.Series | None = None


# ---------------------------------------------------------------- readmission (UCI Diabetes 130)
READMIT_CITE = (
    "Strack B, DeShazo JP, Gennings C, et al. Impact of HbA1c measurement on hospital readmission rates: "
    "analysis of 70,000 clinical database patient records. BioMed Research International 2014;2014:781670."
)
READMIT_LINK = "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008"
# Discharged to hospice or died: these patients cannot be readmitted (Strack et al. 2014)
EXCLUDE_DISPOSITION = [11, 13, 14, 19, 20, 21]
MIN_LEVEL = 50  # category levels with fewer rows are pooled into "other"

NUMERIC = {
    "age": "Age (decade midpoint, years)",
    "female": "Female (1 = yes)",
    "time_in_hospital": "Length of stay (days)",
    "num_lab_procedures": "Lab procedures",
    "num_procedures": "Procedures",
    "num_medications": "Medications",
    "number_outpatient": "Outpatient visits, past year",
    "number_emergency": "Emergency visits, past year",
    "number_inpatient": "Inpatient admissions, past year",
    "number_diagnoses": "Diagnoses recorded",
}
CATEGORICAL = {
    "race": "Race",
    "admission_type_id": "Admission type",
    "admission_source_id": "Admission source",
    "discharge_disposition_id": "Discharge disposition",
    "A1Cresult": "HbA1c result",
    "max_glu_serum": "Max glucose serum",
    "insulin": "Insulin",
    "change": "Medication change",
    "diabetesMed": "Diabetes medication",
    "diag1_group": "Primary diagnosis group",
}


def _icd9_group(code) -> str:
    """Strack et al.'s nine groups for the primary diagnosis."""
    if not isinstance(code, str) or not code:
        return "Other"
    if code[0] in "VE":
        return "Other"
    v = float(code)
    if 390 <= v <= 459 or int(v) == 785:
        return "Circulatory"
    if 460 <= v <= 519 or int(v) == 786:
        return "Respiratory"
    if 520 <= v <= 579 or int(v) == 787:
        return "Digestive"
    if int(v) == 250:
        return "Diabetes"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or int(v) == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


def _diabetes130() -> tuple[pd.DataFrame, list[dict]]:
    d = pd.read_csv(RAW / "diabetic_data.csv", na_values="?", keep_default_na=False, low_memory=False)
    funnel = [{"step": "All encounters in the UCI file", "n": int(len(d))}]
    d = d[~d.discharge_disposition_id.isin(EXCLUDE_DISPOSITION)]
    funnel.append({"step": "Excluding discharge to hospice or death", "n": int(len(d))})
    return d, funnel


def _readmission_features(d: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    X = pd.DataFrame(index=d.index)
    X["age"] = d.age.str.extract(r"\[(\d+)-").astype(float)[0] + 5
    X["female"] = (d.gender == "Female").astype(float)
    for c in list(NUMERIC)[2:]:
        X[c] = d[c].astype(float)
    labels = dict(NUMERIC)
    cats = d[[c for c in CATEGORICAL if c != "diag1_group"]].copy()
    cats["diag1_group"] = d.diag_1.map(_icd9_group)
    for c in cats:
        s = cats[c].astype("string").fillna("None" if c in ("A1Cresult", "max_glu_serum") else "Unknown").astype(str)
        counts = s.value_counts()
        s = s.where(s.map(counts) >= MIN_LEVEL, "other")
        if c in ("change", "diabetesMed"):
            # binary: one indicator column
            on = "Ch" if c == "change" else "Yes"
            X[c] = (s == on).astype(float)
            labels[c] = CATEGORICAL[c] + " (1 = yes)"
            continue
        for level in sorted(s.unique()):
            col = f"{c}={level}"
            X[col] = (s == level).astype(float)
            labels[col] = f"{CATEGORICAL[c]}: {level}"
    return X, labels


def readmission() -> Dataset:
    d, funnel = _diabetes130()
    d = d.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    funnel.append({"step": "First encounter per patient", "n": int(len(d))})
    y_all = (d.readmitted == "<30").astype(int).to_numpy()
    idx, _ = train_test_split(np.arange(len(d)), train_size=20000, stratify=y_all, random_state=SEED)
    d = d.iloc[idx]
    funnel.append({"step": "Stratified random sample (random_state=0)", "n": int(len(d))})
    X, labels = _readmission_features(d)
    y = (d.readmitted == "<30").astype(int)
    return Dataset(
        id="readmission",
        name="30-day readmission (20,000 discharges)",
        story="Weak signal, imbalanced",
        source="UCI Diabetes 130-US hospitals 1999–2008: one discharge per patient, readmitted within 30 days = positive",
        citation=READMIT_CITE,
        url=READMIT_LINK,
        labels={"pos": "Readmitted", "neg": "Not readmitted", "unit": "discharge", "units": "discharges"},
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        feature_labels=labels,
        cohort=funnel,
    )


def readmission_encounters(n_rows: int = 20000) -> Dataset:
    """Whole patients, with every one of their encounters, added in random order until 20,000 rows. No de-duplication."""
    d, funnel = _diabetes130()
    pats = d.patient_nbr.unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(pats)
    sizes = d.groupby("patient_nbr").size().reindex(pats).to_numpy()
    keep = pats[np.cumsum(sizes) <= n_rows]
    d = d[d.patient_nbr.isin(set(keep))].sort_values("encounter_id")
    funnel.append({"step": "Whole patients added in random order (all their encounters)", "n": int(len(d))})
    X, labels = _readmission_features(d)
    return Dataset(
        id="readmission_encounters",
        name="30-day readmission, every encounter",
        story="Repeated patients",
        source="UCI Diabetes 130-US hospitals: all encounters of randomly chosen patients",
        citation=READMIT_CITE,
        url=READMIT_LINK,
        labels={"pos": "Readmitted", "neg": "Not readmitted", "unit": "encounter", "units": "encounters"},
        X=X.reset_index(drop=True),
        y=(d.readmitted == "<30").astype(int).reset_index(drop=True),
        feature_labels=labels,
        cohort=funnel,
        groups=d.patient_nbr.reset_index(drop=True),
    )


def readmission_funnel() -> list[dict]:
    return readmission().cohort


# ---------------------------------------------------------------- NHANES diabetes screening
NHANES_CITE = (
    "Centers for Disease Control and Prevention, National Center for Health "
    "Statistics. National Health and Nutrition Examination Survey 2017-2018."
)
NHANES_LINK = "https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017"
RACE = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other / multiracial",
}
FAIR = {
    "age": "Age (years)",
    "female": "Female (1 = yes)",
    "bmi": "BMI (kg/m²)",
    "waist": "Waist (cm)",
    "sbp": "Systolic BP, mean of 3 (mmHg)",
    "income_ratio": "Income-to-poverty ratio",
}
LEAKY = {
    "hba1c": "HbA1c (%) — defines the label",
    "insulin": "Takes insulin (1 = yes)",
    "pills": "Takes diabetes pills (1 = yes)",
}


def _nhanes() -> tuple[pd.DataFrame, list[dict]]:
    def read(f: str) -> pd.DataFrame:
        return pd.read_sas(RAW / f"{f}.xpt").set_index("SEQN")

    d = (
        read("DEMO_J")[["RIAGENDR", "RIDAGEYR", "RIDRETH3", "INDFMPIR"]]
        .join(read("BMX_J")[["BMXBMI", "BMXWAIST"]])
        .join(read("BPX_J")[["BPXSY1", "BPXSY2", "BPXSY3"]])
        .join(read("GHB_J")[["LBXGH"]])
        .join(read("DIQ_J")[["DIQ010", "DIQ050", "DIQ070"]])
        .join(read("HDL_J")[["LBDHDD"]])
        .join(read("TCHOL_J")[["LBXTC"]])
    )
    funnel = [{"step": "NHANES 2017–2018 participants", "n": int(len(d))}]
    d = d[d.RIDAGEYR >= 18]
    funnel.append({"step": "Adults (18+)", "n": int(len(d))})
    d = d[d.LBXGH.notna()]
    funnel.append({"step": "HbA1c measured", "n": int(len(d))})
    d = d[d.DIQ010.isin([1, 2, 3])].copy()
    funnel.append({"step": "Answered the diabetes question (yes / no / borderline)", "n": int(len(d))})
    out = pd.DataFrame(index=d.index)
    out["age"] = d.RIDAGEYR
    out["female"] = (d.RIAGENDR == 2).astype(float)
    out["bmi"] = d.BMXBMI
    out["waist"] = d.BMXWAIST
    out["sbp"] = d[["BPXSY1", "BPXSY2", "BPXSY3"]].mean(axis=1)
    out["income_ratio"] = d.INDFMPIR
    race = d.RIDRETH3.map(RACE)
    for code, lab in RACE.items():
        if code == 3:
            continue
        out[f"race={lab}"] = (race == lab).astype(float)
    out["hdl"] = d.LBDHDD
    out["total_chol"] = d.LBXTC
    out["hba1c"] = d.LBXGH
    out["insulin"] = (d.DIQ050 == 1).astype(float)
    out["pills"] = (d.DIQ070 == 1).astype(float)
    out["y"] = ((d.DIQ010 == 1) | (d.LBXGH >= 6.5)).astype(int)
    return out, funnel


def fair_columns() -> list[str]:
    return list(FAIR) + [f"race={RACE[c]}" for c in RACE if c != 3] + ["hdl", "total_chol"]


def diabetes(leaky: bool = False) -> Dataset:
    d, funnel = _nhanes()
    cols = fair_columns() + (list(LEAKY) if leaky else [])
    labels = dict(FAIR) | {f"race={RACE[c]}": f"{RACE[c]} (vs NH White)" for c in RACE if c != 3}
    labels |= {"hdl": "HDL cholesterol (mg/dL)", "total_chol": "Total cholesterol (mg/dL)"} | LEAKY
    return Dataset(
        id="diabetes",
        name="Diabetes screening (NHANES adults)",
        story="Moderate signal",
        source="NHANES 2017–2018 (CDC): adults with HbA1c and an answer to “ever told you have diabetes?”. "
               "Positive = told by a doctor, or HbA1c ≥ 6.5%",
        citation=NHANES_CITE,
        url=NHANES_LINK,
        labels={"pos": "Has diabetes", "neg": "No diabetes", "unit": "adult", "units": "adults"},
        X=d[cols].reset_index(drop=True),
        y=d.y.reset_index(drop=True),
        feature_labels={c: labels[c] for c in cols},
        cohort=funnel,
    )


def waist_cohort() -> tuple[pd.DataFrame, list[dict]]:
    d, funnel = _nhanes()
    d = d[d.waist.notna()]
    funnel = funnel + [{"step": "Waist measured", "n": int(len(d))}]
    return d.reset_index(drop=True), funnel


# ---------------------------------------------------------------- UCI heart disease (Cleveland)
HEART_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
HEART_LABELS = {
    "age": "Age (years)",
    "sex": "Male (1 = yes)",
    "cp": "Chest pain type (1–4)",
    "trestbps": "Resting BP (mmHg)",
    "chol": "Cholesterol (mg/dL)",
    "fbs": "Fasting glucose > 120 (1 = yes)",
    "restecg": "Resting ECG (0–2)",
    "thalach": "Maximum heart rate (bpm)",
    "exang": "Exercise angina (1 = yes)",
    "oldpeak": "ST depression",
    "slope": "ST slope (1–3)",
    "ca": "Vessels coloured (0–3)",
    "thal": "Thallium test (3/6/7)",
}


def heart() -> Dataset:
    d = pd.read_csv(RAW / "processed.cleveland.data", header=None, names=HEART_COLS, na_values="?")
    X = d[HEART_COLS[:-1]].astype(float)
    return Dataset(
        id="heart",
        name="Heart disease (Cleveland, 303 patients)",
        story="Small n",
        source="UCI Heart Disease, Cleveland Clinic subset: any angiographic disease (num > 0) = positive",
        citation=(
            "Detrano R, Janosi A, Steinbrunn W, et al. International application of a new probability algorithm "
            "for the diagnosis of coronary artery disease. Am J Cardiol 1989;64(5):304-10."
        ),
        url="https://archive.ics.uci.edu/dataset/45/heart+disease",
        labels={"pos": "Heart disease", "neg": "No heart disease", "unit": "patient", "units": "patients"},
        X=X,
        y=(d.num > 0).astype(int),
        feature_labels=HEART_LABELS,
        cohort=[{"step": "Cleveland patients", "n": int(len(d))}],
    )


if __name__ == "__main__":
    download()
    r = readmission()
    print("readmission", r.X.shape, int(r.y.sum()), [f["n"] for f in r.cohort])
    e = readmission_encounters()
    print("encounters", e.X.shape, int(e.y.sum()), int(e.groups.nunique()))
    dd = diabetes()
    print("diabetes", dd.X.shape, int(dd.y.sum()), round(dd.y.mean(), 4), [f["n"] for f in dd.cohort])
    w, f = waist_cohort()
    print("waist", len(w), int(w.y.sum()))
    h = heart()
    print("heart", h.X.shape, int(h.y.sum()))
