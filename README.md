# Classification Models

An interactive walk-through of the M11 lecture *Classification Models: From a Probability to a Decision* (CAI5735).
It has 17 sections that follow the lecture deck (`M11_01_CAI5735_Classification_Intro.pptx`) slide by slide, each with toggles, sliders and draggable thresholds, and is designed to be recorded for a video.
It is the sibling of last week's [Regression Models Explorer](https://github.com/snsie/regression-app).

The slides send students to **§2** (the accuracy trap), **§6** (move the threshold), **§8** (the ROC curve), **§13** (cross-validation) and **§16** (model comparison).
The homework is the **Try it** prompts in §2, §6 and §8.

| § | Section | Slides |
|---|---|---|
| 1 | From a number to a category (objectives; regression vs classification; feature, target, class; probability → threshold → class) | 2–6 |
| 2 | Accuracy, and the accuracy trap | 7–9 |
| 3 | Positive, negative and the confusion matrix | 10–12 |
| 4 | Each metric describes a different group (sensitivity, specificity, precision, TPR, FPR) | 13–17 |
| 5 | F1, and what it leaves out | 18–19 |
| 6 | Move the threshold | 20 |
| 7 | Pick the threshold for the job | 20 |
| 8 | The ROC curve | 21 |
| 9 | AUROC is about ranking | 22 |
| 10 | What AUROC does not tell you | 23 |
| 11 | Last week's models, now as classifiers | 24 |
| 12 | Logistic regression, a classifier despite its name | 25 |
| 13 | Judge a model on patients it has not seen (training vs unseen; cross-validation vs test set) | 26–27 |
| 14 | Five rules for a fair comparison | 28–29 |
| 15 | Choose the threshold without peeking | 29 |
| 16 | On real data, the models are often close | 30 |
| 17 | Putting it together | 31 |

Each section's header shows its slide range. `sections/index.ts` holds the order, and in development it checks that the cited sections haven't moved.

Every model is trained ahead of time in Python. The React app only reads the resulting JSON, so it deploys as a static site with no server.
The hypothetical running example, **Model A** (100 patients, 10 readmitted), is built deterministically in the browser so that every slide number reproduces exactly.

```
pipeline/   Python: download public data → train models → run the verification analyses → JSON
web/        React + TypeScript (Vite); reads web/public/data/*.json
.github/workflows/deploy.yml   builds web/ and publishes it to GitHub Pages
```

## Data (all public)

| Dataset | Source | Licence | Role in the story |
|---|---|---|---|
| 30-day readmission: 20,000 discharges, one per patient (1,796 readmitted) | UCI Diabetes 130-US hospitals 1999–2008 (Strack et al. 2014) | CC BY 4.0 | The assistant's “91% accuracy”; thresholds for the job; calibration; model comparison |
| Readmission, every encounter: 20,000 encounters from 14,285 patients | Same file, whole patients with all their encounters | CC BY 4.0 | Split by row vs by patient; SMOTE leakage |
| Diabetes screening: 5,257 NHANES adults (19.0% positive) | NHANES 2017–2018, CDC/NCHS public-use files | Public domain | Leakage lab (HbA1c, diabetes medicines); model comparison |
| Waist → diabetes: 4,977 NHANES adults | Same survey, adults with a waist measurement | Public domain | Logistic regression (§12); probability maps (§11) |
| Heart disease: 303 patients (Cleveland) | UCI Heart Disease (Detrano et al. 1989) | CC BY 4.0 | Training vs unseen (§13); threshold peeking (§15); probability maps (§11) |

Readmission cohort: 101,766 encounters → 99,343 after excluding discharge to hospice or death → 69,990 first encounters per patient →
20,000 by a stratified random sample (`random_state=0`). Positive = readmitted within 30 days.
Diabetes positive = told by a doctor (DIQ010 = 1) or HbA1c ≥ 6.5%.

## Models

Baseline (prevalence), logistic regression, decision tree (depth 4), random forest (300 trees, `min_samples_leaf=3`, unlike last week's default of 1),
LightGBM (defaults), and the **TabICL** tabular foundation model (Apache-2.0, open weights). Every model is scored on the same
`StratifiedKFold(5, shuffle=True, random_state=0)` folds, with imputation and scaling inside each model's pipeline.

## Regenerate the results

You need a CUDA GPU for TabICL (or pass `--no-tabicl`, which writes `null` in its place). A full run took about **4 minutes** on one NVIDIA RTX 5070 Ti;
`pyproject.toml` installs the CUDA 12.8 build of PyTorch, which covers recent GPUs.

```bash
cd pipeline
uv sync --python 3.11               # creates .venv with every dependency
uv run python run_all.py            # downloads raw data, writes ../web/public/data/*.json
uv run python test_story.py         # checks every slide number; writes reconciliation.md
```

`run_all.py --quick` uses fewer bootstrap draws and peeking repeats; `--only datasets [readmission diabetes heart] | playground | lab` reruns one part;
`--device cpu` runs TabICL on the CPU (slow).

`test_story.py` compares each number on the slides with the regenerated one and writes [`pipeline/reconciliation.md`](pipeline/reconciliation.md):
**match**, **close**, **story holds** (the number drifted, the slide's claim didn't) or **FAIL** (exit code 1).
Drift is reported, never forced: no seed is tuned to make a number agree. The app's text reads every real-data number from the JSON,
so nothing in `web/src/sections/` needs editing after a rerun; the slides might.

## Run the app locally

```bash
cd web
npm install
npm run dev          # http://localhost:5173
npm test             # Model A, the sandbox and the Try-it answers (vitest)
npm run build        # static site in web/dist
```

In development, every dataset's out-of-fold AUROCs and 0.50 confusion matrices are recomputed in the browser and checked against the pipeline (see the console).

## Deploy to GitHub Pages

1. Push this folder to a GitHub repository, with the generated JSON committed. The lecture deck (`*.pptx`) is git-ignored.
2. Go to **Settings → Pages → Build and deployment** and set **Source** to **GitHub Actions**.
3. Push to `main`. The workflow builds `web/` and publishes it. Vite uses a relative base (`./`), so no further configuration is needed.

## Recording tips

- Use **← / →** (or PageUp/PageDown) to move between sections, and **p** to toggle presenter mode (larger text).
  When a threshold handle or slider has focus, the arrow keys move the threshold instead.
- Each section has its own URL, by name (`#/threshold`) or by the slides' number (`#/2`, `#/6`, `#/8`, `#/13`). The old `#/hook` still opens §2.
- The theme button cycles through auto, dark and light.
- “Pause and think”, quiz and “Check your answers” boxes stay closed until you click **Reveal**.
- Every interactive starts in the default state that reproduces the slide's numbers (for example §6 opens at 0.50: 26 flagged, 80% / 80%), so retakes match.
- Threshold handles: drag, or focus and use arrows (±0.01), Shift+arrows (±0.05), PageUp/PageDown (±0.10), Home/End.
