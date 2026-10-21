import { ComponentType } from "react";
import Framing from "./Framing";
import Accuracy from "./Accuracy";
import Matrix from "./Matrix";
import MetricsSection from "./Metrics";
import F1 from "./F1";
import Threshold from "./Threshold";
import Job from "./Job";
import Roc from "./Roc";
import Auroc from "./Auroc";
import AurocLimits from "./AurocLimits";
import Classifiers from "./Classifiers";
import Logistic from "./Logistic";
import CrossValidation from "./CrossValidation";
import Leakage from "./Leakage";
import Peeking from "./Peeking";
import Compare from "./Compare";
import Checklist from "./Checklist";

export interface Section {
  id: string;
  part: string;
  title: string;
  /** The slides of M11_01_CAI5735_Classification_Intro.pptx that this section follows. */
  slides: string;
  lede?: string;
  /** Homework: the slides send students to this section's "Try it" prompts. */
  tryIt?: boolean;
  Component: ComponentType;
}

// § numbers are array positions (the kicker shows idx + 1). The sections follow the deck's order; the slides cite §2, §6, §8, §13 and §16.
export const SECTIONS: Section[] = [
  { id: "framing", part: "Regression → classification", title: "From a number to a category", slides: "2–6", lede: "Last week the model predicted a number. This week it estimates a probability, and a threshold turns that into a class.", Component: Framing },
  { id: "accuracy", part: "Accuracy and its limits", title: "Accuracy, and the accuracy trap", slides: "7–9", tryIt: true, lede: "Accuracy is the fraction of predictions that were correct. With imbalanced classes, a model that finds no one can still score 90%.", Component: Accuracy },
  { id: "matrix", part: "Read the matrix", title: "Positive, negative and the confusion matrix", slides: "10–12", lede: "Name the positive class, then compare what the model said at discharge with what happened 30 days later. There are only four possible results.", Component: Matrix },
  { id: "metrics", part: "Read the matrix", title: "Each metric describes a different group", slides: "13–17", lede: "Sensitivity, specificity, precision, TPR and FPR all come from the same four counts. What changes is the group of patients each one divides by.", Component: MetricsSection },
  { id: "f1", part: "Read the matrix", title: "F1, and what it leaves out", slides: "18–19", lede: "The harmonic mean of precision and recall, pulled toward whichever is smaller. Useful, but it isn't the whole story.", Component: F1 },
  { id: "threshold", part: "Thresholds & ROC", title: "Move the threshold", slides: "20", tryIt: true, lede: "Same model, same probabilities, no retraining. Only the cut-off moves.", Component: Threshold },
  { id: "job", part: "Thresholds & ROC", title: "Pick the threshold for the job", slides: "20", lede: "The best cut-off depends on how many calls the team can make, and on what a missed readmission costs compared with a wasted call.", Component: Job },
  { id: "roc", part: "Thresholds & ROC", title: "The ROC curve", slides: "21", tryIt: true, lede: "Every threshold at once: each one gives a (false positive rate, true positive rate) point, and the curve joins them.", Component: Roc },
  { id: "auroc", part: "Thresholds & ROC", title: "AUROC is about ranking", slides: "22", lede: "Pick one readmitted and one not-readmitted patient at random. AUROC is the chance the model gives the readmitted one the higher probability.", Component: Auroc },
  { id: "auroc-limits", part: "Thresholds & ROC", title: "What AUROC does not tell you", slides: "23", lede: "AUROC summarises ranking. It is not accuracy, it doesn't choose the threshold, and it can't tell you whether the probabilities are right.", Component: AurocLimits },
  { id: "classifiers", part: "Model families", title: "Last week's models, now as classifiers", slides: "24", lede: "Each family has a classification version that outputs a probability. Here they are on real data with two features, so you can see the probability surface.", Component: Classifiers },
  { id: "logistic", part: "Model families", title: "Logistic regression, a classifier despite its name", slides: "25", lede: "A weighted sum of the features, squeezed into a probability by an S-shaped curve. A threshold then turns it into a class.", Component: Logistic },
  { id: "cv", part: "Compare fairly", title: "Judge a model on patients it has not seen", slides: "26–27", lede: "Training scores flatter every model. Cross-validation is for choosing; a locked test set is for confirming, once.", Component: CrossValidation },
  { id: "leakage", part: "Compare fairly", title: "Five rules for a fair comparison", slides: "28–29", lede: "Without them, the “winner” may just be the model that saw more of the answers. The assistant's code below runs without a single error, and the score is wrong.", Component: Leakage },
  { id: "peeking", part: "Compare fairly", title: "Choose the threshold without peeking", slides: "29", lede: "Set a threshold to catch 90% of cases. If you set it while looking at the test patients, you'll promise more than you deliver.", Component: Peeking },
  { id: "compare", part: "Compare fairly", title: "On real data, the models are often close", slides: "30", lede: "Same five folds for every model, scored on patients they haven't seen. No model always wins.", Component: Compare },
  { id: "checklist", part: "Putting it together", title: "Putting it together", slides: "31", lede: "A classifier gives you a probability. You decide how to turn it into a decision, and how to judge it.", Component: Checklist },
];

/** Old section ids that links (and last year's recordings) may still use. */
export const ALIASES: Record<string, string> = { hook: "accuracy" };

// The slides cite these sections by number; a reorder must not move them.
if (import.meta.env.DEV) {
  const fixed: [string, number][] = [["accuracy", 1], ["threshold", 5], ["roc", 7], ["cv", 12], ["compare", 15]];
  for (const [id, i] of fixed) {
    if (SECTIONS[i]?.id !== id) throw new Error(`Section "${id}" must be §${i + 1} (the slides cite it), found "${SECTIONS[i]?.id}"`);
  }
  if (SECTIONS.length !== 17) throw new Error(`Expected 17 sections (slide 2), found ${SECTIONS.length}`);
}
