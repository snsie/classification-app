import { ComponentType } from "react";
import Hook from "./Hook";
import Framing from "./Framing";
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
  lede?: string;
  /** Homework: the slides send students to this section's "Try it" prompts. */
  tryIt?: boolean;
  Component: ComponentType;
}

// § numbers are array positions (the kicker shows idx + 1). The slides cite §1, §6, §8, §13 and §16, so keep this order.
export const SECTIONS: Section[] = [
  { id: "hook", part: "Frame", title: "The accuracy trap", tryIt: true, lede: "An AI assistant reports 91% accuracy on 20,000 real discharges. First ask what predicting “no” for everyone would score.", Component: Hook },
  { id: "framing", part: "Frame", title: "From a number to a category", lede: "Last week the model predicted a number. This week it estimates a probability, and a threshold turns that into a class.", Component: Framing },
  { id: "matrix", part: "Read the matrix", title: "Four results in a grid", lede: "Compare what the model said at discharge with what happened 30 days later, patient by patient. There are only four possible results.", Component: Matrix },
  { id: "metrics", part: "Read the matrix", title: "Each metric asks about a different group", lede: "Sensitivity, specificity, precision and FPR all come from the same four counts. What changes is the group of patients each one divides by.", Component: MetricsSection },
  { id: "f1", part: "Read the matrix", title: "F1: one score for precision and recall", lede: "The harmonic mean of precision and recall, pulled toward whichever is smaller.", Component: F1 },
  { id: "threshold", part: "Thresholds & ROC", title: "Move the threshold", tryIt: true, lede: "Same model, same probabilities, no retraining. Only the cut-off moves.", Component: Threshold },
  { id: "job", part: "Thresholds & ROC", title: "Pick the threshold for the job", lede: "The best cut-off depends on how many calls the team can make, and on what a missed readmission costs compared with a wasted call.", Component: Job },
  { id: "roc", part: "Thresholds & ROC", title: "The ROC curve", tryIt: true, lede: "Every threshold at once: each one gives a (false positive rate, true positive rate) point, and the curve joins them.", Component: Roc },
  { id: "auroc", part: "Thresholds & ROC", title: "AUROC is about ranking", lede: "Pick one readmitted and one not-readmitted patient at random. AUROC is the chance the model gives the readmitted one the higher probability.", Component: Auroc },
  { id: "auroc-limits", part: "Thresholds & ROC", title: "What AUROC doesn't tell you", lede: "AUROC summarises ranking. It is not accuracy, it doesn't choose the threshold, and it can't tell you whether the probabilities are right.", Component: AurocLimits },
  { id: "classifiers", part: "Model families", title: "Last week's models, now as classifiers", lede: "Each family has a classification version that outputs a probability. Here they are on real data with two features, so you can see the probability surface.", Component: Classifiers },
  { id: "logistic", part: "Model families", title: "Logistic regression, a classifier despite its name", lede: "A weighted sum of the features, squeezed into a probability by an S-shaped curve. A threshold then turns it into a class.", Component: Logistic },
  { id: "cv", part: "Compare fairly", title: "Grade on patients it hasn't seen", lede: "Training scores flatter every model. Cross-validation is for choosing; a locked test set is for confirming, once.", Component: CrossValidation },
  { id: "leakage", part: "Compare fairly", title: "The assistant's pipeline ran. Was it fair?", lede: "“Build the best classifier and tell me its AUROC.” The code runs without a single error, and the score is wrong.", Component: Leakage },
  { id: "peeking", part: "Compare fairly", title: "Choose the threshold without peeking", lede: "Set a threshold to catch 90% of cases. If you set it while looking at the test patients, you'll promise more than you deliver.", Component: Peeking },
  { id: "compare", part: "Compare fairly", title: "No model always wins", lede: "Same five folds for every model, scored on patients they haven't seen. On real data, the models are often close.", Component: Compare },
  { id: "checklist", part: "Compare fairly", title: "Putting it together", lede: "A classifier gives you a probability. You decide how to turn it into a decision, and how to judge it.", Component: Checklist },
];

// The slides cite these sections by number; a reorder must not move them.
if (import.meta.env.DEV) {
  const fixed: [string, number][] = [["hook", 0], ["threshold", 5], ["roc", 7], ["cv", 12], ["compare", 15]];
  for (const [id, i] of fixed) {
    if (SECTIONS[i]?.id !== id) throw new Error(`Section "${id}" must be §${i + 1} (the slides cite it), found "${SECTIONS[i]?.id}"`);
  }
  if (SECTIONS.length !== 17) throw new Error(`Expected 17 sections (slide 2), found ${SECTIONS.length}`);
}
