# R6–R8 Results: IEP/bus and low GPA among tutored students

**Study type:** Observational, retrospective prediction/association study; not a randomized experiment or causal test.

**Hypothesis:** “If the tutored students have a disability or ride the bus, then they are likely to have a low GPA.”

**Operational interpretation:** The district's `iep=Y` is used as the available IEP/special-education indicator, not a comprehensive disability measure. `bus_rider=Y` means *assigned a bus route*, not confirmed daily bus usage. “Low GPA” means an estimated term-3 GPA below 2.0, using six course grades converted to 4–0 points.

## R6 — Experiment (design, fair comparisons, numerical results)

**Population:** Students enrolled at year-end with ≥1 recorded Present, 45-minute tutoring session before 2026-02-18, complete valid grades for six courses in both Terms 2 and 3, and known IEP/bus status.

**Counts:** Original students = 40,000; attended before Term 3 = 10,416; final eligible = **10,416**. Low-GPA outcomes = **6,032**. Unknown IEP/bus records excluded = 0.

**Direct prediction test:** Compare four mutually exclusive student groups and combined IEP-or-bus versus neither. All groups share the same eligibility and grading rules. Wilson confidence intervals describe uncertainty in percentages.

| Group | n | Low GPA n | Low GPA % | 95% CI % | Mean Term 3 GPA | ≥1 F % |
|---|---:|---:|---:|---|---:|---:|
| Neither IEP nor bus | 3,447 | 1,816 | 52.7 | 51.0–54.3 | 1.87 | 46.1 |
| IEP only | 652 | 443 | 67.9 | 64.3–71.4 | 1.46 | 62.0 |
| Bus only | 5,292 | 3,034 | 57.3 | 56.0–58.7 | 1.76 | 49.5 |
| Both IEP and bus | 1,025 | 739 | 72.1 | 69.3–74.8 | 1.37 | 64.2 |
| IEP or bus (combined) | 6,969 | 4,216 | 60.5 | 59.3–61.6 | 1.67 | 52.8 |

![Group low-GPA rates](group_low_gpa_rates.png)

**Primary unadjusted contrast:** Combined IEP-or-bus group low-GPA rate = 60.50%; neither group rate = 52.68%; difference = 7.81 percentage points; 95% bootstrap CI = [5.86, 9.74] percentage points.

**Observed conclusion:** The unadjusted combined group had a higher rate, with a 95% bootstrap CI above zero. Over 50% of the combined group had a low GPA. Because this is unadjusted, differences may reflect prior academic performance, economic disadvantage, absences, school, or other hidden variables.

**Fairness controls:** The prediction comparison tests whether IEP/bus variables improve performance beyond controls: prior-year GPA, Term 2 GPA, Term 2 absences, economic disadvantage, English learner status, school, grade level, and tutoring attendance/subjects observed before Term 3. All models use identical student partitions. Term 2 performance may already be influenced by early tutoring, so conditioning on it may conceal some relationships.

## R7 — Model building (suitable models, consistent comparisons, MLflow)

Only **DecisionTreeClassifier** is used, in three prespecified feature families:

1. `flags_only`: IEP and bus assignment alone (simple direct hypothesis model).
2. `controls_only`: previous achievement and background/context without IEP/bus (baseline).
3. `controls_plus_flags`: same control variables **plus** IEP and bus (main hypothesis model).

Each is fit at depths 3, 5, and 8, with minimum leaf sizes 25 and 75. Fixed Gini splitting and `random_state=42` keep the comparison consistent. Each candidate uses the same unstratified shuffled five folds within an 80% training partition. The prespecified selection metric is **mean cross-validation ROC AUC**. Preprocessing is fit independently inside each training fold to avoid leakage.

| Feature family | Tree depth | Min. leaf | CV ROC AUC mean | CV ROC AUC SD | CV F1 mean |
|---|---:|---:|---:|---:|---:|
| controls_only | 8 | 75 | 0.941 | 0.004 | 0.881 |
| controls_only | 5 | 75 | 0.941 | 0.005 | 0.880 |
| controls_only | 5 | 25 | 0.940 | 0.006 | 0.880 |
| controls_only | 8 | 25 | 0.936 | 0.007 | 0.881 |
| controls_only | 3 | 25 | 0.929 | 0.005 | 0.877 |
| controls_only | 3 | 75 | 0.929 | 0.005 | 0.877 |
| controls_plus_flags | 8 | 75 | 0.941 | 0.004 | 0.882 |
| controls_plus_flags | 5 | 75 | 0.941 | 0.005 | 0.880 |
| controls_plus_flags | 5 | 25 | 0.939 | 0.005 | 0.880 |
| controls_plus_flags | 8 | 25 | 0.935 | 0.007 | 0.879 |
| controls_plus_flags | 3 | 25 | 0.929 | 0.005 | 0.877 |
| controls_plus_flags | 3 | 75 | 0.929 | 0.005 | 0.877 |
| flags_only | 3 | 25 | 0.559 | 0.014 | 0.731 |
| flags_only | 3 | 75 | 0.559 | 0.014 | 0.731 |
| flags_only | 5 | 25 | 0.559 | 0.014 | 0.731 |
| flags_only | 5 | 75 | 0.559 | 0.014 | 0.731 |
| flags_only | 8 | 25 | 0.559 | 0.014 | 0.731 |
| flags_only | 8 | 75 | 0.559 | 0.014 | 0.731 |

**Chosen configurations (CV-only):**

- Flags only: depth 3, leaf 75, CV ROC AUC 0.559.
- Controls only: depth 8, leaf 75, CV ROC AUC 0.941.
- Controls + flags: depth 8, leaf 75, CV ROC AUC 0.941.

The **main model** is the CV-selected `controls_plus_flags` tree; the `controls_only` tree is its comparison. MLflow records every candidate and each of its five training/validation folds, final model runs, parameters, aggregate metrics, code, dependency versions, file hashes, and artifacts. Code and dependencies are also packaged alongside this report for reproducibility.

## R8 — Validation (unseen students, uncertainty, stability)

A seeded **non-stratified 80/20 student-level split** sets aside the holdout. Model configurations are chosen without looking at its outcomes. Five-fold validation uses **`KFold(n_splits=5, shuffle=True, random_state=42)`**, not `StratifiedKFold`. No student occurs in both a training and validation/test partition. The following are computed on the untouched holdout once configurations are fixed.

| Model | Test ROC AUC | Test accuracy | Test precision | Test recall | Test F1 | Test balanced accuracy | Train ROC AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| flags_only | 0.553 | 0.593 | 0.593 | 1.000 | 0.745 | 0.500 | 0.559 |
| controls_only | 0.938 | 0.869 | 0.879 | 0.903 | 0.891 | 0.861 | 0.950 |
| controls_plus_flags | 0.938 | 0.870 | 0.881 | 0.904 | 0.892 | 0.863 | 0.950 |

**Main full model, test ROC AUC:** 0.938 (bootstrap 95% CI 0.927–0.948). **Test F1:** 0.892 (bootstrap 95% CI 0.879–0.904). **Test recall:** 0.904 (bootstrap 95% CI 0.888–0.919).

**Full versus controls-only test ROC AUC difference:** 0.000; paired bootstrap 95% CI -0.000–0.001. The improvement in holdout discrimination is uncertain (paired 95% CI includes zero).

**Main model confusion matrix (test, positive = low GPA):** TP=1117, FP=151, FN=119, TN=697. Prediction threshold = 0.50. Check train versus CV versus test differences for overfitting. CV standard deviations in the R7 table provide another stability estimate.

![Final tree overview](decision_tree_overview.png)

## Overall conclusion and limitations

The descriptive group comparison answers whether the specified IEP/bus categories are associated with low GPA; the model comparison answers whether these features add **predictive** information after controlling for recorded context. Neither can establish that a disability, a bus assignment, or tutoring **causes** failure. The export omits students who left the district, reasons for referral, tutoring content/quality, daily attendance, and home/private support. IEP and bus flags have no recorded effective dates and may not represent status at the beginning of Term 3. Only course absences by term and program attendance are available. Term-3 GPA is an approximation rather than official cumulative GPA.

**Responsible interpretation:** Never use a predicted risk or group membership to determine a student's eligibility for educational services. These results describe associations within the observed cohort, not student potential or the efficacy of tutoring.
