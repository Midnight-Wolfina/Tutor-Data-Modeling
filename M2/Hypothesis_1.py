
"""
Usage:
    python decision_tree_tutoring_experiment.py --data-dir ./district_data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

RANDOM_STATE = 42
CV_SPLITS = 5
TERM3_START = pd.Timestamp('2026-02-18')
PROGRAM_START = pd.Timestamp('2025-08-18')
MIN_COURSES = 6
LOW_GPA_CUTOFF = 2.0

FLAG_FEATURES = ['iep_yes', 'bus_yes']
NUMERIC_CONTROLS = [
    'prior_year_gpa', 'prior_year_gpa_missing', 'term2_gpa',
    'term2_absences', 'tutoring_sessions_before_t3',
    'tutored_math', 'tutored_ela', 'tutored_science',
]
CATEGORICAL_CONTROLS = [
    'school_code', 'grade_level', 'econ_disadvantaged', 'english_learner'
]
FAMILIES = {
    'flags_only': {'numeric': FLAG_FEATURES, 'categorical': []},
    'controls_only': {'numeric': NUMERIC_CONTROLS, 'categorical': CATEGORICAL_CONTROLS},
    'controls_plus_flags': {
        'numeric': NUMERIC_CONTROLS + FLAG_FEATURES,
        'categorical': CATEGORICAL_CONTROLS,
    },
}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, source: str, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f'{source}: missing required column(s) {missing}')


def as_category(s: pd.Series) -> pd.Series:
    cleaned = s.astype('string').str.strip().fillna('Unknown')
    return cleaned.mask(cleaned.eq(''), 'Unknown').astype(str)


def load_cohort(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    paths = {n: data_dir / n for n in [
        'students.csv', 'term_grades.csv', 'tutoring_sessions.csv.gz'
    ]}
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            'Full district CSV files were NOT provided. Place these files in --data-dir: '
            + ', '.join(missing_files) + '. The data dictionary is not itself the data.'
        )

    students = pd.read_csv(
        paths['students.csv'], dtype={'student_id': 'string', 'school_code': 'string'}
    )
    grades = pd.read_csv(
        paths['term_grades.csv'],
        dtype={'student_id': 'string', 'course_code': 'string', 'term_grade': 'string'},
    )
    sessions = pd.read_csv(
        paths['tutoring_sessions.csv.gz'],
        compression='gzip', dtype={'student_id': 'string', 'session_id': 'string'},
    )
    require_columns(students, 'students.csv', [
        'student_id', 'school_code', 'grade_level', 'econ_disadvantaged',
        'english_learner', 'iep', 'bus_rider', 'prior_year_gpa'
    ])
    require_columns(grades, 'term_grades.csv', [
        'student_id', 'course_code', 'term', 'term_grade', 'absences'
    ])
    require_columns(sessions, 'tutoring_sessions.csv.gz', [
        'session_id', 'student_id', 'session_date', 'minutes', 'attendance', 'subject_area'
    ])

    if students.student_id.isna().any() or students.student_id.duplicated().any():
        raise ValueError('students.csv contains missing or duplicated student_id values.')
    original_grade_rows, original_session_rows = len(grades), len(sessions)
    grades = grades.drop_duplicates().copy()
    sessions = sessions.drop_duplicates().copy()
    dropped_exact_grade_rows = original_grade_rows - len(grades)
    dropped_exact_session_rows = original_session_rows - len(sessions)
    for label, frame, keys in [
        ('term_grades.csv', grades, ['student_id', 'course_code', 'term']),
        ('tutoring_sessions.csv.gz', sessions, ['session_id']),
    ]:
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f'{label}: missing/conflicting key values in {keys}. Resolve before running.')

    # Gradebook changed: Term 1 is letters, while Terms 2/3 are 0–100 percentages.
    # Map percentages to points; this is a GPA proxy, not a district cumulative GPA.
    grades['term'] = pd.to_numeric(grades['term'], errors='coerce')
    grades = grades.loc[grades.term.isin([2, 3])].copy()
    percent = pd.to_numeric(grades.term_grade, errors='coerce').to_numpy(dtype=float, na_value=np.nan)
    valid_percent = np.isfinite(percent) & (percent >= 0) & (percent <= 100) & (percent % 1 == 0)
    grades['grade_points'] = np.select([
        valid_percent & (percent >= 90), valid_percent & (percent >= 80),
        valid_percent & (percent >= 70), valid_percent & (percent >= 60),
        valid_percent & (percent >= 0),
    ], [4.0, 3.0, 2.0, 1.0, 0.0], default=np.nan)
    grades['failed_course'] = np.where(
        grades.grade_points.notna(), (grades.grade_points == 0).astype(float), np.nan
    )
    grades['absences'] = pd.to_numeric(grades['absences'], errors='coerce')
    grades.loc[(grades.absences < 0) | (grades.absences % 1 != 0), 'absences'] = np.nan
    term_summary = grades.groupby(['student_id', 'term'], as_index=False).agg(
        gpa=('grade_points', 'mean'), n_valid_grades=('grade_points', 'count'),
        n_distinct_courses=('course_code', 'nunique'),
        n_valid_absences=('absences', 'count'), term_absences=('absences', 'sum'),
        any_f=('failed_course', 'max'),
    )
    term_summary = term_summary.loc[
        (term_summary.n_valid_grades == MIN_COURSES)
        & (term_summary.n_distinct_courses == MIN_COURSES)
        & (term_summary.n_valid_absences == MIN_COURSES)
    ].copy()
    term2 = term_summary.loc[term_summary.term == 2, [
        'student_id', 'gpa', 'term_absences'
    ]].rename(columns={'gpa': 'term2_gpa', 'term_absences': 'term2_absences'})
    term3 = term_summary.loc[term_summary.term == 3, [
        'student_id', 'gpa', 'any_f'
    ]].rename(columns={'gpa': 'term3_gpa', 'any_f': 'term3_any_f'})

    # Tutored means attended at least one actual session before Term 3.
    # A referral by itself or an Absent session does not qualify.
    sessions['session_date'] = pd.to_datetime(sessions.session_date, errors='coerce')
    sessions['minutes'] = pd.to_numeric(sessions.minutes, errors='coerce')
    attendance = sessions.attendance.astype('string').str.strip().str.lower()
    valid_attendance = (
        sessions.session_date.ge(PROGRAM_START)
        & sessions.session_date.lt(TERM3_START)
        & attendance.eq('present')
        & sessions.minutes.eq(45)
    ).fillna(False)
    present = sessions.loc[valid_attendance].copy()
    present['subject_area'] = present.subject_area.astype('string').str.strip()
    present['tutored_math'] = present.subject_area.eq('Mathematics').astype(int)
    present['tutored_ela'] = present.subject_area.eq('English Language Arts').astype(int)
    present['tutored_science'] = present.subject_area.eq('Science').astype(int)
    tutoring_summary = present.groupby('student_id', as_index=False).agg(
        tutoring_sessions_before_t3=('session_id', 'count'),
        tutored_math=('tutored_math', 'max'),
        tutored_ela=('tutored_ela', 'max'),
        tutored_science=('tutored_science', 'max'),
    )

    cohort_with_any_tutoring = students.merge(
        tutoring_summary, on='student_id', how='inner', validate='one_to_one'
    )
    with_t2 = cohort_with_any_tutoring.merge(
        term2, on='student_id', how='inner', validate='one_to_one'
    )
    with_both_terms = with_t2.merge(
        term3, on='student_id', how='inner', validate='one_to_one'
    )
    cohort = with_both_terms.copy()
    cohort['iep'] = cohort.iep.astype('string').str.strip().str.upper()
    cohort['bus_rider'] = cohort.bus_rider.astype('string').str.strip().str.upper()
    cohort = cohort.loc[
        cohort.iep.isin(['Y', 'N']) & cohort.bus_rider.isin(['Y', 'N'])
    ].copy()
    cohort['iep_yes'] = (cohort.iep == 'Y').astype(int)
    cohort['bus_yes'] = (cohort.bus_rider == 'Y').astype(int)
    cohort['any_flag'] = (cohort.iep_yes | cohort.bus_yes).astype(int)
    cohort['low_gpa'] = (cohort.term3_gpa < LOW_GPA_CUTOFF).astype(int)
    cohort['prior_year_gpa'] = pd.to_numeric(cohort.prior_year_gpa, errors='coerce')
    cohort.loc[~cohort.prior_year_gpa.between(0, 4), 'prior_year_gpa'] = np.nan
    cohort['prior_year_gpa_missing'] = cohort.prior_year_gpa.isna().astype(int)
    cohort['grade_level'] = as_category(cohort.grade_level)
    for column in ['school_code', 'econ_disadvantaged', 'english_learner']:
        cohort[column] = as_category(cohort[column])
    cohort = cohort.sort_values('student_id').reset_index(drop=True)
    if len(cohort) < 60 or cohort.low_gpa.nunique() < 2:
        raise ValueError('Too few eligible students or one outcome class: cannot run 5-fold binary validation.')

    # Only aggregate metadata and input hashes are persisted.
    checks = {
        'original_students': int(len(students)),
        'original_grade_rows': int(original_grade_rows),
        'exact_duplicate_grade_rows_removed': int(dropped_exact_grade_rows),
        'original_tutoring_session_rows': int(original_session_rows),
        'exact_duplicate_session_rows_removed': int(dropped_exact_session_rows),
        'valid_present_sessions_before_t3': int(len(present)),
        'students_with_present_session_before_t3': int(len(cohort_with_any_tutoring)),
        'tutored_students_with_complete_term2': int(len(with_t2)),
        'tutored_students_with_complete_term2_and_term3': int(len(with_both_terms)),
        'final_eligible_students': int(len(cohort)),
        'low_gpa_students': int(cohort.low_gpa.sum()),
        'not_low_gpa_students': int(len(cohort) - cohort.low_gpa.sum()),
        'unknown_iep_or_bus_excluded': int(len(with_both_terms) - len(cohort)),
        'missing_prior_year_gpa_imputed_in_training_only': int(cohort.prior_year_gpa_missing.sum()),
        'cohort_definition': '>=1 Present 45-minute session before 2026-02-18; 6 valid course grades each in terms 2/3; known IEP and bus assignment',
        'outcome_definition': 'Mean of six Term 3 course grade points, low if <2.0; A=4,B=3,C=2,D=1,F=0',
        'input_sha256': {name: hash_file(path) for name, path in paths.items()},
        'source_note': 'Only 2025–26 students still enrolled at year-end are included in export; IEP and bus flags are not timestamped',
    }
    return cohort, checks


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float('nan'), float('nan'))
    phat = k/n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n))/denom
    half = z * np.sqrt((phat*(1-phat)/n) + (z*z/(4*n*n))) / denom
    return float(center-half), float(center+half)


def group_rates(cohort: pd.DataFrame) -> pd.DataFrame:
    masks = [
        ('Neither IEP nor bus', (cohort.iep_yes == 0) & (cohort.bus_yes == 0)),
        ('IEP only', (cohort.iep_yes == 1) & (cohort.bus_yes == 0)),
        ('Bus only', (cohort.iep_yes == 0) & (cohort.bus_yes == 1)),
        ('Both IEP and bus', (cohort.iep_yes == 1) & (cohort.bus_yes == 1)),
        ('IEP or bus (combined)', cohort.any_flag == 1),
    ]
    rows = []
    for name, mask in masks:
        group = cohort.loc[mask]
        n, low_count = len(group), int(group.low_gpa.sum())
        lo, hi = wilson_ci(low_count, n)
        rows.append({
            'group': name, 'n_students': n, 'n_low_gpa': low_count,
            'low_gpa_rate': low_count/n if n else float('nan'),
            'low_gpa_95ci_low': lo, 'low_gpa_95ci_high': hi,
            'mean_term3_gpa': float(group.term3_gpa.mean()) if n else float('nan'),
            'at_least_one_F_rate': float(group.term3_any_f.mean()) if n else float('nan'),
            'avg_tutoring_sessions_before_t3': float(group.tutoring_sessions_before_t3.mean()) if n else float('nan'),
        })
    return pd.DataFrame(rows)


def bootstrap_rate_difference(cohort: pd.DataFrame, reps: int = 2000) -> dict:
    a = cohort.loc[cohort.any_flag == 1, 'low_gpa'].to_numpy()
    b = cohort.loc[cohort.any_flag == 0, 'low_gpa'].to_numpy()
    if not len(a) or not len(b):
        return {'risk_difference': None, '95ci_low': None, '95ci_high': None}
    rng = np.random.default_rng(RANDOM_STATE)
    values = np.empty(reps)
    for i in range(reps):
        values[i] = a[rng.integers(0, len(a), size=len(a))].mean() - b[rng.integers(0, len(b), size=len(b))].mean()
    return {
        'risk_difference': float(a.mean()-b.mean()),
        '95ci_low': float(np.percentile(values, 2.5)),
        '95ci_high': float(np.percentile(values, 97.5)),
        'bootstrap_repetitions': reps,
        'interpretation': 'Unadjusted group difference; not a causal effect',
    }


def build_tree(family: str, max_depth: int, min_samples_leaf: int) -> Pipeline:
    config = FAMILIES[family]
    transformers = []
    if config['numeric']:
        transformers.append(('num', SimpleImputer(strategy='median'), config['numeric']))
    if config['categorical']:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), config['categorical']))
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    tree = DecisionTreeClassifier(
        criterion='gini', max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_STATE,
    )
    return Pipeline([('preprocess', preprocessor), ('tree', tree)])


def binary_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    y_pred = (prob >= 0.5).astype(int)
    if np.unique(y_true).size != 2:
        raise ValueError('A fold/test set has only one class; ROC AUC is undefined. Inspect cohort/split.')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        'roc_auc': float(roc_auc_score(y_true, prob)),
        'average_precision': float(average_precision_score(y_true, prob)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def bootstrap_ci(y: np.ndarray, prob: np.ndarray, metric='roc_auc', reps=1000, seed=42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    observed = []
    for _ in range(reps):
        ind = rng.integers(0, len(y), size=len(y))
        yy, pp = y[ind], prob[ind]
        if np.unique(yy).size < 2:
            continue
        if metric == 'roc_auc':
            observed.append(roc_auc_score(yy, pp))
        elif metric == 'f1':
            observed.append(f1_score(yy, (pp>=0.5).astype(int), zero_division=0))
        elif metric == 'recall':
            observed.append(recall_score(yy, (pp>=0.5).astype(int), zero_division=0))
        else:
            raise ValueError('Unsupported bootstrap metric')
    if len(observed) < 100:
        raise ValueError('Not enough two-class resamples to compute confidence intervals.')
    return tuple(map(float, np.percentile(observed, [2.5, 97.5])))


def paired_auc_delta_ci(y: np.ndarray, new_p: np.ndarray, base_p: np.ndarray, reps=1000) -> dict:
    point = float(roc_auc_score(y, new_p) - roc_auc_score(y, base_p))
    rng = np.random.default_rng(RANDOM_STATE+10)
    differences = []
    for _ in range(reps):
        ind = rng.integers(0, len(y), size=len(y))
        if np.unique(y[ind]).size == 2:
            differences.append(roc_auc_score(y[ind], new_p[ind]) - roc_auc_score(y[ind], base_p[ind]))
    if len(differences) < 100:
        raise ValueError('Insufficient paired bootstrap resamples')
    low, high = map(float, np.percentile(differences, [2.5, 97.5]))
    return {'test_auc_improvement_full_minus_control': point,
            'auc_difference_95ci_low': low, 'auc_difference_95ci_high': high,
            'bootstrap_repetitions': reps}


def save_group_chart(group_df: pd.DataFrame, destination: Path) -> None:
    g = group_df.iloc[:4].copy()
    values = g.low_gpa_rate.to_numpy(dtype=float) * 100
    lower = (g.low_gpa_rate - g.low_gpa_95ci_low).to_numpy(dtype=float) * 100
    upper = (g.low_gpa_95ci_high - g.low_gpa_rate).to_numpy(dtype=float) * 100
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(g))
    ax.bar(x, values, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
    if np.isfinite(np.r_[lower, upper]).all():
        ax.errorbar(x, values, yerr=[lower, upper], fmt='none', ecolor='black', capsize=5)
    for xi, yi, n in zip(x, values, g.n_students):
        if np.isfinite(yi):
            ax.text(xi, yi+2, f'{yi:.1f}%  (n={int(n):,})', ha='center', fontsize=9)
    ax.set_xticks(x, ['Neither', 'IEP only', 'Bus only', 'Both'])
    ax.set_ylabel('Students with Term 3 GPA < 2.0 (%)')
    ax.set_ylim(0, max(105, float(np.nanmax(values)+15)))
    ax.set_title('Low-GPA rates among tutored students (Wilson 95% CIs)')
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def save_tree_figure(model: Pipeline, destination: Path) -> None:
    names = model.named_steps['preprocess'].get_feature_names_out()
    tree = model.named_steps['tree']
    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(tree, feature_names=names, class_names=['GPA >= 2.0', 'GPA < 2.0'],
              filled=True, rounded=True, max_depth=3, fontsize=8, ax=ax)
    ax.set_title('Trained decision tree (top 4 levels; deeper nodes truncated)')
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def save_importances(model: Pipeline, destination: Path) -> None:
    names = model.named_steps['preprocess'].get_feature_names_out()
    values = model.named_steps['tree'].feature_importances_
    pd.DataFrame({'encoded_feature': names, 'importance': values}).sort_values(
        'importance', ascending=False
    ).to_csv(destination, index=False)


def fmt(value, digits=3) -> str:
    return 'N/A' if value is None or pd.isna(value) else f'{value:.{digits}f}'


def create_report(
    path: Path, checks: dict, groups: pd.DataFrame, contrast: dict,
    candidates: pd.DataFrame, selected: dict, metrics: dict, gain: dict,
) -> None:
    group_lines = [
        '| Group | n | Low GPA n | Low GPA % | 95% CI % | Mean Term 3 GPA | ≥1 F % |',
        '|---|---:|---:|---:|---|---:|---:|',
    ]
    for row in groups.itertuples(index=False):
        group_lines.append(
            f'| {row.group} | {row.n_students:,} | {row.n_low_gpa:,} | '
            f'{fmt(row.low_gpa_rate*100,1)} | '
            f'{fmt(row.low_gpa_95ci_low*100,1)}–{fmt(row.low_gpa_95ci_high*100,1)} | '
            f'{fmt(row.mean_term3_gpa,2)} | {fmt(row.at_least_one_F_rate*100,1)} |'
        )
    candidate_lines = [
        '| Feature family | Tree depth | Min. leaf | CV ROC AUC mean | CV ROC AUC SD | CV F1 mean |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for row in candidates.sort_values(['family','cv_roc_auc_mean'], ascending=[True,False]).itertuples(index=False):
        candidate_lines.append(
            f'| {row.family} | {row.max_depth} | {row.min_samples_leaf} | '
            f'{fmt(row.cv_roc_auc_mean)} | {fmt(row.cv_roc_auc_sd)} | {fmt(row.cv_f1_mean)} |'
        )
    group_primary = groups.loc[groups.group.eq('IEP or bus (combined)')].iloc[0]
    group_neither = groups.loc[groups.group.eq('Neither IEP nor bus')].iloc[0]
    rd = contrast['risk_difference']
    if rd is None:
        assoc = 'Not evaluable because at least one comparison group is empty.'
    elif contrast['95ci_low'] > 0:
        assoc = 'The unadjusted combined group had a higher rate, with a 95% bootstrap CI above zero.'
    elif contrast['95ci_high'] < 0:
        assoc = 'The unadjusted combined group had a lower rate, with a 95% bootstrap CI below zero.'
    else:
        assoc = 'The 95% bootstrap CI includes zero; the unadjusted rate difference is not conclusive.'
    likely = (
        'Over 50% of the combined group had a low GPA.'
        if pd.notna(group_primary.low_gpa_rate) and group_primary.low_gpa_rate > 0.5 else
        'The observed combined-group low-GPA rate did not exceed 50%.'
    )
    if gain['auc_difference_95ci_low'] > 0:
        gain_statement = 'Adding IEP and bus improved holdout discrimination; the paired bootstrap interval is above zero.'
    elif gain['auc_difference_95ci_high'] < 0:
        gain_statement = 'Adding IEP and bus reduced holdout discrimination; the paired bootstrap interval is below zero.'
    else:
        gain_statement = 'The improvement in holdout discrimination is uncertain (paired 95% CI includes zero).'
    text = f'''# R6–R8 Results: IEP/bus and low GPA among tutored students

**Study type:** Observational, retrospective prediction/association study; not a randomized experiment or causal test.

**Hypothesis:** “If the tutored students have a disability or ride the bus, then they are likely to have a low GPA.”

**Operational interpretation:** The district's `iep=Y` is used as the available IEP/special-education indicator, not a comprehensive disability measure. `bus_rider=Y` means *assigned a bus route*, not confirmed daily bus usage. “Low GPA” means an estimated term-3 GPA below 2.0, using six course grades converted to 4–0 points.

## R6 — Experiment (design, fair comparisons, numerical results)

**Population:** Students enrolled at year-end with ≥1 recorded Present, 45-minute tutoring session before 2026-02-18, complete valid grades for six courses in both Terms 2 and 3, and known IEP/bus status.

**Counts:** Original students = {checks['original_students']:,}; attended before Term 3 = {checks['students_with_present_session_before_t3']:,}; final eligible = **{checks['final_eligible_students']:,}**. Low-GPA outcomes = **{checks['low_gpa_students']:,}**. Unknown IEP/bus records excluded = {checks['unknown_iep_or_bus_excluded']:,}.

**Direct prediction test:** Compare four mutually exclusive student groups and combined IEP-or-bus versus neither. All groups share the same eligibility and grading rules. Wilson confidence intervals describe uncertainty in percentages.

{chr(10).join(group_lines)}

![Group low-GPA rates](group_low_gpa_rates.png)

**Primary unadjusted contrast:** Combined IEP-or-bus group low-GPA rate = {fmt(group_primary.low_gpa_rate*100,2)}%; neither group rate = {fmt(group_neither.low_gpa_rate*100,2)}%; difference = {fmt(rd*100 if rd is not None else None,2)} percentage points; 95% bootstrap CI = [{fmt(contrast['95ci_low']*100 if contrast['95ci_low'] is not None else None,2)}, {fmt(contrast['95ci_high']*100 if contrast['95ci_high'] is not None else None,2)}] percentage points.

**Observed conclusion:** {assoc} {likely} Because this is unadjusted, differences may reflect prior academic performance, economic disadvantage, absences, school, or other hidden variables.

**Fairness controls:** The prediction comparison tests whether IEP/bus variables improve performance beyond controls: prior-year GPA, Term 2 GPA, Term 2 absences, economic disadvantage, English learner status, school, grade level, and tutoring attendance/subjects observed before Term 3. All models use identical student partitions. Term 2 performance may already be influenced by early tutoring, so conditioning on it may conceal some relationships.

## R7 — Model building (suitable models, consistent comparisons, MLflow)

Only **DecisionTreeClassifier** is used, in three prespecified feature families:

1. `flags_only`: IEP and bus assignment alone (simple direct hypothesis model).
2. `controls_only`: previous achievement and background/context without IEP/bus (baseline).
3. `controls_plus_flags`: same control variables **plus** IEP and bus (main hypothesis model).

Each is fit at depths 3, 5, and 8, with minimum leaf sizes 25 and 75. Fixed Gini splitting and `random_state=42` keep the comparison consistent. Each candidate uses the same unstratified shuffled five folds within an 80% training partition. The prespecified selection metric is **mean cross-validation ROC AUC**. Preprocessing is fit independently inside each training fold to avoid leakage.

{chr(10).join(candidate_lines)}

**Chosen configurations (CV-only):**

- Flags only: depth {selected['flags_only']['max_depth']}, leaf {selected['flags_only']['min_samples_leaf']}, CV ROC AUC {fmt(selected['flags_only']['cv_roc_auc_mean'])}.
- Controls only: depth {selected['controls_only']['max_depth']}, leaf {selected['controls_only']['min_samples_leaf']}, CV ROC AUC {fmt(selected['controls_only']['cv_roc_auc_mean'])}.
- Controls + flags: depth {selected['controls_plus_flags']['max_depth']}, leaf {selected['controls_plus_flags']['min_samples_leaf']}, CV ROC AUC {fmt(selected['controls_plus_flags']['cv_roc_auc_mean'])}.

The **main model** is the CV-selected `controls_plus_flags` tree; the `controls_only` tree is its comparison. MLflow records every candidate and each of its five training/validation folds, final model runs, parameters, aggregate metrics, code, dependency versions, file hashes, and artifacts. Code and dependencies are also packaged alongside this report for reproducibility.

## R8 — Validation (unseen students, uncertainty, stability)

A seeded **non-stratified 80/20 student-level split** sets aside the holdout. Model configurations are chosen without looking at its outcomes. Five-fold validation uses **`KFold(n_splits=5, shuffle=True, random_state=42)`**, not `StratifiedKFold`. No student occurs in both a training and validation/test partition. The following are computed on the untouched holdout once configurations are fixed.

| Model | Test ROC AUC | Test accuracy | Test precision | Test recall | Test F1 | Test balanced accuracy | Train ROC AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    for name, info in metrics.items():
        t, tr = info['test'], info['train']
        text += (f"| {name} | {fmt(t['roc_auc'])} | {fmt(t['accuracy'])} | {fmt(t['precision'])} | "
                 f"{fmt(t['recall'])} | {fmt(t['f1'])} | {fmt(t['balanced_accuracy'])} | {fmt(tr['roc_auc'])} |\n")
    full_test = metrics['controls_plus_flags']
    ci = full_test['bootstrap_ci']
    text += f'''
**Main full model, test ROC AUC:** {fmt(full_test['test']['roc_auc'])} (bootstrap 95% CI {fmt(ci['roc_auc'][0])}–{fmt(ci['roc_auc'][1])}). **Test F1:** {fmt(full_test['test']['f1'])} (bootstrap 95% CI {fmt(ci['f1'][0])}–{fmt(ci['f1'][1])}). **Test recall:** {fmt(full_test['test']['recall'])} (bootstrap 95% CI {fmt(ci['recall'][0])}–{fmt(ci['recall'][1])}).

**Full versus controls-only test ROC AUC difference:** {fmt(gain['test_auc_improvement_full_minus_control'])}; paired bootstrap 95% CI {fmt(gain['auc_difference_95ci_low'])}–{fmt(gain['auc_difference_95ci_high'])}. {gain_statement}

**Main model confusion matrix (test, positive = low GPA):** TP={full_test['test']['tp']}, FP={full_test['test']['fp']}, FN={full_test['test']['fn']}, TN={full_test['test']['tn']}. Prediction threshold = 0.50. Check train versus CV versus test differences for overfitting. CV standard deviations in the R7 table provide another stability estimate.

![Final tree overview](decision_tree_overview.png)

## Overall conclusion and limitations

The descriptive group comparison answers whether the specified IEP/bus categories are associated with low GPA; the model comparison answers whether these features add **predictive** information after controlling for recorded context. Neither can establish that a disability, a bus assignment, or tutoring **causes** failure. The export omits students who left the district, reasons for referral, tutoring content/quality, daily attendance, and home/private support. IEP and bus flags have no recorded effective dates and may not represent status at the beginning of Term 3. Only course absences by term and program attendance are available. Term-3 GPA is an approximation rather than official cumulative GPA.

**Responsible interpretation:** Never use a predicted risk or group membership to determine a student's eligibility for educational services. These results describe associations within the observed cohort, not student potential or the efficacy of tutoring.
'''
    path.write_text(text, encoding='utf-8')


def run(data_dir: Path, output_dir: Path, bootstrap_reps: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort, checks = load_cohort(data_dir)
    groups = group_rates(cohort)
    contrast = bootstrap_rate_difference(cohort, reps=bootstrap_reps)
    groups.to_csv(output_dir / 'R6_group_rates.csv', index=False)
    save_group_chart(groups, output_dir / 'group_low_gpa_rates.png')
    (output_dir / 'R6_primary_risk_difference.json').write_text(
        json.dumps(contrast, indent=2), encoding='utf-8'
    )
    checks.update({
        'seed': RANDOM_STATE, 'cv': 'KFold (NOT stratified), shuffle=True, n_splits=5',
        'holdout': 'train_test_split(test_size=0.20, stratify=None, shuffle=True)',
        'selected_metric': 'mean CV ROC AUC',
        'n_cv_folds': CV_SPLITS,
        'packages': {
            'pandas': pd.__version__, 'numpy': np.__version__,
            'scikit-learn': sklearn.__version__, 'scipy': scipy.__version__,
            'mlflow': mlflow.__version__,
        },
    })

    # Split once on unique students, never on their multiple course or session rows.
    indices = np.arange(len(cohort))
    train_i, test_i = train_test_split(
        indices, test_size=0.20, shuffle=True, random_state=RANDOM_STATE,
        stratify=None  # user explicitly requested non-stratified validation
    )
    train = cohort.iloc[train_i].reset_index(drop=True)
    test = cohort.iloc[test_i].reset_index(drop=True)
    y_train = train.low_gpa.to_numpy(dtype=int)
    y_test = test.low_gpa.to_numpy(dtype=int)
    checks['training_n'] = int(len(train))
    checks['holdout_n'] = int(len(test))
    checks['training_low_gpa_n'] = int(y_train.sum())
    checks['holdout_low_gpa_n'] = int(y_test.sum())
    if np.unique(y_train).size != 2 or np.unique(y_test).size != 2:
        raise ValueError('The non-stratified holdout contains only one class; ROC AUC cannot be computed.')
    folds = list(KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(train))
    for j, (tr_i, va_i) in enumerate(folds, start=1):
        if np.unique(y_train[tr_i]).size < 2 or np.unique(y_train[va_i]).size < 2:
            raise ValueError(f'Unstratified fold {j} has a single class; ROC AUC undefined.')
    mlflow.set_tracking_uri("http://10.0.100.20:5000")
    mlflow.set_experiment('R6-R8_Decision_Tree_IEP_Bus_Low_GPA')
    candidate_rows, fold_rows = [], []
    source_script = Path(__file__).resolve()
    shutil.copy2(source_script, output_dir / source_script.name) if source_script != (output_dir / source_script.name).resolve() else None
    req_path = source_script.with_name('requirements_decision_tree.txt')
    if req_path.exists():
        shutil.copy2(req_path, output_dir / req_path.name) if req_path.resolve() != (output_dir / req_path.name).resolve() else None
    checks['script_sha256'] = hash_file(source_script)
    (output_dir / 'methods_and_data_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    pinned_lines = [f'{name}=={version}' for name, version in checks['packages'].items()]
    (output_dir / 'runtime_versions.txt').write_text('\n'.join(pinned_lines) + '\n', encoding='utf-8')

    for family in FAMILIES:
        for depth in (3, 5, 8):
            for leaf in (25, 75):
                label = f'{family}_depth{depth}_leaf{leaf}'
                cv_stats = []
                with mlflow.start_run(run_name=label):
                    mlflow.log_params({
                        'model': 'DecisionTreeClassifier', 'feature_family': family,
                        'max_depth': depth, 'min_samples_leaf': leaf,
                        'criterion': 'gini', 'random_state': RANDOM_STATE,
                        'cv_type': 'KFold_non_stratified', 'cv_n_splits': CV_SPLITS,
                        'cv_shuffle': True, 'data_script_sha256': checks['script_sha256'],
                        'low_gpa_threshold': LOW_GPA_CUTOFF,
                    })
                    for fold_no, (tr_i, va_i) in enumerate(folds, start=1):
                        with mlflow.start_run(run_name=f'{label}_fold{fold_no}', nested=True):
                            tree = build_tree(family, depth, leaf)
                            train_fold = train.iloc[tr_i]
                            valid_fold = train.iloc[va_i]
                            tree.fit(train_fold, y_train[tr_i])
                            prob_train = tree.predict_proba(train_fold)[:, 1]
                            prob_valid = tree.predict_proba(valid_fold)[:, 1]
                            performance = binary_metrics(y_train[va_i], prob_valid)
                            train_auc = float(roc_auc_score(y_train[tr_i], prob_train))
                            row = {
                                'candidate': label, 'family': family, 'fold': fold_no,
                                'n_train': int(len(tr_i)), 'n_valid': int(len(va_i)),
                                'validation_low_gpa_n': int(y_train[va_i].sum()),
                                'train_roc_auc': train_auc,
                                **{f'valid_{name}': value for name, value in performance.items()},
                            }
                            cv_stats.append(row)
                            fold_rows.append(row)
                            mlflow.log_params({'fold': fold_no, 'n_train': len(tr_i), 'n_valid': len(va_i)})
                            mlflow.log_metrics({
                                'train_roc_auc': train_auc,
                                **{f'valid_{name}': value for name, value in performance.items()}
                            })
                    aucs = [r['valid_roc_auc'] for r in cv_stats]
                    fscores = [r['valid_f1'] for r in cv_stats]
                    mean_auc = float(np.mean(aucs))
                    sd_auc = float(np.std(aucs, ddof=1))
                    candidate = {
                        'candidate': label, 'family': family, 'max_depth': depth,
                        'min_samples_leaf': leaf, 'cv_roc_auc_mean': mean_auc,
                        'cv_roc_auc_sd': sd_auc, 'cv_f1_mean': float(np.mean(fscores)),
                        'cv_f1_sd': float(np.std(fscores, ddof=1)),
                        'cv_train_roc_auc_mean': float(np.mean([r['train_roc_auc'] for r in cv_stats])),
                        'cv_train_minus_valid_auc_gap': float(np.mean([r['train_roc_auc'] for r in cv_stats]) - mean_auc),
                    }
                    candidate_rows.append(candidate)
                    mlflow.log_metrics({k: v for k,v in candidate.items() if k.startswith('cv_')})

    comparison = pd.DataFrame(candidate_rows).sort_values(['family', 'cv_roc_auc_mean'], ascending=[True, False])
    comparisons_path = output_dir / 'R7_cv_candidate_comparison.csv'
    folds_path = output_dir / 'R7_cv_fold_details.csv'
    comparison.to_csv(comparisons_path, index=False)
    pd.DataFrame(fold_rows).to_csv(folds_path, index=False)

    selected, result_metrics, probabilities = {}, {}, {}
    for family in FAMILIES:
        best = (comparison.loc[comparison.family == family]
                .sort_values(['cv_roc_auc_mean', 'max_depth', 'min_samples_leaf'],
                             ascending=[False, True, False]).iloc[0].to_dict())
        selected[family] = best
        tree = build_tree(family, int(best['max_depth']), int(best['min_samples_leaf']))
        tree.fit(train, y_train)
        train_p = tree.predict_proba(train)[:, 1]
        test_p = tree.predict_proba(test)[:, 1]
        probs = {'train': train_p, 'test': test_p}
        probabilities[family] = probs
        train_metrics = binary_metrics(y_train, train_p)
        test_metrics = binary_metrics(y_test, test_p)
        cis = {m: bootstrap_ci(y_test, test_p, metric=m, reps=bootstrap_reps,
                               seed=RANDOM_STATE+k)
               for k, m in enumerate(['roc_auc', 'f1', 'recall'])}
        result_metrics[family] = {
            'selected_depth': int(best['max_depth']),
            'selected_min_samples_leaf': int(best['min_samples_leaf']),
            'cv_roc_auc_mean': float(best['cv_roc_auc_mean']),
            'cv_roc_auc_sd': float(best['cv_roc_auc_sd']),
            'train': train_metrics, 'test': test_metrics,
            'bootstrap_ci': cis,
        }
        joblib.dump(tree, output_dir / f'R7_selected_{family}_model.joblib')
        with mlflow.start_run(run_name=f'FINAL_{family}'):
            mlflow.log_params({
                'model': 'DecisionTreeClassifier', 'feature_family': family,
                'max_depth': int(best['max_depth']),
                'min_samples_leaf': int(best['min_samples_leaf']),
                'criterion': 'gini', 'cv_type': 'KFold_non_stratified',
                'split_seed': RANDOM_STATE, 'test_size': 0.20,
                'model_selected_using': 'CV_ROC_AUC_only',
                'input_sha256_combined': hashlib.sha256(json.dumps(checks['input_sha256'], sort_keys=True).encode()).hexdigest(),
                'script_sha256': checks['script_sha256'],
            })
            mlflow.log_metrics({
                'cv_roc_auc_mean': float(best['cv_roc_auc_mean']),
                'cv_roc_auc_sd': float(best['cv_roc_auc_sd']),
                **{f'train_{k}':v for k,v in train_metrics.items()},
                **{f'test_{k}':v for k,v in test_metrics.items()},
                **{f'ci_{m}_{side}': values[i] for m,values in cis.items()
                   for i,side in enumerate(['low','high'])},
            })
            mlflow.log_artifact(str(output_dir / f'R7_selected_{family}_model.joblib'))
            mlflow.log_artifact(str(output_dir / 'methods_and_data_checks.json'))
            mlflow.log_artifact(str(output_dir / 'runtime_versions.txt'))
            mlflow.log_artifact(str(output_dir / source_script.name))
            if (output_dir / 'requirements_decision_tree.txt').is_file():
                mlflow.log_artifact(str(output_dir / 'requirements_decision_tree.txt'))

        if family == 'controls_plus_flags':
            save_tree_figure(tree, output_dir / 'decision_tree_overview.png')
            save_importances(tree, output_dir / 'R7_full_tree_feature_importances.csv')

    gain = paired_auc_delta_ci(y_test, probabilities['controls_plus_flags']['test'],
                               probabilities['controls_only']['test'], reps=bootstrap_reps)
    (output_dir / 'R8_holdout_test_results.json').write_text(
        json.dumps({'models': result_metrics, 'full_minus_controls': gain}, indent=2),
        encoding='utf-8'
    )
    create_report(output_dir / 'R6_R7_R8_RESULTS.md', checks, groups,
                  contrast, comparison, selected, result_metrics, gain)
    with mlflow.start_run(run_name='R6_R7_R8_aggregate_report'):
        mlflow.log_params({'project': 'IEP/bus vs low GPA among tutored',
                           'model_family': 'decision_trees_only',
                           'cv_method': 'unstratified_shuffled_KFold_5'})
        mlflow.log_metrics({'final_eligible_students': checks['final_eligible_students'],
                            'low_gpa_students': checks['low_gpa_students'],
                            'primary_unadjusted_risk_difference': contrast['risk_difference']
                            if contrast['risk_difference'] is not None else float('nan'),
                            **{key: val for key,val in gain.items()
                               if key.startswith('test_') or key.startswith('auc_')}})
        for filename in [
            'R6_group_rates.csv', 'R6_primary_risk_difference.json',
            'R7_cv_candidate_comparison.csv', 'R7_cv_fold_details.csv',
            'R8_holdout_test_results.json', 'R6_R7_R8_RESULTS.md',
            'group_low_gpa_rates.png', 'decision_tree_overview.png',
        ]:
            mlflow.log_artifact(str(output_dir / filename))
    print(f'Completed. Eligible students: {len(cohort):,}. Final artifacts in {output_dir.resolve()}')
    print(f'Full-tree holdout ROC AUC: {result_metrics["controls_plus_flags"]["test"]["roc_auc"]:.3f}')
    print(f'Full-tree holdout recall:  {result_metrics["controls_plus_flags"]["test"]["recall"]:.3f}')
    print(f'5-fold CV: ordinary non-stratified KFold. MLflow: {(output_dir / "mlruns").resolve()}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True, help='Folder with full original district CSV files')
    parser.add_argument('--output-dir', type=Path, default=Path('decision_tree_results'), help='Folder for aggregate results and MLflow history')
    parser.add_argument('--bootstrap-reps', type=int, default=1000, help='Bootstrap resamples (>=200 recommended)')
    args = parser.parse_args()
    if args.bootstrap_reps < 200:
        parser.error('--bootstrap-reps must be at least 200 for stable intervals')
    run(args.data_dir, args.output_dir, args.bootstrap_reps)


if __name__ == '__main__':
    main()
