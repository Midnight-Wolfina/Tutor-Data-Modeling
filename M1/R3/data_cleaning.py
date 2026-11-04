'''
To run, the folder with every csv must be in the same folder as this file.
'''
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_FOLDER = BASE_DIR / "data"
OUTPUT_FOLDER = BASE_DIR / "cleaned_data"
REPORT_FOLDER = BASE_DIR / "cleaning_reports"

# Exact, case-insensitive text tokens; deliberately exclude "NA" and "-"
# because they can be legitimate values. Empty strings are handled separately.
MISSING_MARKERS = {"", "n/a", "null", "none", "nan"}

# Keep numeric-type checks; range, date and category validation were removed.
NUMERIC_COLUMNS = {
    "enrollment", "econ_disadvantaged_pct", "grade_level", "prior_year_gpa",
    "credits", "term", "absences", "minutes"
}
INTEGER_COLUMNS = {"enrollment", "grade_level", "term", "absences", "minutes"}

records = []


def report(filename, step, before, after, affected, reason, action):
    """Report actual outcomes, rather than labeling every attempted operation a change."""
    removed = before - after

    if removed > 0:
        status = "REMOVED"
    elif affected > 0:
        status = "FLAGGED" if action == "FLAGGED" else "MODIFIED"
    else:
        status = "NO CHANGE"

    if status == "REMOVED":
        metric = "Rows removed"
        count = removed
    elif status == "FLAGGED":
        metric = "Columns flagged" if step == "Check fully empty columns" else "Values flagged"
        count = affected
    elif status == "MODIFIED":
        metric = "Columns renamed" if step == "Trim column names" else "Values modified"
        count = affected
    else:
        metric = "Values changed/flagged"
        count = 0

    records.append({
        "file": filename,
        "step": step,
        "action": status,
        "rows_before": before,
        "rows_after": after,
        "rows_removed": removed,
        "values_affected_or_flagged": affected,
        "metric": metric,
        "reason": reason,
    })

    print(f"  {step} [{status}]")
    print(f"    Rows: {before:,} -> {after:,} | removed: {removed:,}")
    print(f"    {metric}: {count:,}")
    if status in {"MODIFIED", "REMOVED"}:
        print(f"    Reason: {reason}")


def canonical_filename(path):
    name = path.name
    if name.endswith(".csv.gz"):
        return name[:-3]  # remove '.gz' for schema lookup
    return name


def read_csv_safely(path):
    # Read all fields as text to preserve student IDs, site codes, and formatting.
    # Pandas handles .csv.gz compression by filename automatically.
    return pd.read_csv(path, dtype="string", keep_default_na=False,
                       compression="infer", encoding="utf-8-sig")


def clean_general(df, filename):
    # Step 1: normalize column labels.
    before = len(df)
    old_names = list(df.columns)
    new_names = [str(name).strip() for name in old_names]
    if len(new_names) != len(set(new_names)):
        raise ValueError("Trimming column names would create duplicate names; manual review required")
    df.columns = new_names
    report(filename, "Trim column names", before, len(df),
           sum(a != b for a, b in zip(old_names, new_names)),
           "Remove surrounding whitespace from column labels.", "MODIFIED")

    # Step 2: trim surrounding whitespace in every cell.
    before = len(df)
    stripped = df.apply(lambda series: series.str.strip())
    changed = int((df.ne(stripped).fillna(False)).to_numpy().sum())
    df = stripped
    report(filename, "Trim text values", before, len(df), changed,
           "Remove incidental leading/trailing whitespace without altering interior text.",
           "MODIFIED")

    # Step 3: normalize missing markers to pd.NA (no imputation/filling).
    before = len(df)
    missing_mask = df.apply(lambda col: col.str.lower().isin(MISSING_MARKERS))
    converted = int(missing_mask.to_numpy().sum())
    df = df.mask(missing_mask, pd.NA)
    report(filename, "Normalize missing markers", before, len(df), converted,
           "Treat configured blank/null text markers as missing; no values are imputed.",
           "MODIFIED")

    # Step 4: only remove rows where EVERY value is missing.
    before = len(df)
    df = df.dropna(how="all").copy()
    report(filename, "Remove fully empty rows", before, len(df), before - len(df),
           "Every field in the removed row was missing.", "REMOVED")

    # Step 5: all-empty columns are flagged, never removed automatically.
    empty_columns = df.columns[df.isna().all()].tolist()
    report(filename, "Check fully empty columns", len(df), len(df), len(empty_columns),
           f"Flagged for review: {empty_columns}" if empty_columns else "No empty columns.",
           "FLAGGED")

    # Step 6: exact duplicate rows. For a transactional/log dataset,
    # identical-looking records may still be legitimate.
    dup_mask = df.duplicated(keep="first")
    duplicates = int(dup_mask.sum())
    before = len(df)
    df = df.loc[~dup_mask].copy()
    action = "REMOVED"
    reason = "Rows were exact duplicates across all columns."
    report(filename, "Check exact duplicate rows", before, len(df), duplicates, reason, action)

    # Step 7: missing-value counts. Missingness may be intentional.
    counts = df.isna().sum()
    total_missing = int(counts.sum())
    detail = ", ".join(f"{key}: {value}" for key, value in counts.items() if value)
    report(filename, "Count missing values", len(df), len(df), total_missing,
           detail or "No missing values found.", "FLAGGED")

    return df


def clean_absences(df, filename):
    """Strip day/days only from nonnegative integer absence counts."""
    if "absences" not in df.columns:
        return df
    original = df["absences"].copy()
    cleaned = original.str.replace(r"^(\d+)\s+days?$", r"\1", regex=True, case=False)
    changed = int(original.ne(cleaned).fillna(False).sum())
    df["absences"] = cleaned
    report(filename, "Clean absences", len(df), len(df), changed,
           "Removed day/days suffix only when preceded by a nonnegative integer; other values retained.",
           "MODIFIED")
    return df


def clean_minutes(df, filename):
    """Correct invalid session minutes from documented attendance states."""
    original = df["minutes"].copy()
    invalid = ~original.isin(["0", "45"])
    df.loc[invalid & (df["attendance"] == "Present"), "minutes"] = "45"
    df.loc[invalid & (df["attendance"] == "Absent"), "minutes"] = "0"
    changed = int(original.ne(df["minutes"]).fillna(False).sum())
    report(filename, "Clean session minutes", len(df), len(df), changed,
           "Corrected invalid minutes using attendance: Present=45, Absent=0.",
           "MODIFIED")
    return df


def validate_numeric_types(df, filename):
    """Flag nonnumeric and noninteger values without changing any rows."""
    for column in df.columns:
        if column not in NUMERIC_COLUMNS:
            continue
        numeric = pd.to_numeric(df[column].dropna(), errors="coerce")
        bad_type = int(numeric.isna().sum())
        if column in INTEGER_COLUMNS:
            bad_type += int(((numeric.dropna() % 1) != 0).sum())
        report(filename, f"Validate numeric type: {column}", len(df), len(df),
               bad_type, "Column should contain numeric values" +
               (" representing integers." if column in INTEGER_COLUMNS else "."),
               "FLAGGED")


def main():
    if not INPUT_FOLDER.is_dir():
        print(f"Input folder not found: {INPUT_FOLDER.resolve()}")
        return
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in INPUT_FOLDER.iterdir()
                   if p.is_file() and (p.name.lower().endswith(".csv")
                                      or p.name.lower().endswith(".csv.gz")))
    if not paths:
        print(f"No matching CSV / CSV.GZ files in {INPUT_FOLDER.resolve()}")
        return

    total_before = total_after = successes = 0
    for path in paths:
        name = path.name
        print(f"\n{'=' * 65}\nProcessing: {name}")
        try:
            df = read_csv_safely(path)
            original_count = len(df)
            print(f"Original rows: {original_count:,}")
            df = clean_general(df, name)
            if canonical_filename(path) == "term_grades.csv":
                df = clean_absences(df, name)
            if canonical_filename(path) == "tutoring_sessions.csv":
                df = clean_minutes(df, name)
            validate_numeric_types(df, name)
            if canonical_filename(path) == "tutoring_sessions.csv":
                invalid = df[~df["minutes"].isin(["0", "45"])]
                invalid_path = REPORT_FOLDER / "invalid_minutes.csv"
                invalid.to_csv(invalid_path, index=False)
                print(f"Invalid minutes records exported: {len(invalid):,}")
            output = OUTPUT_FOLDER / name
            df.to_csv(output, index=False, compression="infer")
            print(f"Saved: {output} | final rows: {len(df):,}")
            total_before += original_count
            total_after += len(df)
            successes += 1
        except Exception as exc:
            print(f"FAILED: {name}: {exc}")
            records.append({"file": name, "step": "FILE ERROR", "action": "FAILED",
                            "rows_before": None, "rows_after": None,
                            "rows_removed": None, "values_affected_or_flagged": None,
                            "reason": str(exc)})

    audit_path = REPORT_FOLDER / "cleaning_audit.csv"
    pd.DataFrame(records).to_csv(audit_path, index=False)
    print(f"\n{'=' * 65}\nSUMMARY")
    print(f"Files processed: {successes}/{len(paths)}")
    print(f"Rows before: {total_before:,} | after: {total_after:,}")
    print(f"Rows removed: {total_before - total_after:,}")
    print(f"Audit report: {audit_path}")


if __name__ == "__main__":
    main()
