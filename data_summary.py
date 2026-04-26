import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "plotting" / "markers" / "unmarked_edited"

# Vicon CSVs have 3 rows before actual data
HEADER_ROWS = 3


def summarize_file(filepath):
    df = pd.read_csv(filepath, skiprows=HEADER_ROWS, header=None)
    n_rows = len(df)
    n_cols = len(df.columns)
    rows_with_nan = df.isna().any(axis=1).sum()
    total_nan = df.isna().sum().sum()
    percent_nan = (total_nan / df.size) * 100
    return n_rows, n_cols, rows_with_nan, percent_nan


def main():
    csv_files = sorted(DATA_DIR.rglob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}")
        return

    print(f"{'File':<50} {'Rows':>6} {'Cols':>6} {'NaN Rows':>10} {'% NaN':>8}")
    print("-" * 76)

    for filepath in csv_files:
        n_rows, n_cols, rows_with_nan, percent_nan = summarize_file(filepath)
        rel_path = filepath.relative_to(DATA_DIR)
        print(
            f"{str(rel_path):<50} {n_rows:>6} {n_cols:>6} {rows_with_nan:>10} {(percent_nan):>7.1f}%"
        )


if __name__ == "__main__":
    main()
