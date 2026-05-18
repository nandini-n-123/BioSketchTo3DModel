import csv
import json
import os
from pathlib import Path
from typing import Dict, Any, List


APP_DIR = Path(__file__).resolve().parent
DEFORM_ROOT = APP_DIR.parent

RESULTS_DIR = DEFORM_ROOT / "presentation_debug" / "result_tables"
RAW_LOG_PATH = RESULTS_DIR / "deformation_results.jsonl"
CSV_PATH = RESULTS_DIR / "deformation_results_table.csv"


CSV_FIELDS = [
    "organ",
    "rms_before",
    "rms_after",
    "rms_improvement_percent",
    "similarity_before_percent",
    "similarity_after_percent",
    "mean_deformation_percent",
    "max_deformation_percent",
    "mean_depth_deformation_percent",
    "max_depth_deformation_percent",
]


def _ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def append_result(result: Dict[str, Any]) -> None:
    """
    Append one pipeline result to a JSONL log file.

    This is called automatically by pipeline.py after each deformation run.
    """
    _ensure_results_dir()

    row = {}

    for key in CSV_FIELDS:
        row[key] = result.get(key, "")

    # Optional useful metadata.
    row["output_path"] = result.get("output_path", "")
    row["debug_path"] = result.get("debug_path", "")
    row["sketch_debug_path"] = result.get("sketch_debug_path", "")

    with open(RAW_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    print(f"[RESULT LOG] Appended result to: {RAW_LOG_PATH}")


def load_results() -> List[Dict[str, Any]]:
    """
    Load all logged pipeline results.
    """
    if not RAW_LOG_PATH.exists():
        return []

    results = []

    with open(RAW_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            results.append(json.loads(line))

    return results


def export_csv(output_path: str = None) -> str:
    """
    Export logged results to a clean CSV table.
    """
    _ensure_results_dir()

    if output_path is None:
        output_path = CSV_PATH
    else:
        output_path = Path(output_path)

    results = load_results()

    if not results:
        raise RuntimeError(
            f"No results found. Run pipeline.py first. Expected log file: {RAW_LOG_PATH}"
        )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for result in results:
            row = {}

            for field in CSV_FIELDS:
                row[field] = result.get(field, "")

            writer.writerow(row)

    print(f"[CSV] Exported result table: {output_path}")
    return str(output_path)


def clear_results() -> None:
    """
    Clear previous logged results.
    Use this before collecting final report values.
    """
    _ensure_results_dir()

    if RAW_LOG_PATH.exists():
        RAW_LOG_PATH.unlink()

    if CSV_PATH.exists():
        CSV_PATH.unlink()

    print("[RESULT LOG] Cleared old result logs.")


if __name__ == "__main__":
    export_csv()