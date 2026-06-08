"""
report_writer.py — CSV Report Utility for the SLD QA Batch Pipeline

Writes one row per processed file to a CSV with columns:
  audio_file, speaker_rttm, language_rttm, lang_labels,
  spk_structural_score, spk_f1_score, spk_agreement,
  lang_structural_score, lang_f1_score, lang_agreement,
  spk_severe_anomalies, lang_severe_anomalies,
  status, error
"""

import csv
import os

# ---------------------------------------------------------------------------
# Column order (used for both the header and DictWriter fieldnames)
# ---------------------------------------------------------------------------
FIELDNAMES = [
    "audio_file",
    "speaker_rttm",
    "language_rttm",
    "lang_labels",
    "spk_structural_score",
    "spk_f1_score",
    "spk_agreement",
    "lang_structural_score",
    "lang_f1_score",
    "lang_agreement",
    "spk_severe_anomalies",
    "lang_severe_anomalies",
    "lang_ambiguous_anomalies",
    "error",
]


def init_csv(output_path: str) -> None:
    """
    Creates (or overwrites) the CSV file and writes the header row.
    Call this once at the start of a batch run.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()


def build_lang_labels(label_map: dict) -> str:
    """
    Converts the label_map dict from LanguageValidator into a human-readable string.

    Args:
        label_map: e.g. {"L1": "mal", "L2": "eng"}

    Returns:
        e.g. "L1:mal, L2:eng"  (sorted by key for consistency)
    """
    if not label_map:
        return "N/A"
    return ", ".join(f"{k}:{v}" for k, v in sorted(label_map.items()))


def append_row(output_path: str, result: dict) -> None:
    """
    Appends a single result row to the CSV.

    Expected keys in `result` (all optional — missing values default to 'N/A'):
      audio_file, speaker_rttm, language_rttm,
      label_map (dict),
      spk_structural_score, spk_f1, spk_agreement,
      lang_structural_score, lang_f1, lang_agreement,
      spk_severe, lang_severe,
      passed (bool), error (str)
    """
    spk_agreement = result.get("spk_agreement")
    lang_agreement = result.get("lang_agreement")

    row = {
        "audio_file":           result.get("audio_file", "N/A"),
        "speaker_rttm":         result.get("speaker_rttm", "N/A"),
        "language_rttm":        result.get("language_rttm", "N/A"),
        "lang_labels":          build_lang_labels(result.get("label_map", {})),
        "spk_structural_score": result.get("spk_structural_score", "N/A"),
        "spk_f1_score":         f"{result['spk_f1']:.4f}" if result.get("spk_f1") is not None else "N/A",
        "spk_agreement":        f"{spk_agreement:.2%}" if spk_agreement is not None else "N/A",
        "lang_structural_score":result.get("lang_structural_score", "N/A"),
        "lang_f1_score":        f"{result['lang_f1']:.4f}" if result.get("lang_f1") is not None else "N/A",
        "lang_agreement":       f"{lang_agreement:.2%}" if lang_agreement is not None else "N/A",
        "spk_severe_anomalies": result.get("spk_severe", "N/A"),
        "lang_severe_anomalies":result.get("lang_severe", "N/A"),
        "lang_ambiguous_anomalies": result.get("lang_ambiguous", "N/A"),
        "error":                result.get("error", ""),
    }

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)


def print_summary(output_path: str) -> None:
    """
    Prints a short terminal summary after the batch run completes.
    """
    rows = []
    with open(output_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total  = len(rows)
    errors = sum(1 for r in rows if r["error"])

    print("\n" + "=" * 60)
    print("BATCH QA SUMMARY")
    print("=" * 60)
    print(f"  Total files processed : {total}")
    print(f"  Processing errors     : {errors}")
    print(f"\n  Report saved to: {os.path.abspath(output_path)}")
    print("=" * 60)
