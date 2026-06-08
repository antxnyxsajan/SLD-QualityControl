"""
batch_runner.py — Batch QA Pipeline for SLD-QualityControl

Processes multiple RTTM+audio file triplets from three separate directories
and generates a consolidated CSV quality report.

Folder layout expected:
  --audio_dir    /path/to/audio/       →  {name}.wav
  --speaker_dir  /path/to/speaker/     →  {name}_SPEAKER.rttm
  --language_dir /path/to/language/    →  {name}_LANGUAGE.rttm  (optional)

Usage:
  python batch_runner.py \\
      --audio_dir    /data/audio \\
      --speaker_dir  /data/rttms/speaker \\
      --language_dir /data/rttms/language \\
      --output_csv   results/qa_report.csv \\
      [--audio_ext .wav] \\
      [--suffix_speaker _SPEAKER.rttm] \\
      [--suffix_lang    _LANGUAGE.rttm] \\
      [--skip_audio]   \\
      [--verbose]

Options:
  --skip_audio   Skip acoustic (Step 2) and language (Step 3) validation.
                 Only runs structural checks (Step 1.1 and 1.2).
                 Much faster — useful for a quick format sanity check.
  --verbose      Print full per-file pipeline output (same as running main.py
                 individually). By default, only a progress line per file is shown.
"""

import argparse
import os
import sys
import traceback

from importlib import import_module

import report_writer

main_module = import_module("main")


# ---------------------------------------------------------------------------
# File Discovery
# ---------------------------------------------------------------------------

def discover_triplets(
    audio_dir: str,
    speaker_dir: str,
    language_dir: str | None,
    audio_ext: str,
    suffix_speaker: str,
    suffix_lang: str,
) -> list[dict]:
    """
    Scans audio_dir for audio files, then looks up the matching speaker and
    (optionally) language RTTM files in their respective directories.

    Returns a list of dicts:
      {
        "name":         str,   # base name without extension, e.g. "B007"
        "audio":        str,   # full path to audio file
        "speaker_rttm": str,   # full path to speaker RTTM
        "language_rttm":str | None,  # full path to language RTTM (or None)
      }

    Files with no matching speaker RTTM are skipped with a warning.
    Files with no matching language RTTM proceed with speaker-only validation.
    """
    triplets = []
    missing_speaker = []
    missing_language = []

    # Normalise extension
    if not audio_ext.startswith("."):
        audio_ext = "." + audio_ext

    # Collect audio files
    audio_files = sorted(
        f for f in os.listdir(audio_dir)
        if f.lower().endswith(audio_ext)
    )

    if not audio_files:
        print(f"[WARNING] No audio files found in '{audio_dir}' with extension '{audio_ext}'.")
        return []

    for audio_filename in audio_files:
        name = audio_filename[: -len(audio_ext)]  # strip extension to get base name
        if name.endswith("_MIX"):
            name = name[:-4]

        audio_path = os.path.join(audio_dir, audio_filename)

        # Speaker RTTM
        spk_filename = name + suffix_speaker
        spk_path = os.path.join(speaker_dir, spk_filename)
        if not os.path.isfile(spk_path):
            missing_speaker.append(audio_filename)
            continue  # Cannot validate without speaker RTTM

        # Language RTTM (optional)
        lang_path = None
        if language_dir:
            lang_filename = name + suffix_lang
            candidate = os.path.join(language_dir, lang_filename)
            if os.path.isfile(candidate):
                lang_path = candidate
            else:
                missing_language.append(audio_filename)

        triplets.append({
            "name":          name,
            "audio":         audio_path,
            "speaker_rttm":  spk_path,
            "language_rttm": lang_path,
        })

    # Report discovery results
    print(f"\n[Discovery] Found {len(triplets)} file(s) to process.")
    if missing_speaker:
        print(f"  [SKIP] {len(missing_speaker)} audio file(s) skipped — no matching Speaker RTTM:")
        for f in missing_speaker:
            print(f"    - {f}")
    if missing_language:
        print(f"  [INFO] {len(missing_language)} file(s) will run speaker-only (no Language RTTM found):")
        for f in missing_language:
            print(f"    - {f}")

    return triplets


# ---------------------------------------------------------------------------
# Per-file Processing
# ---------------------------------------------------------------------------

def process_one(triplet: dict, skip_audio: bool, verbose: bool) -> dict:
    """
    Runs the full QA pipeline on a single audio+RTTM triplet.

    Returns a result dict compatible with report_writer.append_row().
    """
    name         = triplet["name"]
    audio_path   = triplet["audio"] if not skip_audio else None
    speaker_rttm = triplet["speaker_rttm"]
    language_rttm= triplet["language_rttm"]

    print(f"\n{'='*60}")
    print(f"  Processing: {name}")
    print(f"    Audio   : {os.path.basename(audio_path) if audio_path else 'N/A (--skip_audio)'}")
    print(f"    Speaker : {os.path.basename(speaker_rttm)}")
    print(f"    Language: {os.path.basename(language_rttm) if language_rttm else 'N/A'}")
    print(f"{'='*60}")

    try:
        pipeline_results = main_module.run_pipeline(
            rttm_filepath=speaker_rttm,
            audio_filepath=audio_path,
            rttml_filepath=language_rttm,
            silent=not verbose,
        )

        return {
            "audio_file":            os.path.basename(triplet["audio"]),
            "speaker_rttm":          os.path.basename(speaker_rttm),
            "language_rttm":         os.path.basename(language_rttm) if language_rttm else "N/A",
            "label_map":             pipeline_results.get("label_map", {}),
            "spk_structural_score":  pipeline_results.get("spk_structural_score"),
            "spk_f1":                pipeline_results.get("spk_f1"),
            "spk_agreement":         pipeline_results.get("spk_agreement"),
            "spk_severe":            pipeline_results.get("spk_severe"),
            "lang_structural_score": pipeline_results.get("lang_structural_score"),
            "lang_f1":               pipeline_results.get("lang_f1"),
            "lang_agreement":        pipeline_results.get("lang_agreement"),
            "lang_severe":           pipeline_results.get("lang_severe"),
            "lang_ambiguous":        pipeline_results.get("lang_ambiguous"),
            "passed":                pipeline_results.get("passed", False),
            "error":                 "",
        }

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        print(f"\n  [ERROR] Failed to process '{name}': {error_msg}")
        traceback.print_exc()
        return {
            "audio_file":            os.path.basename(triplet["audio"]),
            "speaker_rttm":          os.path.basename(speaker_rttm),
            "language_rttm":         os.path.basename(language_rttm) if language_rttm else "N/A",
            "label_map":             {},
            "spk_structural_score":  None,
            "spk_f1":                None,
            "spk_agreement":         None,
            "spk_severe":            None,
            "lang_structural_score": None,
            "lang_f1":               None,
            "lang_agreement":        None,
            "lang_severe":           None,
            "passed":                False,
            "error":                 error_msg,
        }


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch QA pipeline: processes multiple audio+RTTM pairs and writes a CSV report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--audio_dir", required=True,
        help="Directory containing audio WAV files."
    )
    parser.add_argument(
        "--speaker_dir", required=True,
        help="Directory containing speaker RTTM files ({name}_SPEAKER.rttm)."
    )
    parser.add_argument(
        "--output_csv", required=True,
        help="Path where the output CSV report will be written."
    )

    # Optional arguments
    parser.add_argument(
        "--language_dir", default=None,
        help="Directory containing language RTTM files ({name}_LANGUAGE.rttm). "
             "If not provided, only speaker validation is run."
    )
    parser.add_argument(
        "--audio_ext", default=".wav",
        help="Audio file extension to look for."
    )
    parser.add_argument(
        "--suffix_speaker", default="_RTTM.rttm",
        help="Suffix (including extension) used to derive speaker RTTM filename from audio base name."
    )
    parser.add_argument(
        "--suffix_lang", default="_LANGUAGE.rttm",
        help="Suffix (including extension) used to derive language RTTM filename from audio base name."
    )
    parser.add_argument(
        "--skip_audio", action="store_true",
        help="Skip acoustic and language validation (Steps 2 & 3). "
             "Only runs structural checks. Much faster for format-only sanity checks."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full per-file pipeline output. By default, only a one-line progress "
             "summary per file is shown to keep the terminal clean."
    )

    args = parser.parse_args()

    # Validate directories
    for label, path in [("audio_dir", args.audio_dir), ("speaker_dir", args.speaker_dir)]:
        if not os.path.isdir(path):
            print(f"[ERROR] --{label} path does not exist or is not a directory: '{path}'")
            sys.exit(1)

    if args.language_dir and not os.path.isdir(args.language_dir):
        print(f"[ERROR] --language_dir path does not exist or is not a directory: '{args.language_dir}'")
        sys.exit(1)

    print("=" * 60)
    print("  SLD Quality Control — Batch Runner")
    print("=" * 60)
    print(f"  Audio dir    : {args.audio_dir}")
    print(f"  Speaker dir  : {args.speaker_dir}")
    print(f"  Language dir : {args.language_dir or 'N/A (speaker-only mode)'}")
    print(f"  Output CSV   : {args.output_csv}")
    print(f"  Skip audio   : {args.skip_audio}")
    print(f"  Verbose      : {args.verbose}")

    # ── Discover files ────────────────────────────────────────────────────────
    triplets = discover_triplets(
        audio_dir=args.audio_dir,
        speaker_dir=args.speaker_dir,
        language_dir=args.language_dir,
        audio_ext=args.audio_ext,
        suffix_speaker=args.suffix_speaker,
        suffix_lang=args.suffix_lang,
    )

    if not triplets:
        print("\n[ERROR] No valid file triplets found. Exiting.")
        sys.exit(1)

    # ── Initialise CSV ────────────────────────────────────────────────────────
    report_writer.init_csv(args.output_csv)

    # ── Process each triplet ──────────────────────────────────────────────────
    total = len(triplets)
    for idx, triplet in enumerate(triplets, 1):
        print(f"\n[{idx}/{total}] {triplet['name']}")
        result = process_one(triplet, skip_audio=args.skip_audio, verbose=args.verbose)

        # Brief result line visible even in non-verbose mode
        status_str = "PASSED ✓" if result["passed"] else "FAILED ✗"
        if result["error"]:
            status_str = f"ERROR — {result['error']}"
        spk_agr = f"{result['spk_agreement']:.2%}" if result["spk_agreement"] is not None else "N/A"
        lang_agr = f"{result['lang_agreement']:.2%}" if result["lang_agreement"] is not None else "N/A"
        print(f"  → {status_str}  |  Spk Agr: {spk_agr}  |  Lang Agr: {lang_agr}")

        report_writer.append_row(args.output_csv, result)

    # ── Print final summary ───────────────────────────────────────────────────
    report_writer.print_summary(args.output_csv)


if __name__ == "__main__":
    main()
