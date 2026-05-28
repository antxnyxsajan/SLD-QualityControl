"""
prepare_dataset.py
------------------
Scans extracted .npy feature files, assigns integer language labels,
and creates train.txt / test.txt files in the format expected by
confphone/train_conformer.py.

Each line in the output files has the format:
    <path_to_npy_file> <integer_label>

Usage
-----
    python prepare_dataset.py --features_dir <path_to_npy_features> --output_dir <path_to_save_txt>

Example
-------
    python prepare_dataset.py \
        --features_dir "D:/MANDI INTERNSHIP/Datasets/Training Dataset - shubham/features" \
        --output_dir "D:/MANDI INTERNSHIP/Codebase/LID Model - shubham/confphone"
"""

import argparse
import os
import random
import numpy as np
from collections import Counter


# ---------- Language code → integer label mapping ----------
# This matches the 12-class setup described in the README.
# Labels 0-11 in alphabetical order by language code.
LANG_MAP = {
    "asm": 0,   # Assamese
    "ben": 1,   # Bengali
    "eng": 2,   # English
    "guj": 3,   # Gujarati
    "hin": 4,   # Hindi
    "kan": 5,   # Kannada
    "mal": 6,   # Malayalam
    "mar": 7,   # Marathi
    "odi": 8,   # Odia
    "pun": 9,   # Punjabi
    "tam": 10,  # Tamil
    "tel": 11,  # Telugu
}


def detect_language_from_path(npy_path):
    """
    Determine the language of a .npy file by checking which language
    folder name (e.g. 'hin', 'ben', etc.) appears in its path.
    Returns the integer label or None if no match.
    """
    # Normalise separators
    parts = npy_path.replace("\\", "/").split("/")
    for part in parts:
        lower = part.lower()
        if lower in LANG_MAP:
            return LANG_MAP[lower]
    return None


def collect_samples(features_dir):
    """
    Walk the features directory and collect all (npy_path, label) pairs.
    """
    samples = []
    skipped = 0

    for root, dirs, files in os.walk(features_dir):
        for f in files:
            if not f.endswith(".npy"):
                continue
            npy_path = os.path.join(root, f)
            label = detect_language_from_path(npy_path)
            if label is None:
                skipped += 1
                continue
            samples.append((npy_path, label))

    return samples, skipped


def validate_sample(npy_path, expected_dim=392):
    """
    Quick sanity check: make sure the .npy file can be loaded
    and has the right first dimension (392).
    """
    try:
        arr = np.load(npy_path, allow_pickle=True)
        if arr.shape[0] != expected_dim:
            return False, f"dim mismatch: {arr.shape}"
        if arr.ndim != 2:
            return False, f"ndim={arr.ndim}, expected 2"
        return True, "ok"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare train.txt and test.txt for LID Conformer training"
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        required=True,
        help="Root directory containing extracted .npy feature files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write train.txt and test.txt",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of data to use for testing (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/test split",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="If set, validate each .npy file before including it (slower)",
    )
    args = parser.parse_args()

    print(f"[INFO] Scanning features in: {args.features_dir}")
    samples, skipped = collect_samples(args.features_dir)
    print(f"[INFO] Found {len(samples)} labelled samples ({skipped} skipped - no language match)")

    if len(samples) == 0:
        print("[ERROR] No samples found. Check your features_dir path.")
        return

    # Optional validation
    if args.validate:
        print("[INFO] Validating .npy files...")
        valid_samples = []
        invalid_count = 0
        for npy_path, label in samples:
            ok, msg = validate_sample(npy_path)
            if ok:
                valid_samples.append((npy_path, label))
            else:
                invalid_count += 1
                if invalid_count <= 10:
                    print(f"  [INVALID] {npy_path}: {msg}")
        print(f"[INFO] Valid: {len(valid_samples)}, Invalid: {invalid_count}")
        samples = valid_samples

    # Print distribution
    label_counts = Counter(label for _, label in samples)
    inv_map = {v: k for k, v in LANG_MAP.items()}
    print("\n[INFO] Language distribution:")
    print(f"  {'Language':<12} {'Code':<6} {'Label':<6} {'Count':<8}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*8}")
    for label_id in sorted(label_counts.keys()):
        code = inv_map.get(label_id, "???")
        count = label_counts[label_id]
        lang_names = {
            "asm": "Assamese", "ben": "Bengali", "eng": "English",
            "guj": "Gujarati", "hin": "Hindi", "kan": "Kannada",
            "mal": "Malayalam", "mar": "Marathi", "odi": "Odia",
            "pun": "Punjabi", "tam": "Tamil", "tel": "Telugu",
        }
        name = lang_names.get(code, code)
        print(f"  {name:<12} {code:<6} {label_id:<6} {count:<8}")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(samples)

    split_idx = int(len(samples) * (1 - args.test_ratio))
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    print(f"\n[INFO] Split: {len(train_samples)} train / {len(test_samples)} test")

    # Write files
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.txt")
    test_path = os.path.join(args.output_dir, "test.txt")

    with open(train_path, "w") as f:
        for npy_path, label in train_samples:
            f.write(f"{npy_path} {label}\n")

    with open(test_path, "w") as f:
        for npy_path, label in test_samples:
            f.write(f"{npy_path} {label}\n")

    print(f"\n[DONE] Files written:")
    print(f"  Train: {train_path} ({len(train_samples)} samples)")
    print(f"  Test:  {test_path} ({len(test_samples)} samples)")


if __name__ == "__main__":
    main()
