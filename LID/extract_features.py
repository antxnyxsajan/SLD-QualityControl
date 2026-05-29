"""
extract_features.py
-------------------
Extracts phoneme posterior features from .wav audio files using the
facebook/wav2vec2-lv-60-espeak-cv-ft model (a wav2vec2-based phoneme recogniser).

For each .wav file it produces a .npy file containing a (392, T) matrix of
phoneme posterior probabilities — the exact input format expected by the
ConfPhone Conformer model in confphone/train_conformer.py.

Usage
-----
    python extract_features.py --input_dir <path_to_dataset> --output_dir <path_to_save_npy>

Example
-------
    python extract_features.py \
        --input_dir "D:/MANDI INTERNSHIP/Datasets/Training Dataset - shubham/ekstep_seen" \
        --output_dir "D:/MANDI INTERNSHIP/Datasets/Training Dataset - shubham/features"
"""

import argparse
import os
import sys
import glob
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC


# ---------- configuration ----------
TARGET_SAMPLE_RATE = 16_000
TARGET_FEATURE_DIM = 392  # Conformer model expects exactly 392
MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"


def load_model(device):
    """Load the wav2vec2 phoneme model and feature extractor."""
    print(f"[INFO] Loading model: {MODEL_NAME}")
    print(f"[INFO] Device: {device}")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    print("[INFO] Model loaded successfully.")
    return processor, model


def extract_phoneme_posteriors(wav_path, processor, model, device):
    """
    Load a single .wav file, run it through the phoneme model,
    and return the softmax posteriors as a numpy array of shape (392, T).
    """
    # Load and resample
    waveform, sr = torchaudio.load(wav_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16 kHz if needed
    if sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)

    waveform = waveform.squeeze(0)  # (num_samples,)

    # Skip very short audio (< 0.1s)
    if waveform.shape[0] < TARGET_SAMPLE_RATE * 0.1:
        return None

    # Process through the model
    inputs = processor(
        waveform.numpy(),
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits  # (1, T, vocab_size)

    # Apply softmax to get posteriors
    posteriors = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (T, vocab_size)

    # Adjust to exactly TARGET_FEATURE_DIM (392)
    vocab_size = posteriors.shape[1]
    if vocab_size > TARGET_FEATURE_DIM:
        posteriors = posteriors[:, :TARGET_FEATURE_DIM]
    elif vocab_size < TARGET_FEATURE_DIM:
        padding = np.zeros((posteriors.shape[0], TARGET_FEATURE_DIM - vocab_size), dtype=posteriors.dtype)
        posteriors = np.concatenate([posteriors, padding], axis=1)

    # Transpose to (392, T) — the format expected by the data loader
    posteriors = posteriors.T  # (392, T)

    return posteriors


def process_dataset(input_dir, output_dir, processor, model, device):
    """
    Walk the entire input directory tree, find all .wav files,
    extract features, and save as .npy files mirroring the folder structure.
    """
    from collections import defaultdict
    
    # Collect all .wav files grouped by directory
    folders_to_files = defaultdict(list)
    total_files = 0
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                folders_to_files[root].append(os.path.join(root, f))
                total_files += 1

    print(f"[INFO] Found {total_files} .wav files across {len(folders_to_files)} directories in {input_dir}")

    if total_files == 0:
        print("[WARNING] No .wav files found. Check your input_dir path.")
        return

    skipped = 0
    errors = 0

    # Sort folders to process them in a predictable order
    for folder in sorted(folders_to_files.keys()):
        files = folders_to_files[folder]
        folder_name = os.path.basename(folder)
        dataset_name = os.path.basename(os.path.dirname(folder))
        
        print(f"\n============================================================")
        print(f"[INFO] ---> Starting: {dataset_name} / {folder_name} ({len(files)} files)")
        print(f"============================================================")
        
        for wav_path in tqdm(files, desc=f"Extracting {folder_name}", unit="file", leave=False):
            # Build mirrored output path
            rel_path = os.path.relpath(wav_path, input_dir)
            npy_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".npy")

            # Skip if already processed
            if os.path.exists(npy_path):
                continue

            # Create output directory
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)

            try:
                posteriors = extract_phoneme_posteriors(wav_path, processor, model, device)
                if posteriors is None:
                    skipped += 1
                    continue
                np.save(npy_path, posteriors)
            except Exception as e:
                errors += 1
                tqdm.write(f"[ERROR] {wav_path}: {e}")
                continue
                
        print(f"[INFO] <--- Finished: {dataset_name} / {folder_name}")

    print(f"\n[DONE] Feature extraction complete.")
    print(f"  Total files: {total_files}")
    print(f"  Skipped (too short): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract phoneme posteriors from .wav files for LID training"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory containing .wav files (e.g. ekstep_seen/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the extracted .npy feature files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. 'cuda:0' or 'cpu'). Auto-detects if not specified.",
    )
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    processor, model = load_model(device)
    process_dataset(args.input_dir, args.output_dir, processor, model, device)


if __name__ == "__main__":
    main()
