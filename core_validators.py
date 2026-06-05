import os
import sys
import torch
import torchaudio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, WhisperProcessor, WhisperConfig
import whisper

class SpeakerVerificationSystem:
    def __init__(self, similarity_threshold=0.25):
        """
        Initializes the speaker validation system using SpeechBrain's ECAPA-TDNN model.
        """
        from speechbrain.inference.speaker import SpeakerRecognition
        from speechbrain.utils.fetching import LocalStrategy
        
        self.similarity_threshold = similarity_threshold
        
        print("Loading Core Speaker Verification Model (ECAPA-TDNN)...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.verifier = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": device}
        )

    def compare_speakers(self, audio_tensor_1, audio_tensor_2):
        """
        Compares two audio tensors and returns if they match and their cosine similarity score.
        audio_tensor_1: The anchor segment (first occurrence).
        audio_tensor_2: The test segment.
        """
        # Get embeddings and compute cosine similarity
        score, prediction = self.verifier.verify_batch(audio_tensor_2, audio_tensor_1)
        sim_score = score.item()
        is_match = sim_score >= self.similarity_threshold
        
        return is_match, sim_score


# ---------------------------------------------------------------------------
# Fine-tuned Whisper Encoder definition (must match the training architecture)
# ---------------------------------------------------------------------------
_FT_MODEL_NAME = "openai/whisper-base"
_FT_NUM_CLASSES = 12
_FT_NUM_UNFROZEN_LAYERS = 4   # v1: 4 encoder layers unfrozen
_TARGET_SR = 16000

# Label mapping: must match sklearn LabelEncoder.fit() sort order used at training time.
# Training labels (sorted alphabetically): asm, ben, eng, guj, hin, kan, mal, mar, odi, pun, tam, tel
_FT_LABEL_CLASSES = ["asm", "ben", "eng", "guj", "hin", "kan", "mal", "mar", "odi", "pun", "tam", "tel"]


class _FineTuneWhisperEncoder(nn.Module):
    """Exact replica of the architecture used during fine-tuning (v1)."""

    def __init__(self):
        super().__init__()
        self.config = WhisperConfig.from_pretrained(_FT_MODEL_NAME)
        self.whisper = WhisperModel.from_pretrained(_FT_MODEL_NAME)
        for p in self.whisper.parameters():
            p.requires_grad = False
        for layer in self.whisper.encoder.layers[-_FT_NUM_UNFROZEN_LAYERS:]:
            for p in layer.parameters():
                p.requires_grad = True
        self.classifier = nn.Sequential(
            nn.Linear(self.config.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, _FT_NUM_CLASSES),
        )

    def forward(self, x):
        out = self.whisper.encoder(x).last_hidden_state
        pooled = out.mean(dim=1)
        return self.classifier(pooled)


class LanguageIdentificationSystem:
    def __init__(self, model_size="large"):
        """
        Initializes the language identification system using the fine-tuned
        Whisper encoder checkpoint located at whisper/model_continued_best.pth.
        The `model_size` parameter is kept for API compatibility but is ignored;
        the fine-tuned model is always loaded.
        """
        # Resolve checkpoint path relative to this file's location
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _checkpoint = os.path.join(_script_dir, "whisper", "model_continued_best.pth")

        print("Loading Fine-Tuned Language Identification Model (Whisper-base FT)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model and load weights
        self.model = _FineTuneWhisperEncoder().to(self.device)
        state_dict = torch.load(_checkpoint, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Processor for feature extraction (same as training)
        self.processor = WhisperProcessor.from_pretrained(_FT_MODEL_NAME)

        print(f"  Checkpoint : {_checkpoint}")
        print(f"  Device     : {self.device}")
        print(f"  Classes    : {_FT_LABEL_CLASSES}")

    def identify_language(self, audio_tensor, current_sample_rate):
        """
        Identifies the language of an audio tensor using the fine-tuned model.
        Returns (language_code, confidence) matching the original API.
        """
        # Ensure 1D tensor (mono)
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.mean(dim=0)

        # Resample to 16 kHz if needed (cached resampler for speed)
        if current_sample_rate != _TARGET_SR:
            if not hasattr(self, "_resampler") or self._resampler_orig_freq != current_sample_rate:
                self._resampler = torchaudio.transforms.Resample(current_sample_rate, _TARGET_SR)
                self._resampler_orig_freq = current_sample_rate
            audio_tensor = self._resampler(audio_tensor)

        # Convert to numpy for WhisperProcessor
        audio_np = audio_tensor.numpy()

        # Extract mel features (WhisperProcessor expects a list of arrays)
        inputs = self.processor(
            [audio_np],
            sampling_rate=_TARGET_SR,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        feats = inputs.input_features  # shape: (1, 80, T)

        # Pad/trim time dimension to exactly 3000 frames (30 s at 10 ms/frame)
        T = feats.size(2)
        if T < 3000:
            feats = F.pad(feats, (0, 3000 - T))
        else:
            feats = feats[:, :, :3000]

        feats = feats.to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(feats)          # (1, NUM_CLASSES)
            probs = F.softmax(logits, dim=1)    # (1, NUM_CLASSES)

        class_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, class_idx].item())
        detected_lang = _FT_LABEL_CLASSES[class_idx]

        return detected_lang, confidence


# ---------------------------------------------------------------------------
# Stock Whisper Cross-Checker (second-opinion validator)
# ---------------------------------------------------------------------------
# This class wraps the full stock whisper-large-v3 model and uses its built-in
# language detection to independently verify anomalies flagged by the fine-tuned
# model. Since Whisper was trained on 680,000 hours of diverse multilingual audio
# (including Indian accents and conversational speech), it serves as a reliable
# cross-check to filter out false positives from the domain-limited fine-tuned model.
#
# Usage in the validator:
#   checker = WhisperCrossChecker()
#   lang, prob = checker.detect(audio_np_16khz)   # audio must be 16kHz numpy float32
#
# Whisper detect_language() returns probabilities for all 99 languages it supports.
# We extract the top prediction and its probability, then compare against the
# anchor language to determine if the fine-tuned model's flag was justified.

class WhisperCrossChecker:
    """Second-opinion language detector using stock whisper-large-v3."""

    # Whisper uses ISO 639-1 codes (2-letter). Map them to our 3-letter codes
    # for languages in our 12-class set that have different ISO representations.
    _WHISPER_TO_FT_CODE = {
        "en": "eng",
        "hi": "hin",
        "bn": "ben",
        "gu": "guj",
        "kn": "kan",
        "ml": "mal",
        "mr": "mar",
        "or": "odi",
        "pa": "pun",
        "ta": "tam",
        "te": "tel",
        # Assamese: Whisper uses 'as', our code uses 'asm'
        "as": "asm",
    }

    def __init__(self, model_name: str = "large-v3"):
        """
        Loads the stock Whisper model for cross-validation.

        Args:
            model_name: Whisper model size. 'large-v3' recommended for best accuracy.
        """
        print(f"Loading Stock Whisper Cross-Checker ({model_name})...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_name, device=self.device)
        self.model.eval()
        print(f"  Whisper {model_name} loaded on {self.device}")

    def detect(self, audio_np: np.ndarray, sample_rate: int) -> tuple:
        """
        Detects the language of an audio segment using stock Whisper.

        Args:
            audio_np:    1-D float32 numpy array of audio samples.
            sample_rate: Sample rate of the audio (will be resampled to 16kHz if needed).

        Returns:
            (language_code_3letter, confidence) matching the fine-tuned model's API.
            language_code_3letter is mapped from Whisper's ISO 639-1 2-letter code.
            Returns (None, 0.0) on failure.
        """
        try:
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                audio_tensor = torch.from_numpy(audio_np).float()
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                audio_np = resampler(audio_tensor).numpy()

            # Whisper expects a float32 numpy array at 16kHz, padded/trimmed to 30s
            audio_np = audio_np.astype(np.float32)
            audio_padded = whisper.pad_or_trim(audio_np)

            # FIX 1: whisper-large-v3 uses 128 mel bins (not 80). Use n_mels from the
            # loaded model's own dims to make this work for any model size.
            mel = whisper.log_mel_spectrogram(
                audio_padded, n_mels=self.model.dims.n_mels
            ).to(self.device)

            # FIX 2: detect_language() returns a LIST of dicts (one per batch item),
            # not a single dict. Index [0] to get the {lang: prob} dict for our segment.
            with torch.no_grad():
                _, probs_list = self.model.detect_language(mel.unsqueeze(0))

            probs = probs_list[0]   # dict like {'en': 0.85, 'ml': 0.07, ...}
            top_lang = max(probs, key=probs.get)
            top_prob = float(probs[top_lang])

            # Convert to our 3-letter code
            mapped = self._WHISPER_TO_FT_CODE.get(top_lang, top_lang)
            return mapped, top_prob

        except Exception as e:
            # Surface the error so it can be debugged rather than silently failing
            print(f"  [WhisperCrossChecker] ERROR: {type(e).__name__}: {e}")
            return None, 0.0
