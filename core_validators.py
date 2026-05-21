import torch
import torchaudio
import numpy as np

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


class LanguageIdentificationSystem:
    def __init__(self, model_size="large"):
        """
        Initializes the language identification system using OpenAI's Whisper model.
        Note: The 'large' model is highly accurate but requires a GPU for reasonable speed.
        """
        import whisper
        print(f"Loading Core Language Identification Model (Whisper: {model_size})...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size, device=device)
        
    def identify_language(self, audio_tensor, current_sample_rate):
        """
        Identifies the language of an audio tensor.
        Whisper strictly requires 16000 Hz sample rate.
        """
        import whisper
        
        # Ensure 1D tensor
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.mean(dim=0)
            
        # Resample to 16kHz if needed (using a cached resampler for huge speedups)
        if current_sample_rate != 16000:
            if not hasattr(self, "resampler") or self.resampler_orig_freq != current_sample_rate:
                self.resampler = torchaudio.transforms.Resample(current_sample_rate, 16000)
                self.resampler_orig_freq = current_sample_rate
            audio_tensor = self.resampler(audio_tensor)
            
        # Convert to numpy array as Whisper expects
        audio_np = audio_tensor.numpy()
        
        # Pad or trim to 30 seconds
        audio = whisper.pad_or_trim(audio_np)
        
        # Compute log mel spectrogram
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels).to(self.model.device)
        
        # Detect language
        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        confidence = probs[detected_lang]
        
        return detected_lang, confidence
