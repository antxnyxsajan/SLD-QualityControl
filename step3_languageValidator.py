import torchaudio
import torch
import collections
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

class LanguageValidator:
    def __init__(self):
        """
        Initializes the language validator using SpeechBrain's VoxLingua107 model.
        """
        self.errors = []
        self.warnings = []
        
        print("Loading SpeechBrain VoxLingua107 LID model (this takes a moment)...")
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Running language validator on: {device.upper()}")
        
        # Load the Language Identification model
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa", 
            savedir="pretrained_models/lang-id-voxlingua107-ecapa",
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": device}
        )

    def _slice_tensor(self, full_waveform, sample_rate, start_sec, duration_sec):
        """Slices the audio tensor in memory."""
        frame_offset = int(start_sec * sample_rate)
        num_frames = int(duration_sec * sample_rate)
        return full_waveform[:, frame_offset:frame_offset + num_frames]

    def validate(self, rttm_filepath, audio_filepath, struct_results):
        self.errors = []
        self.warnings = []
        
        error_lines = [int(err.split(":")[0].replace("Line ", "")) for err in struct_results['errors'] if "Line" in err]
        language_segments = collections.defaultdict(list)
        
        # 1. Parse valid segments
        with open(rttm_filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num in error_lines:
                    continue 
                
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "LANGUAGE":
                    lang_id = parts[7] # e.g., L1, L2
                    start = float(parts[3])
                    duration = float(parts[4])
                    language_segments[lang_id].append({
                        "line": line_num, "start": start, "duration": duration
                    })

        # Load audio
        if not audio_filepath:
            self.warnings.append("No audio file provided for Language Validation.")
            return {"is_valid": True, "errors": self.errors, "warnings": self.warnings}
            
        try:
            import soundfile as sf
            waveform_np, sample_rate = sf.read(audio_filepath)
            if len(waveform_np.shape) == 1: 
                full_waveform = torch.from_numpy(waveform_np).unsqueeze(0).float()
            else: 
                full_waveform = torch.from_numpy(waveform_np).T.float()
        except Exception as e:
            self.warnings.append(f"Could not load audio file: {e}")
            return {"is_valid": True, "errors": self.errors, "warnings": self.warnings}

        # 2. Verify Languages
        for lang_id, segments in language_segments.items():
            if len(segments) < 2:
                continue 

            predicted_labels = []

            # Find out what language the AI thinks this group is
            for seg in segments:
                # We need at least 1-2 seconds of audio for accurate language ID
                if seg['duration'] < 1.0:
                    self.warnings.append(f"Line {seg['line']}: Segment too short ({seg['duration']}s) for reliable Language ID.")
                    continue

                test_wave = self._slice_tensor(full_waveform, sample_rate, seg['start'], seg['duration'])
                if test_wave.size(1) == 0:
                    continue

                # The AI classifies the language
                out_prob, score, index, text_lab = self.classifier.classify_batch(test_wave)
                prediction = text_lab[0] # e.g., 'en' for English, 'es' for Spanish
                
                predicted_labels.append((seg['line'], seg['start'], prediction))

            # Check if all predictions match the majority
            if predicted_labels:
                # Find the most common language predicted for this tag (e.g., mostly English)
                all_preds = [p[2] for p in predicted_labels]
                majority_lang = max(set(all_preds), key=all_preds.count)

                # Flag the ones that don't match the majority
                for line, start, pred in predicted_labels:
                    if pred != majority_lang:
                        self.errors.append(
                            f"Line {line}: Language Mismatch. Expected '{majority_lang}' (Majority for {lang_id}), "
                            f"but AI detected '{pred}' at {start}s."
                        )

        return {
            "is_valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings
        }