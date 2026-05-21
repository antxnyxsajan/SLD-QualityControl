import torch
import collections
from core_validators import LanguageIdentificationSystem

class LanguageValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        # Whisper model initialization (large is high accuracy)
        self.language_system = LanguageIdentificationSystem(model_size="large")

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
            return {"is_valid": True, "errors": self.errors, "warnings": self.warnings, "score": 100, "accuracy": 1.0, "total_comparisons": 0}
            
        try:
            import soundfile as sf
            waveform_np, sample_rate = sf.read(audio_filepath)
            if len(waveform_np.shape) == 1: 
                full_waveform = torch.from_numpy(waveform_np).unsqueeze(0).float()
            else: 
                full_waveform = torch.from_numpy(waveform_np).T.float()
        except Exception as e:
            self.warnings.append(f"Could not load audio file: {e}")
            return {"is_valid": True, "errors": self.errors, "warnings": self.warnings, "score": 0, "accuracy": 0.0, "total_comparisons": 0}

        from tqdm import tqdm
        
        # 2. Verify Languages
        total_comparisons = 0
        successful_comparisons = 0
        
        total_test_segments = sum(len(segs) - 1 for segs in language_segments.values() if len(segs) >= 2)
        
        with tqdm(total=total_test_segments, desc="Validating Language Segments") as pbar:
            for lang_id, segments in language_segments.items():
                if len(segments) < 2:
                    continue 
                    
                segments.sort(key=lambda x: x['start'])
                
                # Find the language of the first occurrence
                anchor_seg = segments[0]
                if anchor_seg['duration'] < 1.0:
                    self.warnings.append(f"Line {anchor_seg['line']}: Anchor segment too short ({anchor_seg['duration']}s) for reliable Language ID.")
                    # We'll still try
                
                anchor_wave = self._slice_tensor(full_waveform, sample_rate, anchor_seg['start'], anchor_seg['duration'])
                
                try:
                    anchor_language, anchor_confidence = self.language_system.identify_language(anchor_wave, sample_rate)
                except Exception as e:
                    self.warnings.append(f"Could not identify language for Line {anchor_seg['line']}: {e}")
                    pbar.update(len(segments) - 1)
                    continue
    
                for test_seg in segments[1:]:
                    if test_seg['duration'] < 1.0:
                        self.warnings.append(f"Line {test_seg['line']}: Test segment too short ({test_seg['duration']}s).")
                        
                    test_wave = self._slice_tensor(full_waveform, sample_rate, test_seg['start'], test_seg['duration'])
                    if test_wave.size(1) == 0:
                        pbar.update(1)
                        continue
                        
                    total_comparisons += 1
                    
                    try:
                        test_language, test_confidence = self.language_system.identify_language(test_wave, sample_rate)
                        
                        if test_language == anchor_language:
                            successful_comparisons += 1
                            # Dynamic Anchoring: Update anchor to this successful match
                            anchor_seg = test_seg
                            anchor_language = test_language
                        else:
                            self.errors.append(
                                f"Line {test_seg['line']}: Language Mismatch. Expected '{anchor_language}' (from previous valid occurrence at {anchor_seg['start']}s), "
                                f"but AI detected '{test_language}' (confidence: {test_confidence:.2f})."
                            )
                    except Exception as e:
                        self.errors.append(f"Line {test_seg['line']}: Language identification failed: {e}")
                    
                    pbar.update(1)

        accuracy = successful_comparisons / total_comparisons if total_comparisons > 0 else 1.0
        score = int(accuracy * 100)

        return {
            "is_valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "accuracy": accuracy,
            "score": score,
            "total_comparisons": total_comparisons
        }