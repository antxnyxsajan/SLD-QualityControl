import torch
import collections
from core_validators import LanguageIdentificationSystem

# Severity thresholds for Language Whisper Confidence
HIGH_CONFIDENCE_THRESHOLD = 0.80  # Above this = AI is very sure about its prediction

class LanguageValidator:
    def __init__(self):
        self.anomalies = []
        self.warnings = []
        # Whisper model initialization (large is high accuracy)
        self.language_system = LanguageIdentificationSystem(model_size="large")

    def _slice_tensor(self, full_waveform, sample_rate, start_sec, duration_sec):
        """Slices the audio tensor in memory."""
        frame_offset = int(start_sec * sample_rate)
        num_frames = int(duration_sec * sample_rate)
        return full_waveform[:, frame_offset:frame_offset + num_frames]

    def _anomaly(self, line, severity, confidence, message):
        """Constructs a standardized anomaly dictionary."""
        return {
            "line": line,
            "severity": severity,
            "confidence": confidence,
            "message": message
        }

    def validate(self, rttm_filepath, audio_filepath, struct_results):
        self.anomalies = []
        self.warnings = []
        
        # Extract error lines from structural results (now dict-based)
        error_lines = [err['line'] for err in struct_results['errors'] if isinstance(err, dict) and 'line' in err]
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
            return self._build_result(0, 0)
            
        try:
            import soundfile as sf
            waveform_np, sample_rate = sf.read(audio_filepath)
            if len(waveform_np.shape) == 1: 
                full_waveform = torch.from_numpy(waveform_np).unsqueeze(0).float()
            else: 
                full_waveform = torch.from_numpy(waveform_np).T.float()
        except Exception as e:
            self.warnings.append(f"Could not load audio file: {e}")
            return self._build_result(0, 0)

        from tqdm import tqdm
        
        # 2. Verify Languages with Severity Matrix
        total_comparisons = 0
        successful_comparisons = 0
        
        total_test_segments = sum(len(segs) - 1 for segs in language_segments.values() if len(segs) >= 2)
        
        with tqdm(total=total_test_segments, desc="Validating Language Segments") as pbar:
            for lang_id, segments in language_segments.items():
                if len(segments) < 2:
                    continue 
                    
                segments.sort(key=lambda x: x['start'])
                
                # Combine first few segments until we reach >= 5.0 seconds for a reliable anchor
                anchor_seg = segments[0]
                anchor_duration = 0.0
                anchor_waves = []
                for seg in segments:
                    wave = self._slice_tensor(full_waveform, sample_rate, seg['start'], seg['duration'])
                    if wave.size(1) > 0:
                        anchor_waves.append(wave)
                        anchor_duration += seg['duration']
                    if anchor_duration >= 5.0:
                        break
                
                if anchor_duration < 1.0:
                    self.warnings.append(f"Language ID '{lang_id}': Combined anchor segment too short ({anchor_duration:.2f}s) for reliable Language ID.")
                
                if anchor_waves:
                    anchor_wave = torch.cat(anchor_waves, dim=1)
                else:
                    anchor_wave = self._slice_tensor(full_waveform, sample_rate, anchor_seg['start'], anchor_seg['duration'])
                
                try:
                    anchor_language, anchor_confidence = self.language_system.identify_language(anchor_wave, sample_rate)
                except Exception as e:
                    self.warnings.append(f"Could not identify language for anchor of '{lang_id}': {e}")
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
                            # --- Severity Matrix (Probability-Based) ---
                            if test_confidence > HIGH_CONFIDENCE_THRESHOLD:
                                severity = "SEVERE"
                                detail = (
                                    f"AI is {test_confidence:.0%} confident this is '{test_language}', not '{anchor_language}'. "
                                    f"Likely a human annotation error (wrong language tag)."
                                )
                            else:
                                severity = "MODERATE"
                                detail = (
                                    f"AI detected '{test_language}' but with only {test_confidence:.0%} confidence. "
                                    f"Segment may be code-switched, garbled, or ambiguous. Human tag might be correct."
                                )
                            
                            self.anomalies.append(self._anomaly(
                                line=test_seg['line'],
                                severity=severity,
                                confidence=test_confidence,
                                message=(
                                    f"Language Mismatch for ID '{lang_id}'. Expected '{anchor_language}' "
                                    f"(from previous valid occurrence at {anchor_seg['start']}s). {detail}"
                                )
                            ))
                    except Exception as e:
                        self.anomalies.append(self._anomaly(
                            line=test_seg['line'],
                            severity="MODERATE",
                            confidence=None,
                            message=f"Language identification failed: {e}"
                        ))
                    
                    pbar.update(1)

        return self._build_result(total_comparisons, successful_comparisons)

    def _build_result(self, total_comparisons, successful_comparisons):
        agreement_rate = successful_comparisons / total_comparisons if total_comparisons > 0 else 1.0

        return {
            "anomalies": self.anomalies,
            "warnings": self.warnings,
            "agreement_rate": agreement_rate,
            "total_comparisons": total_comparisons,
            "severe_count": sum(1 for a in self.anomalies if a['severity'] == 'SEVERE'),
            "moderate_count": sum(1 for a in self.anomalies if a['severity'] == 'MODERATE'),
        }