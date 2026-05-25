import torch
import collections
from core_validators import SpeakerVerificationSystem

# Severity thresholds for Speaker Cosine Similarity
SEVERE_THRESHOLD = 0.10   # Below this = AI is certain it's a different person
PASSING_THRESHOLD = 0.25  # At or above this = match

class AcousticValidator:
    def __init__(self):
        self.anomalies = []
        self.warnings = []
        self.verifier_system = SpeakerVerificationSystem(similarity_threshold=PASSING_THRESHOLD)
        
    def _slice_tensor(self, full_waveform, sample_rate, start_sec, duration_sec):
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
        error_lines = set(err['line'] for err in struct_results['errors'] if isinstance(err, dict) and 'line' in err)
        speaker_segments = collections.defaultdict(list)
        
        # 1. Parse valid segments
        with open(rttm_filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num in error_lines:
                    continue 
                
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    speaker_id = parts[7]
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker_segments[speaker_id].append({
                        "line": line_num, "start": start, "duration": duration
                    })

        # Load audio file entirely into memory once to avoid disk I/O bottlenecks
        if not audio_filepath:
            self.warnings.append("No audio file provided for Speaker Validation.")
            return self._build_result(0, 0, 0.0)
            
        try:
            import soundfile as sf
            waveform_np, sample_rate = sf.read(audio_filepath)
            if len(waveform_np.shape) == 1: # Mono
                full_waveform = torch.from_numpy(waveform_np).unsqueeze(0).float()
            else: # Multi-channel
                full_waveform = torch.from_numpy(waveform_np).T.float()
        except Exception as e:
            self.warnings.append(f"Could not load audio file '{audio_filepath}': {e}")
            return self._build_result(0, 0, 0.0)

        from tqdm import tqdm
        
        # 2. Verify Speakers with Severity Matrix
        total_comparisons = 0
        successful_comparisons = 0
        total_similarity = 0.0
        
        total_test_segments = sum(len(segs) - 1 for segs in speaker_segments.values() if len(segs) >= 2)
        
        with tqdm(total=total_test_segments, desc="Validating Speaker Segments") as pbar:
            for speaker_id, segments in speaker_segments.items():
                if len(segments) < 2:
                    continue 
                
                # Sort by start time to find the first occurrence
                segments.sort(key=lambda x: x['start'])
                
                anchor_seg = segments[0]
                anchor_wave = self._slice_tensor(full_waveform, sample_rate, anchor_seg['start'], anchor_seg['duration'])
                
                if anchor_wave.size(1) == 0:
                    self.warnings.append(f"Speaker {speaker_id} anchor segment is empty.")
                    pbar.update(len(segments) - 1)
                    continue
    
                for test_seg in segments[1:]:
                    test_wave = self._slice_tensor(full_waveform, sample_rate, test_seg['start'], test_seg['duration'])
                    if test_wave.size(1) == 0:
                        pbar.update(1)
                        continue
                    
                    is_match, sim_score = self.verifier_system.compare_speakers(anchor_wave, test_wave)
                    
                    total_comparisons += 1
                    total_similarity += sim_score
                    
                    if is_match:
                        successful_comparisons += 1
                        # Dynamic Anchoring: Update anchor to this successful match
                        anchor_seg = test_seg
                        anchor_wave = test_wave
                    else:
                        # --- Severity Matrix ---
                        if sim_score < SEVERE_THRESHOLD:
                            severity = "SEVERE"
                            detail = "AI is mathematically certain a different human is speaking."
                        else:
                            severity = "MODERATE"
                            detail = "AI suspects a mismatch, but could be due to noise, cough, or cross-talk."
                        
                        self.anomalies.append(self._anomaly(
                            line=test_seg['line'],
                            severity=severity,
                            confidence=sim_score,
                            message=(
                                f"Voice mismatch for Speaker {speaker_id}. "
                                f"Does not match previous valid occurrence at {anchor_seg['start']}s. "
                                f"(Cosine Similarity: {sim_score:.4f}). {detail}"
                            )
                        ))
                    pbar.update(1)

        return self._build_result(total_comparisons, successful_comparisons, total_similarity)

    def _build_result(self, total_comparisons, successful_comparisons, total_similarity):
        agreement_rate = successful_comparisons / total_comparisons if total_comparisons > 0 else 1.0
        avg_confidence = total_similarity / total_comparisons if total_comparisons > 0 else 0.0

        return {
            "anomalies": self.anomalies,
            "warnings": self.warnings,
            "agreement_rate": agreement_rate,
            "avg_confidence": avg_confidence,
            "total_comparisons": total_comparisons,
            "severe_count": sum(1 for a in self.anomalies if a['severity'] == 'SEVERE'),
            "moderate_count": sum(1 for a in self.anomalies if a['severity'] == 'MODERATE'),
        }