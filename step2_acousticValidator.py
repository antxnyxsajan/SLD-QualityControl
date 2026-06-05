import torch
import collections
from core_validators import SpeakerVerificationSystem

# ── Thresholds ─────────────────────────────────────────────────────────────────
# Calibrated for telephony/conversational audio (ECAPA-TDNN trained on VoxCeleb
# YouTube videos — embeddings are naturally weaker on phone-quality audio).
PASSING_THRESHOLD    = 0.20  # At or above → same speaker (was 0.25, too high for telephony)
SEVERE_THRESHOLD     = 0.05  # Below this → AI is certain different person (was 0.10)
MIN_SEGMENT_DURATION = 1.5   # Seconds — ECAPA-TDNN needs at least 1.5s for reliable embedding


class AcousticValidator:
    def __init__(self):
        self.anomalies = []
        self.warnings  = []
        self.verifier_system = SpeakerVerificationSystem(similarity_threshold=PASSING_THRESHOLD)

    def _slice_tensor(self, full_waveform, sample_rate, start_sec, duration_sec):
        frame_offset = int(start_sec * sample_rate)
        num_frames   = int(duration_sec * sample_rate)
        return full_waveform[:, frame_offset:frame_offset + num_frames]

    def _anomaly(self, line, severity, confidence, message):
        """Constructs a standardized anomaly dictionary."""
        return {
            "line":       line,
            "severity":   severity,
            "confidence": confidence,
            "message":    message
        }

    def _get_embedding(self, audio_wave):
        """
        Extracts a speaker embedding from an audio tensor using ECAPA-TDNN.
        Returns a 1-D normalised embedding tensor, or None if extraction fails.
        """
        try:
            emb = self.verifier_system.verifier.encode_batch(audio_wave)  # (1, 1, D)
            emb = emb.squeeze()                                            # (D,)
            emb = torch.nn.functional.normalize(emb, dim=0)               # L2-normalise
            return emb
        except Exception as e:
            return None

    def validate(self, rttm_filepath, audio_filepath, struct_results):
        self.anomalies = []
        self.warnings  = []

        # Extract error lines from structural results
        error_lines = set(
            err['line'] for err in struct_results['errors']
            if isinstance(err, dict) and 'line' in err
        )
        speaker_segments = collections.defaultdict(list)

        # 1. Parse valid segments from RTTM
        with open(rttm_filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num in error_lines:
                    continue
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    speaker_id = parts[7]
                    start      = float(parts[3])
                    duration   = float(parts[4])
                    speaker_segments[speaker_id].append({
                        "line": line_num, "start": start, "duration": duration
                    })

        # 2. Load audio
        if not audio_filepath:
            self.warnings.append("No audio file provided for Speaker Validation.")
            return self._build_result(0, 0, 0.0)

        try:
            import soundfile as sf
            waveform_np, sample_rate = sf.read(audio_filepath)
            if len(waveform_np.shape) == 1:
                full_waveform = torch.from_numpy(waveform_np).unsqueeze(0).float()
            else:
                full_waveform = torch.from_numpy(waveform_np).T.float()
        except Exception as e:
            self.warnings.append(f"Could not load audio file '{audio_filepath}': {e}")
            return self._build_result(0, 0, 0.0)

        from tqdm import tqdm

        # ── Phase 1: Build speaker centroids ──────────────────────────────────
        # For each speaker ID, compute a centroid embedding by averaging the
        # embeddings of all qualifying segments (duration >= MIN_SEGMENT_DURATION).
        # The centroid is a robust, noise-averaged speaker profile.
        # Using a single first-segment anchor (old approach) is fragile because
        # one noisy or short segment poisons ALL downstream comparisons.
        #
        # L2-normalised mean → equivalent to the direction of maximum likelihood
        # in a von Mises-Fisher distribution over the unit sphere (i.e. the
        # geometric mean on the hypersphere).
        print("\n[Phase 1] Building speaker centroids...")
        speaker_centroids  = {}  # speaker_id → centroid embedding tensor
        speaker_emb_counts = {}  # speaker_id → number of embeddings averaged

        total_spk_segs = sum(len(s) for s in speaker_segments.values())
        with tqdm(total=total_spk_segs, desc="Computing Embeddings") as pbar:
            for speaker_id, segments in speaker_segments.items():
                segments.sort(key=lambda x: x['start'])
                embeddings = []

                for seg in segments:
                    pbar.update(1)

                    # Skip very short segments — ECAPA-TDNN needs >= 1.5s
                    if seg['duration'] < MIN_SEGMENT_DURATION:
                        self.warnings.append(
                            f"Line {seg['line']} (Speaker {speaker_id}): segment too short "
                            f"({seg['duration']:.2f}s < {MIN_SEGMENT_DURATION}s) — "
                            f"skipped from centroid computation (unreliable embedding)."
                        )
                        continue

                    wave = self._slice_tensor(
                        full_waveform, sample_rate, seg['start'], seg['duration']
                    )
                    if wave.size(1) == 0:
                        continue

                    emb = self._get_embedding(wave)
                    if emb is not None:
                        embeddings.append(emb)

                if embeddings:
                    # Stack all embeddings and take L2-normalised mean
                    stacked  = torch.stack(embeddings, dim=0)   # (N, D)
                    centroid = stacked.mean(dim=0)               # (D,)
                    centroid = torch.nn.functional.normalize(centroid, dim=0)
                    speaker_centroids[speaker_id]  = centroid
                    speaker_emb_counts[speaker_id] = len(embeddings)
                else:
                    self.warnings.append(
                        f"Speaker '{speaker_id}': no qualifying segments found for centroid. "
                        f"All segments were shorter than {MIN_SEGMENT_DURATION}s or empty."
                    )

        # ── Phase 2: Compare each segment to its speaker centroid ─────────────
        # Each segment is independently compared against the stable centroid,
        # NOT against the previous segment (old drifting-anchor approach).
        # This removes the cascade failure mode where one bad early segment
        # causes all downstream comparisons to fail.
        print("\n[Phase 2] Comparing segments to speaker centroids...")
        total_comparisons       = 0
        successful_comparisons  = 0
        total_similarity        = 0.0

        total_test_segs = sum(len(s) for s in speaker_segments.values())
        with tqdm(total=total_test_segs, desc="Validating Speaker Segments") as pbar:
            for speaker_id, segments in speaker_segments.items():
                if speaker_id not in speaker_centroids:
                    pbar.update(len(segments))
                    continue

                centroid    = speaker_centroids[speaker_id]
                n_embs_used = speaker_emb_counts[speaker_id]

                for seg in segments:
                    pbar.update(1)

                    # Skip segments too short for reliable comparison
                    if seg['duration'] < MIN_SEGMENT_DURATION:
                        continue

                    wave = self._slice_tensor(
                        full_waveform, sample_rate, seg['start'], seg['duration']
                    )
                    if wave.size(1) == 0:
                        continue

                    emb = self._get_embedding(wave)
                    if emb is None:
                        continue

                    # Cosine similarity against speaker centroid
                    sim_score = torch.dot(emb, centroid).item()

                    total_comparisons  += 1
                    total_similarity   += sim_score

                    if sim_score >= PASSING_THRESHOLD:
                        successful_comparisons += 1
                    else:
                        # Severity matrix
                        if sim_score < SEVERE_THRESHOLD:
                            severity = "SEVERE"
                            detail   = (
                                "AI is certain a different person is speaking "
                                "(similarity far below telephony baseline)."
                            )
                        else:
                            severity = "MODERATE"
                            detail   = (
                                "AI suspects a mismatch. Could be noise, cross-talk, "
                                "or a genuine speaker confusion error."
                            )

                        self.anomalies.append(self._anomaly(
                            line=seg['line'],
                            severity=severity,
                            confidence=sim_score,
                            message=(
                                f"Voice mismatch for Speaker '{speaker_id}'. "
                                f"Segment at {seg['start']:.2f}s does not match the "
                                f"speaker centroid (averaged from {n_embs_used} segments). "
                                f"Cosine similarity: {sim_score:.4f} "
                                f"(passing threshold: {PASSING_THRESHOLD}). {detail}"
                            )
                        ))

        return self._build_result(total_comparisons, successful_comparisons, total_similarity)

    def _build_result(self, total_comparisons, successful_comparisons, total_similarity):
        agreement_rate = (
            successful_comparisons / total_comparisons if total_comparisons > 0 else 1.0
        )
        avg_confidence = (
            total_similarity / total_comparisons if total_comparisons > 0 else 0.0
        )

        tp       = successful_comparisons
        fn       = total_comparisons - successful_comparisons
        f1_score = (2 * tp) / (2 * tp + fn) if (2 * tp + fn) > 0 else 1.0

        return {
            "anomalies":         self.anomalies,
            "warnings":          self.warnings,
            "agreement_rate":    agreement_rate,
            "avg_confidence":    avg_confidence,
            "total_comparisons": total_comparisons,
            "f1_score":          f1_score,
            "severe_count":      sum(1 for a in self.anomalies if a['severity'] == 'SEVERE'),
            "moderate_count":    sum(1 for a in self.anomalies if a['severity'] == 'MODERATE'),
        }