import torch
import numpy as np
import collections
from core_validators import LanguageIdentificationSystem, WhisperCrossChecker

# Severity thresholds for Fine-Tuned Language Whisper Confidence
# NOTE: The fine-tuned model has softmax over 12 classes (not 99), so probabilities
# are naturally more concentrated. Threshold is set higher accordingly.
HIGH_CONFIDENCE_THRESHOLD = 0.90  # Above this = AI is very sure about its prediction
MIN_TEST_DURATION = 3.0           # Skip segments shorter than this (seconds).
                                  # 2s is the absolute floor for LID but short telephony
                                  # clips at low SNR produce random predictions in any model.
                                  # 3s is a safer minimum for reliable language identification.

# Cross-check agreement threshold for stock Whisper.
# Only trust Whisper's verdict when it reaches this confidence.
# Below this threshold, Whisper is too uncertain to reliably override the fine-tuned model
# (e.g., telephony audio often gets low-confidence, wrong-language predictions like
# Spanish/Tagalog for Indian speech — those should NOT dismiss real anomalies).
CROSS_CHECK_WHISPER_MIN_CONFIDENCE = 0.50

# When Whisper's top-1 confidence is below this floor, it is so uncertain that
# it cannot identify ANY clear language in the segment. For normal single-language
# Indian audio, Whisper (trained on 680K hours) reliably gives >= 50% on a known
# language. Extreme uncertainty (< 35%) is almost always caused by code-switching,
# background noise, or genuinely mixed content.
# In this case, the SEVERE label is downgraded to AMBIGUOUS: both models are
# confused, so treating the segment as a definitive annotation error is wrong.
WHISPER_VERY_LOW_CONFIDENCE = 0.35

# ── Anchor Election Quality Controls ───────────────────────────────────────────
#
# Fix 1 — Confidence floor: only predictions at or above this threshold
# participate in the anchor vote. Low-confidence predictions (noisy 3-4s clips
# at ~55% confidence) add noise and can swing the election to the wrong language.
ANCHOR_ELECTION_MIN_CONF = 0.70

# Fix 2 — Weak-majority skip: if the elected anchor's weighted share falls
# below this ratio, the anchor itself is unreliable. Phase-2 comparisons are
# skipped entirely for that ID rather than producing false alarms against a bad anchor.
ANCHOR_WEAK_MAJORITY_THRESHOLD = 0.60

# Fix 5 — Whisper anchor verification: when the anchor ratio is between
# ANCHOR_WEAK_MAJORITY_THRESHOLD and this value, run Whisper on the single
# most-confident segment to independently confirm the anchor language before
# starting phase-2. If Whisper confidently disagrees, it overrides the anchor.
ANCHOR_WHISPER_VERIFY_RATIO = 0.80

# Languages that the fine-tuned model can actually predict (mirrors _FT_LABEL_CLASSES
# in core_validators.py). Whisper knows 99 languages; if it predicts 'ur', 'es', 'tl'
# etc., the fine-tuned model can NEVER match it in phase-2 → 100% false-positive rate
# for that label ID. Anchor overrides are silently rejected for out-of-vocabulary langs.
_FT_KNOWN_LANGUAGES = frozenset([
    "asm", "ben", "eng", "guj", "hin", "kan", "mal", "mar", "odi", "pun", "tam", "tel"
])

# How many top-confident segments Whisper must check (and agree on) before it is
# allowed to override the fine-tuned anchor. A single-segment check is too fragile —
# one misidentified clip corrupts the anchor for the entire label ID.
ANCHOR_WHISPER_VERIFY_TOP_N = 3

# ── Split-Half Code-Switch Detection ───────────────────────────────────────────
# When a segment is long enough, split it in half and run LID on each half.
# If the two halves predict different languages the segment likely contains a
# genuine language switch and should not be treated as an annotation error.
# Each half must be >= MIN_TEST_DURATION (3.0s) for reliable LID, so the full
# segment must be >= 2 × MIN_TEST_DURATION.
MIN_SPLIT_DURATION = MIN_TEST_DURATION * 2   # 6.0 seconds

# Confidence gap threshold for short-segment entropy-based code-switch detection.
# When the fine-tuned model's top-1 and top-2 softmax probabilities are within this
# margin, the model is genuinely uncertain between two languages — a strong signal that
# the segment contains mixed content (e.g. hin=52%, mal=41% → gap=0.11 < 0.20).
# Only applied to segments that are too short to split (< MIN_SPLIT_DURATION).
AMBIGUOUS_GAP_THRESHOLD = 0.20


class LanguageValidator:
    def __init__(self):
        self.anomalies = []
        self.warnings = []
        # Fine-tuned model: primary language identification (12 Indic classes)
        self.language_system = LanguageIdentificationSystem(model_size="large")
        # Stock Whisper large-v3: second-opinion cross-checker for anomaly verification
        self.cross_checker = WhisperCrossChecker(model_name="large-v3")

    def _slice_tensor(self, full_waveform, sample_rate, start_sec, duration_sec):
        """Slices the audio tensor in memory."""
        frame_offset = int(start_sec * sample_rate)
        num_frames = int(duration_sec * sample_rate)
        return full_waveform[:, frame_offset:frame_offset + num_frames]

    def _detect_code_switch(self, wave, sample_rate, duration):
        """
        Tries multiple split points (1/3, 1/2, 2/3) and returns info about the
        FIRST split where the two parts predict different languages.

        Using multiple split points catches code-switches that happen near the
        start or end of a segment — the midpoint-only split would miss those
        because one language still dominates both halves.

        Each part must be >= MIN_TEST_DURATION for reliable LID. Skips any split
        where either part is too short.

        Returns a dict on detection:
          {
            'split_ratio': float,       # e.g. 0.33
            'h1_lang': str, 'h1_conf': float,
            'h2_lang': str, 'h2_conf': float,
            'method': 'multi_split',
          }
        Returns None if no code-switch detected or all splits are too short.
        """
        for ratio in [0.33, 0.50, 0.67]:
            left_dur  = ratio * duration
            right_dur = (1.0 - ratio) * duration
            if left_dur < MIN_TEST_DURATION or right_dur < MIN_TEST_DURATION:
                continue   # this split produces parts too short for reliable LID

            mid        = int(ratio * wave.size(1))
            left_wave  = wave[:, :mid]
            right_wave = wave[:, mid:]

            try:
                lang_l, conf_l = self.language_system.identify_language(left_wave, sample_rate)
            except Exception:
                continue

            try:
                lang_r, conf_r = self.language_system.identify_language(right_wave, sample_rate)
            except Exception:
                continue

            if lang_l != lang_r:
                return {
                    "split_ratio": ratio,
                    "h1_lang": lang_l, "h1_conf": conf_l,
                    "h2_lang": lang_r, "h2_conf": conf_r,
                    "method": "multi_split",
                }

        return None   # no code-switch found at any split point

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
        self.label_map = {}   # lang_id → majority-voted language, e.g. {"L1": "mal", "L2": "eng"}

        # Extract error lines from structural results (now dict-based)
        error_lines = set(err['line'] for err in struct_results['errors'] if isinstance(err, dict) and 'line' in err)
        language_segments = collections.defaultdict(list)

        # 1. Parse valid segments
        with open(rttm_filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num in error_lines:
                    continue

                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "LANGUAGE":
                    lang_id = parts[7]   # e.g., L1, L2
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
                full_waveform_np = waveform_np.astype(np.float32)
            else:
                full_waveform = torch.from_numpy(waveform_np).T.float()
                # Convert to mono for cross-checker
                full_waveform_np = waveform_np.mean(axis=1).astype(np.float32)
        except Exception as e:
            self.warnings.append(f"Could not load audio file: {e}")
            return self._build_result(0, 0)

        from tqdm import tqdm

        # 2. Verify Languages with Majority-Vote Anchor Strategy
        #
        # WHY MAJORITY VOTE instead of a single/concatenated anchor:
        # The old approach built an anchor by concatenating the first N seconds of
        # (often very short, <1s) segments. This produced a garbled, unnatural audio
        # clip that the model misidentified — poisoning ALL comparisons for that ID.
        # Majority vote runs LID on every qualified segment independently, elects the
        # most-agreed language as the reference, and flags outliers against it.
        total_comparisons = 0
        successful_comparisons = 0

        # Collect all candidates first (Phase 1 + 2), then cross-check (Phase 3)
        # candidate = (lang_id, anchor_language, anchor_votes, total_votes, ft_lang,
        #              ft_conf, seg, proposed_severity, base_detail)
        candidates = []

        total_test_segments = sum(
            sum(1 for s in segs if s['duration'] >= MIN_TEST_DURATION)
            for segs in language_segments.values()
        )

        print("\n[Phase 1+2] Fine-tuned model: majority-vote anchor + outlier detection")
        with tqdm(total=total_test_segments, desc="Fine-Tuned Model Pass") as pbar:
            for lang_id, segments in language_segments.items():
                segments.sort(key=lambda x: x['start'])

                # ── Phase 1: Elect anchor language via majority vote ──────────────
                # Run LID on all segments that are long enough to be reliable.
                qualified = [s for s in segments if s['duration'] >= MIN_TEST_DURATION]

                if len(qualified) < 2:
                    skipped = len(segments) - len(qualified)
                    self.warnings.append(
                        f"Language ID '{lang_id}': fewer than 2 qualified segments "
                        f"(>= {MIN_TEST_DURATION}s). {skipped} segment(s) skipped as too short."
                    )
                    pbar.update(len(qualified))
                    continue

                lang_votes = []   # (predicted_lang, confidence, seg)
                for seg in qualified:
                    wave = self._slice_tensor(full_waveform, sample_rate, seg['start'], seg['duration'])
                    if wave.size(1) == 0:
                        continue
                    try:
                        lang, conf = self.language_system.identify_language(wave, sample_rate)
                        lang_votes.append((lang, conf, seg))
                    except Exception as e:
                        self.warnings.append(
                            f"Line {seg['line']}: Language identification failed during "
                            f"majority-vote pass for '{lang_id}': {e}"
                        )

                if not lang_votes:
                    pbar.update(len(qualified))
                    continue

                # ── Fix 1 + 3 + 4: Improved Anchor Election ──────────────────────────
                #
                # Fix 1 — Confidence floor: only let predictions at or above
                # ANCHOR_ELECTION_MIN_CONF participate. Uncertain predictions on noisy
                # clips (e.g. 55% on a 3s telephony segment) add noise and can swing
                # the election to the wrong language.
                #
                # Fix 3 — Duration scaling: weight each vote by conf × min(dur/10s, 1).
                # A 10s segment produces a more reliable embedding than a 3s clip;
                # duration scaling prevents many short clips from numerically drowning
                # out a few long, highly-reliable segments.
                #
                # Fix 4 — Outlier-filtered pool: if too few segments pass the confidence
                # floor (< 2), fall back to the top-50% most confident segments (minimum
                # 5). This excludes the noisiest outliers without discarding all data.
                from collections import defaultdict, Counter

                # Raw count (for reporting: "X/Y segments agree")
                vote_counts = Counter(v[0] for v in lang_votes)

                # Build election pool — high-confidence predictions only
                high_conf_votes = [(l, c, s) for l, c, s in lang_votes
                                   if c >= ANCHOR_ELECTION_MIN_CONF]

                if len(high_conf_votes) >= 2:
                    election_pool = high_conf_votes
                else:
                    # Not enough high-confidence votes: fall back to top-50% by confidence
                    sorted_by_conf = sorted(lang_votes, key=lambda x: x[1], reverse=True)
                    election_pool  = sorted_by_conf[:max(5, len(sorted_by_conf) // 2)]
                    self.warnings.append(
                        f"Language ID '{lang_id}': only {len(high_conf_votes)} segment(s) met "
                        f"the confidence floor ({ANCHOR_ELECTION_MIN_CONF:.0%}). "
                        f"Falling back to top-{len(election_pool)} most-confident segments "
                        f"for anchor election."
                    )

                # Duration-scaled confidence weights for anchor election
                weight_sums = defaultdict(float)
                for lang, conf, seg in election_pool:
                    duration_weight = min(seg['duration'] / 10.0, 1.0)  # cap at 10s → 1.0
                    weight_sums[lang] += conf * duration_weight

                # Elect anchor: language with highest total confidence-duration weight
                anchor_language = max(weight_sums, key=weight_sums.get)
                anchor_weight   = weight_sums[anchor_language]
                total_weight    = sum(weight_sums.values())
                anchor_ratio    = anchor_weight / total_weight  # weighted share (0–1)

                # For human-readable reporting, keep raw vote count for anchor language
                anchor_votes = vote_counts[anchor_language]

                # Record majority-voted language for this label ID (used for CSV report)
                self.label_map[lang_id] = anchor_language

                # ── Fix 2: Skip phase-2 entirely when anchor majority is too weak ────
                # A ratio below ANCHOR_WEAK_MAJORITY_THRESHOLD means the elected anchor
                # is genuinely ambiguous (e.g. hin=35%, mal=30%, eng=20%). Flagging
                # outliers against such an anchor generates more false alarms than it
                # prevents. Skip rather than cap-to-MODERATE as before.
                if anchor_ratio < ANCHOR_WEAK_MAJORITY_THRESHOLD:
                    self.warnings.append(
                        f"Language ID '{lang_id}': weak weighted majority for '{anchor_language}' "
                        f"({anchor_votes}/{len(lang_votes)} raw votes, {anchor_ratio:.0%} "
                        f"confidence-duration weight). Anchor is unreliable — skipping "
                        f"phase-2 comparisons entirely to avoid false alarms. Label may "
                        f"cover multiple languages or model needs more training data."
                    )
                    pbar.update(len(qualified))
                    continue   # ← skip phase-2 entirely for this lang_id

                # ── Fix 5: Whisper anchor verification (moderate majority) ──────────
                # When anchor_ratio is in [ANCHOR_WEAK_MAJORITY_THRESHOLD,
                # ANCHOR_WHISPER_VERIFY_RATIO), run Whisper on the top-N most-confident
                # segments to confirm — or override — the elected anchor before phase-2.
                #
                # Three guards prevent wrong overrides:
                #   A) Only override if a MAJORITY of the top-N Whisper runs agree on
                #      the same alternative language (not just one segment).
                #   B) Reject override if Whisper's language is outside _FT_KNOWN_LANGUAGES
                #      (e.g. 'ur', 'es') — the fine-tuned model can never predict it,
                #      so using it as anchor causes 100% false-positive rate in phase-2.
                #   C) Only count Whisper predictions that meet CROSS_CHECK_WHISPER_MIN_CONFIDENCE.
                anchor_whisper_overridden = False
                if anchor_ratio < ANCHOR_WHISPER_VERIFY_RATIO:
                    # Pick top-N most confident segments from the election pool
                    top_n_votes = sorted(election_pool, key=lambda x: x[1], reverse=True)
                    top_n_votes = top_n_votes[:ANCHOR_WHISPER_VERIFY_TOP_N]

                    wh_tally  = collections.Counter()   # lang → number of confident agrees
                    wh_best_prob = {}                   # lang → highest Whisper prob seen
                    for _, _, tv_seg in top_n_votes:
                        tv_start = int(tv_seg["start"] * sample_rate)
                        tv_end   = tv_start + int(tv_seg["duration"] * sample_rate)
                        tv_slice = full_waveform_np[tv_start:tv_end]
                        wh_l, wh_p = self.cross_checker.detect(tv_slice, sample_rate)
                        if wh_l is None or wh_p < CROSS_CHECK_WHISPER_MIN_CONFIDENCE:
                            continue
                        # Guard B: reject out-of-vocabulary languages
                        if wh_l not in _FT_KNOWN_LANGUAGES:
                            self.warnings.append(
                                f"Language ID '{lang_id}': Whisper predicted '{wh_l}' "
                                f"({wh_p:.0%}) which is outside the fine-tuned model's "
                                f"12-class vocabulary — override rejected for this segment."
                            )
                            continue
                        wh_tally[wh_l] += 1
                        wh_best_prob[wh_l] = max(wh_best_prob.get(wh_l, 0.0), wh_p)

                    # Guard A: override only if Whisper majority agrees on one language
                    majority_needed = len(top_n_votes) // 2 + 1  # strict majority
                    if wh_tally:
                        best_wh_lang, best_wh_count = wh_tally.most_common(1)[0]
                        best_wh_prob = wh_best_prob[best_wh_lang]

                        if best_wh_count >= majority_needed and best_wh_lang != anchor_language:
                            # Override: Whisper majority confidently disagrees with FT anchor
                            self.warnings.append(
                                f"Language ID '{lang_id}': Whisper majority "
                                f"({best_wh_count}/{len(top_n_votes)} segments, "
                                f"best {best_wh_prob:.0%}) overrides fine-tuned anchor "
                                f"'{anchor_language}' (FT ratio {anchor_ratio:.0%}, "
                                f"{anchor_votes}/{len(lang_votes)} raw votes) → "
                                f"'{best_wh_lang}'. Anchor corrected."
                            )
                            anchor_language          = best_wh_lang
                            self.label_map[lang_id]  = anchor_language
                            anchor_votes             = vote_counts.get(anchor_language, 0)
                            anchor_whisper_overridden = True
                        else:
                            # No clear Whisper majority — keep FT anchor
                            wh_summary = ", ".join(
                                f"'{l}'×{c}" for l, c in wh_tally.most_common()
                            ) or "none (all below confidence floor or OOV)"
                            self.warnings.append(
                                f"Language ID '{lang_id}': anchor '{anchor_language}' "
                                f"({anchor_ratio:.0%} FT majority). Whisper top-"
                                f"{len(top_n_votes)} check: {wh_summary}. "
                                f"No Whisper majority — fine-tuned anchor kept."
                            )
                    else:
                        self.warnings.append(
                            f"Language ID '{lang_id}': anchor '{anchor_language}' "
                            f"({anchor_ratio:.0%} FT majority). Whisper top-"
                            f"{len(top_n_votes)} check produced no usable predictions "
                            f"(low confidence or out-of-vocabulary) — FT anchor kept."
                        )

                # ── Phase 2: Flag outliers against the elected anchor ─────────────
                # effective_threshold is always HIGH_CONFIDENCE_THRESHOLD here because
                # weak-majority IDs are now skipped entirely above (Fix 2).
                effective_threshold = HIGH_CONFIDENCE_THRESHOLD

                for ft_lang, ft_conf, seg in lang_votes:
                    pbar.update(1)
                    total_comparisons += 1

                    if ft_lang == anchor_language:
                        successful_comparisons += 1
                    else:
                        # ── Initial severity (SEVERE / MODERATE) ─────────────────────
                        if ft_conf > effective_threshold:
                            proposed_severity = "SEVERE"
                            base_detail = (
                                f"Fine-tuned model is {ft_conf:.0%} confident this is '{ft_lang}', "
                                f"not '{anchor_language}' (majority language for this ID)."
                            )
                        else:
                            proposed_severity = "MODERATE"
                            base_detail = (
                                f"Fine-tuned model detected '{ft_lang}' with {ft_conf:.0%} confidence "
                                f"against majority '{anchor_language}'."
                            )

                        # ── Code-Switch Detection: two complementary methods ──────────────────
                        #
                        # Method A — Multi-Point Split (long segments ≥ 6s):
                        #   Tries split points at 1/3, 1/2, 2/3 of the segment.
                        #   If ANY split shows the two parts predicting different
                        #   languages → AMBIGUOUS. Using 3 split points catches
                        #   code-switches near the start/end that a midpoint-only
                        #   split would miss (one language would still dominate both
                        #   halves at the midpoint).
                        #
                        # Method B — Entropy / Confidence Gap (short segments 3–6s):
                        #   Short segments can't be split reliably (halves < 3s).
                        #   Instead, re-run identify_language_with_gap() to get the
                        #   gap between top-1 and top-2 softmax probabilities.
                        #   If gap < AMBIGUOUS_GAP_THRESHOLD (0.20), the model is
                        #   genuinely torn between two languages → AMBIGUOUS.
                        code_switch_info = None

                        if seg['duration'] >= MIN_SPLIT_DURATION:
                            # Method A: multi-point split
                            wave_seg = self._slice_tensor(
                                full_waveform, sample_rate, seg['start'], seg['duration']
                            )
                            cs = self._detect_code_switch(wave_seg, sample_rate, seg['duration'])
                            if cs is not None:
                                proposed_severity = "AMBIGUOUS"
                                code_switch_info  = cs

                        elif seg['duration'] >= MIN_TEST_DURATION:
                            # Method B: entropy / confidence gap
                            wave_seg = self._slice_tensor(
                                full_waveform, sample_rate, seg['start'], seg['duration']
                            )
                            try:
                                _, _, ft_gap = self.language_system.identify_language_with_gap(
                                    wave_seg, sample_rate
                                )
                                if ft_gap < AMBIGUOUS_GAP_THRESHOLD:
                                    proposed_severity = "AMBIGUOUS"
                                    code_switch_info  = {
                                        "method":      "entropy",
                                        "ft_gap":       ft_gap,
                                        "h1_lang":      ft_lang,
                                        "h1_conf":      ft_conf,
                                        "split_ratio":  None,
                                    }
                            except Exception:
                                pass   # gap check failed — keep original severity

                        candidates.append({
                            "lang_id":         lang_id,
                            "anchor_language":  anchor_language,
                            "anchor_votes":     anchor_votes,
                            "total_votes":      len(lang_votes),
                            "ft_lang":          ft_lang,
                            "ft_conf":          ft_conf,
                            "seg":              seg,
                            "proposed_severity":proposed_severity,
                            "base_detail":      base_detail,
                            "code_switch_info": code_switch_info,
                        })

        # ── Phase 3: Cross-check all flagged candidates with stock Whisper ───────
        # When Whisper is confident (>= CROSS_CHECK_WHISPER_MIN_CONFIDENCE), it takes
        # authority over the fine-tuned model's prediction:
        #
        #   Fine-tuned  │  Whisper (confident)      │  Outcome
        #   ────────────┼───────────────────────────┼─────────────────────────────────────────
        #   ≠ anchor    │  == anchor                │  DISMISS as SUCCESS — Whisper says it
        #               │                           │  IS the anchor lang; fine-tuned was wrong
        #   ≠ anchor    │  ≠ anchor (any language)  │  USE Whisper's language as the detected
        #               │                           │  language in the report (more reliable)
        #   ≠ anchor    │  low confidence (< 50%)   │  Keep fine-tuned prediction, mark inconclusive
        #   ≠ anchor    │  failed                   │  Keep fine-tuned prediction, note failure
        #
        # WHY: Stock Whisper (680K hours, purpose-built multilingual) is more reliable
        # than the fine-tuned model for conversational telephony audio. When it speaks
        # with >= 50% confidence, trust it over the fine-tuned model.

        if candidates:
            print(f"\n[Phase 3] Stock Whisper cross-check on {len(candidates)} flagged segment(s)...")
            with tqdm(total=len(candidates), desc="Whisper Cross-Check") as pbar:
                for c in candidates:
                    seg = c["seg"]
                    anchor_language = c["anchor_language"]
                    ft_lang = c["ft_lang"]
                    ft_conf = c["ft_conf"]
                    proposed_severity = c["proposed_severity"]
                    base_detail = c["base_detail"]

                    # Slice raw numpy audio for Whisper
                    frame_start = int(seg["start"] * sample_rate)
                    frame_end = frame_start + int(seg["duration"] * sample_rate)
                    audio_slice = full_waveform_np[frame_start:frame_end]

                    wh_lang, wh_prob = self.cross_checker.detect(audio_slice, sample_rate)
                    pbar.update(1)

                    code_switch_info = c["code_switch_info"]

                    # ── AMBIGUOUS fast-path: code-switch already detected ─────────────
                    # Split-half check in Phase 2 found that the two halves of this
                    # segment predicted different languages. We still run Whisper to
                    # gather evidence, but severity is locked to AMBIGUOUS regardless
                    # of Whisper's verdict — the segment is genuinely bilingual.
                    if code_switch_info is not None:
                        cs = code_switch_info
                        wh_lang, wh_prob = self.cross_checker.detect(audio_slice, sample_rate)
                        pbar.update(1)

                        if wh_lang and wh_prob >= CROSS_CHECK_WHISPER_MIN_CONFIDENCE:
                            wh_note = f"Whisper cross-check: '{wh_lang}' ({wh_prob:.0%}). "
                        else:
                            wh_note = "Whisper cross-check: inconclusive. "

                        # Dismiss if Whisper says this IS the anchor language
                        if wh_lang == anchor_language and wh_prob >= CROSS_CHECK_WHISPER_MIN_CONFIDENCE:
                            successful_comparisons += 1
                            self.warnings.append(
                                f"Line {seg['line']} (ID '{c['lang_id']}'): Code-switch signal "
                                f"detected but Whisper authoritatively confirms "
                                f"'{anchor_language}' ({wh_prob:.0%}) as dominant. "
                                f"DISMISSED — counted as MATCH."
                            )
                            continue

                        # Build detection-method-specific detail
                        if cs.get("method") == "entropy":
                            cs_detail = (
                                f"Short-segment entropy check: model confidence gap between "
                                f"top-2 languages is only {cs['ft_gap']:.0%} "
                                f"(threshold: {AMBIGUOUS_GAP_THRESHOLD:.0%}). "
                                f"Model is nearly tied between two languages — likely mixed content."
                            )
                        else:   # multi_split
                            pct_l = int(cs['split_ratio'] * 100)
                            pct_r = 100 - pct_l
                            cs_detail = (
                                f"Multi-point split ({pct_l}% / {pct_r}%): "
                                f"first part → '{cs['h1_lang']}' ({cs['h1_conf']:.0%}), "
                                f"second part → '{cs['h2_lang']}' ({cs['h2_conf']:.0%})."
                            )

                        self.anomalies.append(self._anomaly(
                            line=seg["line"],
                            severity="AMBIGUOUS",
                            confidence=ft_conf,
                            message=(
                                f"Possible Code-Switch in ID '{c['lang_id']}'. "
                                f"Majority language is '{anchor_language}' "
                                f"({c['anchor_votes']}/{c['total_votes']} segments agree). "
                                f"{cs_detail} {wh_note}"
                                f"Segment may contain two languages — review before treating "
                                f"as an annotation error."
                            )
                        ))
                        continue

                    # ── Normal Whisper cross-check (no code-switch detected) ───────────
                    if wh_lang is None:
                        # ── Cross-check failed ────────────────────────────────────────
                        final_severity = proposed_severity
                        detected_lang  = ft_lang
                        detected_conf  = ft_conf
                        verdict = (
                            f"Detected '{detected_lang}' ({detected_conf:.0%}, fine-tuned model). "
                            f"Cross-check unavailable (Whisper failed on this segment). "
                            f"Likely a human annotation error (wrong language tag)."
                        )

                    elif wh_prob < CROSS_CHECK_WHISPER_MIN_CONFIDENCE:
                        # ── Whisper not confident enough to override FT model ─────────

                        if wh_prob < WHISPER_VERY_LOW_CONFIDENCE and proposed_severity == "SEVERE":
                            # ── Mutual confusion: Whisper is extremely uncertain ────────
                            # For clear single-language Indian audio, Whisper reliably gives
                            # >= 50% on a known language. When it falls below 35% (e.g.,
                            # predicts Italian at 24% on Hindi+English audio), it means
                            # neither model can cleanly identify the segment. This is a
                            # strong indicator of code-switching or mixed-language content.
                            # Downgrade SEVERE → AMBIGUOUS: both models are confused, so
                            # this is not safe to call a definitive annotation error.
                            final_severity = "AMBIGUOUS"
                            detected_lang  = ft_lang
                            detected_conf  = ft_conf
                            verdict = (
                                f"Detected '{detected_lang}' ({detected_conf:.0%}, fine-tuned model). "
                                f"However, Stock Whisper is extremely uncertain (predicted "
                                f"'{wh_lang}' at only {wh_prob:.0%} — well below the "
                                f"{WHISPER_VERY_LOW_CONFIDENCE:.0%} floor for reliable detection). "
                                f"When Whisper cannot identify any language with confidence, the "
                                f"segment is likely code-switched or mixed-language. The fine-tuned "
                                f"model may be mapping mixed content (e.g. hin+eng) onto a "
                                f"linguistically adjacent class (e.g. pun). "
                                f"Review manually — do NOT treat as a definitive annotation error."
                            )
                        else:
                            # Whisper between WHISPER_VERY_LOW_CONFIDENCE and
                            # CROSS_CHECK_WHISPER_MIN_CONFIDENCE — unreliable but not
                            # extreme enough to change the severity.
                            final_severity = proposed_severity
                            detected_lang  = ft_lang
                            detected_conf  = ft_conf
                            verdict = (
                                f"Detected '{detected_lang}' ({detected_conf:.0%}, fine-tuned model). "
                                f"Cross-check inconclusive: Stock Whisper predicted '{wh_lang}' "
                                f"with only {wh_prob:.0%} confidence (minimum required: "
                                f"{CROSS_CHECK_WHISPER_MIN_CONFIDENCE:.0%}). "
                                f"Whisper unreliable on this audio. Likely annotation error."
                            )

                    elif wh_lang == anchor_language:
                        # ── Whisper is confident AND says it IS the anchor language ──
                        successful_comparisons += 1
                        self.warnings.append(
                            f"Line {seg['line']} (ID '{c['lang_id']}'): Fine-tuned predicted "
                            f"'{ft_lang}' ({ft_conf:.0%}) against anchor '{anchor_language}', "
                            f"but Stock Whisper (large-v3) authoritatively says this IS "
                            f"'{anchor_language}' ({wh_prob:.0%}). "
                            f"DISMISSED as fine-tuned domain error — segment counted as MATCH."
                        )
                        continue

                    else:
                        # ── Whisper is confident AND says it is NOT the anchor language ──
                        final_severity = proposed_severity
                        detected_lang  = wh_lang
                        detected_conf  = wh_prob
                        ft_note = (
                            f"Fine-tuned also disagrees (predicted '{ft_lang}' / {ft_conf:.0%})."
                            if ft_lang != wh_lang else
                            f"Fine-tuned agrees: also predicted '{ft_lang}' ({ft_conf:.0%})."
                        )
                        verdict = (
                            f"Detected language: '{detected_lang}' "
                            f"(Stock Whisper large-v3: {detected_conf:.0%} confidence). "
                            f"{ft_note} "
                            f"This segment appears to be '{detected_lang}', not '{anchor_language}'. "
                            f"Likely a human annotation error (wrong language tag)."
                        )

                    self.anomalies.append(self._anomaly(
                        line=seg["line"],
                        severity=final_severity,
                        confidence=ft_conf,
                        message=(
                            f"Language Mismatch for ID '{c['lang_id']}'. "
                            # If anchor was Whisper-overridden (0 FT votes), make that explicit
                            # so the reader understands why anchor_votes is 0.
                            + (
                                f"Whisper-elected anchor is '{anchor_language}' "
                                f"(fine-tuned model never predicted this language for this ID "
                                f"— Whisper override applied). "
                                if c['anchor_votes'] == 0 else
                                f"Majority language is '{anchor_language}' "
                                f"({c['anchor_votes']}/{c['total_votes']} segments agree). "
                            )
                            + verdict
                        )
                    ))

        return self._build_result(total_comparisons, successful_comparisons)

    def _build_result(self, total_comparisons, successful_comparisons):
        agreement_rate = successful_comparisons / total_comparisons if total_comparisons > 0 else 1.0

        # F1 Score: All comparisons are within the same language ID (ground truth = positive).
        # TP = AI correctly confirms a match, FN = AI flags a mismatch, FP = 0 by design.
        # F1 = 2*TP / (2*TP + FP + FN) => 2*TP / (2*TP + FN)
        tp = successful_comparisons
        fn = total_comparisons - successful_comparisons
        f1_score = (2 * tp) / (2 * tp + fn) if (2 * tp + fn) > 0 else 1.0

        return {
            "anomalies":      self.anomalies,
            "warnings":       self.warnings,
            "agreement_rate": agreement_rate,
            "total_comparisons": total_comparisons,
            "f1_score":       f1_score,
            "severe_count":    sum(1 for a in self.anomalies if a['severity'] == 'SEVERE'),
            "moderate_count":  sum(1 for a in self.anomalies if a['severity'] == 'MODERATE'),
            "ambiguous_count": sum(1 for a in self.anomalies if a['severity'] == 'AMBIGUOUS'),
            "label_map":       dict(self.label_map),   # e.g. {"L1": "mal", "L2": "eng"}
        }