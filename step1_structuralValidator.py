import collections

class RTTMValidator:
    def __init__(self, 
                 short_segment_threshold=0.2, 
                 density_window=10.0, 
                 density_max_segments=20,
                 audio_duration_map=None):
        """
        Initializes the validator with configurable thresholds.
        
        :param short_segment_threshold: Minimum valid duration in seconds.
        :param density_window: Window size in seconds to check for density anomalies.
        :param density_max_segments: Max allowed segments within the density window.
        :param audio_duration_map: Dict mapping file_id to total audio duration in seconds.
        """
        self.short_segment_threshold = short_segment_threshold
        self.density_window = density_window
        self.density_max_segments = density_max_segments
        self.audio_duration_map = audio_duration_map or {}
        
        self.errors = []
        self.warnings = []
        self.count = 0
        self.countover = 0

    def _anomaly(self, line, severity, message, confidence=None):
        """Constructs a standardized anomaly dictionary."""
        return {
            "line": line,
            "severity": severity,
            "confidence": confidence,
            "message": message
        }

    def validate_file(self, filepath):
        self.errors = []
        self.warnings = []
        self.count = 0
        self.countover = 0

        segments_by_file = collections.defaultdict(list)
        
        # Step 1: Syntax and Segment-Level Checks
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith(';'):
                    continue  # Skip empty lines or comments
                
                segment = self._parse_and_validate_syntax(line, line_num)
                if segment:
                    self._validate_segment_logic(segment, line_num)
                    segments_by_file[segment['file_id']].append((line_num, segment))

        # Step 2: File-Level Checks (Overlaps, Gaps, Consistency, Density)
        for file_id, segments in segments_by_file.items():
            self._validate_file_level(file_id, segments)

        score = max(0, 100 - (len(self.errors) * 5) - (len(self.warnings) * 1))
        
        return {
            "is_valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "short_segment_count": self.count,
            "overlaped segments": self.countover,
            "score": score
        }

    def _parse_and_validate_syntax(self, line, line_num):
        parts = line.split()
        
        # 1. RTTM Syntax Check (Adapted for your 10-column LANGUAGE format)
        if len(parts) not in [9, 10]:
            self.errors.append(self._anomaly(
                line_num, "SEVERE",
                f"Invalid column count. Expected 9 or 10, got {len(parts)}."
            ))
            return None
            
        if parts[0] not in ["SPEAKER", "LANGUAGE"]:
            self.errors.append(self._anomaly(
                line_num, "SEVERE",
                f"Invalid type. Expected 'SPEAKER' or 'LANGUAGE', got '{parts[0]}'."
            ))
            return None

        try:
            start_time = float(parts[3])
            duration = float(parts[4])
        except ValueError:
            self.errors.append(self._anomaly(
                line_num, "SEVERE",
                "Start time and duration must be numeric floats."
            ))
            return None

        return {
            "type": parts[0],
            "file_id": parts[1],
            "channel": parts[2],
            "start": start_time,
            "duration": duration,
            "end": start_time + duration,
            "ortho": parts[5],
            "subtype": parts[6],
            "language": parts[7],
            "speaker_id": parts[7],
            "confidence": parts[8] if len(parts) > 8 else "<NA>"
        }

    def _validate_segment_logic(self, seg, line_num):
        # 3. Impossible timestamps
        if seg['start'] < 0:
            self.errors.append(self._anomaly(
                line_num, "SEVERE",
                f"Impossible timestamp. Start time is negative ({seg['start']})."
            ))
            
        # 4. Negative durations
        if seg['duration'] <= 0:
            self.errors.append(self._anomaly(
                line_num, "SEVERE",
                f"Invalid duration. Duration is <= 0 ({seg['duration']})."
            ))

        # 6. Out-of-bounds timestamps
        max_dur = self.audio_duration_map.get(seg['file_id'])
        if max_dur and seg['end'] > max_dur:
            self.errors.append(self._anomaly(
                line_num, "SEVERE",
                f"Out of bounds. Segment ends at {seg['end']}s, audio length is {max_dur}s."
            ))

        # 9. Extremely short segments
        if 0 < seg['duration'] < self.short_segment_threshold:
            self.errors.append(self._anomaly(
                line_num, "MODERATE",
                f"Extremely short segment detected ({seg['duration']}s). Skipped in acoustic/language validation."
            ))
            self.count += 1

    def _validate_file_level(self, file_id, segments):
        # Sort by start time for sequential checks
        segments.sort(key=lambda x: x[1]['start'])
        
        speaker_history = collections.defaultdict(list)
        language_map = {}
        
        for i in range(len(segments)):
            line_num, seg = segments[i]
            
            # 8. Language Tag Consistency
            spk = seg['speaker_id']
            lang = seg['language']
            if spk in language_map and language_map[spk] != lang and lang != "<NA>":
                self.errors.append(self._anomaly(
                    line_num, "SEVERE",
                    f"Language tag consistency error. Speaker '{spk}' was previously tagged as '{language_map[spk]}' but is now '{lang}'."
                ))
            elif lang != "<NA>":
                language_map[spk] = lang

            # Cross-segment checks with previous segments
            if i > 0:
                prev_line_num, prev_seg = segments[i-1]
                
                # 5. Gaps (Warning if larger than expected)
                gap = seg['start'] - prev_seg['end']
                if gap > 10.0:
                    self.warnings.append(self._anomaly(
                        line_num, "LOW",
                        f"Large gap detected ({gap:.2f}s) between Lines {prev_line_num} and {line_num}."
                    ))

                # 2. Segment overlaps (General overlapping speech)
                # Cross-speaker overlaps are normal in real conversations (interruptions, backchannels).
                # We count them for informational purposes only but do NOT penalize the score.
                if seg['start'] < prev_seg['end']:
                    self.countover += 1

            # 7. Speaker Duplication (Same speaker overlaps themselves)
            for prev_spk_line, prev_spk_seg in speaker_history[spk]:
                if prev_spk_seg['end'] > seg['start']:
                    self.errors.append(self._anomaly(
                        line_num, "SEVERE",
                        f"Speaker duplication. Speaker '{spk}' overlaps themselves between Lines {prev_spk_line} and {line_num}."
                    ))
            
            speaker_history[spk].append((line_num, seg))

        # 10. Annotation density anomalies
        self._check_density(segments)

    def _check_density(self, sorted_segments):
        """Checks if there are too many segments packed into a small time window."""
        for i in range(len(sorted_segments)):
            window_start = sorted_segments[i][1]['start']
            window_end = window_start + self.density_window
            
            count = 0
            for j in range(i, len(sorted_segments)):
                if sorted_segments[j][1]['start'] <= window_end:
                    count += 1
                else:
                    break
                    
            if count > self.density_max_segments:
                line_nums = [str(sorted_segments[k][0]) for k in range(i, i+count)]
                self.warnings.append(self._anomaly(
                    int(line_nums[0]), "MODERATE",
                    f"Density anomaly: {count} segments found between {window_start}s and {window_end}s (Lines {line_nums[0]} to {line_nums[-1]})."
                ))
