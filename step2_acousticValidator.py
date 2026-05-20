import torchaudio
import torch
import collections
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy

class AcousticValidator:
    def __init__(self, similarity_threshold=0.25):
        """
        Initializes the acoustic validator and loads the neural network.
        SpeechBrain's ECAPA-TDNN usually uses ~0.25 as a threshold for cosine similarity.
        """
        self.similarity_threshold = similarity_threshold
        self.errors = []
        self.warnings = []
        
        print("Loading SpeechBrain ECAPA-TDNN model (this takes a moment)...")
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 1:
                device = "cuda:1"
                try:
                    device_name = torch.cuda.get_device_name(1)
                    print(f"Running acoustic validator on: {device.upper()} ({device_name})")
                except Exception:
                    print(f"Running acoustic validator on: {device.upper()}")
            else:
                device = "cuda:0"
                try:
                    device_name = torch.cuda.get_device_name(0)
                    print(f"Running acoustic validator on: {device.upper()} ({device_name}) - Note: Defaulting to cuda:0 because a second GPU was not found.")
                except Exception:
                    print(f"Running acoustic validator on: {device.upper()} - Note: Defaulting to cuda:0 because a second GPU was not found.")
        else:
            device = "cpu"
            print(f"Running acoustic validator on: {device.upper()}")
        
        # Load the pre-trained model once in memory
        self.verifier = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": device}
        )

    def _slice_tensor(self, full_waveform, sample_rate, start_sec, duration_sec):
        """Slices the audio tensor in memory."""
        frame_offset = int(start_sec * sample_rate)
        num_frames = int(duration_sec * sample_rate)
        return full_waveform[:, frame_offset:frame_offset + num_frames]

    def validate(self, rttm_filepath, audio_filepath, struct_results):
        """Runs acoustic validation using speaker embeddings."""
        self.errors = []
        self.warnings = []
        
        # Get the list of lines that had hard errors in Step 1 so we can skip them safely
        error_lines = [int(err.split(":")[0].replace("Line ", "")) for err in struct_results['errors'] if "Line" in err]
        
        speaker_segments = collections.defaultdict(list)
        
        # 1. Parse valid segments
        with open(rttm_filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num in error_lines:
                    continue # Skip structurally broken lines to prevent crashes
                
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] in ["SPEAKER", "LANGUAGE"]:
                    speaker_id = parts[7]
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker_segments[speaker_id].append({
                        "line": line_num, "start": start, "duration": duration
                    })

        # Load audio file entirely into memory once to avoid disk I/O bottlenecks
        if audio_filepath:
            try:
                import soundfile as sf
                waveform_np, sample_rate = sf.read(audio_filepath)
                # Convert to torch tensor of shape [channels, time]
                if len(waveform_np.shape) == 1: # Mono
                    full_waveform = torch.from_numpy(waveform_np).unsqueeze(0).float()
                else: # Multi-channel
                    full_waveform = torch.from_numpy(waveform_np).T.float()
            except Exception as e:
                self.warnings.append(f"Could not load audio file '{audio_filepath}': {e}")
                return {"is_valid": True, "errors": self.errors, "warnings": self.warnings}
        else:
            self.warnings.append("No audio file provided.")
            return {"is_valid": True, "errors": self.errors, "warnings": self.warnings}

        # 2. Verify Speakers
        for speaker_id, segments in speaker_segments.items():
            if len(segments) < 2:
                continue # Need at least 2 segments to compare a speaker against themselves

            # Find the longest segment to use as the "Anchor" identity
            anchor_seg = max(segments, key=lambda x: x['duration'])
            anchor_wave = self._slice_tensor(full_waveform, sample_rate, anchor_seg['start'], anchor_seg['duration'])
            
            if anchor_wave.size(1) == 0:
                self.warnings.append(f"Anchor segment for Speaker '{speaker_id}' is empty.")
                continue

            # Compare all other segments to the anchor
            for seg in segments:
                if seg['line'] == anchor_seg['line']:
                    continue # Don't compare anchor to itself
                
                test_wave = self._slice_tensor(full_waveform, sample_rate, seg['start'], seg['duration'])
                if test_wave.size(1) == 0:
                    continue

                # Get embeddings and compute cosine similarity
                score, prediction = self.verifier.verify_batch(test_wave, anchor_wave)
                sim_score = score.item()
                
                if sim_score < self.similarity_threshold:
                    self.errors.append(
                        f"Line {seg['line']}: Acoustic Mismatch. Segment at {seg['start']}s "
                        f"does not match the anchor profile for '{speaker_id}'. (Score: {sim_score:.2f})"
                    )

        return {
            "is_valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings
        }