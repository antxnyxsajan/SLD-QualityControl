import sys
import os
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

# Add the confphone folder to sys.path so we can import the model
sys.path.append(os.path.join(os.path.dirname(__file__), "LID", "confphone"))

from model import Conformer
from data_load import get_atten_mask

# ---------- Language code → integer label mapping ----------
LANG_MAP = {
    0: "asm (Assamese)",
    1: "ben (Bengali)",
    2: "eng (English)",
    3: "guj (Gujarati)",
    4: "hin (Hindi)",
    5: "kan (Kannada)",
    6: "mal (Malayalam)",
    7: "mar (Marathi)",
    8: "odi (Odia)",
    9: "pun (Punjabi)",
    10: "tam (Tamil)",
    11: "tel (Telugu)",
}

TARGET_SAMPLE_RATE = 16000
TARGET_FEATURE_DIM = 392

def load_wav2vec2(device):
    print("[INFO] Loading Wav2Vec2 Feature Extractor...")
    model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device).eval()
    return processor, model

def load_conformer(ckpt_path, device):
    print(f"[INFO] Loading trained Conformer from {ckpt_path}...")
    model = Conformer(
        input_dim=392,
        feat_dim=32,
        d_k=32,
        d_v=32,
        n_heads=4,
        d_ff=1024,
        max_len=100000,
        dropout=0.1,
        device=device,
        n_lang=12
    )
    # Load weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model

def extract_features(audio_path, processor, w2v_model, device):
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        
    # Resample
    if sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
        
    waveform = waveform.squeeze(0)
    
    # [CRITICAL OOM FIX] 
    # Conformer's attention mechanism uses O(T^2) memory. 
    # We must trim audio to max 10 seconds to avoid blowing up the GPU.
    max_length_samples = TARGET_SAMPLE_RATE * 60  # 10 seconds
    if waveform.shape[0] > max_length_samples:
        print(f"[WARNING] Audio is very long! Trimming to first 10 seconds to prevent GPU OOM.")
        waveform = waveform[:max_length_samples]
    
    # Extract features
    inputs = processor(waveform.numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    
    with torch.no_grad():
        logits = w2v_model(input_values).logits
        
    posteriors = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    
    # Adjust to exactly 392 dimensions
    vocab_size = posteriors.shape[1]
    if vocab_size > TARGET_FEATURE_DIM:
        posteriors = posteriors[:, :TARGET_FEATURE_DIM]
    elif vocab_size < TARGET_FEATURE_DIM:
        padding = np.zeros((posteriors.shape[0], TARGET_FEATURE_DIM - vocab_size), dtype=posteriors.dtype)
        posteriors = np.concatenate([posteriors, padding], axis=1)
        
    # Return as tensor with batch dim (1, T, 392)
    posteriors = torch.tensor(posteriors).unsqueeze(0).to(device, dtype=torch.float)
    return posteriors

def test_model(audio_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ckpt_path = os.path.join(os.path.dirname(__file__), "LID", "confphone", "TransformerconfPhone_iv79.ckpt")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found at {ckpt_path}")
        return
        
    w2v_processor, w2v_model = load_wav2vec2(device)
    conformer_model = load_conformer(ckpt_path, device)
    
    print(f"\n[INFO] Processing audio file: {audio_path}")
    features = extract_features(audio_path, w2v_processor, w2v_model, device)
    
    # sequence length is the time dimension (dim 1)
    seq_len = [features.size(1)]
    atten_mask = get_atten_mask(seq_len, features.size(0)).to(device)
    
    with torch.no_grad():
        outputs = conformer_model(features, atten_mask)
        probs = torch.softmax(outputs, dim=-1).squeeze(0)
        
    # Get top 3 predictions
    top_probs, top_indices = torch.topk(probs, 3)
    
    print("\n" + "="*50)
    print("🎯 PREDICTION RESULTS")
    print("="*50)
    
    for i in range(3):
        lang_idx = top_indices[i].item()
        prob = top_probs[i].item()
        lang_name = LANG_MAP.get(lang_idx, "Unknown")
        print(f"#{i+1}: {lang_name.ljust(20)} | Confidence: {prob*100:.2f}%")
        
    print("="*50 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test trained LID model on an audio file")
    parser.add_argument("audio_path", type=str, help="Path to .wav file to test")
    args = parser.parse_args()
    
    test_model(args.audio_path)
