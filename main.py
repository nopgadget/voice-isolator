#!/usr/bin/env python3
"""
Target-Speaker Extraction with Overlap Handling

Pipeline:
1) Load enrollment clip (target speaker) and compute its speaker embedding.
2) Slide over the input audio (e.g., 15s window, 5s hop).
3) For each window, run blind source separation (2-speaker) using Asteroid.
4) For each separated source, compute an embedding and compare to the target embedding.
5) Keep the source with the higher cosine similarity (if above threshold).
6) Concatenate all kept windows to produce a single 'target_only.wav'.

Requirements:
  pip install torch torchaudio numpy soundfile librosa tqdm asteroid speechbrain

Optional (faster with GPU if available).
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import torch
import torchaudio
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

# Asteroid separation model loader
from asteroid.models import BaseModel

# SpeechBrain embedding model
from speechbrain.pretrained import EncoderClassifier


def check_ffmpeg():
    """Check if ffmpeg is available in the system PATH."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def convert_audio_with_ffmpeg(input_path, output_path, target_sr=24000):
    """Convert audio to WAV format using ffmpeg."""
    cmd = [
        'ffmpeg', '-i', input_path,
        '-ar', str(target_sr),
        '-ac', '1',  # mono
        '-c:a', 'pcm_s16le',  # 16-bit PCM
        '-y',  # overwrite output
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] FFmpeg conversion failed: {e}")
        return False


def load_wav_mono(path, sr=24000):
    """Load audio file with fallback to ffmpeg for unsupported formats."""
    # First try torchaudio directly
    try:
        wav, file_sr = torchaudio.load(path)
        # Convert to mono
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        # Resample if needed
        if file_sr != sr:
            wav = torchaudio.functional.resample(wav, orig_freq=file_sr, new_freq=sr)
        return wav.squeeze(0), sr
    except Exception as e:
        print(f"[!] Torchaudio failed to load {path}: {e}")
        
        # Fallback to ffmpeg if available
        if check_ffmpeg():
            print(f"[+] Attempting conversion with ffmpeg...")
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            if convert_audio_with_ffmpeg(path, tmp_path, sr):
                try:
                    wav, file_sr = torchaudio.load(tmp_path)
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                    # Convert to mono and resample if needed
                    if wav.shape[0] > 1:
                        wav = torch.mean(wav, dim=0, keepdim=True)
                    if file_sr != sr:
                        wav = torchaudio.functional.resample(wav, orig_freq=file_sr, new_freq=sr)
                    return wav.squeeze(0), sr
                except Exception as e2:
                    print(f"[!] Failed to load converted file: {e2}")
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            else:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            print("[!] FFmpeg not found. Please install ffmpeg to support additional audio formats.")
            print("[!] Supported formats by torchaudio: WAV, FLAC, MP3, OGG, M4A (with codecs)")
        
        raise RuntimeError(f"Could not load audio file {path} with either torchaudio or ffmpeg")


def chunk_indices(total_len, win_len, hop_len):
    """Yield (start, end) sample indices for sliding windows."""
    starts = np.arange(0, max(1, total_len - win_len + 1), hop_len)
    for s in starts:
        e = int(min(s + win_len, total_len))
        yield int(s), e
    if total_len <= win_len:
        # handled above; but ensure at least one window
        pass


def cosine_similarity(a, b, eps=1e-9):
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)
    return float(np.dot(a, b))


def get_embedding(encoder: EncoderClassifier, wav_24k: torch.Tensor, sr=24000, device="cpu"):
    """Return numpy embedding for a mono waveform tensor [T]."""
    wav_24k = wav_24k.unsqueeze(0).to(device)  # [1, T]
    with torch.no_grad():
        emb = encoder.encode_batch(wav_24k)  # [1, 1, D] or [1, D]
    emb = emb.squeeze().cpu().numpy()
    return emb


def separate_chunk(model, chunk_wav: torch.Tensor, device="cpu"):
    """
    Run separation on a mono chunk [T] -> list of separated sources [S x T].
    """
    # Model expects [batch, time]
    with torch.no_grad():
        est_sources = model.forward(chunk_wav.unsqueeze(0).to(device))  # [1, S, T]
    est_sources = est_sources.squeeze(0).cpu()  # [S, T]
    return [est_sources[i] for i in range(est_sources.shape[0])]


def main():
    ap = argparse.ArgumentParser(description="Extract target speaker with overlap handling.")
    ap.add_argument("--audio", required=True, help="Input audio file (WAV, FLAC, MP3, OGG, M4A, etc. - ffmpeg fallback for unsupported formats).")
    ap.add_argument("--target-speaker", required=True, help="Audio clip of target speaker (5+ seconds of clean speech, longer clips like 20s work fine). Supports WAV, FLAC, MP3, OGG, M4A, etc.")
    ap.add_argument("--output", default="target_only.wav", help="Output path for extracted target audio.")
    ap.add_argument("--window-length", type=float, default=15.0, help="Analysis window length in seconds.")
    ap.add_argument("--hop-size", type=float, default=5.0, help="Hop size between windows in seconds.")
    ap.add_argument("--similarity-threshold", type=float, default=0.65, help="Cosine similarity threshold to keep a window (0.0-1.0).")
    ap.add_argument("--separation-model", default="JorisCos/ConvTasNet_Libri2Mix_sepclean_16k",
                    help="Asteroid HF model id (2-speaker separation at 16k, will be resampled to 24k).")
    ap.add_argument("--save-segments", action="store_true", help="Also save kept window clips as individual files.")
    ap.add_argument("--cpu-only", action="store_true", help="Force CPU usage (disable GPU acceleration).")
    args = ap.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not args.cpu_only) else "cpu"
    sr = 24000
    win_len = int(args.window_length * sr)
    hop_len = int(args.hop_size * sr)

    # ---- Load models
    print(f"[+] Loading separation model: {args.separation_model} on {device}")
    sep_model = BaseModel.from_pretrained(args.separation_model).to(device).eval()

    print("[+] Loading speaker embedding model (SpeechBrain ECAPA)…")
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )

    # ---- Load audio
    print(f"[+] Loading input audio: {args.audio}")
    wav, _ = load_wav_mono(args.audio, sr=sr)  # torch [T]
    total_len = wav.shape[0]

    print(f"[+] Loading target speaker audio: {args.target_speaker}")
    enroll_wav, _ = load_wav_mono(args.target_speaker, sr=sr)  # torch [T]

    # Normalize (optional but helps)
    def rms_norm(x, eps=1e-8):
        return x / (x.pow(2).mean().sqrt() + eps)

    wav = rms_norm(wav)
    enroll_wav = rms_norm(enroll_wav)

    # ---- Target embedding
    print("[+] Computing target speaker embedding…")
    target_emb = get_embedding(enc, enroll_wav, sr=sr, device=device)

    # ---- Process by windows
    kept_chunks = []
    seg_dir = None
    if args.save_segments:
        seg_dir = Path(args.output).with_suffix("").as_posix() + "_segments"
        os.makedirs(seg_dir, exist_ok=True)

    print("[+] Running separation & selection over windows…")
    for i, (s, e) in enumerate(tqdm(list(chunk_indices(total_len, win_len, hop_len)))):
        chunk = wav[s:e]  # [Twin]
        if chunk.shape[0] < 1000:
            continue  # skip tiny tail

        # Separate
        try:
            sources = separate_chunk(sep_model, chunk, device=device)  # list of [T]
        except RuntimeError as ex:
            print(f"[!] Separation failed on window {i}: {ex}")
            continue

        # Compute embeddings for each source & score
        best_sim = -1.0
        best_src = None
        for src in sources:
            emb = get_embedding(enc, src, sr=sr, device=device)
            sim = cosine_similarity(target_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best_src = src

        # Keep the best source if above threshold
        if best_sim >= args.similarity_threshold and best_src is not None:
            kept_chunks.append(best_src.cpu().numpy())

            if args.save_segments:
                seg_path = Path(seg_dir) / f"seg_{i:05d}_{s}_{e}_sim{best_sim:.3f}.wav"
                sf.write(seg_path.as_posix(), best_src.cpu().numpy(), sr, format='WAV', subtype='PCM_24')

    if not kept_chunks:
        print("[!] No windows passed the similarity threshold. Try lowering --similarity-threshold.")
        return

    # ---- Simple overlap-add style stitching
    # Since windows hop, we just concatenate the chosen target streams.
    # For a fancier reconstruction, you could cross-fade; this keeps it simple.
    target_track = np.concatenate(kept_chunks, axis=0)

    # Optional: light-weight VAD-like trimming using energy gating
    def energy_trim(x, frame=2048, hop=512, thresh_db=-45.0):
        rms = librosa.feature.rms(y=x, frame_length=frame, hop_length=hop, center=True)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        mask = librosa.util.frame(x, frame_length=frame, hop_length=hop).mean(axis=0)  # dummy to align lengths
        # Build keep mask based on rms_db threshold
        keep = np.repeat(rms_db > thresh_db, hop)
        keep = keep[:len(x)]
        return x[keep] if keep.any() else x

    # You can comment this out if you prefer raw concatenation
    # target_track = energy_trim(target_track, thresh_db=-50.0)

    # ---- Save result
    # Ensure output is saved as WAV format with 24kHz mono
    output_path = args.output
    if not output_path.lower().endswith('.wav'):
        output_path = output_path + '.wav'
    
    sf.write(output_path, target_track, sr, format='WAV', subtype='PCM_24')
    print(f"[+] Wrote extracted target audio to: {output_path} (24kHz mono WAV)")
    if args.save_segments:
        print(f"[+] Wrote kept segments to: {seg_dir}")


if __name__ == "__main__":
    main()