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
import hashlib
import time
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
from speechbrain.inference import EncoderClassifier


def generate_unique_filename(input_audio_path, target_speaker_path, base_output_dir="output"):
    """Generate a unique filename based on input files and timestamp."""
    # Get base names without extensions
    input_name = Path(input_audio_path).stem
    target_name = Path(target_speaker_path).stem
    
    # Create a hash of the input paths for uniqueness
    path_hash = hashlib.md5(f"{input_audio_path}_{target_speaker_path}".encode()).hexdigest()[:8]
    
    # Add timestamp for additional uniqueness
    timestamp = int(time.time()) % 100000  # Last 5 digits of timestamp
    
    # Create the filename
    filename = f"{input_name}_extracted_from_{target_name}_{path_hash}_{timestamp}.wav"
    
    # Ensure output directory exists
    output_path = Path(base_output_dir) / filename
    os.makedirs(output_path.parent, exist_ok=True)
    
    return output_path.as_posix()


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


def separate_chunk(model, chunk_wav: torch.Tensor, device="cpu", normalization_strength=0.1):
    """
    Run separation on a mono chunk [T] -> list of separated sources [S x T].
    """
    # Model expects [batch, time]
    with torch.no_grad():
        est_sources = model.forward(chunk_wav.unsqueeze(0).to(device))  # [1, S, T]
    
    # Ensure we get the right shape and handle potential issues
    if est_sources.dim() == 3:
        est_sources = est_sources.squeeze(0)  # Remove batch dimension
    elif est_sources.dim() == 2:
        # If we only get 2D, assume it's [S, T]
        pass
    else:
        raise RuntimeError(f"Unexpected separation output shape: {est_sources.shape}")
    
    # Convert to CPU and normalize each source to prevent extreme values
    est_sources = est_sources.cpu()
    
    # Apply light normalization to each separated source to prevent bass boost
    def light_norm_source(x, target_rms=None, eps=1e-8):
        if target_rms is None:
            target_rms = normalization_strength
        current_rms = x.pow(2).mean().sqrt()
        if current_rms > eps:
            scale_factor = min(target_rms / current_rms, 3.0)  # Max 3x boost for sources
            return x * scale_factor
        return x
    
    normalized_sources = []
    for i in range(est_sources.shape[0]):
        src = est_sources[i]
        # Check for extreme values that might indicate separation issues
        if torch.isnan(src).any() or torch.isinf(src).any():
            print(f"[!] Warning: Source {i} contains NaN or Inf values, skipping")
            continue
        
        # Apply light normalization
        src_norm = light_norm_source(src)
        normalized_sources.append(src_norm)
    
    if not normalized_sources:
        # Don't fall back to original audio - this could include non-target speech
        print(f"[!] All separation sources had issues, skipping this window")
        return []
    
    return normalized_sources


def main():
    ap = argparse.ArgumentParser(description="Extract target speaker with overlap handling.")
    ap.add_argument("--audio", required=True, help="Input audio file (WAV, FLAC, MP3, OGG, M4A, etc. - ffmpeg fallback for unsupported formats).")
    ap.add_argument("--target-speaker", required=True, help="Audio clip of target speaker (5+ seconds of clean speech, longer clips like 20s work fine). Supports WAV, FLAC, MP3, OGG, M4A, etc.")
    ap.add_argument("--output", help="Output path for extracted target audio (auto-generates unique filename if not specified).")
    ap.add_argument("--window-length", type=float, default=15.0, help="Analysis window length in seconds (larger = more coverage, default: 15.0).")
    ap.add_argument("--hop-size", type=float, default=5.0, help="Hop size between windows in seconds (default: 5.0).")
    ap.add_argument("--min-overlap", type=float, default=0.0, help="Minimum overlap between windows as fraction (0.0 = no overlap, 0.5 = 50%% overlap, default: 0.0).")
    ap.add_argument("--skip-start", type=float, default=0.0, help="Skip the first N seconds of audio (useful if target speaker doesn't start immediately, default: 0.0).")
    ap.add_argument("--similarity-threshold", type=float, default=0.80, help="Cosine similarity threshold to keep a window (0.0-1.0, higher = more strict, default: 0.85).")
    ap.add_argument("--normalization-strength", type=float, default=0.05, help="Audio normalization strength (0.05-0.3, lower = less aggressive, default: 0.1).")
    ap.add_argument("--separation-model", default="JorisCos/ConvTasNet_Libri2Mix_sepclean_16k",
                    help="Asteroid HF model id (2-speaker separation at 16k, will be resampled to 24k).")
    ap.add_argument("--save-segments", action="store_true", help="Also save kept window clips as individual files.")
    ap.add_argument("--cpu-only", action="store_true", help="Force CPU usage (disable GPU acceleration).")
    args = ap.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not args.cpu_only) else "cpu"
    sr = 24000
    win_len = int(args.window_length * sr)
    
    # Calculate hop size based on overlap preference
    if args.min_overlap > 0.0:
        # If min_overlap is specified, calculate hop size to achieve that overlap
        overlap_samples = int(args.min_overlap * win_len)
        hop_len = win_len - overlap_samples
    else:
        # Use the specified hop size
        hop_len = int(args.hop_size * sr)
    
    # Ensure hop size is at least 1 sample
    hop_len = max(1, hop_len)

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
    print(f"[+] Input audio length: {total_len/sr:.2f}s ({total_len} samples)")

    print(f"[+] Loading target speaker audio: {args.target_speaker}")
    enroll_wav, _ = load_wav_mono(args.target_speaker, sr=sr)  # torch [T]
    print(f"[+] Target speaker audio length: {enroll_wav.shape[0]/sr:.2f}s")

    # Light normalization - avoid aggressive RMS normalization that can cause bass boost
    def light_norm(x, target_rms=None, eps=1e-8):
        """Apply light normalization to avoid extreme bass boost."""
        if target_rms is None:
            target_rms = args.normalization_strength
        current_rms = x.pow(2).mean().sqrt()
        if current_rms > eps:
            # Scale to target RMS, but limit the scaling factor
            scale_factor = min(target_rms / current_rms, 5.0)  # Max 5x boost
            return x * scale_factor
        return x

    # Apply light normalization
    wav = light_norm(wav)
    enroll_wav = light_norm(enroll_wav)

    # ---- Target embedding
    print("[+] Computing target speaker embedding…")
    target_emb = get_embedding(enc, enroll_wav, sr=sr, device=device)

    # ---- Process by windows
    kept_chunks = []
    kept_timestamps = []  # Track which windows were kept
    seg_dir = None
    
    # Determine final output path early for directory creation
    final_output_path = args.output
    if not final_output_path:
        final_output_path = generate_unique_filename(args.audio, args.target_speaker)
        print(f"[+] Auto-generated output filename: {final_output_path}")
    elif not final_output_path.lower().endswith('.wav'):
        final_output_path = final_output_path + '.wav'
    
    # Create output directory
    output_dir = Path(final_output_path).parent
    if output_dir != Path('.'):
        os.makedirs(output_dir, exist_ok=True)
        print(f"[+] Created output directory: {output_dir}")
    
    if args.save_segments:
        # Create segments directory inside output directory
        seg_dir = output_dir / Path(final_output_path).stem + "_segments"
        os.makedirs(seg_dir, exist_ok=True)
        print(f"[+] Created segments directory: {seg_dir}")

    print(f"[+] Processing {total_len} samples with {win_len} sample windows, {hop_len} sample hops")
    if args.skip_start > 0:
        skip_samples = int(args.skip_start * sr)
        print(f"[+] Skipping first {args.skip_start}s ({skip_samples} samples)")
        total_len = total_len - skip_samples
        wav = wav[skip_samples:]  # Remove the skipped portion
    print(f"[+] Window length: {win_len/sr:.2f}s, Hop size: {hop_len/sr:.2f}s")
    
    window_count = 0
    kept_count = 0
    
    print("[+] Running separation & selection over windows…")
    for i, (s, e) in enumerate(tqdm(list(chunk_indices(total_len, win_len, hop_len)))):
        window_count += 1
        chunk = wav[s:e]  # [Twin]
        if chunk.shape[0] < 1000:
            continue  # skip tiny tail

        # Separate
        try:
            sources = separate_chunk(sep_model, chunk, device=device, normalization_strength=args.normalization_strength)  # list of [T]
        except RuntimeError as ex:
            print(f"[!] Separation failed on window {i}: {ex}")
            continue

        # Compute embeddings for each source & score
        best_sim = -1.0
        best_src = None
        best_source_idx = -1
        
        for src_idx, src in enumerate(sources):
            emb = get_embedding(enc, src, sr=sr, device=device)
            sim = cosine_similarity(target_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best_src = src
                best_source_idx = src_idx

        # Keep the best source if above threshold
        if best_sim >= args.similarity_threshold and best_src is not None:
            # Enhanced duplicate detection - check against last few chunks, not just the previous one
            is_duplicate = False
            if kept_chunks and len(kept_chunks) > 0:
                # Check against the last 3 chunks to catch more duplicates
                for check_idx in range(max(0, len(kept_chunks) - 3), len(kept_chunks)):
                    prev_chunk = kept_chunks[check_idx]
                    # Only compare if lengths are similar (within 10%)
                    if abs(len(prev_chunk) - len(best_src.cpu().numpy())) / len(prev_chunk) < 0.1:
                        chunk_sim = cosine_similarity(prev_chunk, best_src.cpu().numpy())
                        if chunk_sim > 0.90:  # Lowered threshold to catch more duplicates
                            if i < 10:  # Debug output
                                print(f"[!] Window {i}: Skipped duplicate chunk (similarity to chunk {check_idx}: {chunk_sim:.3f})")
                            is_duplicate = True
                            break
            
            if is_duplicate:
                continue
            
            kept_count += 1
            kept_chunks.append(best_src.cpu().numpy())
            kept_timestamps.append((s, e, best_sim))
            
            # Debug output for first few windows to help diagnose issues
            if kept_count <= 5:
                print(f"[+] Window {i}: Kept source {best_source_idx} with similarity {best_sim:.3f} (time: {s/sr:.1f}s-{e/sr:.1f}s)")
            elif kept_count == 6:
                print(f"[+] ... (similar debug output continues for remaining windows)")

            if args.save_segments:
                seg_path = seg_dir / f"seg_{i:05d}_{s}_{e}_sim{best_sim:.3f}.wav"
                sf.write(seg_path.as_posix(), best_src.cpu().numpy(), sr, format='WAV', subtype='PCM_24')
        else:
            # Debug output for rejected windows
            if i < 10:  # Show first 10 rejected windows
                print(f"[!] Window {i}: Rejected (best similarity: {best_sim:.3f}, threshold: {args.similarity_threshold})")

    print(f"[+] Processed {window_count} windows, kept {kept_count} windows above threshold {args.similarity_threshold}")
    
    if not kept_chunks:
        print("[!] No windows passed the similarity threshold. Try lowering --similarity-threshold.")
        return

    # ---- Simple overlap-add style stitching
    # Since windows hop, we just concatenate the chosen target streams.
    # For a fancier reconstruction, you could cross-fade; this keeps it simple.
    target_track = np.concatenate(kept_chunks, axis=0)
    
    # Apply final light normalization to the output
    target_track = light_norm(torch.tensor(target_track)).numpy()
    
    print(f"[+] Final output length: {len(target_track)/sr:.2f}s ({len(target_track)} samples)")
    print(f"[+] Kept windows covered: {kept_timestamps[0][0]/sr:.2f}s to {kept_timestamps[-1][1]/sr:.2f}s")

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
    sf.write(final_output_path, target_track, sr, format='WAV', subtype='PCM_24')
    print(f"[+] Wrote extracted target audio to: {final_output_path} (24kHz mono WAV)")
    if args.save_segments:
        print(f"[+] Wrote kept segments to: {seg_dir}")


if __name__ == "__main__":
    main()