# Voice Isolator

A Python tool for extracting target speaker audio from mixed audio using speaker separation and speaker recognition.

## Features

- **Target Speaker Extraction**: Uses enrollment audio to identify and extract a specific speaker
- **Blind Source Separation**: Leverages Asteroid models for 2-speaker separation
- **Speaker Recognition**: Uses SpeechBrain ECAPA-TDNN for speaker embedding and similarity scoring
- **Overlap Handling**: Processes audio in sliding windows with configurable parameters
- **High Quality Output**: Outputs 24kHz mono WAV files with 24-bit PCM encoding
- **Wide Format Support**: Handles WAV, FLAC, MP3, OGG, M4A, and more with ffmpeg fallback

## Input Audio Formats

The tool supports a wide range of audio formats:

**Direct Support (via torchaudio):**
- WAV, FLAC, MP3, OGG, M4A (with compatible codecs)

**Extended Support (via ffmpeg fallback):**
- Any format supported by ffmpeg (AAC, WMA, RealAudio, etc.)
- Automatic conversion to WAV format for processing

## Requirements

### Python Dependencies
```bash
pip install torch torchaudio numpy soundfile librosa tqdm asteroid speechbrain
```

### System Dependencies
- **FFmpeg** (recommended for extended format support)
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - CentOS/RHEL: `sudo yum install ffmpeg`

## Output Format

The tool outputs audio in the following format:
- **Sample Rate**: 24kHz (24,000 Hz)
- **Channels**: Mono (single channel)
- **Format**: WAV with 24-bit PCM encoding
- **File Extension**: Automatically adds .wav if not specified

## Usage

```bash
python main.py --audio input_audio.m4a --target-speaker reference.wav --output extracted_target.wav
```

### Parameters

- `--audio`: Input audio file (WAV, FLAC, MP3, OGG, M4A, etc. - ffmpeg fallback for unsupported formats)
- `--target-speaker`: Audio clip of target speaker (5+ seconds of clean speech, longer clips like 20s work fine). Supports WAV, FLAC, MP3, OGG, M4A, etc.
- `--output`: Output path for extracted target audio (default: target_only.wav)
- `--window-length`: Analysis window length in seconds (default: 15.0)
- `--hop-size`: Hop size between windows in seconds (default: 5.0)
- `--similarity-threshold`: Cosine similarity threshold to keep a window (0.0-1.0, default: 0.65)
- `--separation-model`: Asteroid HF model id for source separation (default: JorisCos/ConvTasNet_Libri2Mix_sepclean_16k)
- `--save-segments`: Also save kept window clips as individual files
- `--cpu-only`: Force CPU usage (disable GPU acceleration)