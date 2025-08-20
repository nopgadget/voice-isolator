# Voice Isolator

A Python tool for extracting target speaker audio from mixed audio using speaker separation and speaker recognition.

## Features

- **Target Speaker Extraction**: Uses enrollment audio to identify and extract a specific speaker
- **Blind Source Separation**: Leverages Asteroid models for 2-speaker separation
- **Speaker Recognition**: Uses SpeechBrain ECAPA-TDNN for speaker embedding and similarity scoring
- **Overlap Handling**: Processes audio in sliding windows with configurable parameters
- **High Quality Output**: Outputs 24kHz mono WAV files with 24-bit PCM encoding
- **Wide Format Support**: Handles WAV, FLAC, MP3, OGG, M4A, and more with ffmpeg fallback
- **Smart Output Management**: Auto-generates unique filenames and organizes output in dedicated folders
- **Advanced Audio Processing**: Configurable normalization, overlap control, and start position skipping

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
- **Output Location**: Creates `output/` directory with unique, descriptive filenames

## Usage

```bash
python main.py --audio input_audio.m4a --target-speaker reference.wav --output extracted_target.wav
```

### Parameters

- `--audio`: Input audio file (WAV, FLAC, MP3, OGG, M4A, etc. - ffmpeg fallback for unsupported formats)
- `--target-speaker`: Audio clip of target speaker (5+ seconds of clean speech, longer clips like 20s work fine). Supports WAV, FLAC, MP3, OGG, M4A, etc.
- `--output`: Output path for extracted target audio (auto-generates unique filename if not specified)
- `--window-length`: Analysis window length in seconds (larger = more coverage, default: 15.0)
- `--hop-size`: Hop size between windows in seconds (default: 5.0)
- `--min-overlap`: Minimum overlap between windows as fraction (0.0 = no overlap, 0.5 = 50% overlap, default: 0.0)
- `--skip-start`: Skip the first N seconds of audio (useful if target speaker doesn't start immediately, default: 30.0)
- `--similarity-threshold`: Cosine similarity threshold to keep a window (0.0-1.0, higher = more strict, default: 0.80)
- `--normalization-strength`: Audio normalization strength (0.05-0.3, lower = less aggressive, default: 0.05)
- `--separation-model`: Asteroid HF model id for source separation (default: JorisCos/ConvTasNet_Libri2Mix_sepclean_16k)
- `--save-segments`: Also save kept window clips as individual files
- `--cpu-only`: Force CPU usage (disable GPU acceleration)

## Examples

### Basic Usage
```bash
# Extract target speaker with auto-generated filename
python main.py --audio interview.m4a --target-speaker chris_voice.wav

# Specify custom output path
python main.py --audio interview.m4a --target-speaker chris_voice.wav --output chris_interview.wav
```

### Advanced Usage
```bash
# Skip first 30 seconds, use strict similarity threshold
python main.py --audio long_audio.m4a --target-speaker speaker.wav --skip-start 30 --similarity-threshold 0.90

# Custom window processing with overlap
python main.py --audio audio.m4a --target-speaker target.wav --window-length 20 --hop-size 3 --min-overlap 0.3

# Gentle normalization for sensitive audio
python main.py --audio audio.m4a --target-speaker target.wav --normalization-strength 0.02
```

## Output Structure

The tool automatically creates an `output/` directory and generates unique filenames:
- **Format**: `{input_name}_extracted_from_{target_name}_{hash}_{timestamp}.wav`
- **Example**: `interview_extracted_from_chris_voice_a1b2c3d4_12345.wav`
- **Segments**: If `--save-segments` is used, individual window clips are saved in the same directory