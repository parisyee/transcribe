# Transcribe

Bilingual (English/Spanish) video/audio transcription tool using OpenAI Whisper with optional speaker diarization via pyannote-audio.

## Setup

```bash
pip install -r requirements.txt
```

Diarization requires a HuggingFace token with access to `pyannote/speaker-diarization-3.1`. Store it in `.env` as `HF_TOKEN` (see `.env.example`). You must accept the model terms at https://huggingface.co/pyannote/speaker-diarization-3.1.

## Usage

Basic transcription (no speaker labels):
```bash
python3 transcribe.py path/to/file.mp3
```

With speaker diarization:
```bash
python3 transcribe.py path/to/file.mp3 --speakers --num-speakers 2
```

With custom speaker names:
```bash
python3 transcribe.py path/to/file.mp3 --speakers --num-speakers 2 --speaker-names "SPEAKER_00=Tutor,SPEAKER_01=Me"
```

Use `--model small` for faster testing; `--model large-v3` (default) for best bilingual accuracy.

## Output

Produces two files alongside the input (or in `--output-dir`):
- `.txt` — plain text transcript (with speaker labels if diarization is enabled)
- `.srt` — subtitle file with timestamps

## Tips

- **Extract audio first** for large video files — Whisper/pyannote only use audio, and video files are orders of magnitude larger:
  ```bash
  ffmpeg -i video.mov -vn -acodec libmp3lame -q:a 2 audio.mp3
  ```
- **Test with short clips** before running full files:
  ```bash
  ffmpeg -ss 00:01:00 -i input.mov -t 5 -c copy test_clip.mov
  ```
- On macOS, if Whisper model download fails with SSL errors, run:
  ```bash
  /Applications/Python\ 3.14/Install\ Certificates.command
  ```

## Known Issues

- The `std()` warning from pyannote during diarization is harmless (numerical edge case on short segments).
- Diarization is the slowest step — progress bars are displayed via pyannote's `ProgressHook`.

## Architecture

Single-file script (`transcribe.py`). Key functions:
- `transcribe()` — main entry point, orchestrates diarization + Whisper
- `diarize()` — runs pyannote speaker diarization pipeline
- `assign_speakers()` — maps diarization turns to Whisper segments by max overlap
