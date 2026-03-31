#!/usr/bin/env python3
"""Transcribe bilingual (English/Spanish) video/audio files using Whisper.

Optionally adds speaker diarization via pyannote-audio.
"""

import argparse
import sys
from pathlib import Path

import whisper


def load_audio_for_diarization(file_path: str):
    """Convert video/audio to a format pyannote can process."""
    import torch
    import torchaudio

    waveform, sample_rate = torchaudio.load(file_path)
    # pyannote expects mono audio
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return {"waveform": waveform, "sample_rate": sample_rate}


def diarize(file_path: str, hf_token: str, num_speakers: int = None):
    """Run speaker diarization and return list of (start, end, speaker) turns."""
    from pyannote.audio import Pipeline

    print("Loading diarization model...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    print("Running speaker diarization...")
    audio = load_audio_for_diarization(file_path)
    params = {}
    if num_speakers:
        params["num_speakers"] = num_speakers

    # Progress hook to show diarization status
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    with ProgressHook() as hook:
        result = pipeline(audio, hook=hook, **params)

    # Newer pyannote returns a DiarizeOutput dataclass; older versions return Annotation directly
    diarization = getattr(result, "speaker_diarization", result)

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append((turn.start, turn.end, speaker))
    return turns


def assign_speakers(segments, speaker_turns):
    """Assign a speaker label to each Whisper segment based on overlap with diarization turns."""
    for seg in segments:
        seg_start, seg_end = seg["start"], seg["end"]
        best_speaker = None
        best_overlap = 0.0

        for turn_start, turn_end, speaker in speaker_turns:
            overlap_start = max(seg_start, turn_start)
            overlap_end = min(seg_end, turn_end)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        seg["speaker"] = best_speaker or "Unknown"
    return segments


def transcribe(
    file_path: str,
    model_size: str = "large-v3",
    output_dir: str = None,
    speakers: bool = False,
    hf_token: str = None,
    num_speakers: int = None,
    speaker_names: dict = None,
):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: file not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(output_dir) if output_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Speaker diarization (run first so it overlaps mentally with transcription wait)
    speaker_turns = None
    if speakers:
        if not hf_token:
            print(
                "Error: --hf-token is required for speaker diarization.\n"
                "Get a free token at https://huggingface.co/settings/tokens\n"
                "and accept the model terms at https://huggingface.co/pyannote/speaker-diarization-3.1",
                file=sys.stderr,
            )
            sys.exit(1)
        speaker_turns = diarize(str(path), hf_token, num_speakers)

    # Transcription
    print(f"Loading Whisper model '{model_size}'... (first run downloads ~3GB)")
    model = whisper.load_model(model_size)

    print(f"Transcribing: {path.name}")
    result = model.transcribe(
        str(path),
        language=None,  # auto-detect — lets Whisper handle code-switching
        verbose=True,
    )

    segments = result["segments"]

    # Merge speaker labels into segments
    if speaker_turns:
        segments = assign_speakers(segments, speaker_turns)
        # Apply custom speaker names if provided
        if speaker_names:
            for seg in segments:
                seg["speaker"] = speaker_names.get(seg["speaker"], seg["speaker"])

    # Save plain text
    txt_path = out_dir / f"{path.stem}.txt"
    with open(txt_path, "w") as f:
        if speaker_turns:
            current_speaker = None
            for seg in segments:
                if seg["speaker"] != current_speaker:
                    current_speaker = seg["speaker"]
                    f.write(f"\n[{current_speaker}]\n")
                f.write(seg["text"].strip() + "\n")
        else:
            f.write(result["text"])
    print(f"\nPlain text saved to: {txt_path}")

    # Save SRT with speaker labels
    srt_path = out_dir / f"{path.stem}.srt"
    with open(srt_path, "w") as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp_srt(seg["start"])
            end = format_timestamp_srt(seg["end"])
            text = seg["text"].strip()
            if speaker_turns:
                text = f"[{seg['speaker']}] {text}"
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    print(f"SRT subtitles saved to: {srt_path}")

    return result


def format_timestamp_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_speaker_names(value: str) -> dict:
    """Parse 'SPEAKER_00=Tutor,SPEAKER_01=Me' into a dict."""
    mapping = {}
    for pair in value.split(","):
        if "=" in pair:
            key, name = pair.split("=", 1)
            mapping[key.strip()] = name.strip()
    return mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe bilingual video/audio with Whisper")
    parser.add_argument("file", help="Path to video or audio file")
    parser.add_argument(
        "--model", default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size (default: large-v3, best for bilingual)",
    )
    parser.add_argument("--output-dir", "-o", help="Output directory (default: same as input file)")
    parser.add_argument(
        "--speakers", action="store_true",
        help="Enable speaker diarization (requires --hf-token)",
    )
    parser.add_argument(
        "--hf-token",
        help="HuggingFace token for pyannote models (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--num-speakers", type=int,
        help="Exact number of speakers (helps accuracy — use 2 for tutor sessions)",
    )
    parser.add_argument(
        "--speaker-names",
        help="Rename speakers: 'SPEAKER_00=Tutor,SPEAKER_01=Me'",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    hf_token = args.hf_token
    if not hf_token:
        import os
        hf_token = os.environ.get("HF_TOKEN")

    speaker_names = parse_speaker_names(args.speaker_names) if args.speaker_names else None

    transcribe(
        args.file,
        model_size=args.model,
        output_dir=args.output_dir,
        speakers=args.speakers,
        hf_token=hf_token,
        num_speakers=args.num_speakers,
        speaker_names=speaker_names,
    )
