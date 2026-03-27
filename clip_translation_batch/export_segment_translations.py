#!/usr/bin/env python3

import argparse
import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

try:
    import webrtcvad  # type: ignore
except ImportError:
    webrtcvad = None


DEFAULT_MODEL_NAME = os.getenv("BATCH_WHISPER_MODEL", "large-v3").strip() or "large-v3"
DEFAULT_SOURCE_LANGUAGE = os.getenv("BATCH_SOURCE_LANGUAGE", "te").strip() or "te"
DEFAULT_OUTPUT_DIR = "translated_text"
DEFAULT_TEXT_MODEL = os.getenv("BATCH_TEXT_MODEL", "mistral:7b").strip() or "mistral:7b"
DEFAULT_VISION_MODEL = os.getenv("BATCH_VISION_MODEL", "llava:7b").strip() or "llava:7b"
DEFAULT_COMPUTE_TYPE = os.getenv("BATCH_COMPUTE_TYPE", "int8").strip() or "int8"
DEFAULT_CONTEXT_WINDOW = max(1, int(os.getenv("BATCH_CONTEXT_WINDOW", "2")))
DEFAULT_VAD_AGGRESSIVENESS = min(3, max(0, int(os.getenv("BATCH_VAD_AGGRESSIVENESS", "2"))))
DEFAULT_VAD_FRAME_MS = 30
DEFAULT_VAD_SPEECH_RATIO_THRESHOLD = float(os.getenv("BATCH_VAD_SPEECH_RATIO_THRESHOLD", "0.12"))
DEFAULT_VAD_MIN_SPEECH_FRAMES = max(1, int(os.getenv("BATCH_VAD_MIN_SPEECH_FRAMES", "3")))
DEFAULT_LOW_CONFIDENCE_THRESHOLD = float(os.getenv("BATCH_LOW_CONFIDENCE_THRESHOLD", "0.45"))
DEFAULT_OUTPUT_SENTENCE = "."
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH_BYTES = 2
DEFAULT_OLLAMA_TIMEOUT_SECONDS = int(os.getenv("BATCH_OLLAMA_TIMEOUT_SECONDS", "60"))
DEFAULT_OLLAMA_URL = os.getenv("BATCH_OLLAMA_URL", "http://127.0.0.1:11434/api/generate").strip()
SUPPORTED_MEDIA_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
}
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
}
WHISPER_INITIAL_PROMPT = (
    "This is a tutorial about using a mobile app. "
    "The speaker explains steps like clicking buttons, navigating screens, "
    "login, logout, dashboard, menu, survey."
)
WHISPER_NO_SPEECH_THRESHOLD = 0.6


@dataclass
class WhisperResult:
    text: str
    confidence: float


def natural_sort_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def normalize_text(text: str) -> str:
    cleaned = (text or "").replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_translated_text(text: str) -> str:
    cleaned = normalize_text(text)
    cleaned = re.sub(r"\band then\b", ", then", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bso\b", ", so", cleaned, count=1, flags=re.IGNORECASE)
    cleaned = re.sub(r",\s*,+", ", ", cleaned)
    cleaned = cleaned.strip(" ,")

    if not cleaned:
        return ""

    words = cleaned.split()
    if words and re.fullmatch(r"[A-Za-z]{1,2}", words[-1].rstrip(".!?,")):
        cleaned = " ".join(words[:-1]).strip()

    if not cleaned:
        return ""

    cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
    if cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


def final_output_text(text: str) -> str:
    cleaned = clean_translated_text(text)
    return cleaned if cleaned else DEFAULT_OUTPUT_SENTENCE


def choose_folder_with_finder(prompt: str) -> Path | None:
    script = (
        'tell application "Finder"\n'
        "activate\n"
        f'set chosenFolder to choose folder with prompt "{prompt}"\n'
        "POSIX path of chosenFolder\n"
        "end tell\n"
    )

    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    selected_path = (result.stdout or "").strip()
    if not selected_path:
        return None
    return Path(selected_path).expanduser().resolve()


def prompt_for_input_directory() -> Path:
    selected_folder = choose_folder_with_finder("Choose the folder containing your segment clips")
    if selected_folder is not None and selected_folder.is_dir():
        print(f"Selected clips folder: {selected_folder}")
        return selected_folder

    while True:
        raw_value = input("Enter the folder path containing your segment clips: ").strip()
        candidate = Path(raw_value).expanduser()
        if candidate.is_dir():
            return candidate.resolve()
        print("That folder was not found. Please enter a valid folder path.", file=sys.stderr)


def prompt_for_output_directory(default_output: Path) -> Path:
    selected_folder = choose_folder_with_finder("Choose the folder where English text files should be saved")
    if selected_folder is not None:
        print(f"Selected output folder: {selected_folder}")
        return selected_folder

    raw_value = input(
        f"Enter output folder path for English text files [{default_output}]: "
    ).strip()
    if not raw_value:
        return default_output
    return Path(raw_value).expanduser().resolve()


def collect_media_files(input_dir: Path) -> list[Path]:
    files = [
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS
    ]
    return sorted(files, key=natural_sort_key)


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True)


def ensure_required_tools() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found in PATH.")


def extract_audio_to_wav(media_path: Path, output_wav_path: Path) -> Path:
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        "ffmpeg",
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        str(DEFAULT_CHANNELS),
        "-ar",
        str(DEFAULT_SAMPLE_RATE),
        "-acodec",
        "pcm_s16le",
        str(output_wav_path),
    ])
    return output_wav_path


def read_wav_bytes(wav_path: Path) -> tuple[bytes, int]:
    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if channels != DEFAULT_CHANNELS:
        raise ValueError(f"Expected mono wav, got {channels} channels: {wav_path}")
    if sample_width != DEFAULT_SAMPLE_WIDTH_BYTES:
        raise ValueError(f"Expected 16-bit wav, got sample width {sample_width}: {wav_path}")
    if sample_rate != DEFAULT_SAMPLE_RATE:
        raise ValueError(f"Expected 16kHz wav, got {sample_rate}: {wav_path}")

    return frames, sample_rate


def iter_audio_frames(audio_bytes: bytes, sample_rate: int, frame_ms: int):
    frame_size = int(sample_rate * (frame_ms / 1000.0) * DEFAULT_SAMPLE_WIDTH_BYTES)
    if frame_size <= 0:
        return

    for offset in range(0, len(audio_bytes) - frame_size + 1, frame_size):
        yield audio_bytes[offset: offset + frame_size]


def has_speech_webrtcvad(wav_path: Path) -> bool:
    if webrtcvad is None:
        return False

    audio_bytes, sample_rate = read_wav_bytes(wav_path)
    vad = webrtcvad.Vad(DEFAULT_VAD_AGGRESSIVENESS)

    total_frames = 0
    speech_frames = 0
    for frame in iter_audio_frames(audio_bytes, sample_rate, DEFAULT_VAD_FRAME_MS):
        total_frames += 1
        if vad.is_speech(frame, sample_rate):
            speech_frames += 1

    if total_frames == 0:
        return False

    speech_ratio = speech_frames / total_frames
    return speech_frames >= DEFAULT_VAD_MIN_SPEECH_FRAMES and speech_ratio >= DEFAULT_VAD_SPEECH_RATIO_THRESHOLD


def has_speech_energy_fallback(wav_path: Path) -> bool:
    audio_bytes, sample_rate = read_wav_bytes(wav_path)
    frame_size = int(sample_rate * (DEFAULT_VAD_FRAME_MS / 1000.0) * DEFAULT_SAMPLE_WIDTH_BYTES)
    if frame_size <= 0:
        return False

    total_frames = 0
    speech_like_frames = 0
    rms_values = []
    for frame in iter_audio_frames(audio_bytes, sample_rate, DEFAULT_VAD_FRAME_MS):
        if len(frame) < frame_size:
            continue
        total_frames += 1
        rms = audioop_rms(frame)
        rms_values.append(rms)

    if total_frames == 0 or not rms_values:
        return False

    max_rms = max(rms_values)
    if max_rms <= 0:
        return False

    threshold = max(500, int(max_rms * 0.18))
    for rms in rms_values:
        if rms >= threshold:
            speech_like_frames += 1

    speech_ratio = speech_like_frames / total_frames
    return speech_like_frames >= DEFAULT_VAD_MIN_SPEECH_FRAMES and speech_ratio >= DEFAULT_VAD_SPEECH_RATIO_THRESHOLD


def audioop_rms(frame: bytes) -> int:
    import audioop

    return int(audioop.rms(frame, DEFAULT_SAMPLE_WIDTH_BYTES))


def detect_speech(wav_path: Path) -> bool:
    if webrtcvad is not None:
        return has_speech_webrtcvad(wav_path)
    return has_speech_energy_fallback(wav_path)


def segment_confidence(segment) -> float:
    avg_logprob = getattr(segment, "avg_logprob", None)
    if avg_logprob is None:
        return 0.0
    return max(0.0, min(1.0, math.exp(float(avg_logprob))))


def transcribe_clip(model: WhisperModel, wav_path: Path, source_language: str) -> WhisperResult:
    segments, _ = model.transcribe(
        str(wav_path),
        language=source_language,
        task="translate",
        word_timestamps=False,
        initial_prompt=WHISPER_INITIAL_PROMPT,
        temperature=0.0,
        condition_on_previous_text=False,
        no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
    )

    transcript_parts = []
    confidence_values = []
    for segment in segments:
        text = normalize_text(getattr(segment, "text", ""))
        if text:
            transcript_parts.append(text)
        confidence_values.append(segment_confidence(segment))

    cleaned_text = clean_translated_text(" ".join(transcript_parts))
    average_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    return WhisperResult(text=cleaned_text, confidence=average_confidence)


def clip_has_video(media_path: Path) -> bool:
    return media_path.suffix.lower() in VIDEO_EXTENSIONS


def extract_keyframes(media_path: Path, frame_dir: Path, clip_index: int) -> list[Path]:
    if not clip_has_video(media_path):
        return []

    clip_frame_dir = frame_dir / f"clip_{clip_index:03d}"
    clip_frame_dir.mkdir(parents=True, exist_ok=True)
    frame_pattern = clip_frame_dir / "frame_%02d.jpg"

    try:
        run_command([
            "ffmpeg",
            "-y",
            "-i",
            str(media_path),
            "-vf",
            "fps=1",
            "-frames:v",
            "3",
            str(frame_pattern),
        ])
    except subprocess.CalledProcessError:
        return []

    return sorted(clip_frame_dir.glob("frame_*.jpg"))


def ollama_cli_available() -> bool:
    return shutil.which("ollama") is not None


def run_text_llm(prompt: str, model: str) -> str | None:
    if not ollama_cli_available():
        return None

    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            check=True,
            timeout=DEFAULT_OLLAMA_TIMEOUT_SECONDS,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None

    output = normalize_text(result.stdout or "")
    return output or None


def run_vision_llm(prompt: str, image_paths: list[Path], model: str) -> str | None:
    if not image_paths:
        return None

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "images": [base64.b64encode(path.read_bytes()).decode("ascii") for path in image_paths if path.is_file()],
    }
    if not payload["images"]:
        return None

    request = urllib.request.Request(
        DEFAULT_OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=DEFAULT_OLLAMA_TIMEOUT_SECONDS) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None

    output = normalize_text(response_payload.get("response", ""))
    return output or None


def cleaner_prompt(whisper_text: str) -> str:
    return (
        "You are a transcription cleaner.\n\n"
        f"Input:\n{whisper_text}\n\n"
        "Task:\n"
        "* Fix grammar and clarity\n"
        "* Keep meaning EXACTLY the same\n"
        "* Do NOT add new words\n"
        "* Remove filler words\n"
        "* Keep it short and natural\n\n"
        "Output:\n"
        "Return only cleaned sentence."
    )


def continuity_prompt(previous_text: str, current_text: str) -> str:
    return (
        "You are a dialogue continuity editor.\n\n"
        f"Previous text:\n{previous_text}\n\n"
        f"Current text:\n{current_text}\n\n"
        "Task:\n"
        "* Make current text flow naturally from previous text\n"
        "* Avoid broken sentences\n"
        "* Avoid repetition\n"
        "* Keep it short (TTS friendly)\n\n"
        "Rules:\n"
        "* Modify ONLY current text\n"
        "* Do not add extra meaning\n\n"
        "Output:\n"
        "Return improved current text."
    )


def video_description_prompt() -> str:
    return (
        "Describe this short video clip in one short sentence. "
        "Only mention what is clearly visible. "
        "Do not invent actions or speech."
    )


def validation_prompt(current_text: str, video_description: str) -> str:
    return (
        "You are a multimodal validator.\n\n"
        "Inputs:\n"
        f"Text:\n{current_text}\n\n"
        f"Video description:\n{video_description}\n\n"
        "Task:\n"
        "* Check if text matches visual content\n"
        "* If incorrect -> fix it\n"
        "* Keep it short and natural\n\n"
        "Rules:\n"
        "* Do not invent content\n"
        "* Stay aligned with audio + visuals\n\n"
        "Output:\n"
        "Return final corrected text."
    )


def cleaner_fallback(whisper_text: str) -> str:
    return final_output_text(whisper_text)


def continuity_fallback(previous_text: str, current_text: str) -> str:
    current = final_output_text(current_text)
    if current == DEFAULT_OUTPUT_SENTENCE:
        return current
    if not previous_text:
        return current

    current_normalized = normalize_text(current).lower()
    previous_normalized = normalize_text(previous_text).lower()
    if current_normalized == previous_normalized:
        return current

    previous_tail = set(re.findall(r"[a-z0-9']+", previous_normalized)[-5:])
    current_words = re.findall(r"[a-z0-9']+", current_normalized)
    trimmed_words = list(current_words)
    while len(trimmed_words) > 2 and trimmed_words[0] in previous_tail:
        trimmed_words.pop(0)

    if trimmed_words and len(trimmed_words) != len(current_words):
        rebuilt = " ".join(trimmed_words)
        return final_output_text(rebuilt)

    return current


def validator_fallback(current_text: str, _video_description: str) -> str:
    return final_output_text(current_text)


def stage_clean_text(whisper_text: str) -> str:
    response = run_text_llm(cleaner_prompt(whisper_text), DEFAULT_TEXT_MODEL)
    return final_output_text(response or cleaner_fallback(whisper_text))


def stage_stitch_text(previous_text: str, current_text: str, low_confidence: bool) -> str:
    if current_text == DEFAULT_OUTPUT_SENTENCE:
        return current_text

    if not previous_text and not low_confidence:
        return final_output_text(current_text)

    response = run_text_llm(continuity_prompt(previous_text, current_text), DEFAULT_TEXT_MODEL)
    return final_output_text(response or continuity_fallback(previous_text, current_text))


def build_video_description(media_path: Path, frame_dir: Path, clip_index: int) -> str:
    frame_paths = extract_keyframes(media_path, frame_dir, clip_index)
    if not frame_paths:
        return "Visual description unavailable."

    response = run_vision_llm(video_description_prompt(), frame_paths[:3], DEFAULT_VISION_MODEL)
    if response:
        return response

    return "Visual description unavailable."


def stage_validate_text(current_text: str, video_description: str) -> str:
    if current_text == DEFAULT_OUTPUT_SENTENCE:
        return current_text

    if video_description == "Visual description unavailable.":
        return validator_fallback(current_text, video_description)

    response = run_text_llm(validation_prompt(current_text, video_description), DEFAULT_TEXT_MODEL)
    return final_output_text(response or validator_fallback(current_text, video_description))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Translate short Telugu segment clips into English and create one text file per clip."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Folder containing the segment clips.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Folder where English text files will be created.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Whisper model name to use. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--source-language",
        default=DEFAULT_SOURCE_LANGUAGE,
        help=f"Source language code for the clips. Default: {DEFAULT_SOURCE_LANGUAGE}",
    )
    parser.add_argument(
        "--compute-type",
        default=DEFAULT_COMPUTE_TYPE,
        help=f"faster-whisper compute type. Default: {DEFAULT_COMPUTE_TYPE}",
    )
    return parser


def process_clip(
    model: WhisperModel,
    media_path: Path,
    clip_index: int,
    source_language: str,
    temp_audio_dir: Path,
    temp_frames_dir: Path,
    previous_texts: deque[str],
) -> str:
    clip_label = f"clip_{clip_index:03d}"
    wav_path = temp_audio_dir / f"{clip_label}.wav"

    extract_audio_to_wav(media_path, wav_path)
    has_speech = detect_speech(wav_path)
    if not has_speech:
        return DEFAULT_OUTPUT_SENTENCE

    whisper_result = transcribe_clip(model, wav_path, source_language)
    if not whisper_result.text:
        return DEFAULT_OUTPUT_SENTENCE

    cleaned_text = stage_clean_text(whisper_result.text)
    previous_text = " ".join(previous_texts).strip()
    stitched_text = stage_stitch_text(
        previous_text=previous_text,
        current_text=cleaned_text,
        low_confidence=whisper_result.confidence < DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    )
    video_description = build_video_description(media_path, temp_frames_dir, clip_index)
    validated_text = stage_validate_text(stitched_text, video_description)
    return final_output_text(validated_text)


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve() if args.input_dir else prompt_for_input_directory()
    if not input_dir.is_dir():
        parser.error(f"Input folder does not exist: {input_dir}")

    default_output_dir = (input_dir / DEFAULT_OUTPUT_DIR).resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else prompt_for_output_directory(default_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_audio_dir = output_dir / "temp_audio"
    temp_frames_dir = output_dir / "temp_frames"
    temp_audio_dir.mkdir(parents=True, exist_ok=True)
    temp_frames_dir.mkdir(parents=True, exist_ok=True)

    media_files = collect_media_files(input_dir)
    if not media_files:
        print(f"No supported media files found in {input_dir}", file=sys.stderr)
        return 1

    ensure_required_tools()

    print(f"Loading Whisper model: {args.model}")
    model = WhisperModel(args.model, compute_type=args.compute_type)
    print(f"Found {len(media_files)} clip(s) in {input_dir}")
    print(f"Writing English text files to {output_dir}")
    if webrtcvad is None:
        print("webrtcvad is not installed, so the script is using an energy-based VAD fallback.")
    if not ollama_cli_available():
        print("Ollama is not available, so LLM cleanup stages will use safe local fallbacks.")

    previous_texts: deque[str] = deque(maxlen=DEFAULT_CONTEXT_WINDOW)

    for index, media_path in enumerate(media_files, start=1):
        clip_label = f"clip_{index:03d}"
        print(f"[{index}/{len(media_files)}] Processing {media_path.name} -> {clip_label}.txt")
        try:
            final_text = process_clip(
                model=model,
                media_path=media_path,
                clip_index=index,
                source_language=args.source_language,
                temp_audio_dir=temp_audio_dir,
                temp_frames_dir=temp_frames_dir,
                previous_texts=previous_texts,
            )
        except subprocess.CalledProcessError as exc:
            stderr_output = normalize_text(exc.stderr or "")
            print(f"Failed on {media_path.name}: {stderr_output or exc}", file=sys.stderr)
            final_text = DEFAULT_OUTPUT_SENTENCE
        except Exception as exc:  # noqa: BLE001
            print(f"Failed on {media_path.name}: {exc}", file=sys.stderr)
            final_text = DEFAULT_OUTPUT_SENTENCE

        output_path = output_dir / f"{clip_label}.txt"
        output_path.write_text(final_text + "\n", encoding="utf-8")
        print(f"Saved {output_path.name}: {final_text}")

        if final_text != DEFAULT_OUTPUT_SENTENCE:
            previous_texts.append(final_text)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
