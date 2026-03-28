import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import wave
import zipfile
import audioop
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import quote

from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel

from xtts_service import SPEAKER_PATH, tts, tts_lock

app = FastAPI()
templates = Jinja2Templates(directory="templates")
DEV_AUTO_RELOAD = os.getenv("AUTO_RELOAD", "1").strip().lower() not in {"0", "false", "no", "off"}
templates.env.auto_reload = DEV_AUTO_RELOAD

TEMP_DIR = "temp"
DEFAULT_FPS = 30.0
DEFAULT_AUDIO_SAMPLE_RATE = 16000
CLIP_PADDING_SECONDS = 0.08
TIMESTAMP_PRECISION_DECIMALS = 6
WHISPER_MODEL_NAME = "large-v3"
CLIP_TEXT_MODEL_NAME = os.getenv("CLIP_TEXT_MODEL_NAME", "tiny").strip() or "tiny"
CLIP_TEXT_RETRY_EXTENSION_SECONDS = 0.3
CLIP_TEXT_WORKERS = max(1, int(os.getenv("CLIP_TEXT_WORKERS", "4")))
FAST_CLIP_TEXT_REFRESH_MODE = (os.getenv("FAST_CLIP_TEXT_REFRESH_MODE", "selective").strip().lower() or "selective")
WHISPER_INITIAL_PROMPT = (
    "This is a tutorial about using a mobile app. "
    "The speaker explains steps like clicking buttons, navigating screens, "
    "login, logout, dashboard, menu, survey."
)
WHISPER_NO_SPEECH_THRESHOLD = 0.6
WHISPER_STRICT_MODE = os.getenv("WHISPER_STRICT_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}
AUDIO_AUTO_ALIGN_NOISE_DB = -35
AUDIO_AUTO_ALIGN_MIN_SILENCE_SECONDS = 0.18
AUDIO_AUTO_ALIGN_MAX_EDGE_LOOK_SECONDS = 1.25
WORD_GAP_SPLIT_SECONDS = 0.3
WORD_GAP_PUNCTUATION_SPLIT_SECONDS = 0.16
SENTENCE_PAUSE_SPLIT_SECONDS = 0.6
SILENCE_SNAP_THRESHOLD_SECONDS = 0.5
START_SILENCE_SNAP_THRESHOLD_SECONDS = 0.7
SEGMENT_ASSERT_TOLERANCE_SECONDS = 0.04
CLIP_EDGE_TRIM_BUFFER_SECONDS = 0.05
MIN_CLIP_DURATION_SECONDS = 0.8
MERGE_SHORT_CLIP_SECONDS = 1.0
TARGET_CLIP_DURATION_SECONDS = 4.8
MAX_CLIP_DURATION_SECONDS = 6.0
SOURCE_TRANSCRIBE_LANGUAGE = "te"
DISPLAY_TRANSCRIBE_LANGUAGE = "te"
ALIGNMENT_MODE = (os.getenv("ALIGNMENT_MODE", "fast").strip().lower() or "fast")
WHISPERX_REQUIRED = os.getenv("WHISPERX_REQUIRED", "1").strip().lower() not in {"0", "false", "no", "off"}
ALLOW_WHISPERX_FALLBACK = os.getenv("ALLOW_WHISPERX_FALLBACK", "1").strip().lower() in {"1", "true", "yes", "on"}
FORCED_ALIGNMENT_ENABLED = True
LOCAL_TEXT_CORRECTION_ENABLED = os.getenv("ENABLE_LOCAL_TEXT_CORRECTION", "0").strip().lower() in {"1", "true", "yes", "on"}
LOCAL_TEXT_CORRECTION_MODEL = os.getenv("LOCAL_TEXT_CORRECTION_MODEL", "mistral:7b").strip() or "mistral:7b"
ENERGY_WINDOW_MS = 20
BOUNDARY_ENERGY_WINDOW_MS = 10
BOUNDARY_ENERGY_HOP_MS = 5
BOUNDARY_ENERGY_THRESHOLD_RATIO = 0.1
BOUNDARY_REFINEMENT_SEARCH_SECONDS = 0.1
BOUNDARY_REFINEMENT_START_FORWARD_SECONDS = 0.05
BOUNDARY_REFINEMENT_END_BACKWARD_SECONDS = 0.05
BOUNDARY_REFINEMENT_MAX_SHIFT_SECONDS = 0.12
BOUNDARY_REFINEMENT_START_PADDING_SECONDS = 0.02
BOUNDARY_REFINEMENT_END_PADDING_SECONDS = 0.03
ENERGY_THRESHOLD_RATIO = 0.3
ENERGY_MIN_REGION_SECONDS = 0.08
ENERGY_RISE_RATIO = 0.2
ENERGY_DROP_RATIO = 0.16
LOW_CONFIDENCE_THRESHOLD = 0.55
VERY_LOW_CONFIDENCE_THRESHOLD = 0.4
MIN_CLIP_WORD_COUNT = 3
MIN_CONFIDENCE_WORD_DURATION_SECONDS = 0.1
MICRO_PADDING_START_SECONDS = 0.04
MICRO_PADDING_END_SECONDS = 0.04
FAST_SPEECH_THRESHOLD_WPS = 2.8
SLOW_SPEECH_THRESHOLD_WPS = 1.5
FAST_SPEECH_PADDING_SCALE = 0.7
IDEAL_CLIP_MIN_SECONDS = 2.0
IDEAL_CLIP_MAX_SECONDS = 6.0
HARD_CLIP_MAX_SECONDS = 7.0
SEMANTIC_MERGE_GAP_SECONDS = 0.4
LOCAL_SEMANTIC_GROUPING_ENABLED = os.getenv("ENABLE_LOCAL_SEMANTIC_GROUPING", "1").strip().lower() not in {"0", "false", "no", "off"}
SLOW_SPEECH_PADDING_SCALE = 1.35
SUBTITLE_MAX_LINE_CHARS = 42
SUBTITLE_MAX_LINES = 2
READING_SPEED_MIN_CPS = 12.0
READING_SPEED_MAX_CPS = 18.0
VISUAL_MIN_DURATION_SECONDS = 1.0
VISUAL_MAX_DURATION_SECONDS = 7.0
UNCOVERED_SPEECH_MIN_GAP_SECONDS = 0.5
LOCAL_LINE_BREAK_FORMATTING_ENABLED = os.getenv("ENABLE_LOCAL_LINE_FORMATTING", "1").strip().lower() not in {"0", "false", "no", "off"}
ENABLE_SCENE_DETECTION = os.getenv("ENABLE_SCENE_DETECTION", "1").strip().lower() not in {"0", "false", "no", "off"}
SCENE_DETECTION_THRESHOLD = float(os.getenv("SCENE_DETECTION_THRESHOLD", "0.3"))
SUBTITLE_START_DELAY_SECONDS = 0.08
SUBTITLE_END_DELAY_SECONDS = 0.12
MAX_START_BACKTRACK_SECONDS = 0.08
AUDIO_CACHE_DIRNAME = "audio_cache"
TRANSCRIPTION_CACHE_DIRNAME = "transcription_cache"
ENGLISH_OUTPUT_DIRNAME = "english_output"
MERGED_VIDEO_FILE_PREFIX = "english_video"
MERGED_AUDIO_FILE_PREFIX = "english_audio"
SMALL_VIDEO_EXTENSION_THRESHOLD_SECONDS = 1.0
FINAL_VIDEO_BUFFER_SECONDS = 2.0
FINAL_VIDEO_MAX_OVERAGE_SECONDS = 3.0
FINAL_VIDEO_TARGET_OVERAGE_SECONDS = 2.0
VOICE_STYLE_DEFAULT = "tutorial"
ENGLISH_AUDIO_PIPELINE_VERSION = "v2_consistent_voice"
USE_ALIGNMENT_DEBUG_MODE = os.getenv("USE_ALIGNMENT_DEBUG_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}
VOICE_STYLE_CONFIG = {
    "tutorial": {
        "comma_pause": 0.24,
        "sentence_pause": 0.42,
        "thinking_pause": 0.55,
        "connector_pause": 0.20,
        "gain_db": 1.3,
    },
    "friendly": {
        "comma_pause": 0.18,
        "sentence_pause": 0.32,
        "thinking_pause": 0.42,
        "connector_pause": 0.14,
        "gain_db": 1.0,
    },
    "formal": {
        "comma_pause": 0.14,
        "sentence_pause": 0.26,
        "thinking_pause": 0.30,
        "connector_pause": 0.08,
        "gain_db": 0.8,
    },
}
PAUSE_KEYWORDS = ("after that", "next", "now", "then", "and")
CONNECTOR_WORDS = ("and", "so", "because", "but", "or", "if", "then", "after that", "now")
BAD_LINE_BREAK_WORDS = (
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "from",
    "and", "but", "so", "or", "because",
)
EMPHASIS_KEYWORDS = ("click", "enter", "select", "open", "login", "submit")
DRIVE_FOLDER_PATTERN = re.compile(
    r"^https://drive\.google\.com/drive/folders/([A-Za-z0-9_-]+)(?:\?.*)?$"
)
SILENCE_START_PATTERN = re.compile(r"silence_start:\s*([0-9.]+)")
SILENCE_END_PATTERN = re.compile(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)")
SCENE_TIME_PATTERN = re.compile(r"pts_time:([0-9.]+)")
DOMAIN_TEXT_REPLACEMENTS = (
    ("survey appi", "survey app"),
    ("APPI", "APPS"),
    ("appi", "app"),
    ("apps survey app", "APPS survey app"),
    ("app survey app", "APPS survey app"),
    ("log in screen", "login screen"),
    ("login page", "login screen"),
    ("log out", "logout"),
)
HALLUCINATION_PHRASES = (
    "subscribe",
    "like this video",
    "bell icon",
    "channel",
)

os.makedirs(TEMP_DIR, exist_ok=True)
app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")

whisper_model = None
whisper_model_lock = threading.Lock()
clip_text_model = None
clip_text_model_lock = threading.Lock()
clip_text_transcribe_lock = threading.Lock()
audio_generation_state = {}
audio_generation_state_lock = threading.Lock()
audio_generation_workers = {}
audio_generation_workers_lock = threading.Lock()
translation_cache = {}
translation_cache_lock = threading.Lock()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"🔥 ERROR: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)},
    )


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def run_command(command):
    subprocess.run(command, check=True)


def normalize_alignment_mode(mode):
    normalized = (mode or ALIGNMENT_MODE or "fast").strip().lower()
    if normalized not in {"fast", "accurate"}:
        return "fast"
    return normalized


def is_fast_alignment_mode(mode):
    return normalize_alignment_mode(mode) == "fast"


def should_use_heavy_refinement(mode):
    return normalize_alignment_mode(mode) == "accurate"


def build_audio_job_key(video_path, clip_id, voice_style):
    return f"{os.path.realpath(video_path)}::{int(clip_id)}::{normalize_voice_style(voice_style)}"


def set_audio_generation_state(video_path, clip_id, voice_style, **updates):
    job_key = build_audio_job_key(video_path, clip_id, voice_style)
    with audio_generation_state_lock:
        current = audio_generation_state.get(job_key, {})
        current.update(updates)
        current["updated_at"] = time.time()
        audio_generation_state[job_key] = current
        return dict(current)


def get_audio_generation_state(video_path, clip_id, voice_style):
    job_key = build_audio_job_key(video_path, clip_id, voice_style)
    with audio_generation_state_lock:
        return dict(audio_generation_state.get(job_key, {}))


def clear_audio_generation_state(video_path, clip_id, voice_style):
    job_key = build_audio_job_key(video_path, clip_id, voice_style)
    with audio_generation_state_lock:
        audio_generation_state.pop(job_key, None)


def build_translation_cache_key(text, target_lang):
    cleaned_text = normalize_text_for_hash(text)
    return f"{target_lang}:{cleaned_text}"


def translate_to_telugu(text):
    cleaned_text = normalize_text_for_hash(text)
    if not cleaned_text:
        return ""

    for _ in range(2):
        try:
            translated = GoogleTranslator(source="en", target="te").translate(cleaned_text)
            translated_text = normalize_text_for_hash(translated or "")
            if translated_text:
                return translated_text
        except Exception:
            continue

    return None


def translate_text(text, target_lang):
    cleaned_text = normalize_text_for_hash(text)
    normalized_target_lang = (target_lang or "").strip().lower()

    if not cleaned_text:
        return ""

    if normalized_target_lang != "te":
        raise ValueError("Only Telugu translation is supported right now")

    cache_key = build_translation_cache_key(cleaned_text, normalized_target_lang)
    with translation_cache_lock:
        cached_value = translation_cache.get(cache_key)
    if cached_value:
        return cached_value

    translated_text = translate_to_telugu(cleaned_text)

    if not translated_text:
        return None

    with translation_cache_lock:
        translation_cache[cache_key] = translated_text

    return translated_text


def create_session_dir():
    session_id = uuid.uuid4().hex
    session_dir = os.path.join(TEMP_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def resolve_safe_path(path):
    if not path:
        raise HTTPException(status_code=400, detail="Path is required")

    resolved = os.path.realpath(path)
    temp_root = os.path.realpath(TEMP_DIR)
    if not resolved.startswith(temp_root + os.sep):
        raise HTTPException(status_code=400, detail="Invalid session path")

    return resolved


def get_session_dir(video_path):
    resolved_video_path = resolve_safe_path(video_path)
    if not os.path.exists(resolved_video_path):
        raise HTTPException(status_code=404, detail="Source video no longer exists")
    return os.path.dirname(resolved_video_path)


def get_audio_cache_dir(video_path):
    audio_cache_dir = os.path.join(get_session_dir(video_path), AUDIO_CACHE_DIRNAME)
    os.makedirs(audio_cache_dir, exist_ok=True)
    return audio_cache_dir


def get_transcription_cache_dir():
    cache_dir = os.path.join(TEMP_DIR, TRANSCRIPTION_CACHE_DIRNAME)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_english_output_dir(video_path):
    output_dir = os.path.join(get_session_dir(video_path), ENGLISH_OUTPUT_DIRNAME)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def compute_file_sha256(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_transcription_cache_path(video_path, alignment_mode):
    video_hash = compute_file_sha256(video_path)
    return os.path.join(
        get_transcription_cache_dir(),
        f"{video_hash}_{normalize_alignment_mode(alignment_mode)}.json",
    )


def load_transcription_cache(video_path, alignment_mode):
    cache_path = build_transcription_cache_path(video_path, alignment_mode)
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except (OSError, json.JSONDecodeError):
        return None


def write_transcription_cache(video_path, alignment_mode, payload):
    cache_path = build_transcription_cache_path(video_path, alignment_mode)
    try:
        with open(cache_path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=True, indent=2)
    except OSError:
        return


def build_clip_text_cache_path(audio_path, clip_id, start, end, alignment_mode, audio_signature=None):
    audio_hash = audio_signature or compute_file_sha256(audio_path)
    payload = f"{audio_hash}:{clip_id}:{round_timestamp_value(start)}:{round_timestamp_value(end)}:{CLIP_TEXT_MODEL_NAME}:{normalize_alignment_mode(alignment_mode)}"
    cache_key = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return os.path.join(get_transcription_cache_dir(), f"clip_text_{cache_key}.json")


def load_clip_text_cache(audio_path, clip_id, start, end, alignment_mode, audio_signature=None):
    cache_path = build_clip_text_cache_path(
        audio_path,
        clip_id,
        start,
        end,
        alignment_mode,
        audio_signature=audio_signature,
    )
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except (OSError, json.JSONDecodeError):
        return None


def write_clip_text_cache(audio_path, clip_id, start, end, alignment_mode, payload, audio_signature=None):
    cache_path = build_clip_text_cache_path(
        audio_path,
        clip_id,
        start,
        end,
        alignment_mode,
        audio_signature=audio_signature,
    )
    try:
        with open(cache_path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=True, indent=2)
    except OSError:
        return


def get_whisper_model():
    global whisper_model

    if whisper_model is not None:
        return whisper_model

    with whisper_model_lock:
        if whisper_model is None:
            log(f"Loading Whisper model: {WHISPER_MODEL_NAME}")
            whisper_model = WhisperModel(WHISPER_MODEL_NAME, compute_type="int8")

    return whisper_model


def get_clip_text_model():
    global clip_text_model

    if clip_text_model is not None:
        return clip_text_model

    with clip_text_model_lock:
        if clip_text_model is None:
            log(f"Loading clip text Whisper model: {CLIP_TEXT_MODEL_NAME}")
            clip_text_model = WhisperModel(CLIP_TEXT_MODEL_NAME, compute_type="int8")

    return clip_text_model


def build_whisper_transcribe_kwargs(task, word_timestamps, strict_mode=False):
    kwargs = {
        "task": task,
        "word_timestamps": word_timestamps,
        "initial_prompt": WHISPER_INITIAL_PROMPT,
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "no_speech_threshold": WHISPER_NO_SPEECH_THRESHOLD,
    }
    if strict_mode or WHISPER_STRICT_MODE:
        kwargs["beam_size"] = 5
        kwargs["patience"] = 1
    return kwargs


def extract_audio(input_video):
    audio_path = os.path.join(os.path.dirname(input_video), "audio.wav")

    run_command([
        "ffmpeg", "-y",
        "-i", input_video,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(DEFAULT_AUDIO_SAMPLE_RATE),
        "-ac", "1",
        audio_path
    ])

    return audio_path


def get_video_fps(input_video):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                input_video
            ],
            capture_output=True,
            text=True,
            check=True
        )
        rate = result.stdout.strip()
        if "/" in rate:
            numerator, denominator = rate.split("/", maxsplit=1)
            denominator_value = float(denominator)
            if denominator_value:
                return float(numerator) / denominator_value
        if rate:
            return float(rate)
    except (subprocess.CalledProcessError, ValueError, ZeroDivisionError):
        log(f"Falling back to default FPS: {DEFAULT_FPS}")

    return DEFAULT_FPS


def get_media_duration(media_path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            media_path
        ],
        capture_output=True,
        text=True,
        check=True
    )
    return float(result.stdout.strip() or 0.0)


def get_media_stream_start_times(media_path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_streams",
            "-of", "json",
            media_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout or "{}")
    video_start = 0.0
    audio_start = 0.0
    for stream in payload.get("streams", []):
        stream_type = stream.get("codec_type")
        try:
            start_time = float(stream.get("start_time", 0.0) or 0.0)
        except (TypeError, ValueError):
            start_time = 0.0
        if stream_type == "video" and video_start == 0.0:
            video_start = start_time
        elif stream_type == "audio" and audio_start == 0.0:
            audio_start = start_time
    return {
        "video_start": video_start,
        "audio_start": audio_start,
        "offset_seconds": audio_start - video_start,
    }


def detect_scene_cuts(video_path, threshold=SCENE_DETECTION_THRESHOLD):
    if not ENABLE_SCENE_DETECTION:
        return []

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", video_path,
                "-filter:v", f"select='gt(scene,{threshold})',metadata=print",
                "-an",
                "-f", "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        output = f"{exc.stdout or ''}\n{exc.stderr or ''}"
    else:
        output = f"{result.stdout or ''}\n{result.stderr or ''}"

    cuts = []
    for line in output.splitlines():
        match = SCENE_TIME_PATTERN.search(line)
        if not match:
            continue
        try:
            cuts.append(round_timestamp_value(float(match.group(1))))
        except ValueError:
            continue

    deduped = []
    for cut in cuts:
        if not deduped or abs(cut - deduped[-1]) > 0.15:
            deduped.append(cut)

    log(f"Detected {len(deduped)} scene cuts")
    return deduped


def snap_to_frame(timestamp, fps, method="nearest"):
    frame = timestamp * fps

    if method == "floor":
        return max(int(frame) / fps, 0.0)

    if method == "ceil":
        frame_number = int(frame)
        if frame_number < frame:
            frame_number += 1
        return frame_number / fps

    return max(round(frame) / fps, 0.0)


def snap_to_audio_frame(timestamp, sample_rate=DEFAULT_AUDIO_SAMPLE_RATE):
    frame_time = 1.0 / sample_rate
    return max(round(timestamp / frame_time) * frame_time, 0.0)


def seconds_to_milliseconds(timestamp):
    return max(int(round(float(timestamp) * 1000)), 0)


def milliseconds_to_seconds(milliseconds):
    return round(milliseconds / 1000.0, TIMESTAMP_PRECISION_DECIMALS)


def format_timestamp_for_ffmpeg(milliseconds):
    return f"{milliseconds_to_seconds(milliseconds):.{TIMESTAMP_PRECISION_DECIMALS}f}"


def format_seconds_for_ffmpeg(seconds):
    return f"{max(float(seconds), 0.0):.{TIMESTAMP_PRECISION_DECIMALS}f}"


def normalize_segments(segments, sample_rate=DEFAULT_AUDIO_SAMPLE_RATE):
    del sample_rate
    normalized = []

    for index, segment in enumerate(segments):
        segment_payload = dict(segment or {})
        start_raw = segment_payload.get("start")
        end_raw = segment_payload.get("end")

        # Defensive reconstruction for segments created in intermediate stages.
        if start_raw is None or end_raw is None:
            segment_words = list(segment_payload.get("words") or segment_payload.get("source_words") or [])
            if segment_words:
                start_raw = segment_words[0].get("start", start_raw)
                end_raw = segment_words[-1].get("end", end_raw)
            else:
                if start_raw is None:
                    start_raw = segment_payload.get("word_start")
                if end_raw is None:
                    end_raw = segment_payload.get("word_end")

        try:
            start = max(float(start_raw), 0.0)
            end = max(float(end_raw), start)
        except (TypeError, ValueError):
            log(
                f"Invalid segment skipped during normalize at index={index}: "
                f"reason=missing_or_non_numeric_timestamps segment={segment_payload!r}"
            )
            continue
        start_ms = seconds_to_milliseconds(start)
        end_ms = seconds_to_milliseconds(end)

        if end_ms <= start_ms:
            log(
                f"Skipping segment with non-positive duration: "
                f"start={format_timestamp_for_ffmpeg(start_ms)} "
                f"end={format_timestamp_for_ffmpeg(end_ms)} "
                f"text={segment.get('text', '')[:40]!r}"
            )
            continue

        normalized_segment = {
            "clip_id": int(segment_payload.get("clip_id", index)),
            "start": round_timestamp_value(start),
            "end": round_timestamp_value(end),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": segment_payload.get("text", "")
        }
        for key, value in segment_payload.items():
            if key not in normalized_segment:
                normalized_segment[key] = value

        normalized.append(normalized_segment)

    return normalized


def normalize_words(words, sample_rate=DEFAULT_AUDIO_SAMPLE_RATE):
    del sample_rate
    normalized = []
    min_step = 1.0 / DEFAULT_AUDIO_SAMPLE_RATE

    for word in words:
        text = (word.get("text") or "").strip()
        if not text:
            continue

        start = max(float(word["start"]), 0.0)
        end = max(float(word["end"]), start + min_step)
        start_ms = seconds_to_milliseconds(start)
        end_ms = seconds_to_milliseconds(end)

        if end_ms <= start_ms:
            end_ms = start_ms + 1

        normalized_word = {
            "text": text,
            "start": round_timestamp_value(start),
            "end": round_timestamp_value(end),
            "start_ms": start_ms,
            "end_ms": end_ms,
        }

        for key, value in word.items():
            if key not in normalized_word:
                normalized_word[key] = value

        normalized.append(normalized_word)

    return normalized


def detect_silence_regions(audio_path, media_duration, noise_db=AUDIO_AUTO_ALIGN_NOISE_DB, min_duration=AUDIO_AUTO_ALIGN_MIN_SILENCE_SECONDS):
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", audio_path,
                "-af", f"silencedetect=n={noise_db}dB:d={min_duration}",
                "-f", "null",
                "-"
            ],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as exc:
        output = f"{exc.stdout or ''}\n{exc.stderr or ''}"
    else:
        output = f"{result.stdout or ''}\n{result.stderr or ''}"

    silences = []
    active_start = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        start_match = SILENCE_START_PATTERN.search(line)
        if start_match:
            active_start = max(float(start_match.group(1)), 0.0)
            continue

        end_match = SILENCE_END_PATTERN.search(line)
        if not end_match:
            continue

        silence_end = max(float(end_match.group(1)), 0.0)
        silence_duration = max(float(end_match.group(2)), 0.0)
        silence_start = active_start
        if silence_start is None:
            silence_start = max(silence_end - silence_duration, 0.0)

        if silence_end > silence_start and silence_duration >= min_duration:
            silences.append({
                "start": round_timestamp_value(silence_start),
                "end": round_timestamp_value(min(silence_end, media_duration)),
            })
        active_start = None

    if active_start is not None and media_duration > active_start:
        silences.append({
            "start": round_timestamp_value(active_start),
            "end": round_timestamp_value(media_duration),
        })

    log(f"Detected {len(silences)} silence regions for auto-alignment")
    return silences


def detect_silence(audio_path, media_duration=None):
    try:
        effective_duration = float(media_duration) if media_duration is not None else get_media_duration(audio_path)
        silence_regions = detect_silence_regions(audio_path, effective_duration)
    except Exception as exc:
        log(f"Silence detection failed, continuing without silence points: {exc}")
        return []

    silence_points = []
    for silence in silence_regions or []:
        try:
            start = float(silence.get("start", 0.0))
            end = float(silence.get("end", start))
        except (TypeError, ValueError, AttributeError):
            continue

        if end <= start:
            continue
        silence_points.append({
            "start": round_timestamp_value(start),
            "end": round_timestamp_value(end),
            "mid": round_timestamp_value((start + end) / 2.0),
        })

    if not silence_points:
        return []

    return sorted(silence_points, key=lambda point: float(point["start"]))


def snap_start_to_silence(start_time, silence_points, threshold=START_SILENCE_SNAP_THRESHOLD_SECONDS):
    original_start = round_timestamp_value(start_time)
    if not silence_points:
        return original_start

    candidates = [
        point
        for point in silence_points
        if float(point.get("end", 0.0)) <= original_start
    ]
    if not candidates:
        return original_start

    nearest = max(candidates, key=lambda point: float(point.get("end", 0.0)))
    snapped_start = round_timestamp_value(float(nearest.get("end", original_start)))
    if (original_start - snapped_start) <= float(threshold):
        return max(snapped_start, 0.0)
    return original_start


def snap_end_to_silence(end_time, silence_points, min_end_time=None, threshold=SILENCE_SNAP_THRESHOLD_SECONDS):
    original_end = round_timestamp_value(end_time)
    minimum_end = original_end if min_end_time is None else round_timestamp_value(min_end_time)
    if not silence_points:
        return max(original_end, minimum_end)

    candidates = [
        point
        for point in silence_points
        if float(point.get("start", original_end)) >= original_end
    ]
    if not candidates:
        return max(original_end, minimum_end)

    nearest = min(candidates, key=lambda point: float(point.get("start", original_end)) - original_end)
    snapped_end = round_timestamp_value(float(nearest.get("start", original_end)))
    if (snapped_end - original_end) <= float(threshold):
        return max(snapped_end, minimum_end)
    return max(original_end, minimum_end)


def snap_to_nearest_silence(timestamp, silence_points, threshold=SILENCE_SNAP_THRESHOLD_SECONDS):
    target_time = round_timestamp_value(timestamp)
    if not silence_points:
        return target_time

    nearest = min(
        silence_points,
        key=lambda point: abs(float(point.get("mid", target_time)) - float(target_time)),
    )
    nearest_mid = round_timestamp_value(float(nearest.get("mid", target_time)))
    if abs(float(nearest_mid) - float(target_time)) <= float(threshold):
        return nearest_mid
    return target_time


def detect_energy_regions(
    audio_path,
    window_ms=ENERGY_WINDOW_MS,
    threshold_ratio=ENERGY_THRESHOLD_RATIO,
    min_region_seconds=ENERGY_MIN_REGION_SECONDS,
):
    try:
        with wave.open(audio_path, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            window_frames = max(int(sample_rate * (window_ms / 1000.0)), 1)
            frame_index = 0
            windows = []

            while True:
                raw_frames = wav_file.readframes(window_frames)
                if not raw_frames:
                    break

                if channels > 1:
                    raw_frames = audioop.tomono(raw_frames, sample_width, 0.5, 0.5)
                rms = float(audioop.rms(raw_frames, sample_width))
                start = frame_index / sample_rate
                end = min((frame_index + window_frames) / sample_rate, start + (len(raw_frames) / max(sample_width, 1)) / sample_rate)
                windows.append({
                    "start": round_timestamp_value(start),
                    "end": round_timestamp_value(end),
                    "rms": rms,
                })
                frame_index += window_frames
    except (wave.Error, OSError) as exc:
        log(f"Energy detection failed: {exc}")
        return {
            "windows": [],
            "regions": [],
            "avg_energy": 0.0,
            "threshold": 0.0,
            "peaks": [],
        }

    if not windows:
        return {
            "windows": [],
            "regions": [],
            "avg_energy": 0.0,
            "threshold": 0.0,
            "peaks": [],
        }

    avg_energy = sum(window["rms"] for window in windows) / len(windows)
    threshold = avg_energy * threshold_ratio
    rise_threshold = avg_energy * ENERGY_RISE_RATIO
    drop_threshold = avg_energy * ENERGY_DROP_RATIO
    regions = []
    active_start = None
    active_peak = 0.0
    active_peak_time = 0.0
    onset_peaks = []
    offset_peaks = []

    for index, window in enumerate(windows):
        previous_rms = float(windows[index - 1]["rms"]) if index > 0 else 0.0
        delta = float(window["rms"]) - previous_rms
        window["delta"] = round(delta, 4)
        if delta >= rise_threshold:
            onset_peaks.append({
                "time": round_timestamp_value(window["start"]),
                "delta": round(delta, 4),
                "rms": round(float(window["rms"]), 4),
            })
        if (-delta) >= drop_threshold:
            offset_peaks.append({
                "time": round_timestamp_value(window["end"]),
                "delta": round(-delta, 4),
                "rms": round(float(window["rms"]), 4),
            })

    for window in windows:
        if window["rms"] >= threshold:
            if active_start is None:
                active_start = float(window["start"])
                active_peak = window["rms"]
                active_peak_time = float(window["start"])
            elif window["rms"] >= active_peak:
                active_peak = window["rms"]
                active_peak_time = float(window["start"])
            continue

        if active_start is None:
            continue

        region_end = float(window["start"])
        if (region_end - active_start) >= min_region_seconds:
            regions.append({
                "start": round_timestamp_value(active_start),
                "end": round_timestamp_value(region_end),
                "peak_rms": round(active_peak, 4),
                "peak_time": round_timestamp_value(active_peak_time),
            })
        active_start = None
        active_peak = 0.0
        active_peak_time = 0.0

    if active_start is not None:
        region_end = float(windows[-1]["end"])
        if (region_end - active_start) >= min_region_seconds:
            regions.append({
                "start": round_timestamp_value(active_start),
                "end": round_timestamp_value(region_end),
                "peak_rms": round(active_peak, 4),
                "peak_time": round_timestamp_value(active_peak_time),
            })

    peaks = sorted(regions, key=lambda item: item["peak_rms"], reverse=True)[:10]
    log(
        f"Detected {len(regions)} energy speech regions "
        f"(avg={avg_energy:.2f}, threshold={threshold:.2f}, rise={rise_threshold:.2f}, drop={drop_threshold:.2f})"
    )
    return {
        "windows": windows,
        "regions": regions,
        "avg_energy": round(avg_energy, 4),
        "threshold": round(threshold, 4),
        "rise_threshold": round(rise_threshold, 4),
        "drop_threshold": round(drop_threshold, 4),
        "peaks": peaks,
        "onset_peaks": onset_peaks[:24],
        "offset_peaks": offset_peaks[:24],
    }


def refine_segment_boundaries_with_energy(audio_path, segments):
    if not segments:
        return []

    safe_segments = [dict(segment) for segment in normalize_segments(segments)]
    if not safe_segments:
        return []

    try:
        with wave.open(audio_path, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            total_frames = wav_file.getnframes()
            audio_duration = total_frames / max(sample_rate, 1)

            window_frames = max(int(sample_rate * (BOUNDARY_ENERGY_WINDOW_MS / 1000.0)), 1)
            hop_frames = max(int(sample_rate * (BOUNDARY_ENERGY_HOP_MS / 1000.0)), 1)
            frame_index = 0
            windows = []

            while frame_index < total_frames:
                wav_file.setpos(frame_index)
                raw_frames = wav_file.readframes(window_frames)
                if not raw_frames:
                    break
                if channels > 1:
                    raw_frames = audioop.tomono(raw_frames, sample_width, 0.5, 0.5)
                rms = float(audioop.rms(raw_frames, sample_width))
                center_time = (frame_index + (window_frames / 2.0)) / max(sample_rate, 1)
                windows.append({
                    "time": round_timestamp_value(center_time),
                    "rms": rms,
                })
                frame_index += hop_frames
    except (wave.Error, OSError) as exc:
        log(f"Energy boundary refinement skipped (audio read failed): {exc}")
        return safe_segments

    if not windows:
        return safe_segments

    mean_rms = sum(window["rms"] for window in windows) / len(windows)
    silence_threshold = mean_rms * BOUNDARY_ENERGY_THRESHOLD_RATIO
    min_step = 1.0 / DEFAULT_AUDIO_SAMPLE_RATE

    def select_windows(left, right):
        return [
            window for window in windows
            if float(left) <= float(window["time"]) <= float(right)
        ]

    refined = []
    for index, segment in enumerate(safe_segments):
        original_start = float(segment.get("start", 0.0))
        original_end = float(segment.get("end", original_start))
        start_search_left = max(0.0, original_start - BOUNDARY_REFINEMENT_SEARCH_SECONDS)
        start_search_right = min(audio_duration, original_start + BOUNDARY_REFINEMENT_START_FORWARD_SECONDS)
        end_search_left = max(0.0, original_end - BOUNDARY_REFINEMENT_END_BACKWARD_SECONDS)
        end_search_right = min(audio_duration, original_end + BOUNDARY_REFINEMENT_SEARCH_SECONDS)

        start_windows = select_windows(start_search_left, start_search_right)
        end_windows = select_windows(end_search_left, end_search_right)
        adjusted_start = original_start
        adjusted_end = original_end

        if start_windows:
            speech_index = next(
                (i for i, window in enumerate(start_windows) if float(window["rms"]) >= silence_threshold),
                None,
            )
            if speech_index is not None and speech_index > 0:
                silent_before_speech = [
                    window for window in start_windows[:speech_index]
                    if float(window["rms"]) < silence_threshold
                ]
                if silent_before_speech:
                    adjusted_start = float(silent_before_speech[-1]["time"])

        if end_windows:
            speech_indices = [
                i for i, window in enumerate(end_windows)
                if float(window["rms"]) >= silence_threshold
            ]
            if speech_indices:
                last_speech_index = speech_indices[-1]
                silent_after_speech = [
                    window for window in end_windows[last_speech_index + 1:]
                    if float(window["rms"]) < silence_threshold
                ]
                if silent_after_speech:
                    adjusted_end = float(silent_after_speech[0]["time"])

        adjusted_start = max(0.0, adjusted_start - BOUNDARY_REFINEMENT_START_PADDING_SECONDS)
        adjusted_end = min(audio_duration, adjusted_end + BOUNDARY_REFINEMENT_END_PADDING_SECONDS)

        if abs(adjusted_start - original_start) > BOUNDARY_REFINEMENT_MAX_SHIFT_SECONDS:
            adjusted_start = original_start
        if abs(adjusted_end - original_end) > BOUNDARY_REFINEMENT_MAX_SHIFT_SECONDS:
            adjusted_end = original_end

        if refined:
            previous_end = float(refined[-1]["end"])
            adjusted_start = max(adjusted_start, previous_end + min_step)

        next_original_start = None
        if index + 1 < len(safe_segments):
            next_original_start = float(safe_segments[index + 1].get("start", adjusted_end))
            adjusted_end = min(adjusted_end, next_original_start - min_step)

        if adjusted_end <= adjusted_start:
            adjusted_start = original_start
            adjusted_end = original_end
            if refined:
                adjusted_start = max(adjusted_start, float(refined[-1]["end"]) + min_step)
            if next_original_start is not None:
                adjusted_end = min(adjusted_end, next_original_start - min_step)

        if adjusted_end <= adjusted_start:
            adjusted_end = adjusted_start + min_step

        adjusted_start = round_timestamp_value(adjusted_start)
        adjusted_end = round_timestamp_value(adjusted_end)

        log(
            "Energy refine clip "
            f"{get_clip_number(segment.get('clip_id', index))}: "
            f"original=({format_seconds_for_ffmpeg(original_start)}, {format_seconds_for_ffmpeg(original_end)}) "
            f"adjusted=({format_seconds_for_ffmpeg(adjusted_start)}, {format_seconds_for_ffmpeg(adjusted_end)}) "
            f"delta=({format_seconds_for_ffmpeg(adjusted_start - original_start)}, {format_seconds_for_ffmpeg(adjusted_end - original_end)})"
        )

        updated_segment = dict(segment)
        updated_segment["start"] = adjusted_start
        updated_segment["end"] = adjusted_end
        updated_segment["energy_refined"] = True
        refined.append(updated_segment)

    return normalize_segments(refined)


def validate_and_repair_segments(segments, words=None, media_duration=None, stage_label="segment_validation"):
    repaired = []
    safe_words = list(words or [])
    media_limit = float(media_duration) if media_duration is not None else None

    for index, raw_segment in enumerate(segments or []):
        segment = dict(raw_segment or {})
        reason = None
        word_indices = [
            int(word_index)
            for word_index in (segment.get("word_indices") or [])
            if 0 <= int(word_index) < len(safe_words)
        ]
        segment_words = list(segment.get("words") or segment.get("source_words") or [])

        start = segment.get("start")
        end = segment.get("end")

        if (start is None or end is None) and word_indices and safe_words:
            first_word = safe_words[word_indices[0]]
            last_word = safe_words[word_indices[-1]]
            start = first_word.get("start", start)
            end = last_word.get("end", end)

        if start is None or end is None:
            if segment_words:
                start = segment_words[0].get("start", start)
                end = segment_words[-1].get("end", end)
            else:
                start = segment.get("word_start", start)
                end = segment.get("word_end", end)

        try:
            start_value = max(float(start), 0.0)
            end_value = float(end)
        except (TypeError, ValueError):
            reason = "missing_or_non_numeric_start_end"
            log(
                f"{stage_label}: invalid segment index={index} reason={reason} segment={segment!r}"
            )
            continue

        if media_limit is not None and media_limit > 0.0:
            start_value = min(max(start_value, 0.0), media_limit)
            end_value = min(max(end_value, 0.0), media_limit)

        if end_value <= start_value:
            reason = "start_greater_or_equal_end"
            log(
                f"{stage_label}: invalid segment index={index} reason={reason} "
                f"start={start_value} end={end_value} segment={segment!r}"
            )
            continue

        segment["start"] = round_timestamp_value(start_value)
        segment["end"] = round_timestamp_value(end_value)
        if word_indices:
            segment["word_indices"] = word_indices
        repaired.append(segment)

    return repaired


def enforce_strict_timeline_continuity(segments, media_duration=None, stage_label="timeline_continuity"):
    if not segments:
        return []

    ordered = [dict(segment) for segment in normalize_segments(segments)]
    if not ordered:
        return []

    ordered.sort(key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))
    min_step = 1.0 / DEFAULT_AUDIO_SAMPLE_RATE
    duration_limit = float(media_duration) if media_duration is not None else None

    fixed = []
    previous_end = 0.0
    for index, segment in enumerate(ordered):
        start = max(float(segment.get("start", 0.0)), 0.0)
        end = max(float(segment.get("end", start)), start)

        if start < previous_end:
            log(
                f"{stage_label}: overlap fixed at index={index} "
                f"clip_id={segment.get('clip_id')} old_start={start:.6f} "
                f"new_start={previous_end:.6f} previous_end={previous_end:.6f}"
            )
            start = previous_end

        if end <= start:
            proposed_end = start + min_step
            if duration_limit is not None:
                proposed_end = min(proposed_end, duration_limit)
            if proposed_end <= start:
                log(
                    f"{stage_label}: skipping segment index={index} clip_id={segment.get('clip_id')} "
                    f"reason=non_positive_duration_after_continuity_adjustment "
                    f"start={start:.6f} end={end:.6f}"
                )
                continue
            end = proposed_end

        if duration_limit is not None:
            start = min(start, duration_limit)
            end = min(end, duration_limit)
            if end <= start:
                log(
                    f"{stage_label}: skipping segment index={index} clip_id={segment.get('clip_id')} "
                    f"reason=clamped_to_media_end start={start:.6f} end={end:.6f}"
                )
                continue

        segment["start"] = round_timestamp_value(start)
        segment["end"] = round_timestamp_value(end)
        fixed.append(segment)
        previous_end = float(segment["end"])

    for index, segment in enumerate(fixed):
        segment["clip_id"] = index

    return normalize_segments(fixed)


def enforce_full_word_coverage(segments, words, media_duration):
    if not segments:
        return []

    safe_words = list(words or [])
    if not safe_words:
        return normalize_segments(segments)

    validated_segments = validate_and_repair_segments(
        segments,
        words=safe_words,
        media_duration=media_duration,
        stage_label="enforce_full_word_coverage_precheck",
    )
    normalized_segments = [dict(segment) for segment in normalize_segments(validated_segments)]
    min_step = 1.0 / DEFAULT_AUDIO_SAMPLE_RATE
    spoken_word_total = len(safe_words)
    word_to_segment = [-1] * spoken_word_total
    reconciled_segments = []

    for segment in normalized_segments:
        raw_indices = [
            int(word_index)
            for word_index in (segment.get("word_indices") or [])
            if 0 <= int(word_index) < spoken_word_total
        ]

        if not raw_indices:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
            for word_index, word in enumerate(safe_words):
                midpoint = (float(word.get("start", 0.0)) + float(word.get("end", 0.0))) / 2.0
                if start <= midpoint < end:
                    raw_indices.append(word_index)

        unique_indices = []
        for word_index in sorted(set(raw_indices)):
            if word_to_segment[word_index] == -1:
                word_to_segment[word_index] = len(reconciled_segments)
                unique_indices.append(word_index)

        if not unique_indices:
            continue

        updated = dict(segment)
        updated["word_indices"] = unique_indices
        reconciled_segments.append(updated)

    # Any unassigned words become dedicated continuity segments so no speech is lost.
    missing_indices = [index for index, owner in enumerate(word_to_segment) if owner == -1]
    if missing_indices:
        runs = []
        current_run = [missing_indices[0]]
        for index in missing_indices[1:]:
            if index == current_run[-1] + 1:
                current_run.append(index)
            else:
                runs.append(current_run)
                current_run = [index]
        runs.append(current_run)

        for run in runs:
            run_words = [safe_words[index] for index in run]
            fallback_text = build_preserved_text_from_words(run_words) or build_text_from_word_list(run_words)
            reconciled_segments.append({
                "clip_id": len(reconciled_segments),
                "word_indices": list(run),
                "words": run_words,
                "source_words": run_words,
                "text": fallback_text or " ",
                "raw_text": fallback_text or " ",
                "text_source": "coverage_recovery",
                "alignment_method": "coverage_recovery",
                "alignment_mode": ALIGNMENT_MODE,
                "forced_alignment_used": False,
                "auto_aligned": True,
            })

    normalized_segments = [
        segment for segment in reconciled_segments
        if segment.get("word_indices")
    ]
    if not normalized_segments:
        return []

    normalized_segments.sort(key=lambda item: int(item["word_indices"][0]))
    covered_indices = []
    for index, segment in enumerate(normalized_segments):
        covered_indices.extend(int(word_index) for word_index in segment["word_indices"])
        segment["clip_id"] = int(segment.get("clip_id", index))

    # Build continuous segment chain from sequential words. This avoids large uncovered gaps.
    first_segment = normalized_segments[0]
    first_word_start = float(safe_words[first_segment["word_indices"][0]].get("start", 0.0))
    first_segment["start"] = round_timestamp_value(max(0.0, min(float(first_segment.get("start", first_word_start)), first_word_start)))

    for index in range(len(normalized_segments) - 1):
        current = normalized_segments[index]
        next_segment = normalized_segments[index + 1]
        current_last_word = safe_words[int(current["word_indices"][-1])]
        next_first_word = safe_words[int(next_segment["word_indices"][0])]
        current_word_end = float(current_last_word.get("end", current.get("end", 0.0)))
        next_word_start = float(next_first_word.get("start", next_segment.get("start", current_word_end)))

        # Keep continuity anchored to neighboring word timings, not prior adjusted clip edges.
        boundary_hint = (current_word_end + next_word_start) / 2.0
        boundary = min(max(boundary_hint, current_word_end), next_word_start)
        boundary = round_timestamp_value(max(boundary, current_word_end))

        current["end"] = boundary
        next_segment["start"] = boundary

    last_segment = normalized_segments[-1]
    last_word_end = float(safe_words[last_segment["word_indices"][-1]].get("end", 0.0))
    if float(media_duration) > 0.0:
        last_segment["end"] = round_timestamp_value(float(media_duration))
    else:
        last_segment["end"] = round_timestamp_value(max(float(last_segment.get("end", last_word_end)), last_word_end))

    for segment in normalized_segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        word_indices = [int(word_index) for word_index in segment.get("word_indices", [])]
        if not word_indices:
            continue
        first_word = safe_words[word_indices[0]]
        last_word = safe_words[word_indices[-1]]
        first_word_start = float(first_word.get("start", start))
        last_word_end = float(last_word.get("end", end))

        if start > first_word_start:
            start = first_word_start
        if end < last_word_end:
            end = last_word_end

        if end <= start:
            end = min(float(media_duration), start + min_step)

        segment_words = [safe_words[word_index] for word_index in word_indices]
        segment["start"] = round_timestamp_value(max(0.0, start))
        segment["end"] = round_timestamp_value(min(float(media_duration), end))
        segment["word_start"] = round_timestamp_value(first_word_start)
        segment["word_end"] = round_timestamp_value(last_word_end)
        segment["words"] = segment_words
        segment["source_words"] = segment_words

        preserved_existing_text = clean_transcript_text(segment.get("text", "") or segment.get("raw_text", ""))
        rebuilt_text = preserved_existing_text or build_preserved_text_from_words(segment_words)
        if not rebuilt_text:
            rebuilt_text = build_text_from_word_list(segment_words) or " "
        segment["raw_text"] = rebuilt_text
        segment["text"] = rebuilt_text
        segment["lines"] = [rebuilt_text]

        words_per_second = get_words_per_second(segment_words)
        display_start, display_end = compute_subtitle_display_timing(
            float(segment["start"]),
            float(segment["end"]),
            segment.get("text", ""),
            words_per_second,
        )
        segment["subtitleStart"] = display_start
        segment["subtitleEnd"] = display_end
        segment["readingSpeed"] = compute_reading_speed(
            segment.get("text", ""),
            float(segment["start"]),
            float(segment["end"]),
        )

    covered_unique = sorted(set(int(index) for index in covered_indices))
    covered_set = set(covered_unique)
    missing = [index for index in range(spoken_word_total) if index not in covered_set]
    duplicate_count = max(len(covered_indices) - len(covered_unique), 0)
    log(
        "Coverage check: "
        f"words_total={spoken_word_total} covered_unique={len(covered_unique)} "
        f"missing={len(missing)} duplicates={duplicate_count} segments={len(normalized_segments)}"
    )

    if missing:
        log(f"Coverage warning: missing word indices sample={missing[:12]}")
    else:
        log("Coverage validation passed: all words assigned exactly once.")

    if duplicate_count != 0 or missing:
        log("Coverage integrity issue detected after reconciliation; applying deterministic full-word fallback segment.")
        fallback_words = list(safe_words)
        fallback_start = float(fallback_words[0].get("start", 0.0))
        fallback_end = float(media_duration) if float(media_duration) > 0.0 else float(fallback_words[-1].get("end", fallback_start))
        fallback_text = build_preserved_text_from_words(fallback_words) or build_text_from_word_list(fallback_words) or " "
        normalized_segments = [{
            "clip_id": 0,
            "start": round_timestamp_value(max(0.0, fallback_start)),
            "end": round_timestamp_value(fallback_end),
            "word_start": round_timestamp_value(fallback_start),
            "word_end": round_timestamp_value(float(fallback_words[-1].get("end", fallback_start))),
            "words": fallback_words,
            "source_words": fallback_words,
            "word_indices": list(range(spoken_word_total)),
            "text": fallback_text,
            "raw_text": fallback_text,
            "lines": [fallback_text],
            "text_source": "coverage_integrity_fallback",
            "alignment_method": "coverage_integrity_fallback",
            "auto_aligned": True,
        }]

    normalized_segments.sort(key=lambda item: int(item["word_indices"][0]))
    for index, segment in enumerate(normalized_segments):
        segment["clip_id"] = index
    return normalize_segments(normalized_segments)


def _segment_duration_from_indices(word_indices, words):
    if not word_indices:
        return 0.0
    first = words[int(word_indices[0])]
    last = words[int(word_indices[-1])]
    return max(float(last.get("end", 0.0)) - float(first.get("start", 0.0)), 0.0)


def _segment_text_incomplete_by_words(word_indices, words):
    if not word_indices:
        return True
    segment_words = [words[int(index)] for index in word_indices]
    text = build_text_from_word_list(segment_words).strip()
    if not text:
        return True
    if is_sentence_boundary_token(text):
        return False
    tokens = text.lower().split()
    if len(tokens) < 4:
        return True
    tail = tokens[-1].strip(",.;:!?")
    incomplete_tails = {
        "the", "a", "an", "to", "on", "in", "at", "for", "of", "with",
        "and", "or", "but", "then", "now",
    }
    return tail in incomplete_tails


def _should_semantically_merge_runs(current_indices, next_indices, words, max_seconds):
    if not current_indices or not next_indices:
        return False
    current_duration = _segment_duration_from_indices(current_indices, words)
    combined_indices = list(current_indices) + list(next_indices)
    combined_duration = _segment_duration_from_indices(combined_indices, words)
    if combined_duration > max_seconds:
        return False

    if current_duration < IDEAL_CLIP_MIN_SECONDS:
        return True
    if _segment_text_incomplete_by_words(current_indices, words):
        return True

    current_last = words[int(current_indices[-1])]
    next_first = words[int(next_indices[0])]
    gap = max(float(next_first.get("start", 0.0)) - float(current_last.get("end", 0.0)), 0.0)
    if gap <= SENTENCE_PAUSE_SPLIT_SECONDS and not _segment_text_incomplete_by_words(next_indices, words):
        return True
    return False


def _find_duration_split_position(word_indices, words, target_seconds, min_seconds):
    if len(word_indices) < 6:
        return None
    start_time = float(words[int(word_indices[0])].get("start", 0.0))
    target_time = start_time + target_seconds
    best_position = None
    best_score = None

    for position in range(2, len(word_indices) - 2):
        left_indices = word_indices[:position]
        right_indices = word_indices[position:]
        left_duration = _segment_duration_from_indices(left_indices, words)
        right_duration = _segment_duration_from_indices(right_indices, words)
        if left_duration < min_seconds or right_duration < min_seconds:
            continue

        previous_word = words[int(word_indices[position - 1])]
        boundary_time = float(previous_word.get("end", start_time))
        score = abs(boundary_time - target_time)
        if is_sentence_boundary_token(get_word_token(previous_word)):
            score -= 0.2
        next_word = words[int(word_indices[position])]
        if clean_text(get_word_token(next_word)).lower().strip(",.;:!?") in {token.lower() for token in CONNECTOR_WORDS}:
            score += 0.1

        if best_score is None or score < best_score:
            best_score = score
            best_position = position

    return best_position


def _is_function_like_token(token):
    normalized = clean_text(token).lower().strip(",.;:!?")
    return normalized in {
        "the", "a", "an", "to", "on", "in", "at", "for", "of", "with", "from",
        "by", "into", "onto", "over", "under", "and", "or", "but", "so",
        "if", "then", "than", "as", "that", "this", "these", "those", "it",
        "its", "is", "are", "was", "were", "be", "been", "being", "am", "do",
        "does", "did", "have", "has", "had", "can", "could", "will", "would",
        "should", "may", "might", "must",
    }


def _looks_like_clause_start_token(token):
    normalized = clean_text(token).strip()
    if not normalized:
        return False
    core = normalized.strip(",.;:!?").lower()
    if not core or _is_function_like_token(core):
        return False
    if not re.search(r"[A-Za-z]", core):
        return False
    return True


def _find_multi_action_split_position(word_indices, words, min_seconds):
    if len(word_indices) < 8:
        return None

    run_words = [words[int(index)] for index in word_indices]
    gap_profile = compute_gap_profile(run_words)
    median_gap = float(gap_profile.get("median_gap", 0.0))
    large_pause_threshold = float(gap_profile.get("large_pause_threshold", 0.0))
    conjunction_markers = {"and", "then", "next", "afterward", "afterwards", "meanwhile", "also", "so"}

    best_position = None
    best_score = None
    for position in range(2, len(word_indices) - 2):
        left_indices = word_indices[:position]
        right_indices = word_indices[position:]
        if len(left_indices) < 3 or len(right_indices) < 3:
            continue

        left_duration = _segment_duration_from_indices(left_indices, words)
        right_duration = _segment_duration_from_indices(right_indices, words)
        if left_duration < min_seconds or right_duration < min_seconds:
            continue

        previous_word = words[int(word_indices[position - 1])]
        current_word = words[int(word_indices[position])]
        next_word = words[int(word_indices[position + 1])] if (position + 1) < len(word_indices) else None

        previous_token = clean_text(get_word_token(previous_word))
        current_token = clean_text(get_word_token(current_word))
        next_token = clean_text(get_word_token(next_word)) if next_word else ""
        current_core = current_token.lower().strip(",.;:!?")

        marker_score = 0.0
        if previous_token.endswith((",", ";", ":")) and _looks_like_clause_start_token(current_token):
            marker_score += 2.0
        if current_core in conjunction_markers and _looks_like_clause_start_token(next_token):
            marker_score += 2.2
        elif previous_token.lower().strip(",.;:!?") in conjunction_markers and _looks_like_clause_start_token(current_token):
            marker_score += 1.8

        if marker_score <= 0.0:
            continue

        gap = max(get_word_gap(previous_word, current_word), 0.0)
        if gap >= large_pause_threshold and gap > 0.0:
            marker_score += 0.8
        elif gap >= median_gap and gap > 0.0:
            marker_score += 0.4

        if _segment_text_incomplete_by_words(left_indices, words):
            marker_score -= 1.6
        if _segment_text_incomplete_by_words(right_indices, words):
            marker_score -= 1.2

        if marker_score < 1.8:
            continue

        center_bias = abs((left_duration - right_duration)) * 0.08
        score = marker_score - center_bias
        if best_score is None or score > best_score:
            best_score = score
            best_position = position

    return best_position


def postprocess_segments_for_quality(segments, words, media_duration):
    safe_words = list(words or [])
    if not segments or not safe_words:
        return normalize_segments(segments)

    initial = [dict(segment) for segment in normalize_segments(segments)]
    runs = []
    for segment in sorted(initial, key=lambda item: int((item.get("word_indices") or [10**9])[0])):
        indices = [int(index) for index in (segment.get("word_indices") or []) if 0 <= int(index) < len(safe_words)]
        if not indices:
            continue
        runs.append(sorted(set(indices)))

    if not runs:
        return normalize_segments(initial)

    # 1) Semantic merge for incomplete/broken sentence fragments.
    semantic_runs = []
    pointer = 0
    semantic_limit = max(HARD_CLIP_MAX_SECONDS, 8.0)
    while pointer < len(runs):
        current = list(runs[pointer])
        while pointer + 1 < len(runs) and _should_semantically_merge_runs(current, runs[pointer + 1], safe_words, semantic_limit):
            current.extend(runs[pointer + 1])
            pointer += 1
        semantic_runs.append(current)
        pointer += 1

    # 2) Adaptive multi-action split (linguistic clause/conjunction cues, no fixed action verbs).
    action_split_runs = []
    for run in semantic_runs:
        queue = [run]
        while queue:
            current = queue.pop(0)
            split_position = _find_multi_action_split_position(current, safe_words, min_seconds=1.2)
            if split_position is None:
                action_split_runs.append(current)
                continue
            left = current[:split_position]
            right = current[split_position:]
            if not left or not right:
                action_split_runs.append(current)
                continue
            queue.insert(0, right)
            queue.insert(0, left)

    # 3) Grammar-safe cleanup is word-preserving: rebuilt later directly from assigned words.

    # 4) Hard duration control: split long runs; then merge tiny runs.
    max_seconds = max(HARD_CLIP_MAX_SECONDS, 7.8)
    min_seconds = IDEAL_CLIP_MIN_SECONDS
    target_seconds = min(IDEAL_CLIP_MAX_SECONDS, 5.8)

    split_runs = []
    for run in action_split_runs:
        pending = [run]
        while pending:
            current = pending.pop(0)
            duration = _segment_duration_from_indices(current, safe_words)
            if duration <= max_seconds:
                split_runs.append(current)
                continue

            split_position = _find_duration_split_position(current, safe_words, target_seconds, min_seconds)
            if split_position is None:
                split_runs.append(current)
                continue

            left = current[:split_position]
            right = current[split_position:]
            if not left or not right:
                split_runs.append(current)
                continue

            pending.insert(0, right)
            pending.insert(0, left)

    balanced_runs = []
    index = 0
    while index < len(split_runs):
        current = list(split_runs[index])
        current_duration = _segment_duration_from_indices(current, safe_words)
        if current_duration >= min_seconds or len(split_runs) == 1:
            balanced_runs.append(current)
            index += 1
            continue

        if index + 1 < len(split_runs):
            merged = current + list(split_runs[index + 1])
            if _segment_duration_from_indices(merged, safe_words) <= max_seconds + 0.8:
                balanced_runs.append(merged)
                index += 2
                continue

        if balanced_runs:
            candidate = list(balanced_runs[-1]) + current
            if _segment_duration_from_indices(candidate, safe_words) <= max_seconds + 0.8:
                balanced_runs[-1] = candidate
                index += 1
                continue

        balanced_runs.append(current)
        index += 1

    rebuilt = []
    for clip_id, run in enumerate(balanced_runs):
        ordered_indices = sorted(set(int(index) for index in run))
        if not ordered_indices:
            continue
        run_words = [safe_words[word_index] for word_index in ordered_indices]
        start_time = float(run_words[0].get("start", 0.0))
        end_time = float(run_words[-1].get("end", start_time))
        if float(media_duration or 0.0) > 0.0:
            end_time = min(end_time, float(media_duration))
        if end_time <= start_time:
            log(
                "postprocess_segments_for_quality: invalid rebuilt run skipped "
                f"clip_id={clip_id} start={start_time} end={end_time} indices={ordered_indices[:8]}"
            )
            continue
        text = build_preserved_text_from_words(run_words) or build_text_from_word_list(run_words) or " "
        rebuilt.append({
            "clip_id": clip_id,
            "start": round_timestamp_value(start_time),
            "end": round_timestamp_value(end_time),
            "word_indices": ordered_indices,
            "words": run_words,
            "source_words": run_words,
            "text": text,
            "raw_text": text,
            "lines": [text],
            "text_source": "postprocessed_quality",
            "alignment_method": "postprocessed_quality",
            "auto_aligned": True,
        })

    rebuilt = validate_and_repair_segments(
        rebuilt,
        words=safe_words,
        media_duration=media_duration,
        stage_label="postprocess_segments_for_quality_precheck",
    )
    return enforce_full_word_coverage(rebuilt, safe_words, media_duration)


def assign_words_to_segments(segments, words):
    assigned = []
    word_index = 0

    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        current_words = []

        while word_index < len(words):
            word = words[word_index]
            word_start = float(word.get("start", 0.0))
            word_end = float(word.get("end", word_start))

            if word_end <= start:
                word_index += 1
                continue

            if word_start >= end:
                break

            midpoint = (word_start + word_end) / 2.0
            if start <= midpoint < end:
                current_words.append(word)
                word_index += 1
                continue

            break

        assigned.append(current_words)

    return assigned


def find_previous_silence_end(anchor_time, silence_regions, max_look_seconds=AUDIO_AUTO_ALIGN_MAX_EDGE_LOOK_SECONDS):
    best = None
    for silence in silence_regions:
        silence_end = float(silence["end"])
        if silence_end <= anchor_time and (anchor_time - silence_end) <= max_look_seconds:
            best = silence_end
    return best


def find_next_silence_start(anchor_time, silence_regions, max_look_seconds=AUDIO_AUTO_ALIGN_MAX_EDGE_LOOK_SECONDS):
    for silence in silence_regions:
        silence_start = float(silence["start"])
        if silence_start >= anchor_time and (silence_start - anchor_time) <= max_look_seconds:
            return silence_start
    return None


def find_previous_energy_rise(anchor_time, energy_regions, max_look_seconds=AUDIO_AUTO_ALIGN_MAX_EDGE_LOOK_SECONDS):
    best = None
    for region in energy_regions or []:
        region_start = float(region["start"])
        if region_start <= anchor_time and (anchor_time - region_start) <= max_look_seconds:
            best = region_start
    return best


def find_next_energy_drop(anchor_time, energy_regions, max_look_seconds=AUDIO_AUTO_ALIGN_MAX_EDGE_LOOK_SECONDS):
    for region in energy_regions or []:
        region_end = float(region["end"])
        if region_end >= anchor_time and (region_end - anchor_time) <= max_look_seconds:
            return region_end
    return None


def find_previous_onset_peak(anchor_time, onset_peaks, max_look_seconds=AUDIO_AUTO_ALIGN_MAX_EDGE_LOOK_SECONDS):
    best = None
    for peak in onset_peaks or []:
        peak_time = float(peak["time"])
        if peak_time <= anchor_time and (anchor_time - peak_time) <= max_look_seconds:
            best = peak_time
    return best


def find_next_offset_peak(anchor_time, offset_peaks, max_look_seconds=AUDIO_AUTO_ALIGN_MAX_EDGE_LOOK_SECONDS):
    for peak in offset_peaks or []:
        peak_time = float(peak["time"])
        if peak_time >= anchor_time and (peak_time - anchor_time) <= max_look_seconds:
            return peak_time
    return None


def gap_has_boundary_signal(gap_start, gap_end, silence_regions, energy_regions):
    if gap_end <= gap_start:
        return False

    for silence in silence_regions or []:
        overlap_start = max(gap_start, float(silence["start"]))
        overlap_end = min(gap_end, float(silence["end"]))
        if overlap_end > overlap_start:
            return True

    for region in energy_regions or []:
        region_start = float(region["start"])
        region_end = float(region["end"])
        if gap_start <= region_start <= gap_end or gap_start <= region_end <= gap_end:
            return True

    return False


def choose_boundary_from_silence(gap_start, gap_end, silence_regions):
    if gap_end <= gap_start:
        return None

    best_overlap = None
    for silence in silence_regions:
        overlap_start = max(gap_start, float(silence["start"]))
        overlap_end = min(gap_end, float(silence["end"]))
        if overlap_end <= overlap_start:
            continue

        overlap_duration = overlap_end - overlap_start
        if best_overlap is None or overlap_duration > best_overlap[0]:
            best_overlap = (overlap_duration, overlap_start, overlap_end)

    if best_overlap is not None:
        _, overlap_start, overlap_end = best_overlap
        return (overlap_start + overlap_end) / 2.0

    return (gap_start + gap_end) / 2.0


def refine_segments_with_waveform(segments, transcript_words, audio_path, media_duration):
    normalized_segments = normalize_segments(segments)
    normalized_words = normalize_words(transcript_words or [])
    if not normalized_segments:
        return normalized_segments

    silence_regions = detect_silence_regions(audio_path, media_duration)
    if not silence_regions:
        return [
            {
                **segment,
                "auto_aligned": False,
            }
            for segment in normalized_segments
        ]

    words_per_segment = assign_words_to_segments(normalized_segments, normalized_words)
    min_duration = 1.0 / DEFAULT_AUDIO_SAMPLE_RATE
    refined_segments = [dict(segment) for segment in normalized_segments]

    for index, segment in enumerate(refined_segments):
        segment["auto_aligned"] = False

        current_words = words_per_segment[index]
        if not current_words:
            continue

        first_word_start = float(current_words[0]["start"])
        last_word_end = float(current_words[-1]["end"])

        if index == 0:
            silence_end = find_previous_silence_end(first_word_start, silence_regions)
            if silence_end is not None:
                segment["start"] = round_timestamp_value(max(0.0, silence_end))
                segment["auto_aligned"] = True

        if index == len(refined_segments) - 1:
            silence_start = find_next_silence_start(last_word_end, silence_regions)
            if silence_start is not None:
                segment["end"] = round_timestamp_value(min(media_duration, silence_start))
                segment["auto_aligned"] = True

    for index in range(len(refined_segments) - 1):
        current_segment = refined_segments[index]
        next_segment = refined_segments[index + 1]
        current_words = words_per_segment[index]
        next_words = words_per_segment[index + 1]

        current_anchor = float(current_words[-1]["end"]) if current_words else float(current_segment["end"])
        next_anchor = float(next_words[0]["start"]) if next_words else float(next_segment["start"])

        gap_start = max(current_anchor, float(current_segment["start"]))
        gap_end = min(next_anchor, float(next_segment["end"]))
        boundary = choose_boundary_from_silence(gap_start, gap_end, silence_regions)

        if boundary is None:
            continue

        boundary = snap_to_audio_frame(boundary)
        min_boundary = float(current_segment["start"]) + min_duration
        max_boundary = float(next_segment["end"]) - min_duration
        boundary = min(max(boundary, min_boundary), max_boundary)

        if boundary <= float(current_segment["start"]) or boundary >= float(next_segment["end"]):
            continue

        current_segment["end"] = round_timestamp_value(boundary)
        next_segment["start"] = round_timestamp_value(boundary)
        current_segment["auto_aligned"] = True
        next_segment["auto_aligned"] = True

    refined_segments = normalize_segments(refined_segments)
    validate_continuity(refined_segments)
    log(f"Auto-aligned {sum(1 for segment in refined_segments if segment.get('auto_aligned'))}/{len(refined_segments)} clips using waveform silence analysis")
    return refined_segments


def validate_continuity(segments):
    previous_end_ms = None

    for index, segment in enumerate(segments, start=1):
        start_ms = int(segment["start_ms"])
        end_ms = int(segment["end_ms"])

        log(
            f"Segment {index}: "
            f"start={format_timestamp_for_ffmpeg(start_ms)} "
            f"end={format_timestamp_for_ffmpeg(end_ms)}"
        )

        if previous_end_ms is not None and start_ms < previous_end_ms:
            raise ValueError(
                f"Segment overlap error at {index}: "
                f"previous end {format_timestamp_for_ffmpeg(previous_end_ms)} > "
                f"current start {format_timestamp_for_ffmpeg(start_ms)}"
            )

        if end_ms <= start_ms:
            raise ValueError(
                f"Segment duration error at {index}: "
                f"start {format_timestamp_for_ffmpeg(start_ms)} "
                f"end {format_timestamp_for_ffmpeg(end_ms)}"
            )

        previous_end_ms = end_ms


def log_pipeline_observability(
    video_duration,
    audio_duration,
    whisper_segments,
    aligned_words,
    alignment_metadata,
    sentence_groups,
    final_segments,
):
    safe_video_duration = float(video_duration or 0.0)
    safe_audio_duration = float(audio_duration or 0.0)
    safe_whisper_segments = list(whisper_segments or [])
    safe_aligned_words = list(aligned_words or [])
    safe_sentence_groups = list(sentence_groups or [])
    safe_final_segments = sorted(
        (dict(segment) for segment in (final_segments or [])),
        key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))),
    )
    min_step = 1.0 / DEFAULT_AUDIO_SAMPLE_RATE

    print("\n=== PIPELINE OBSERVABILITY ===")

    print("1) INPUT SUMMARY")
    print(f"video_duration={round_timestamp_value(safe_video_duration)}")
    print(f"audio_duration={round_timestamp_value(safe_audio_duration)}")
    print(f"whisper_segments_total={len(safe_whisper_segments)}")

    print("\n2) WHISPER OUTPUT (SOURCE OF TRUTH)")
    for index, segment in enumerate(safe_whisper_segments, start=1):
        segment_text = clean_text(getattr(segment, "text", ""))
        segment_start = round_timestamp_value(float(getattr(segment, "start", 0.0)))
        segment_end = round_timestamp_value(float(getattr(segment, "end", segment_start)))
        segment_duration = round_timestamp_value(max(segment_end - segment_start, 0.0))
        print(
            f"[W{index}] start={segment_start} end={segment_end} duration={segment_duration} "
            f"text={segment_text!r}"
        )

    print("\n3) ALIGNMENT SUMMARY")
    alignment_used = bool((alignment_metadata or {}).get("used"))
    alignment_mode = "WHISPERX" if alignment_used else "WHISPER_FALLBACK"
    words_per_second = (
        float(len(safe_aligned_words)) / safe_audio_duration
        if safe_audio_duration > 0.0
        else 0.0
    )
    print(f"alignment_source={alignment_mode}")
    print(f"aligned_words_total={len(safe_aligned_words)}")
    print(f"aligned_word_density_wps={round(words_per_second, 4)}")

    print("\n4) SENTENCE SEGMENT VIEW")
    for index, sentence in enumerate(safe_sentence_groups, start=1):
        sentence_text = clean_text(sentence.get("text", ""))
        original_start = round_timestamp_value(float(sentence.get("start", 0.0)))
        original_end = round_timestamp_value(float(sentence.get("end", original_start)))
        refined_start = round_timestamp_value(float(sentence.get("start_time", original_start)))
        refined_end = round_timestamp_value(float(sentence.get("end_time", original_end)))
        delta_start = round_timestamp_value(refined_start - original_start)
        delta_end = round_timestamp_value(refined_end - original_end)
        print(
            f"[S{index}] original=({original_start}->{original_end}) "
            f"refined=({refined_start}->{refined_end}) "
            f"delta=({delta_start},{delta_end}) text={sentence_text!r}"
        )

    print("\n5) AUDIO COVERAGE ANALYSIS")
    total_covered = 0.0
    total_gaps = 0.0
    total_overlap = 0.0
    coverage_flags = []
    previous_end = 0.0
    for index, segment in enumerate(safe_final_segments, start=1):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        total_covered += max(end - start, 0.0)
        delta = start - previous_end
        if abs(delta) > min_step:
            if delta > 0.0:
                total_gaps += delta
                coverage_flags.append(
                    f"GAP before segment {index}: {round_timestamp_value(delta)}s "
                    f"({round_timestamp_value(previous_end)}->{round_timestamp_value(start)})"
                )
            else:
                overlap = abs(delta)
                total_overlap += overlap
                coverage_flags.append(
                    f"OVERLAP before segment {index}: {round_timestamp_value(overlap)}s "
                    f"({round_timestamp_value(start)}<{round_timestamp_value(previous_end)})"
                )
        previous_end = max(previous_end, end)

    if safe_video_duration > 0.0:
        tail_delta = safe_video_duration - previous_end
        if abs(tail_delta) > min_step:
            if tail_delta > 0.0:
                total_gaps += tail_delta
                coverage_flags.append(
                    f"GAP at end: {round_timestamp_value(tail_delta)}s "
                    f"({round_timestamp_value(previous_end)}->{round_timestamp_value(safe_video_duration)})"
                )
            else:
                overlap = abs(tail_delta)
                total_overlap += overlap
                coverage_flags.append(
                    f"OVERLAP beyond video end: {round_timestamp_value(overlap)}s"
                )

    print(f"total_clip_coverage={round_timestamp_value(total_covered)}")
    print(f"total_uncovered_gaps={round_timestamp_value(total_gaps)}")
    print(f"total_overlap={round_timestamp_value(total_overlap)}")
    if coverage_flags:
        for flag in coverage_flags:
            print(flag)
    else:
        print("No coverage gaps or overlaps detected.")

    print("\n6) FINAL SEGMENT CHAIN VALIDATION")
    chain_flags = []
    previous_end = None
    for index, segment in enumerate(safe_final_segments, start=1):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        duration = max(end - start, 0.0)
        if previous_end is None:
            diff = 0.0
            state = "START"
        else:
            diff = start - previous_end
            if abs(diff) <= min_step:
                state = "CONTIGUOUS"
            elif diff > 0.0:
                state = "GAP"
                chain_flags.append(f"Segment {index} has GAP of {round_timestamp_value(diff)}s")
            else:
                state = "OVERLAP"
                chain_flags.append(f"Segment {index} has OVERLAP of {round_timestamp_value(abs(diff))}s")

        print(
            f"[C{index}] start={round_timestamp_value(start)} "
            f"end={round_timestamp_value(end)} "
            f"duration={round_timestamp_value(duration)} "
            f"prev_end_diff={round_timestamp_value(diff)} status={state}"
        )
        previous_end = end

    if chain_flags:
        print("Chain flags:")
        for flag in chain_flags:
            print(flag)
    else:
        print("No discontinuity, gap, or overlap flags in segment chain.")

    print("\n7) LAST SEGMENT CHECK")
    if safe_final_segments:
        last_end = float(safe_final_segments[-1].get("end", 0.0))
    else:
        last_end = 0.0
    end_difference = round_timestamp_value(safe_video_duration - last_end)
    print(f"last_segment_end={round_timestamp_value(last_end)}")
    print(f"video_duration={round_timestamp_value(safe_video_duration)}")
    print(f"difference={end_difference}")
    if abs(end_difference) > min_step:
        print("FINAL SEGMENT DOES NOT COVER VIDEO END")
    else:
        print("Final segment reaches video end within frame precision.")

    print("\n8) HUMAN-READABLE SUMMARY")
    has_chain_issues = bool(chain_flags)
    has_coverage_issues = bool(coverage_flags)
    if not has_chain_issues and not has_coverage_issues:
        pipeline_status = "Stable"
    elif has_chain_issues or has_coverage_issues:
        pipeline_status = "Partially unstable" if safe_final_segments else "Misaligned"
    else:
        pipeline_status = "Misaligned"

    if not has_chain_issues and not has_coverage_issues and alignment_used:
        alignment_quality = "High consistency"
    elif not has_chain_issues and not has_coverage_issues:
        alignment_quality = "Consistent fallback"
    elif safe_sentence_groups:
        alignment_quality = "Mixed consistency"
    else:
        alignment_quality = "Low consistency"

    print(f"alignment_quality={alignment_quality}")
    print(f"pipeline_status={pipeline_status}")
    print("=== END PIPELINE OBSERVABILITY ===\n")


def transcribe(audio, fps, media_duration, video_path=None, alignment_mode=ALIGNMENT_MODE):
    model = get_whisper_model()
    del fps
    resolved_mode = normalize_alignment_mode(alignment_mode)

    log("Running Whisper transcription with word timestamps...")
    source_segments, _ = model.transcribe(
        audio,
        language=SOURCE_TRANSCRIBE_LANGUAGE,
        task="translate",
        word_timestamps=True,
        initial_prompt=WHISPER_INITIAL_PROMPT,
        temperature=0.0,
        condition_on_previous_text=False,
        no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
    )
    source_segments = list(source_segments)
    source_words = extract_words_from_whisper_segments(source_segments)

    aligned_words, force_alignment_metadata = try_force_align_words(
        audio_path=audio,
        words=source_words,
        language_code=SOURCE_TRANSCRIBE_LANGUAGE,
    )
    source_words = aligned_words or source_words
    display_words = source_words
    alignment_metadata = {
        "enabled": True,
        "used": bool(force_alignment_metadata.get("used")),
        "method": force_alignment_metadata.get("method", "whisper_segment_timing"),
        "mode": resolved_mode,
    }
    if force_alignment_metadata.get("warning"):
        alignment_metadata["warning"] = force_alignment_metadata.get("warning")
    if force_alignment_metadata.get("error"):
        alignment_metadata["error"] = force_alignment_metadata.get("error")

    alignment_metadata["method"] = "text_first_rewrite_grouping"
    word_source = "WHISPERX" if alignment_metadata.get("used") else "WHISPER"
    log(f"Word source: {word_source}, words={len(source_words)}")
    for index, word in enumerate(source_words[:20], start=1):
        log(
            f"Word[{index}] token={get_word_token(word)!r} "
            f"start={format_seconds_for_ffmpeg(float(word.get('start', 0.0)))} "
            f"end={format_seconds_for_ffmpeg(float(word.get('end', 0.0)))}"
        )

    sentence_groups = build_text_first_sentence_groups(
        source_words,
        whisper_segments=source_segments,
    )
    log(
        f"Built {len(sentence_groups)} text-first sentence groups from rewritten transcript"
    )
    aligned_segments = []
    min_step = 1.0 / DEFAULT_AUDIO_SAMPLE_RATE
    for index, sentence in enumerate(sentence_groups):
        mapped_words = list(sentence.get("words") or [])
        if not mapped_words:
            continue

        first_word_start = float(mapped_words[0]["start"])
        last_word_end = float(mapped_words[-1]["end"])
        start_time = max(0.0, first_word_start)
        end_time = min(float(media_duration), max(last_word_end, start_time + min_step))
        if end_time <= start_time:
            continue

        tolerance = SEGMENT_ASSERT_TOLERANCE_SECONDS
        assert start_time <= (first_word_start + tolerance), "sentence start exceeds first word boundary tolerance"
        assert end_time >= (last_word_end - tolerance), "sentence end precedes last word boundary tolerance"

        log(
            f"Sentence[{index + 1}] start={format_seconds_for_ffmpeg(start_time)} "
            f"end={format_seconds_for_ffmpeg(end_time)} words={len(mapped_words)} "
            f"reason={sentence.get('split_reason', 'unknown')}"
        )

        raw_text = build_preserved_text_from_words(mapped_words)
        if not raw_text:
            raw_text = sentence.get("text", "") or build_text_from_word_list(mapped_words) or " "

        words_per_second = get_words_per_second(mapped_words)
        confidence = average_word_confidence(mapped_words)
        confidence_status = get_confidence_status(confidence)
        display_start, display_end = compute_subtitle_display_timing(
            start_time,
            end_time,
            raw_text,
            words_per_second,
        )
        segment_payload = {
            "clip_id": index,
            "start": round_timestamp_value(start_time),
            "end": round_timestamp_value(end_time),
            "subtitleStart": display_start,
            "subtitleEnd": display_end,
            "word_start": round_timestamp_value(first_word_start),
            "word_end": round_timestamp_value(last_word_end),
            "original_start": round_timestamp_value(first_word_start),
            "original_end": round_timestamp_value(last_word_end),
            "natural_start": round_timestamp_value(first_word_start),
            "natural_end": round_timestamp_value(last_word_end),
            "words_per_second": words_per_second,
            "silenceAdjustment": 0.0,
            "energyAdjustment": 0.0,
            "startPadding": 0.0,
            "endPadding": 0.0,
            "raw_text": raw_text,
            "text": raw_text,
            "lines": [raw_text],
            "readingSpeed": compute_reading_speed(raw_text, start_time, end_time),
            "confidence": confidence,
            "confidenceStatus": confidence_status,
            "needsReview": confidence_status == "needs_review",
            "auto_aligned": True,
            "alignment_mode": normalize_alignment_mode(resolved_mode),
            "alignment_method": "text_first_rewrite_grouping",
            "forced_alignment_used": bool(alignment_metadata.get("used")),
            "text_corrected": False,
            "text_source": "text_first_rewrite_grouping",
            "confidence_expanded": False,
            "words": mapped_words,
            "source_words": mapped_words,
            "word_indices": list(sentence.get("word_indices") or []),
        }
        aligned_segments.append(segment_payload)

    aligned_segments = refine_segment_boundaries_with_energy(audio, aligned_segments)
    aligned_segments = enforce_full_word_coverage(aligned_segments, source_words, media_duration)
    aligned_segments = validate_and_repair_segments(
        aligned_segments,
        words=source_words,
        media_duration=media_duration,
        stage_label="transcribe_final_precheck",
    )
    aligned_segments = enforce_strict_timeline_continuity(
        aligned_segments,
        media_duration=media_duration,
        stage_label="transcribe_timeline_continuity",
    )

    assigned_indices = []
    for segment in aligned_segments:
        assigned_indices.extend(int(index) for index in (segment.get("word_indices") or []))
    assigned_unique = set(assigned_indices)
    expected_total = len(source_words)
    duplicate_assignments = len(assigned_indices) - len(assigned_unique)
    missing_indices = [index for index in range(expected_total) if index not in assigned_unique]
    log(
        "Post-segmentation integrity: "
        f"words_total={expected_total} assigned={len(assigned_indices)} "
        f"unique={len(assigned_unique)} duplicates={duplicate_assignments} "
        f"missing={len(missing_indices)}"
    )
    if missing_indices or duplicate_assignments:
        log(f"Integrity warning details: missing_sample={missing_indices[:12]}")

    for segment in aligned_segments[:20]:
        log(
            f"Final clip boundary clip_id={segment.get('clip_id')} "
            f"start={format_seconds_for_ffmpeg(float(segment.get('start', 0.0)))} "
            f"end={format_seconds_for_ffmpeg(float(segment.get('end', 0.0)))}"
        )

    audio_duration = 0.0
    try:
        audio_duration = float(get_media_duration(audio))
    except Exception as exc:
        log(f"Audio duration check failed for observability: {exc}")
        audio_duration = 0.0

    log_pipeline_observability(
        video_duration=media_duration,
        audio_duration=audio_duration,
        whisper_segments=source_segments,
        aligned_words=source_words,
        alignment_metadata=alignment_metadata,
        sentence_groups=sentence_groups,
        final_segments=aligned_segments,
    )

    cache_payload = {
        "segments": aligned_segments,
        "transcript_words": display_words,
        "alignment": alignment_metadata,
    }
    del cache_payload

    return normalize_segments(aligned_segments), display_words, alignment_metadata


def transcribe_segment_audio(audio_path):
    model = get_whisper_model()
    segments, _ = model.transcribe(
        audio_path,
        language=SOURCE_TRANSCRIBE_LANGUAGE,
        task="translate",
        word_timestamps=False,
        initial_prompt=WHISPER_INITIAL_PROMPT,
        temperature=0.0,
        condition_on_previous_text=False,
        no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
    )

    transcript_parts = []
    for segment in segments:
        text = (segment.text or "").strip()
        if text:
            transcript_parts.append(text)

    return clean_whisper_translated_text(" ".join(transcript_parts))


def extract_audio_clip_from_audio(audio_path, start_time, end_time, output_path):
    safe_start = max(float(start_time), 0.0)
    safe_end = max(float(end_time), safe_start + (1.0 / DEFAULT_AUDIO_SAMPLE_RATE))
    duration = max(safe_end - safe_start, 1.0 / DEFAULT_AUDIO_SAMPLE_RATE)
    run_command([
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ss", f"{safe_start:.3f}",
        "-t", f"{duration:.3f}",
        "-acodec", "pcm_s16le",
        "-ar", str(DEFAULT_AUDIO_SAMPLE_RATE),
        "-ac", "1",
        output_path,
    ])
    return output_path


def transcribe_clip_text(audio_path, clip_id, start, end, alignment_mode, media_duration, audio_signature=None):
    del audio_signature
    session_dir = os.path.dirname(audio_path)
    base_start = round_timestamp_value(max(float(start), 0.0))
    base_end = round_timestamp_value(max(float(end), float(start)))
    clip_duration = max(base_end - base_start, 0.0)
    attempts = []

    if clip_duration < 3.0:
        attempts.append((
            round_timestamp_value(max(base_start - 0.2, 0.0)),
            round_timestamp_value(min(base_end + 0.5, float(media_duration or end))),
        ))

    attempts.append((base_start, base_end))

    extended_end = round_timestamp_value(min(base_end + CLIP_TEXT_RETRY_EXTENSION_SECONDS, float(media_duration or end)))
    if extended_end > base_end:
        attempts.append((base_start, extended_end))

    shortened_end = round_timestamp_value(max(base_end - 0.2, base_start + 0.6))
    if shortened_end < base_end:
        attempts.append((base_start, shortened_end))

    model = get_clip_text_model()
    best_text = ""

    for attempt_index, (attempt_start, attempt_end) in enumerate(attempts):
        clip_audio_path = os.path.join(session_dir, f"clip_text_{clip_id}_{uuid.uuid4().hex}.wav")
        try:
            extract_audio_clip_from_audio(audio_path, attempt_start, attempt_end, clip_audio_path)
            with clip_text_transcribe_lock:
                segments, _ = model.transcribe(
                    clip_audio_path,
                    language=DISPLAY_TRANSCRIBE_LANGUAGE,
                    **build_whisper_transcribe_kwargs("translate", False, strict_mode=should_use_heavy_refinement(alignment_mode)),
                )
            transcript_parts = []
            for segment in segments:
                text = clean_text(segment.text or "")
                if text:
                    transcript_parts.append(text)
            candidate_text = finalize_domain_transcript_text(" ".join(transcript_parts))
            if candidate_text:
                best_text = candidate_text
            if candidate_text and not contains_hallucination_phrase(candidate_text) and not looks_like_incomplete_clip_text(candidate_text):
                break
            if attempt_index < len(attempts) - 1 and candidate_text:
                reason = "hallucination-like phrase" if contains_hallucination_phrase(candidate_text) else "incomplete-looking result"
                log(
                    f"Retrying clip text for clip {get_clip_number(clip_id)} "
                    f"after {reason}: {candidate_text!r}"
                )
        finally:
            if os.path.exists(clip_audio_path):
                try:
                    os.remove(clip_audio_path)
                except OSError:
                    pass

    return best_text


def apply_clip_text_transcriptions(audio_path, segments, alignment_mode, media_duration):
    if not segments:
        return []

    def process_segment(segment):
        if not should_refresh_clip_text(segment, alignment_mode):
            return {
                **segment,
                "text_source": segment.get("text_source", "aligned_words"),
            }

        clip_id = int(segment.get("clip_id", 0))
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        whisper_text = transcribe_clip_text(
            audio_path,
            clip_id,
            start,
            end,
            alignment_mode,
            media_duration,
        )
        if not whisper_text:
            return segment

        raw_text = whisper_text
        refined_text = refine_clip_text(raw_text, segment.get("confidence", 1.0))
        formatted_text, subtitle_lines, reading_speed = apply_subtitle_formatting(
            refined_text,
            start,
            end,
            alignment_mode,
        )
        return {
            **segment,
            "raw_text": raw_text,
            "text": formatted_text,
            "lines": subtitle_lines,
            "readingSpeed": reading_speed,
            "text_source": "whisper_clip",
        }

    max_workers = min(CLIP_TEXT_WORKERS, len(segments))
    if max_workers <= 1:
        return [process_segment(segment) for segment in segments]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(process_segment, segments))


def extract_audio_segment(input_video, start_time, end_time):
    resolved_input_video = resolve_safe_path(input_video)
    safe_start = max(float(start_time), 0.0)
    safe_end = max(float(end_time), safe_start + (1.0 / DEFAULT_AUDIO_SAMPLE_RATE))
    session_dir = get_session_dir(resolved_input_video)
    output_path = os.path.join(session_dir, f"segment_{uuid.uuid4().hex}.wav")
    duration = max(safe_end - safe_start, 1.0 / DEFAULT_AUDIO_SAMPLE_RATE)
    log(
        "Extracting segment audio "
        f"start={safe_start:.3f}s end={safe_end:.3f}s duration={duration:.3f}s "
        f"source={os.path.basename(resolved_input_video)}"
    )

    run_command([
        "ffmpeg", "-y",
        "-i", resolved_input_video,
        "-ss", f"{safe_start:.3f}",
        "-t", f"{duration:.3f}",
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(DEFAULT_AUDIO_SAMPLE_RATE),
        "-ac", "1",
        output_path,
    ])

    return output_path


def apply_stream_offset_to_clip_range(start_seconds, end_seconds, stream_offset_seconds):
    start = max(float(start_seconds) + float(stream_offset_seconds), 0.0)
    end = max(float(end_seconds) + float(stream_offset_seconds), start)
    return start, end


def create_clips(input_video, segments):
    base_name = sanitize_export_base_name(input_video)
    base_dir = os.path.dirname(input_video)
    normalized_segments = normalize_segments(segments)
    validate_continuity(normalized_segments)
    stream_times = get_media_stream_start_times(input_video)
    stream_offset_seconds = float(stream_times.get("offset_seconds", 0.0))
    log(
        "Clip export stream alignment "
        f"video_start={format_seconds_for_ffmpeg(stream_times.get('video_start', 0.0))} "
        f"audio_start={format_seconds_for_ffmpeg(stream_times.get('audio_start', 0.0))} "
        f"offset={format_seconds_for_ffmpeg(stream_offset_seconds)}"
    )

    for i, seg in enumerate(normalized_segments):
        start_seconds = float(seg["start"])
        end_seconds = float(seg["end"])
        duration_seconds = end_seconds - start_seconds

        if duration_seconds <= 0:
            log(f"Skipping segment {i+1} because duration is not positive")
            continue

        clip_start, clip_end = apply_stream_offset_to_clip_range(
            start_seconds,
            end_seconds,
            stream_offset_seconds,
        )
        if clip_end <= clip_start:
            log(f"Skipping segment {i+1} due to invalid offset-adjusted range")
            continue

        log(
            f"Creating clip {i + 1} "
            f"segment_start={format_seconds_for_ffmpeg(start_seconds)} "
            f"segment_end={format_seconds_for_ffmpeg(end_seconds)} "
            f"ffmpeg_start={format_seconds_for_ffmpeg(clip_start)} "
            f"ffmpeg_end={format_seconds_for_ffmpeg(clip_end)}"
        )

        clip_path = os.path.join(base_dir, f"{base_name}_clip_{i+1}.mp4")

        run_command([
            "ffmpeg", "-y",
            "-ss", format_seconds_for_ffmpeg(clip_start),
            "-to", format_seconds_for_ffmpeg(clip_end),
            "-i", input_video,
            "-map", "0:v:0",
            "-map", "0:a:0",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "fast",
            "-crf", "23",
            "-avoid_negative_ts", "make_zero",
            clip_path
        ])

        txt_path = os.path.join(base_dir, f"{base_name}_clip_{i+1}_en.txt")

        with open(txt_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(seg.get("text") or "")


def get_generated_files(video_path):
    base_name = sanitize_export_base_name(video_path)
    base_dir = os.path.dirname(video_path)
    clip_files = []
    text_files = []

    for file_name in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, file_name)
        if not os.path.isfile(full_path):
            continue

        if file_name.startswith(f"{base_name}_clip_") and file_name.endswith(".mp4"):
            clip_files.append(full_path)
        elif file_name.startswith(f"{base_name}_clip_") and file_name.endswith("_en.txt"):
            text_files.append(full_path)

    return clip_files, text_files


def build_output_summary(video_path, segments_count):
    clip_files, text_files = get_generated_files(video_path)

    return {
        "clips": len(clip_files),
        "text_files": len(text_files),
        "segments": segments_count,
        "clip_files": clip_files,
        "text_file_paths": text_files,
    }


def build_preview_url(video_path):
    if not video_path:
        return ""

    relative_path = os.path.relpath(video_path, TEMP_DIR).replace(os.sep, "/")
    if relative_path.startswith(".."):
        return ""

    return f"/temp/{relative_path}"


def build_zip_archive(video_path):
    clip_files, text_files = get_generated_files(video_path)
    archive_files = clip_files + text_files

    if not archive_files:
        raise FileNotFoundError("No generated output files were found.")

    base_name = sanitize_export_base_name(video_path)
    base_dir = os.path.dirname(video_path)
    zip_path = os.path.join(base_dir, f"{base_name}.zip")

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file_path in clip_files:
            archive_name = os.path.join(base_name, "videos", os.path.basename(file_path))
            zipf.write(file_path, archive_name)
        for file_path in text_files:
            archive_name = os.path.join(base_name, "text", os.path.basename(file_path))
            zipf.write(file_path, archive_name)

    return zip_path


def normalize_text_for_hash(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def sanitize_export_base_name(file_name_or_path):
    original_name = os.path.splitext(os.path.basename(file_name_or_path or ""))[0]
    normalized = re.sub(r"\s+", "_", original_name.strip())
    normalized = re.sub(r"[^A-Za-z0-9_]+", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "video"


def clean_text(text):
    return (text or "").strip()


def get_word_token(word):
    return clean_text(word.get("word", "") or word.get("text", ""))


def round_timestamp_value(value):
    return round(float(value), TIMESTAMP_PRECISION_DECIMALS)


def rebuild_segment_text_from_words(words, segment, fallback_text=""):
    if not isinstance(words, list) or not words:
        return fallback_text or ""

    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", 0.0))
    matched_words = []

    for word in words:
        try:
            midpoint = (float(word.get("start", 0.0)) + float(word.get("end", 0.0))) / 2.0
        except (TypeError, ValueError):
            continue

        if start <= midpoint < end:
            token = clean_text(word.get("text", ""))
            if token:
                matched_words.append(token)

    if not matched_words:
        return fallback_text or ""

    return clean_transcript_text(" ".join(matched_words)) or (fallback_text or "")


def clean_transcript_text(text):
    cleaned = normalize_text_for_hash(text)
    cleaned = re.sub(r"\s+([,.;!?])", r"\1", cleaned)
    cleaned = re.sub(r"([(])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([)\]%])", r"\1", cleaned)
    cleaned = re.sub(r"\s+'\s*", "'", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def build_preserved_text_from_words(words):
    tokens = [clean_text(get_word_token(word)) for word in (words or [])]
    tokens = [token for token in tokens if token]
    if not tokens:
        return ""

    # Keep token sequence exactly as assigned words; only normalize whitespace/punctuation spacing.
    rebuilt = clean_transcript_text(" ".join(tokens))
    if not rebuilt:
        return ""
    rebuilt = rebuilt[0].upper() + rebuilt[1:] if len(rebuilt) > 1 else rebuilt.upper()
    if rebuilt[-1] not in ".!?":
        rebuilt = f"{rebuilt}."
    return rebuilt


def refine_clip_text(text: str, confidence: float) -> str:
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 1.0

    if confidence_value < 0.65:
        return "⚠️ Low confidence audio"

    cleaned = clean_transcript_text(text)
    if not cleaned:
        return ""

    filler_patterns = (
        r"\buh\b",
        r"\bum\b",
        r"\bah\b",
        r"\byou know\b",
        r"\blike\b",
    )
    for pattern in filler_patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    repeated_word_pattern = re.compile(r"\b(\w+)( \1\b)+", flags=re.IGNORECASE)
    previous = None
    while cleaned != previous:
        previous = cleaned
        cleaned = repeated_word_pattern.sub(r"\1", cleaned)

    cleaned = re.sub(r",\s*,+", ", ", cleaned)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
    if not cleaned:
        return ""

    word_tokens = re.findall(r"[^\W\d_]+", cleaned, flags=re.UNICODE)
    meaningful_words = [token for token in word_tokens if len(token) > 1]
    normalized_tokens = [token.lower() for token in meaningful_words]
    unique_tokens = set(normalized_tokens)
    repeated_ratio = 1.0
    if normalized_tokens:
        repeated_ratio = len(unique_tokens) / len(normalized_tokens)

    if len(meaningful_words) < 2 or repeated_ratio <= 0.45:
        return ""

    final_words = cleaned.split()
    if len(final_words) > 15:
        cleaned = " ".join(final_words[:15]).rstrip(" ,.;!?") + "..."

    cleaned = clean_transcript_text(cleaned)
    if not cleaned:
        return ""

    cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
    if cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned.strip()


def apply_domain_text_replacements(text):
    cleaned = clean_transcript_text(text)
    if not cleaned:
        return ""

    updated = cleaned
    for source, target in DOMAIN_TEXT_REPLACEMENTS:
        updated = re.sub(rf"\b{re.escape(source)}\b", target, updated, flags=re.IGNORECASE)
    updated = re.sub(r"\bapps\b", "APPS", updated, flags=re.IGNORECASE)
    return clean_transcript_text(updated)


def remove_duplicate_phrases(text):
    cleaned = clean_transcript_text(text)
    if not cleaned:
        return ""

    updated = re.sub(r"\b(.+?)\s+\1\b", r"\1", cleaned, flags=re.IGNORECASE)
    sentences = re.split(r"(?<=[.!?])\s+", updated)
    deduped = []
    for sentence in sentences:
        normalized = clean_transcript_text(sentence)
        if not normalized:
            continue
        if deduped and deduped[-1].lower() == normalized.lower():
            continue
        deduped.append(normalized)
    return " ".join(deduped)


def finalize_domain_transcript_text(text):
    cleaned = apply_domain_text_replacements(text)
    cleaned = remove_duplicate_phrases(cleaned)
    cleaned = clean_transcript_text(cleaned)
    if not cleaned:
        return ""
    cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
    if cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


def clean_whisper_translated_text(text):
    cleaned = (text or "").replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = finalize_domain_transcript_text(cleaned)
    if not cleaned:
        return ""

    words = cleaned.split()
    if words and re.fullmatch(r"[A-Za-z]{1,2}", words[-1].rstrip(".!?,")):
        cleaned = " ".join(words[:-1]).strip()
        cleaned = finalize_domain_transcript_text(cleaned)

    cleaned = cleaned.replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\band then\b", ", then", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bso\b", ", so", cleaned, count=1, flags=re.IGNORECASE)
    cleaned = re.sub(r",\s*,+", ", ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")

    if not cleaned:
        return ""

    cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
    if cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."

    return cleaned


def normalize_token_for_alignment(token):
    normalized = clean_text(token).lower()
    normalized = re.sub(r"^[^\w]+|[^\w]+$", "", normalized)
    return normalized


def contains_hallucination_phrase(text):
    normalized = clean_transcript_text(text).lower()
    return any(phrase in normalized for phrase in HALLUCINATION_PHRASES)


def looks_like_incomplete_clip_text(text):
    cleaned = clean_transcript_text(text)
    if not cleaned:
        return True

    if cleaned[-1] in "-/'":
        return True

    if cleaned[-1] not in ".!?":
        if len(cleaned) < 18:
            return True

        trailing_match = re.search(r"([\w']+)$", cleaned, flags=re.UNICODE)
        trailing_token = (trailing_match.group(1) if trailing_match else "").strip()
        if trailing_token and len(trailing_token) <= 3:
            return True

    return False


def _starts_with_transition_phrase(text):
    normalized = clean_transcript_text(text).lower()
    return bool(re.match(r"^(first|next|then|after that|finally|now)\b[\s,.:;-]*", normalized))


def _tokenize_meaningful(text):
    tokens = re.findall(r"[A-Za-z']+", clean_transcript_text(text).lower())
    stopwords = {
        "the", "a", "an", "to", "of", "on", "in", "at", "for", "with", "and",
        "or", "but", "if", "then", "this", "that", "these", "those", "is",
        "are", "was", "were", "be", "been", "being", "it", "its", "as", "by",
    }
    return [token for token in tokens if len(token) > 2 and token not in stopwords]


def _strip_leading_transition_phrase(text):
    cleaned = clean_transcript_text(text)
    if not cleaned:
        return ""
    updated = re.sub(
        r"^(first|next|then|after that|finally|now)\b[\s,.:;-]*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    # Avoid connectors like "Then, and click..."
    updated = re.sub(r"^(and|then|so)\b[\s,.:;-]*", "", updated, flags=re.IGNORECASE).strip()
    return clean_transcript_text(updated)


def _strip_dangling_connector_tail(text):
    cleaned = clean_transcript_text(text)
    if not cleaned:
        return ""
    updated = re.sub(r"[\s,;:]+(and|or|then|so|now)\.?$", "", cleaned, flags=re.IGNORECASE).strip()
    return clean_transcript_text(updated)


def _normalize_segment_text_for_sequence(text):
    cleaned = clean_transcript_text(text)
    cleaned = _strip_leading_transition_phrase(cleaned)
    cleaned = _strip_dangling_connector_tail(cleaned)
    cleaned = re.sub(r",\s*([.!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s+([,.;!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


def _rewrite_narrative_sequence(text):
    cleaned = clean_transcript_text(text)
    if not cleaned:
        return ""

    if shutil.which("ollama") is None:
        return cleaned

    prompt = (
        "Rewrite the following tutorial transcript into a smooth, coherent step-by-step explanation. "
        "Preserve meaning and instruction order. Do not add new steps. Avoid repetitive connectors. "
        "Return only the rewritten tutorial text.\n\n"
        f"{cleaned}"
    )
    try:
        result = subprocess.run(
            ["ollama", "run", LOCAL_TEXT_CORRECTION_MODEL, prompt],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        log(f"Sequence rewriting skipped: {exc}")
        return cleaned

    rewritten = clean_transcript_text(result.stdout or "")
    return rewritten or cleaned


def _split_text_into_assignable_units(text):
    cleaned = clean_transcript_text(text)
    if not cleaned:
        return []
    units = [unit.strip() for unit in re.split(r"(?<=[.!?])\s+", cleaned) if unit.strip()]
    if len(units) <= 1:
        units = [part.strip() for part in re.split(r"(?<=[,;:])\s+", cleaned) if part.strip()]
    normalized = []
    for unit in units:
        item = clean_transcript_text(unit)
        if item and item[-1] not in ".!?":
            item = f"{item}."
        if item:
            normalized.append(item)
    return normalized


def _redistribute_rewritten_units_to_segments(ordered_segments, rewritten_text):
    units = _split_text_into_assignable_units(rewritten_text)
    if not units:
        return [segment.get("text", "") for segment in ordered_segments]

    base_lengths = []
    for segment in ordered_segments:
        base = _normalize_segment_text_for_sequence(segment.get("text", "") or segment.get("raw_text", ""))
        base_lengths.append(max(len(base), 1))

    total_length = sum(base_lengths) or 1
    remaining_units = len(units)
    assigned = []
    cursor = 0

    for index, segment in enumerate(ordered_segments):
        segments_left = len(ordered_segments) - index
        if cursor >= len(units):
            assigned.append(_normalize_segment_text_for_sequence(segment.get("text", "") or segment.get("raw_text", "")))
            continue

        target = max(18, int(round((base_lengths[index] / total_length) * len(clean_transcript_text(rewritten_text)))))
        chunk = []
        char_count = 0
        while cursor < len(units):
            unit = units[cursor]
            projected = char_count + (1 if chunk else 0) + len(unit)
            must_keep_for_rest = (len(units) - (cursor + 1)) < (segments_left - 1)
            if chunk and projected > target and not must_keep_for_rest:
                break
            chunk.append(unit)
            char_count = projected
            cursor += 1
            remaining_units -= 1
            if char_count >= target and not must_keep_for_rest:
                break

        if not chunk:
            chunk = [units[cursor]]
            cursor += 1
            remaining_units -= 1

        merged = clean_transcript_text(" ".join(chunk))
        if merged and merged[-1] not in ".!?":
            merged = f"{merged}."
        assigned.append(merged or _normalize_segment_text_for_sequence(segment.get("text", "") or segment.get("raw_text", "")))

    return assigned


def apply_cross_segment_flow_text(segments):
    if not segments:
        return []

    ordered = [dict(segment) for segment in normalize_segments(segments)]
    if len(ordered) <= 1:
        return ordered

    normalized_texts = [
        _normalize_segment_text_for_sequence(segment.get("text", "") or segment.get("raw_text", ""))
        for segment in ordered
    ]
    combined = clean_transcript_text(" ".join(text for text in normalized_texts if text))
    rewritten = _rewrite_narrative_sequence(combined)
    reassigned_texts = _redistribute_rewritten_units_to_segments(ordered, rewritten)

    previous_prefix = ""
    for index, segment in enumerate(ordered):
        updated_text = _normalize_segment_text_for_sequence(reassigned_texts[index] if index < len(reassigned_texts) else "")
        if not updated_text:
            updated_text = normalized_texts[index]
        if not updated_text:
            continue

        # Prevent repetitive transition starts across neighboring segments.
        current_prefix = re.match(r"^(first|next|then|after that|finally|now)\b", updated_text.lower())
        if current_prefix:
            prefix_value = current_prefix.group(1)
            if prefix_value == previous_prefix:
                updated_text = _strip_leading_transition_phrase(updated_text)
                if updated_text and updated_text[-1] not in ".!?":
                    updated_text = f"{updated_text}."
            else:
                previous_prefix = prefix_value
        else:
            previous_prefix = ""

        segment["text"] = updated_text
        segment["raw_text"] = updated_text
        segment["lines"] = [updated_text]
        segment["readingSpeed"] = compute_reading_speed(
            updated_text,
            float(segment.get("start", 0.0)),
            float(segment.get("end", 0.0)),
        )

    return ordered


def count_sentence_endings(text):
    return len(re.findall(r"[.!?]", clean_transcript_text(text)))


def should_refresh_clip_text(segment, alignment_mode):
    if should_use_heavy_refinement(alignment_mode):
        return True

    refresh_mode = FAST_CLIP_TEXT_REFRESH_MODE
    if refresh_mode == "all":
        return True
    if refresh_mode == "none":
        return False

    text = clean_transcript_text(segment.get("text", ""))
    confidence = segment.get("confidence")
    duration = max(float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)), 0.0)

    if not text:
        return True
    if looks_like_incomplete_clip_text(text):
        return True
    if segment.get("needsReview"):
        return True
    if confidence is not None and float(confidence) < 0.7:
        return True
    if duration >= 3.0 and count_sentence_endings(text) == 0:
        return True
    if len(text) < 14 and duration >= 1.5:
        return True

    return False


def normalize_subtitle_punctuation(text):
    cleaned = clean_transcript_text(text)
    cleaned = re.sub(r"([!?.,])\1+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([,.;!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:])([A-Za-z])", r"\1 \2", cleaned)
    if cleaned and cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


def compute_reading_speed(text, start, end):
    duration = max(float(end) - float(start), 0.001)
    return round(len((text or "").replace("\n", "")) / duration, 3)


def is_bad_break_token(token):
    return clean_text(token).lower().strip(" ,.;:!?") in BAD_LINE_BREAK_WORDS


def choose_balanced_break_index(words):
    if len(words) < 4:
        return None

    joined_lengths = [len(word) for word in words]
    total_chars = sum(joined_lengths) + max(len(words) - 1, 0)
    target = total_chars / 2.0
    running = 0
    best_index = None
    best_score = None

    for index in range(1, len(words)):
        running += len(words[index - 1]) + (1 if index > 1 else 0)
        left = " ".join(words[:index]).strip()
        right = " ".join(words[index:]).strip()
        if not left or not right:
            continue
        if len(left) > SUBTITLE_MAX_LINE_CHARS or len(right) > SUBTITLE_MAX_LINE_CHARS:
            continue

        previous_token = words[index - 1]
        next_token = words[index]
        punctuation_bonus = 10 if previous_token.endswith((",", ".", "!", "?", ";", ":")) else 0
        bad_break_penalty = 20 if is_bad_break_token(previous_token) or is_bad_break_token(next_token) else 0
        number_penalty = 18 if any(char.isdigit() for char in previous_token + next_token) else 0
        name_penalty = 14 if (
            previous_token[:1].isupper()
            and next_token[:1].isupper()
            and previous_token.isalpha()
            and next_token.isalpha()
        ) else 0
        balance_penalty = abs(len(left) - len(right))
        midpoint_penalty = abs(running - target)
        score = (
            punctuation_bonus
            - bad_break_penalty
            - number_penalty
            - name_penalty
            - balance_penalty
            - midpoint_penalty * 0.2
        )

        if best_score is None or score > best_score:
            best_score = score
            best_index = index

    return best_index


def format_subtitle_lines_locally(text, alignment_mode=ALIGNMENT_MODE):
    cleaned = normalize_subtitle_punctuation(text)
    if (
        not cleaned
        or not should_use_heavy_refinement(alignment_mode)
        or not LOCAL_LINE_BREAK_FORMATTING_ENABLED
        or shutil.which("ollama") is None
    ):
        return None

    prompt = (
        "Format this subtitle into one or two lines for readability. "
        "Do not change meaning. Return only the formatted subtitle text.\n\n"
        f"{cleaned}"
    )

    try:
        result = subprocess.run(
            ["ollama", "run", LOCAL_TEXT_CORRECTION_MODEL, prompt],
            capture_output=True,
            text=True,
            check=True,
            timeout=45,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        log(f"Local line formatting skipped: {exc}")
        return None

    formatted = normalize_text_for_hash(result.stdout or "")
    if not formatted:
        return None

    lines = [clean_text(line) for line in formatted.splitlines() if clean_text(line)]
    if 1 <= len(lines) <= SUBTITLE_MAX_LINES and all(len(line) <= SUBTITLE_MAX_LINE_CHARS for line in lines):
        return lines

    return None


def build_subtitle_lines(text, alignment_mode=ALIGNMENT_MODE):
    cleaned = normalize_subtitle_punctuation(text)
    if not cleaned:
        return []

    llm_lines = format_subtitle_lines_locally(cleaned, alignment_mode)
    if llm_lines:
        return llm_lines

    if len(cleaned) <= SUBTITLE_MAX_LINE_CHARS:
        return [cleaned]

    words = cleaned.split()
    break_index = choose_balanced_break_index(words)
    if break_index is not None:
        return [
            " ".join(words[:break_index]).strip(),
            " ".join(words[break_index:]).strip(),
        ]

    line_one = cleaned[:SUBTITLE_MAX_LINE_CHARS].rstrip()
    remainder = cleaned[SUBTITLE_MAX_LINE_CHARS:].lstrip()
    if not remainder:
        return [line_one]
    return [line_one, remainder[:SUBTITLE_MAX_LINE_CHARS].rstrip()]


def apply_subtitle_formatting(text, start, end, alignment_mode=ALIGNMENT_MODE):
    normalized_text = normalize_subtitle_punctuation(text)
    lines = build_subtitle_lines(normalized_text, alignment_mode)
    final_text = "\n".join(lines) if lines else normalized_text
    reading_speed = compute_reading_speed(final_text, start, end)
    return final_text, lines, reading_speed


def compute_subtitle_display_timing(start, end, text, words_per_second=0.0):
    duration = max(float(end) - float(start), 0.001)
    char_count = len((text or "").replace("\n", ""))

    start_delay = SUBTITLE_START_DELAY_SECONDS
    end_delay = SUBTITLE_END_DELAY_SECONDS

    if words_per_second >= FAST_SPEECH_THRESHOLD_WPS:
        start_delay *= 0.55
        end_delay *= 0.75
    elif words_per_second <= SLOW_SPEECH_THRESHOLD_WPS:
        start_delay *= 1.15
        end_delay *= 1.2

    if char_count > 60:
        end_delay += 0.12
    elif char_count < 24:
        end_delay *= 0.85

    display_start = max(0.0, float(start) + start_delay)
    display_end = min(float(end) + end_delay, float(end) + max(0.0, VISUAL_MAX_DURATION_SECONDS - duration))
    if display_end <= display_start:
        display_end = display_start + max(0.2, min(duration, 1.0))

    return round_timestamp_value(display_start), round_timestamp_value(display_end)


def build_segment_payload(segment, fallback_text=""):
    start = round_timestamp_value(segment.get("start", 0.0))
    end = round_timestamp_value(segment.get("end", 0.0))
    return {
        **segment,
        "clip_id": int(segment.get("clip_id", 0)),
        "start": start,
        "end": end,
        "start_ms": seconds_to_milliseconds(start),
        "end_ms": seconds_to_milliseconds(end),
        "text": segment.get("text", fallback_text or ""),
    }


def is_sentence_boundary_token(text):
    stripped = clean_text(text)
    return stripped.endswith((".", "!", "?", "।", "॥"))


def split_text_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', clean_text(text or ""))
    return [sentence.strip() for sentence in sentences if sentence and sentence.strip()]


def group_has_sentence_boundary(words):
    if not words:
        return False
    return is_sentence_boundary_token(words[-1].get("text", ""))


def average_word_confidence(words):
    weighted_sum = 0.0
    total_weight = 0.0

    for word in words or []:
        try:
            confidence = word.get("confidence", word.get("probability"))
            if confidence is None:
                continue
            duration = max(float(word.get("end", 0.0)) - float(word.get("start", 0.0)), 0.0)
            if duration < MIN_CONFIDENCE_WORD_DURATION_SECONDS:
                continue
            weighted_sum += float(confidence) * duration
            total_weight += duration
        except (TypeError, ValueError):
            continue

    if total_weight <= 0:
        return None

    base_confidence = weighted_sum / total_weight
    text = build_text_from_word_list(words)
    if text:
        if any(symbol in text for symbol in ".!?"):
            base_confidence += 0.03
        if text[-1:] in ".!?" and len(text.split()) >= 3:
            base_confidence += 0.02

    return round(min(max(base_confidence, 0.0), 1.0), 4)


def get_confidence_status(confidence):
    if confidence is None:
        return "needs_review"
    return "needs_review" if float(confidence) < 0.6 else "good"


def get_word_gap(previous_word, next_word):
    try:
        return max(float(next_word["start"]) - float(previous_word["end"]), 0.0)
    except (KeyError, TypeError, ValueError):
        return 0.0


def get_word_group_duration(words):
    if not words:
        return 0.0
    return max(float(words[-1]["end"]) - float(words[0]["start"]), 0.0)


def compute_gap_profile(words):
    gaps = []
    for index in range(1, len(words or [])):
        gaps.append(get_word_gap(words[index - 1], words[index]))

    if not gaps:
        return {
            "gaps": [],
            "average_gap": 0.0,
            "median_gap": 0.0,
            "std_gap": 0.0,
            "large_pause_threshold": 0.0,
            "adaptive_silence_cap": 0.0,
        }

    sorted_gaps = sorted(gaps)
    average_gap = sum(gaps) / len(gaps)
    median_index = len(sorted_gaps) // 2
    if len(sorted_gaps) % 2 == 0:
        median_gap = (sorted_gaps[median_index - 1] + sorted_gaps[median_index]) / 2.0
    else:
        median_gap = sorted_gaps[median_index]

    variance = sum((gap - average_gap) ** 2 for gap in gaps) / len(gaps)
    std_gap = math.sqrt(max(variance, 0.0))
    large_pause_threshold = max(sorted_gaps[-1], average_gap) if len(gaps) == 1 else max(average_gap + std_gap, median_gap + std_gap)
    adaptive_silence_cap = median_gap + (std_gap * 0.5)

    return {
        "gaps": gaps,
        "average_gap": average_gap,
        "median_gap": median_gap,
        "std_gap": std_gap,
        "large_pause_threshold": large_pause_threshold,
        "adaptive_silence_cap": adaptive_silence_cap,
    }


def get_gap_profile_for_words(words, fallback_profile=None):
    profile = compute_gap_profile(words)
    if profile["gaps"]:
        return profile
    return fallback_profile or profile


def get_split_confidence_drop(previous_word, next_word):
    previous_confidence = previous_word.get("confidence", previous_word.get("probability"))
    next_confidence = next_word.get("confidence", next_word.get("probability"))
    if previous_confidence is None or next_confidence is None:
        return 0.0
    try:
        return max(float(previous_confidence) - float(next_confidence), 0.0)
    except (TypeError, ValueError):
        return 0.0


def get_semantic_completion_score(current_group, previous_word, next_word):
    score = 0.0
    current_text = get_group_text(current_group).strip().lower()
    next_text = clean_text(next_word.get("text", "")).strip().lower()
    if current_text and len(current_text.split()) >= 3:
        score += 0.3
    if group_has_sentence_boundary(current_group):
        score += 3.0
    elif current_text.endswith((",", ";", ":")):
        score += 0.15
    if next_text in CONNECTOR_WORDS:
        score -= 0.2
    return score


def score_split_boundary(previous_word, next_word, current_group, gap_profile):
    gap = get_word_gap(previous_word, next_word)
    punctuation_score = 4.0 if is_sentence_boundary_token(previous_word.get("text", "")) else 0.0
    average_gap = float(gap_profile.get("average_gap", 0.0))
    std_gap = float(gap_profile.get("std_gap", 0.0))
    relative_pause_score = 0.0
    if gap > 0.0:
        denominator = std_gap if std_gap > 0.0 else max(average_gap, gap, 0.001)
        relative_pause_score = max((gap - average_gap) / denominator, 0.0)

    confidence_drop_score = get_split_confidence_drop(previous_word, next_word) * 1.5
    semantic_score = get_semantic_completion_score(current_group, previous_word, next_word)

    return punctuation_score + relative_pause_score + confidence_drop_score + semantic_score


def is_large_pause(gap, gap_profile):
    if gap <= 0.0:
        return False
    large_pause_threshold = float(gap_profile.get("large_pause_threshold", 0.0))
    average_gap = float(gap_profile.get("average_gap", 0.0))
    std_gap = float(gap_profile.get("std_gap", 0.0))
    if gap > large_pause_threshold:
        return True
    if std_gap <= 0.0:
        return gap > average_gap
    return gap > (average_gap + std_gap)


def get_trailing_silence_buffer(current_group, next_group=None, media_duration=None, gap_profile=None):
    if not current_group:
        return 0.0

    last_word_end = float(current_group[-1]["end"])
    silence = 0.0
    if next_group:
        silence = max(float(next_group[0]["start"]) - last_word_end, 0.0)
    elif media_duration is not None:
        silence = max(float(media_duration) - last_word_end, 0.0)

    allowed_silence = 0.5
    return min(silence, allowed_silence)


def has_large_silence_between_groups(current_group, next_group, gap_profile=None):
    if not current_group or not next_group:
        return False
    pair_words = list(current_group) + list(next_group)
    profile = get_gap_profile_for_words(pair_words, gap_profile)
    return is_large_pause(get_word_gap(current_group[-1], next_group[0]), profile)


def find_words_in_range(words, start, end):
    matched = []
    for word in words or []:
        midpoint = (float(word.get("start", 0.0)) + float(word.get("end", 0.0))) / 2.0
        if float(start) <= midpoint < float(end):
            matched.append(word)
    return matched


def build_text_from_word_list(words):
    tokens = [clean_text(word.get("text", "")) for word in (words or [])]
    tokens = [token for token in tokens if token]
    if not tokens:
        return ""
    return clean_transcript_text(" ".join(tokens))


def _compute_percentile(values, percentile):
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = max(0.0, min(1.0, float(percentile))) * (len(sorted_values) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _build_whisper_prior_boundary_indices(indexed_words, whisper_segments):
    if not indexed_words or not whisper_segments:
        return set()

    timed_segments = extract_timed_segments_from_whisper_segments(whisper_segments)
    if not timed_segments:
        return set()

    boundaries = set()
    word_pointer = 0
    last_consumed_index = -1
    total_words = len(indexed_words)

    for segment in timed_segments:
        segment_end = float(segment.get("end", 0.0))
        current_last = last_consumed_index
        while word_pointer < total_words:
            global_index, word = indexed_words[word_pointer]
            midpoint = (float(word.get("start", 0.0)) + float(word.get("end", 0.0))) / 2.0
            if midpoint <= segment_end + 0.08:
                current_last = global_index
                word_pointer += 1
            else:
                break
        if 0 <= current_last < (indexed_words[-1][0]):
            boundaries.add(current_last)
        last_consumed_index = max(last_consumed_index, current_last)

    return boundaries


def _distribute_counts_to_total(weights, total):
    if total <= 0:
        return [0 for _ in weights]
    if not weights:
        return []
    safe_weights = [max(float(weight), 0.0) for weight in weights]
    weight_sum = sum(safe_weights)
    if weight_sum <= 0:
        base = [0 for _ in safe_weights]
        for i in range(total):
            base[i % len(base)] += 1
        return base

    raw = [(weight / weight_sum) * total for weight in safe_weights]
    counts = [max(int(math.floor(value)), 0) for value in raw]
    remainders = [(raw[index] - counts[index], index) for index in range(len(raw))]
    allocated = sum(counts)

    # Ensure each bucket has at least one item when possible.
    if total >= len(counts):
        for index in range(len(counts)):
            if counts[index] == 0:
                counts[index] = 1
                allocated += 1

    if allocated > total:
        for _, index in sorted(remainders):
            if allocated <= total:
                break
            if counts[index] > 1:
                counts[index] -= 1
                allocated -= 1
    elif allocated < total:
        for _, index in sorted(remainders, reverse=True):
            if allocated >= total:
                break
            counts[index] += 1
            allocated += 1
        index = 0
        while allocated < total:
            counts[index % len(counts)] += 1
            allocated += 1
            index += 1

    return counts


def build_text_first_sentence_groups(words, whisper_segments=None):
    indexed_words = [
        (index, word)
        for index, word in enumerate(words or [])
        if clean_text(word.get("text", ""))
    ]
    if not indexed_words:
        return []

    full_text = build_preserved_text_from_words([word for _, word in indexed_words]) or build_text_from_word_list(
        [word for _, word in indexed_words]
    )
    rewritten = _rewrite_narrative_sequence(full_text)
    rewritten = clean_transcript_text(rewritten) or clean_transcript_text(full_text)
    if rewritten and rewritten[-1] not in ".!?":
        rewritten = f"{rewritten}."

    step_texts = split_text_into_sentences(rewritten)
    if len(step_texts) <= 1:
        step_texts = [part.strip() for part in re.split(r"(?<=[;:])\s+", rewritten) if part.strip()]
    if not step_texts:
        step_texts = [rewritten] if rewritten else [full_text]

    # Remove immediate repeated steps while preserving order.
    deduped_steps = []
    for step in step_texts:
        cleaned = clean_transcript_text(step)
        if not cleaned:
            continue
        if deduped_steps and deduped_steps[-1].lower() == cleaned.lower():
            continue
        deduped_steps.append(cleaned if cleaned[-1] in ".!?" else f"{cleaned}.")
    step_texts = deduped_steps or [rewritten or full_text]

    token_counts = []
    for step in step_texts:
        tokens = [token for token in re.findall(r"[A-Za-z']+", step) if token.strip()]
        token_counts.append(max(len(tokens), 1))

    counts = _distribute_counts_to_total(token_counts, len(indexed_words))
    groups = []
    cursor = 0

    for step_index, step_text in enumerate(step_texts):
        if cursor >= len(indexed_words):
            break
        take = counts[step_index] if step_index < len(counts) else 0
        remaining_steps = len(step_texts) - step_index
        remaining_words = len(indexed_words) - cursor
        if step_index == len(step_texts) - 1:
            take = remaining_words
        else:
            min_keep_for_rest = max(remaining_steps - 1, 0)
            take = max(take, 1)
            take = min(take, max(1, remaining_words - min_keep_for_rest))

        chunk = indexed_words[cursor:cursor + take]
        if not chunk:
            continue

        chunk_indices = [global_index for global_index, _ in chunk]
        chunk_words = [word for _, word in chunk]
        groups.append({
            "words": chunk_words,
            "word_indices": chunk_indices,
            "split_reason": "text_first_rewrite",
            "start": float(chunk_words[0]["start"]),
            "end": float(chunk_words[-1]["end"]),
            "text": step_text,
        })
        cursor += take

    if cursor < len(indexed_words) and groups:
        # Append leftovers to the last step to preserve full coverage.
        extra_chunk = indexed_words[cursor:]
        groups[-1]["words"].extend([word for _, word in extra_chunk])
        groups[-1]["word_indices"].extend([index for index, _ in extra_chunk])
        groups[-1]["end"] = float(groups[-1]["words"][-1]["end"])
        merged_text = clean_transcript_text(groups[-1].get("text", ""))
        if merged_text and merged_text[-1] not in ".!?":
            merged_text = f"{merged_text}."
        groups[-1]["text"] = merged_text or build_preserved_text_from_words(groups[-1]["words"])

    whisper_prior_boundaries = _build_whisper_prior_boundary_indices(indexed_words, whisper_segments)
    if whisper_prior_boundaries:
        for group in groups:
            first_index = int(group["word_indices"][0])
            last_index = int(group["word_indices"][-1])
            if first_index <= last_index and (last_index in whisper_prior_boundaries):
                group["split_reason"] = "text_first_with_whisper_prior"

    log(
        "Text-first grouping: "
        f"words={len(indexed_words)} steps={len(step_texts)} groups={len(groups)}"
    )
    return groups


def build_sentence_groups_from_words(words, pause_threshold=SENTENCE_PAUSE_SPLIT_SECONDS, whisper_segments=None):
    indexed_words = [
        (index, word)
        for index, word in enumerate(words or [])
        if clean_text(word.get("text", ""))
    ]
    if not indexed_words:
        return []

    gaps = []
    for idx in range(1, len(indexed_words)):
        previous_word = indexed_words[idx - 1][1]
        current_word = indexed_words[idx][1]
        gaps.append(max(get_word_gap(previous_word, current_word), 0.0))

    median_gap = _compute_percentile(gaps, 0.5) if gaps else float(pause_threshold)
    high_gap = _compute_percentile(gaps, 0.75) if gaps else float(pause_threshold)
    very_high_gap = _compute_percentile(gaps, 0.9) if gaps else max(float(pause_threshold), median_gap)

    whisper_prior_boundaries = _build_whisper_prior_boundary_indices(indexed_words, whisper_segments)
    word_count = len(indexed_words)
    adaptive_soft_words = max(6, int(round(_compute_percentile([word_count], 0.5) ** 0.5)) + 8)
    adaptive_hard_words = max(adaptive_soft_words + 5, 24)
    adaptive_soft_duration = max(5.0, 1.8 + (high_gap * 12.0))
    adaptive_hard_duration = max(8.0, adaptive_soft_duration + 2.6)

    sentence_groups = []
    current_words = [indexed_words[0][1]]
    current_indices = [indexed_words[0][0]]
    best_split_index = None
    best_split_score = None
    best_split_reason = "semantic_score"

    for entry_index in range(1, len(indexed_words)):
        current_word_index, word = indexed_words[entry_index]
        previous_word = indexed_words[entry_index - 1][1]
        pause_gap = max(get_word_gap(previous_word, word), 0.0)
        current_duration = max(
            float(previous_word.get("end", 0.0)) - float(current_words[0].get("start", 0.0)),
            0.0,
        )
        current_length = len(current_words)
        previous_token = get_word_token(previous_word)
        next_token = get_word_token(word)

        split_score = 0.0
        split_reason = "semantic_context"
        if is_sentence_boundary_token(previous_token):
            split_score += 4.2
            split_reason = "punctuation"
        elif clean_text(previous_token).endswith((",", ";", ":")):
            split_score += 2.1
            split_reason = "clause_punctuation"

        if pause_gap >= very_high_gap and pause_gap > 0.0:
            split_score += 2.4
        elif pause_gap >= high_gap and pause_gap > 0.0:
            split_score += 1.2
        elif pause_gap >= max(median_gap, float(pause_threshold)):
            split_score += 0.6

        if current_word_index - 1 in whisper_prior_boundaries:
            split_score += 2.5
            if split_reason == "semantic_context":
                split_reason = "whisper_prior"

        if current_length >= adaptive_soft_words:
            split_score += 0.8
        if current_duration >= adaptive_soft_duration:
            split_score += 0.8

        next_is_sentence_startish = bool(next_token and next_token[:1].isupper())
        if next_is_sentence_startish and pause_gap >= median_gap:
            split_score += 0.6
        if (
            not clean_text(previous_token).endswith((".", "!", "?", ",", ";", ":"))
            and next_token
            and next_token[:1].islower()
            and pause_gap <= median_gap
            and current_length <= 4
        ):
            split_score -= 0.7

        if best_split_score is None or split_score > best_split_score:
            best_split_score = split_score
            best_split_index = current_length
            best_split_reason = split_reason

        should_split = split_score >= 3.0 and current_length >= 3
        overflow = (
            current_length >= adaptive_hard_words
            or current_duration >= adaptive_hard_duration
        )

        if should_split or overflow:
            split_index = best_split_index if overflow and best_split_index is not None and best_split_index >= 2 else current_length
            sentence_groups.append({
                "words": list(current_words[:split_index]),
                "word_indices": list(current_indices[:split_index]),
                "split_reason": best_split_reason if overflow else split_reason,
            })
            current_words = list(current_words[split_index:])
            current_indices = list(current_indices[split_index:])
            best_split_index = None
            best_split_score = None
            best_split_reason = "semantic_score"

        current_words.append(word)
        current_indices.append(current_word_index)

    if current_words:
        sentence_groups.append({
            "words": list(current_words),
            "word_indices": list(current_indices),
            "split_reason": "end_of_transcript",
        })

    merged_groups = []
    for group in sentence_groups:
        group_words = list(group.get("words") or [])
        if not group_words:
            continue
        if (
            merged_groups
            and (len(group_words) < 3 or len(build_text_from_word_list(group_words)) < 12)
            and not is_sentence_boundary_token(get_word_token(merged_groups[-1]["words"][-1]))
        ):
            merged_groups[-1]["words"].extend(group_words)
            merged_groups[-1]["word_indices"].extend(list(group.get("word_indices") or []))
            merged_groups[-1]["split_reason"] = "merged_short_fragment"
            continue
        merged_groups.append({
            "words": list(group_words),
            "word_indices": list(group.get("word_indices") or []),
            "split_reason": group.get("split_reason", "word_continuity"),
        })

    for group in merged_groups:
        group_words = list(group.get("words") or [])
        if not group_words:
            continue
        group["start"] = float(group_words[0]["start"])
        group["end"] = float(group_words[-1]["end"])
        preserved_text = build_preserved_text_from_words(group_words)
        group["text"] = preserved_text or build_text_from_word_list(group_words) or " "

    return merged_groups


def _clone_sentence_group(group, words, word_indices, split_reason):
    cloned = dict(group)
    cloned_words = list(words or [])
    cloned_indices = list(word_indices or [])
    cloned["words"] = cloned_words
    cloned["word_indices"] = cloned_indices
    cloned["split_reason"] = split_reason
    if cloned_words:
        cloned["start"] = float(cloned_words[0]["start"])
        cloned["end"] = float(cloned_words[-1]["end"])
        cloned["text"] = build_preserved_text_from_words(cloned_words) or build_text_from_word_list(cloned_words) or " "
    else:
        cloned["start"] = 0.0
        cloned["end"] = 0.0
        cloned["text"] = " "
    return cloned


def _split_group_by_scene_cut(group, scene_cut):
    words = list(group.get("words") or [])
    indices = list(group.get("word_indices") or [])
    if len(words) < 4 or len(indices) != len(words):
        return [group]

    start = float(words[0].get("start", 0.0))
    end = float(words[-1].get("end", start))
    if not (start < float(scene_cut) < end):
        return [group]

    candidates = []
    for split_index in range(2, len(words) - 1):
        left_end = float(words[split_index - 1].get("end", start))
        right_start = float(words[split_index].get("start", left_end))
        if right_start < left_end:
            continue
        boundary_midpoint = (left_end + right_start) / 2.0
        score = abs(boundary_midpoint - float(scene_cut))
        if is_sentence_boundary_token(get_word_token(words[split_index - 1])):
            score -= 0.08
        if clean_text(get_word_token(words[split_index])).lower().strip(",.;:!?") in {token.lower() for token in CONNECTOR_WORDS}:
            score += 0.06
        candidates.append((score, split_index))

    if not candidates:
        return [group]

    _, split_index = min(candidates, key=lambda item: item[0])
    left_words = words[:split_index]
    right_words = words[split_index:]
    left_indices = indices[:split_index]
    right_indices = indices[split_index:]
    if len(left_words) < 2 or len(right_words) < 2:
        return [group]

    left = _clone_sentence_group(group, left_words, left_indices, "scene_cut")
    right = _clone_sentence_group(group, right_words, right_indices, "scene_cut")
    left["scene_split_after"] = round_timestamp_value(float(scene_cut))
    right["scene_split_before"] = round_timestamp_value(float(scene_cut))
    return [left, right]


def _split_group_for_duration(group, target_seconds=5.4, max_seconds=7.8):
    words = list(group.get("words") or [])
    indices = list(group.get("word_indices") or [])
    if len(words) < 6:
        return [group]

    output = []
    pending_words = words
    pending_indices = indices

    while pending_words:
        segment_start = float(pending_words[0].get("start", 0.0))
        segment_end = float(pending_words[-1].get("end", segment_start))
        duration = max(segment_end - segment_start, 0.0)
        if duration <= max_seconds or len(pending_words) < 6:
            output.append(_clone_sentence_group(group, pending_words, pending_indices, group.get("split_reason", "sentence")))
            break

        target_boundary = segment_start + target_seconds
        candidates = []
        for split_index in range(3, len(pending_words) - 2):
            left_end = float(pending_words[split_index - 1].get("end", segment_start))
            left_duration = max(left_end - segment_start, 0.0)
            if left_duration < IDEAL_CLIP_MIN_SECONDS:
                continue
            score = abs(left_end - target_boundary)
            if is_sentence_boundary_token(get_word_token(pending_words[split_index - 1])):
                score -= 0.12
            if clean_text(get_word_token(pending_words[split_index])).lower().strip(",.;:!?") in {token.lower() for token in CONNECTOR_WORDS}:
                score += 0.08
            candidates.append((score, split_index))

        if not candidates:
            output.append(_clone_sentence_group(group, pending_words, pending_indices, group.get("split_reason", "sentence")))
            break

        _, split_index = min(candidates, key=lambda item: item[0])
        left_words = pending_words[:split_index]
        left_indices = pending_indices[:split_index]
        output.append(_clone_sentence_group(group, left_words, left_indices, "max_duration_split"))
        pending_words = pending_words[split_index:]
        pending_indices = pending_indices[split_index:]

    return output


def build_hybrid_sentence_groups(sentence_groups, scene_cuts, media_duration):
    if not sentence_groups:
        return []

    # Scene changes are treated as strong split hints to align clips with UI/screen transitions.
    cuts = sorted(
        round_timestamp_value(float(cut))
        for cut in (scene_cuts or [])
        if 0.0 < float(cut) < float(media_duration)
    )

    scene_split_groups = []
    for group in sentence_groups:
        working = [group]
        for cut in cuts:
            next_working = []
            for item in working:
                next_working.extend(_split_group_by_scene_cut(item, cut))
            working = next_working
        scene_split_groups.extend(working)

    duration_groups = []
    for group in scene_split_groups:
        duration_groups.extend(
            _split_group_for_duration(
                group,
                target_seconds=IDEAL_CLIP_MAX_SECONDS - 0.6,
                max_seconds=max(HARD_CLIP_MAX_SECONDS, 7.8),
            )
        )

    # Merge tiny leftovers to keep clips natural and readable while preserving chronology.
    merged = []
    for group in duration_groups:
        words = list(group.get("words") or [])
        if not words:
            continue
        if merged:
            previous = merged[-1]
            previous_duration = max(float(previous.get("end", 0.0)) - float(previous.get("start", 0.0)), 0.0)
            current_duration = max(float(group.get("end", 0.0)) - float(group.get("start", 0.0)), 0.0)
            combined_duration = max(float(group.get("end", 0.0)) - float(previous.get("start", 0.0)), 0.0)
            if (
                current_duration < IDEAL_CLIP_MIN_SECONDS
                and combined_duration <= (HARD_CLIP_MAX_SECONDS + 0.8)
                and previous_duration < IDEAL_CLIP_MAX_SECONDS
                and "scene_split_after" not in previous
                and "scene_split_before" not in group
            ):
                previous_words = list(previous.get("words") or [])
                previous_indices = list(previous.get("word_indices") or [])
                merged[-1] = _clone_sentence_group(
                    previous,
                    previous_words + words,
                    previous_indices + list(group.get("word_indices") or []),
                    "merged_short_segment",
                )
                continue
        merged.append(group)

    for index, group in enumerate(merged):
        group["group_id"] = index
    return merged


def get_group_time_range(words):
    if not words:
        return 0.0, 0.0
    return float(words[0]["start"]), float(words[-1]["end"])


def get_group_text(words, display_words=None):
    if display_words:
        start, end = get_group_time_range(words)
        matched_display_words = find_words_in_range(display_words, start, end)
        text = build_text_from_word_list(matched_display_words)
        if text:
            return text
    return build_text_from_word_list(words)


def extract_words_from_whisper_segments(segments):
    words = []
    for segment in segments:
        for word in (segment.words or []):
            token = clean_text(getattr(word, "word", ""))
            if not token:
                continue

            start = getattr(word, "start", None)
            end = getattr(word, "end", None)
            if start is None or end is None:
                continue

            payload = {
                "text": token,
                "start": float(start),
                "end": float(end),
            }
            probability = getattr(word, "probability", None)
            if probability is not None:
                payload["confidence"] = float(probability)
            words.append(payload)

    return normalize_words(words)


def extract_text_segments_from_whisper_segments(segments):
    extracted = []
    for segment in segments:
        text = clean_text(getattr(segment, "text", ""))
        if not text:
            continue
        start = getattr(segment, "start", None)
        end = getattr(segment, "end", None)
        if start is None or end is None:
            continue
        extracted.append({
            "start": round_timestamp_value(float(start)),
            "end": round_timestamp_value(float(end)),
            "text": clean_transcript_text(text),
        })
    return extracted


def extract_timed_segments_from_whisper_segments(segments):
    extracted = []
    for segment in segments:
        segment_words = []
        for word in (getattr(segment, "words", None) or []):
            token = clean_text(getattr(word, "word", ""))
            start = getattr(word, "start", None)
            end = getattr(word, "end", None)
            if not token or start is None or end is None:
                continue
            payload = {
                "text": token,
                "start": float(start),
                "end": float(end),
            }
            probability = getattr(word, "probability", None)
            if probability is not None:
                payload["confidence"] = float(probability)
            segment_words.append(payload)

        if not segment_words:
            continue

        text = clean_text(getattr(segment, "text", "")) or build_text_from_word_list(segment_words)
        extracted.append({
            "start": round_timestamp_value(float(segment_words[0]["start"])),
            "end": round_timestamp_value(float(segment_words[-1]["end"])),
            "text": clean_transcript_text(text),
            "words": normalize_words(segment_words),
        })

    return extracted


def build_text_from_text_segments(text_segments, start, end, fallback_text=""):
    if not text_segments:
        return fallback_text or ""

    matched = []
    for segment in text_segments:
        segment_start = float(segment.get("start", 0.0))
        segment_end = float(segment.get("end", segment_start))
        overlap_start = max(float(start), segment_start)
        overlap_end = min(float(end), segment_end)
        if overlap_end <= overlap_start:
            continue
        text = clean_text(segment.get("text", ""))
        if text:
            matched.append(text)

    if not matched:
        return fallback_text or ""

    return clean_transcript_text(" ".join(matched))


def split_timed_segment_on_punctuation(segment_payload):
    words = list(segment_payload.get("words") or [])
    if not words:
        return []

    sentence_groups = split_word_group_into_sentence_groups(words)
    if len(sentence_groups) <= 1:
        return [{
            "start": round_timestamp_value(float(words[0]["start"])),
            "end": round_timestamp_value(float(words[-1]["end"])),
            "text": clean_transcript_text(segment_payload.get("text", "") or build_text_from_word_list(words)),
            "words": words,
        }]

    split_segments = []
    for sentence_words in sentence_groups:
        split_segments.append({
            "start": round_timestamp_value(float(sentence_words[0]["start"])),
            "end": round_timestamp_value(float(sentence_words[-1]["end"])),
            "text": build_text_from_word_list(sentence_words),
            "words": sentence_words,
        })
    return split_segments


def merge_short_whisper_segments(segment_payloads):
    if not segment_payloads:
        return []

    merged = []
    index = 0
    while index < len(segment_payloads):
        current = {
            "start": float(segment_payloads[index]["start"]),
            "end": float(segment_payloads[index]["end"]),
            "text": segment_payloads[index].get("text", ""),
            "words": list(segment_payloads[index].get("words") or []),
        }

        if len(current["words"]) < 2 and index + 1 < len(segment_payloads):
            next_segment = segment_payloads[index + 1]
            merged_words = current["words"] + list(next_segment.get("words") or [])
            merged.append({
                "start": round_timestamp_value(float(merged_words[0]["start"])),
                "end": round_timestamp_value(float(merged_words[-1]["end"])),
                "text": clean_transcript_text(
                    " ".join(part for part in [current.get("text", ""), next_segment.get("text", "")] if part)
                ) or build_text_from_word_list(merged_words),
                "words": merged_words,
            })
            index += 2
            continue

        merged.append({
            "start": round_timestamp_value(float(current["words"][0]["start"])),
            "end": round_timestamp_value(float(current["words"][-1]["end"])),
            "text": clean_transcript_text(current.get("text", "")) or build_text_from_word_list(current["words"]),
            "words": current["words"],
        })
        index += 1

    return merged


def build_word_groups_from_whisper_segments(whisper_segments):
    if not whisper_segments:
        return []

    groups = []
    for segment in whisper_segments:
        groups.extend(split_timed_segment_on_punctuation(segment))

    return groups


def get_whisper_segment_clip_end(current_segment, next_segment, media_duration):
    words = list(current_segment.get("words") or [])
    if not words:
        if next_segment and next_segment.get("words"):
            next_words = list(next_segment.get("words") or [])
            return round_timestamp_value(float(next_words[0]["start"]))
        if next_segment:
            return round_timestamp_value(float(next_segment.get("start", current_segment.get("end", 0.0))))
        return round_timestamp_value(float(media_duration))

    if next_segment and next_segment.get("words"):
        next_words = list(next_segment.get("words") or [])
        return round_timestamp_value(float(next_words[0]["start"]))
    if next_segment:
        return round_timestamp_value(float(next_segment.get("start", words[-1]["end"])))

    return round_timestamp_value(float(media_duration))


def is_text_too_short(words):
    text = build_text_from_word_list(words)
    return len(words or []) < MIN_CLIP_WORD_COUNT or len(text) < 12


def get_words_per_second(words):
    duration = get_word_group_duration(words)
    if duration <= 0:
        return 0.0
    return len(words or []) / duration


def get_adaptive_padding(words):
    words_per_second = get_words_per_second(words)
    scale = 1.0
    if words_per_second >= FAST_SPEECH_THRESHOLD_WPS:
        scale = FAST_SPEECH_PADDING_SCALE
    elif words_per_second <= SLOW_SPEECH_THRESHOLD_WPS:
        scale = SLOW_SPEECH_PADDING_SCALE

    return (
        MICRO_PADDING_START_SECONDS * scale,
        MICRO_PADDING_END_SECONDS * scale,
        round(words_per_second, 4),
    )


def should_split_word_group(previous_word, next_word, current_group, silence_regions, energy_regions, gap_profile=None):
    gap = get_word_gap(previous_word, next_word)
    profile = gap_profile or get_gap_profile_for_words(current_group + [next_word])
    score = score_split_boundary(previous_word, next_word, current_group, profile)
    split_on_punctuation = is_sentence_boundary_token(previous_word.get("text", ""))
    split_on_pause = is_large_pause(gap, profile) and score >= 1.0
    return split_on_punctuation or split_on_pause


def split_words_on_pauses(words, silence_regions, energy_regions):
    if not words:
        return []

    global_gap_profile = compute_gap_profile(words)
    groups = []
    current_group = [words[0]]

    for word in words[1:]:
        previous_word = current_group[-1]
        gap = get_word_gap(previous_word, word)
        group_gap_profile = get_gap_profile_for_words(current_group + [word], global_gap_profile)
        split_on_gap = should_split_word_group(
            previous_word,
            word,
            current_group,
            silence_regions,
            energy_regions,
            group_gap_profile,
        )
        log(
            f"Word gap {clean_text(previous_word.get('text', ''))} -> {clean_text(word.get('text', ''))}: "
            f"{gap:.3f}s avg={group_gap_profile['average_gap']:.3f}s "
            f"std={group_gap_profile['std_gap']:.3f}s threshold={group_gap_profile['large_pause_threshold']:.3f}s "
            f"(split={split_on_gap})"
        )
        if split_on_gap:
            groups.append(current_group)
            current_group = [word]
            continue
        current_group.append(word)

    if current_group:
        groups.append(current_group)

    return groups


def split_word_group_into_sentence_groups(words):
    if not words:
        return []

    sentence_groups = []
    current_group = []

    for word in words:
        current_group.append(word)
        if is_sentence_boundary_token(word.get("text", "")):
            sentence_groups.append(current_group)
            current_group = []

    if current_group:
        sentence_groups.append(current_group)

    if len(sentence_groups) <= 1:
        return sentence_groups

    merged_groups = []
    index = 0
    while index < len(sentence_groups):
        current_group = list(sentence_groups[index])
        if len(current_group) < 2:
            if index + 1 < len(sentence_groups):
                sentence_groups[index + 1] = current_group + list(sentence_groups[index + 1])
            elif merged_groups:
                merged_groups[-1].extend(current_group)
            else:
                merged_groups.append(current_group)
            index += 1
            continue

        merged_groups.append(current_group)
        index += 1

    return merged_groups


def enforce_single_sentence_word_groups(groups):
    if not groups:
        return []

    sentence_groups = []
    for group in groups:
        sentence_groups.extend(split_word_group_into_sentence_groups(group))
    return sentence_groups


def find_best_split_index(words):
    if len(words) < 2:
        return None

    group_start = float(words[0]["start"])
    target_boundary = group_start + min(TARGET_CLIP_DURATION_SECONDS, get_word_group_duration(words) / 2.0)
    gap_profile = compute_gap_profile(words)
    best_index = None
    best_score = None

    for index in range(1, len(words)):
        left = words[:index]
        right = words[index:]
        left_duration = get_word_group_duration(left)
        right_duration = get_word_group_duration(right)
        if left_duration < MIN_CLIP_DURATION_SECONDS or right_duration < MIN_CLIP_DURATION_SECONDS:
            continue

        previous_word = words[index - 1]
        next_word = words[index]
        gap = get_word_gap(previous_word, next_word)
        split_score = score_split_boundary(previous_word, next_word, left, gap_profile)
        has_sentence_boundary = is_sentence_boundary_token(previous_word.get("text", ""))
        has_strong_pause = is_large_pause(gap, gap_profile)
        if not has_sentence_boundary and not has_strong_pause:
            continue
        boundary_time = float(previous_word["end"])
        distance_penalty = abs(boundary_time - target_boundary)
        score = split_score - (distance_penalty * 0.12)

        if best_score is None or score > best_score:
            best_score = score
            best_index = index

    return best_index


def split_long_word_groups(groups):
    output = []

    for group in groups:
        pending = [group]
        while pending:
            current = pending.pop(0)
            if get_word_group_duration(current) <= MAX_CLIP_DURATION_SECONDS:
                output.append(current)
                continue

            split_index = find_best_split_index(current)
            if split_index is None:
                output.append(current)
                continue

            left = current[:split_index]
            right = current[split_index:]
            if not left or not right:
                output.append(current)
                continue

            pending.insert(0, right)
            pending.insert(0, left)

    return output


def merge_short_word_groups(groups):
    if not groups:
        return []

    merged = []
    index = 0

    while index < len(groups):
        current = list(groups[index])
        current_duration = get_word_group_duration(current)

        if (
            current_duration >= MERGE_SHORT_CLIP_SECONDS
            and not is_text_too_short(current)
        ) or len(groups) == 1:
            merged.append(current)
            index += 1
            continue

        if (
            merged
            and not group_has_sentence_boundary(merged[-1])
            and not has_large_silence_between_groups(merged[-1], current)
        ):
            previous = merged[-1]
            previous.extend(current)
            index += 1
            continue

        if (
            index + 1 < len(groups)
            and not group_has_sentence_boundary(current)
            and not has_large_silence_between_groups(current, groups[index + 1])
        ):
            next_group = list(groups[index + 1])
            merged.append(current + next_group)
            index += 2
            continue

        merged.append(current)
        index += 1

    return merged


def merge_low_confidence_word_groups(groups):
    if len(groups) < 2:
        return groups

    merged = []
    index = 0

    while index < len(groups):
        current = list(groups[index])
        confidence = average_word_confidence(current)
        edge_confidences = []
        if current:
            if current[0].get("confidence") is not None:
                edge_confidences.append(float(current[0]["confidence"]))
            if current[-1].get("confidence") is not None:
                edge_confidences.append(float(current[-1]["confidence"]))
        weakest_edge = min(edge_confidences) if edge_confidences else None

        should_merge = (
            confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD
        ) or (
            weakest_edge is not None and weakest_edge < VERY_LOW_CONFIDENCE_THRESHOLD
        )

        if (
            should_merge
            and index + 1 < len(groups)
            and not group_has_sentence_boundary(current)
            and not has_large_silence_between_groups(current, groups[index + 1])
        ):
            merged.append(current + list(groups[index + 1]))
            index += 2
            continue

        if (
            should_merge
            and merged
            and not group_has_sentence_boundary(merged[-1])
            and not has_large_silence_between_groups(merged[-1], current)
        ):
            merged[-1].extend(current)
            index += 1
            continue

        merged.append(current)
        index += 1

    return merged


def group_starts_with_connector(words, display_words=None):
    text = get_group_text(words, display_words).lower().strip()
    return any(text.startswith(f"{connector} ") or text == connector for connector in CONNECTOR_WORDS)


def group_ends_with_connector(words, display_words=None):
    text = get_group_text(words, display_words).lower().strip(" ,;:-")
    return any(text.endswith(f" {connector}") or text == connector for connector in CONNECTOR_WORDS)


def groups_are_semantically_connected(current_group, next_group, display_words=None):
    current_text = get_group_text(current_group, display_words)
    next_text = get_group_text(next_group, display_words)
    gap = get_word_gap(current_group[-1], next_group[0])
    combined_duration = float(next_group[-1]["end"]) - float(current_group[0]["start"])
    gap_profile = get_gap_profile_for_words(list(current_group) + list(next_group))

    if group_has_sentence_boundary(current_group):
        return False

    if is_large_pause(gap, gap_profile):
        return False

    if group_ends_with_connector(current_group, display_words) or group_starts_with_connector(next_group, display_words):
        return True

    if (
        not is_sentence_boundary_token(current_text)
        and combined_duration <= HARD_CLIP_MAX_SECONDS
        and (is_text_too_short(current_group) or is_text_too_short(next_group))
    ):
        return True

    return False


def merge_semantically_connected_groups(groups, display_words=None):
    if len(groups) < 2:
        return groups

    merged = []
    index = 0
    while index < len(groups):
        current = list(groups[index])
        if index + 1 < len(groups):
            next_group = list(groups[index + 1])
            current_duration = get_word_group_duration(current)
            combined_duration = float(next_group[-1]["end"]) - float(current[0]["start"])
            current_wps = get_words_per_second(current)
            dynamic_limit = IDEAL_CLIP_MAX_SECONDS if current_wps >= FAST_SPEECH_THRESHOLD_WPS else HARD_CLIP_MAX_SECONDS
            if (
                not has_large_silence_between_groups(current, next_group)
                and groups_are_semantically_connected(current, next_group, display_words)
                and combined_duration <= dynamic_limit
            ):
                merged.append(current + next_group)
                index += 2
                continue
            if (
                current_duration < 1.0
                and combined_duration <= dynamic_limit
                and not group_has_sentence_boundary(current)
                and not has_large_silence_between_groups(current, next_group)
            ):
                merged.append(current + next_group)
                index += 2
                continue

        merged.append(current)
        index += 1

    return merged


def apply_llm_semantic_grouping(groups, display_words=None, alignment_mode=ALIGNMENT_MODE):
    if (
        not groups
        or not should_use_heavy_refinement(alignment_mode)
        or not LOCAL_SEMANTIC_GROUPING_ENABLED
        or shutil.which("ollama") is None
    ):
        return groups

    payload_lines = []
    for index, group in enumerate(groups, start=1):
        payload_lines.append(f"{index}. {get_group_text(group, display_words)}")

    prompt = (
        "Group the following transcript fragments into natural spoken subtitle sentences. "
        "Do not rewrite the text. Only merge fragments that belong to the same sentence. "
        "Return only JSON as an array of contiguous 1-based index groups like [[1,2],[3],[4,5]].\n\n"
        + "\n".join(payload_lines)
    )

    try:
        result = subprocess.run(
            ["ollama", "run", LOCAL_TEXT_CORRECTION_MODEL, prompt],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        raw = clean_text(result.stdout)
        start_index = raw.find("[")
        end_index = raw.rfind("]")
        if start_index == -1 or end_index == -1:
            return groups
        grouping = json.loads(raw[start_index:end_index + 1])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as exc:
        log(f"Local semantic grouping skipped: {exc}")
        return groups

    merged = []
    consumed = set()
    for block in grouping:
        if not isinstance(block, list) or not block:
            continue
        indices = [int(value) - 1 for value in block]
        if any(index < 0 or index >= len(groups) for index in indices):
            return groups
        if indices != list(range(indices[0], indices[-1] + 1)):
            return groups
        if any(index in consumed for index in indices):
            return groups

        combined = []
        for left_index, right_index in zip(indices, indices[1:]):
            if has_large_silence_between_groups(groups[left_index], groups[right_index]):
                return groups
        for index in indices:
            consumed.add(index)
            combined.extend(groups[index])
        merged.append(combined)

    if len(consumed) != len(groups):
        return groups

    return merged


def normalize_groups_for_story_flow(groups, display_words=None):
    if not groups:
        return groups

    normalized = groups
    for _ in range(3):
        updated = merge_semantically_connected_groups(normalized, display_words)
        updated = merge_short_word_groups(updated)
        if len(updated) == len(normalized):
            normalized = updated
            break
        normalized = updated

    return normalized


def estimate_group_reading_speed(words, display_words=None):
    text = get_group_text(words, display_words)
    duration = max(get_word_group_duration(words), 0.001)
    return len(text) / duration if text else 0.0


def normalize_groups_for_subtitle_readability(groups, display_words=None):
    if not groups:
        return groups

    normalized = []
    index = 0
    while index < len(groups):
        current = list(groups[index])
        current_duration = get_word_group_duration(current)
        reading_speed = estimate_group_reading_speed(current, display_words)

        if (
            index + 1 < len(groups)
            and (
                current_duration < VISUAL_MIN_DURATION_SECONDS
                or reading_speed < READING_SPEED_MIN_CPS
            )
        ):
            next_group = list(groups[index + 1])
            if groups_are_semantically_connected(current, next_group, display_words):
                normalized.append(current + next_group)
                index += 2
                continue

        normalized.append(current)
        index += 1

    return normalized


def split_groups_at_scene_cuts(groups, scene_cuts):
    if not groups or not scene_cuts:
        return groups

    output = []
    for group in groups:
        pending = [group]
        while pending:
            current = pending.pop(0)
            start, end = get_group_time_range(current)
            split_done = False
            for scene_cut in scene_cuts:
                if not (start < scene_cut < end):
                    continue

                split_index = None
                for index in range(1, len(current)):
                    previous_end = float(current[index - 1]["end"])
                    next_start = float(current[index]["start"])
                    if previous_end <= scene_cut <= next_start:
                        split_index = index
                        break

                if split_index is None:
                    continue

                left = current[:split_index]
                right = current[split_index:]
                if left and right:
                    pending.insert(0, right)
                    pending.insert(0, left)
                    split_done = True
                break

            if not split_done:
                output.append(current)

    return output


def build_clip_ranges_from_word_groups(word_groups, silence_regions, energy_regions, media_duration):
    if not word_groups:
        return []

    clip_ranges = []
    for index, group in enumerate(word_groups):
        first_word_start = float(group[0]["start"])
        last_word_end = float(group[-1]["end"])
        next_group = word_groups[index + 1] if index + 1 < len(word_groups) else None
        silence_buffer = get_trailing_silence_buffer(group, next_group, media_duration)
        clip_end = last_word_end + silence_buffer

        clip_ranges.append({
            "start": round_timestamp_value(first_word_start),
            "end": round_timestamp_value(min(media_duration, clip_end)),
            "suggested_start": round_timestamp_value(first_word_start),
            "suggested_end": round_timestamp_value(min(media_duration, clip_end)),
        })

    return clip_ranges


def expand_words_for_low_confidence(word_groups, index):
    current = list(word_groups[index])
    current_confidence = average_word_confidence(current)
    edge_confidences = []
    if current:
        if current[0].get("confidence") is not None:
            edge_confidences.append(float(current[0]["confidence"]))
        if current[-1].get("confidence") is not None:
            edge_confidences.append(float(current[-1]["confidence"]))

    weakest_edge = min(edge_confidences) if edge_confidences else None
    if current_confidence is not None and current_confidence >= LOW_CONFIDENCE_THRESHOLD and (
        weakest_edge is None or weakest_edge >= VERY_LOW_CONFIDENCE_THRESHOLD
    ):
        return current, False

    expanded = list(current)
    expanded_flag = False
    if index > 0 and word_groups[index - 1]:
        expanded.insert(0, word_groups[index - 1][-1])
        expanded_flag = True
    if index + 1 < len(word_groups) and word_groups[index + 1]:
        expanded.append(word_groups[index + 1][0])
        expanded_flag = True

    deduped = []
    seen = set()
    for word in expanded:
        key = (float(word.get("start", 0.0)), float(word.get("end", 0.0)), word.get("text", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(word)

    deduped.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return deduped, expanded_flag


def compute_natural_clip_boundary(group_words, index, word_groups, silence_regions, energy_analysis, media_duration):
    effective_words, confidence_expanded = expand_words_for_low_confidence(word_groups, index)
    first_word_start = float(group_words[0]["start"])
    last_word_end = float(group_words[-1]["end"])
    next_group = word_groups[index + 1] if index + 1 < len(word_groups) else None
    silence_buffer = get_trailing_silence_buffer(group_words, next_group, media_duration)
    clip_end = min(media_duration, last_word_end + silence_buffer)
    energy_regions = energy_analysis.get("regions", [])
    onset_peaks = energy_analysis.get("onset_peaks", [])
    offset_peaks = energy_analysis.get("offset_peaks", [])

    silence_start_candidate = find_previous_silence_end(first_word_start, silence_regions)
    energy_rise_candidate = find_previous_energy_rise(first_word_start, energy_regions)
    peak_onset_candidate = find_previous_onset_peak(first_word_start, onset_peaks)
    start_floor = max(0.0, first_word_start - MAX_START_BACKTRACK_SECONDS)
    bounded_start_candidates = [
        candidate
        for candidate in (silence_start_candidate, energy_rise_candidate, peak_onset_candidate)
        if candidate is not None and start_floor <= float(candidate) <= first_word_start
    ]
    boundary_start = first_word_start
    boundary_end = clip_end

    start_padding, end_padding, words_per_second = get_adaptive_padding(effective_words)
    padded_start = first_word_start
    padded_end = clip_end

    silence_adjustment = 0.0
    energy_adjustment = 0.0
    if silence_start_candidate is not None and start_floor <= float(silence_start_candidate) <= first_word_start:
        silence_adjustment = max(0.0, first_word_start - float(silence_start_candidate))
    if energy_rise_candidate is not None and start_floor <= float(energy_rise_candidate) <= first_word_start:
        energy_adjustment = max(0.0, first_word_start - float(energy_rise_candidate))

    return {
        "start": round_timestamp_value(padded_start),
        "end": round_timestamp_value(padded_end),
        "word_start": round_timestamp_value(first_word_start),
        "word_end": round_timestamp_value(last_word_end),
        "suggested_start": round_timestamp_value(boundary_start),
        "suggested_end": round_timestamp_value(boundary_end),
        "words_per_second": words_per_second,
        "confidence_expanded": confidence_expanded,
        "effective_words": effective_words,
        "silence_adjustment": round_timestamp_value(silence_adjustment),
        "energy_adjustment": round_timestamp_value(energy_adjustment),
        "start_padding": 0.0,
        "end_padding": 0.0,
    }


def reconcile_natural_boundaries(boundaries, media_duration):
    if not boundaries:
        return boundaries

    reconciled = [dict(boundary) for boundary in boundaries]
    for boundary in reconciled:
        boundary["start"] = round_timestamp_value(max(0.0, float(boundary["word_start"])))
        boundary["end"] = round_timestamp_value(min(media_duration, float(boundary["end"])))
    return reconciled


def maybe_correct_text_locally(text, alignment_mode=ALIGNMENT_MODE):
    cleaned = clean_transcript_text(text)
    if not cleaned:
        return cleaned, False

    if not should_use_heavy_refinement(alignment_mode) or not LOCAL_TEXT_CORRECTION_ENABLED:
        return cleaned, False

    if shutil.which("ollama") is None:
        return cleaned, False

    prompt = (
        "Merge broken fragments into a natural sentence and lightly fix grammar, "
        "but do not change meaning. Return only the corrected text.\n\n"
        f"{cleaned}"
    )

    try:
        result = subprocess.run(
            ["ollama", "run", LOCAL_TEXT_CORRECTION_MODEL, prompt],
            capture_output=True,
            text=True,
            check=True,
            timeout=45,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        log(f"Local text correction skipped: {exc}")
        return cleaned, False

    corrected = clean_transcript_text(result.stdout or "")
    if not corrected:
        return cleaned, False

    return corrected, corrected != cleaned


def try_force_align_words(audio_path, words, language_code):
    normalized = normalize_words(words or [])
    if not normalized:
        return normalized, {"enabled": True, "used": False, "method": "none"}

    try:
        import whisperx
    except ImportError:
        message = (
            "WhisperX is not installed, so the app is using Whisper word timestamps as a fallback. "
            "Install WhisperX locally for the highest alignment accuracy."
        )
        if WHISPERX_REQUIRED and not ALLOW_WHISPERX_FALLBACK:
            raise RuntimeError(message)
        log(message)
        return normalized, {"enabled": True, "used": False, "method": "whisperx_missing", "warning": message}

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        audio = whisperx.load_audio(audio_path)
        align_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        align_segments = [
            {
                "start": float(word["start"]),
                "end": float(word["end"]),
                "text": word["text"],
            }
            for word in normalized
        ]
        result = whisperx.align(
            align_segments,
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
    except Exception as exc:
        message = f"WhisperX alignment failed, so the app is using Whisper word timestamps instead: {exc}"
        if WHISPERX_REQUIRED and not ALLOW_WHISPERX_FALLBACK:
            raise RuntimeError(message)
        log(f"{message}; falling back to Whisper timestamps")
        return normalized, {"enabled": True, "used": False, "method": "whisperx_failed", "error": str(exc), "warning": message}

    aligned_words = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []) or []:
            token = clean_text(word.get("word", ""))
            start = word.get("start")
            end = word.get("end")
            if not token or start is None or end is None:
                continue

            payload = {
                "text": token,
                "start": float(start),
                "end": float(end),
            }
            if word.get("score") is not None:
                payload["confidence"] = float(word["score"])
            aligned_words.append(payload)

    if not aligned_words:
        message = "WhisperX returned no aligned words, so the app is using Whisper word timestamps instead."
        if WHISPERX_REQUIRED and not ALLOW_WHISPERX_FALLBACK:
            raise RuntimeError(message)
        log(message)
        return normalized, {"enabled": True, "used": False, "method": "whisperx_empty", "warning": message}

    # Preserve every Whisper token even if WhisperX misses or misaligns some words.
    aligned_words = normalize_words(aligned_words)
    recovered_words = []
    aligned_pointer = 0
    fallback_count = 0

    for source_word in normalized:
        source_token = normalize_token_for_alignment(source_word.get("text", ""))
        chosen = None

        for probe in range(aligned_pointer, min(len(aligned_words), aligned_pointer + 8)):
            candidate = aligned_words[probe]
            candidate_token = normalize_token_for_alignment(candidate.get("text", ""))
            if not source_token or source_token == candidate_token:
                chosen = candidate
                aligned_pointer = probe + 1
                break

        if chosen is None and aligned_pointer < len(aligned_words):
            candidate = aligned_words[aligned_pointer]
            candidate_start = float(candidate.get("start", 0.0))
            candidate_end = float(candidate.get("end", candidate_start))
            source_start = float(source_word.get("start", 0.0))
            source_end = float(source_word.get("end", source_start))
            if abs(candidate_start - source_start) <= 0.2 and abs(candidate_end - source_end) <= 0.2:
                chosen = candidate
                aligned_pointer += 1

        if chosen is None:
            chosen = source_word
            fallback_count += 1

        merged = {
            "text": clean_text(source_word.get("text", "")) or clean_text(chosen.get("text", "")),
            "start": float(chosen.get("start", source_word.get("start", 0.0))),
            "end": float(chosen.get("end", source_word.get("end", source_word.get("start", 0.0)))),
        }
        if chosen.get("confidence") is not None:
            merged["confidence"] = float(chosen.get("confidence"))
        elif source_word.get("confidence") is not None:
            merged["confidence"] = float(source_word.get("confidence"))
        recovered_words.append(merged)

    if fallback_count > 0:
        log(f"WhisperX partial alignment fallback used for {fallback_count} words (preserved from Whisper timing).")

    return normalize_words(recovered_words), {"enabled": True, "used": True, "method": "whisperx_with_fallback"}


def build_aligned_segments(
    source_words,
    display_words,
    silence_regions,
    energy_analysis,
    media_duration,
    alignment_metadata,
    source_text_segments=None,
    source_timed_segments=None,
    scene_cuts=None,
    alignment_mode=ALIGNMENT_MODE,
    display_text_segments=None,
):
    if not source_words:
        return []
    del silence_regions, energy_analysis, scene_cuts

    timed_segments = [dict(segment) for segment in (source_timed_segments or []) if segment.get("words")]
    if not timed_segments:
        timed_segments = [{
            "start": round_timestamp_value(float(source_words[0]["start"])),
            "end": round_timestamp_value(float(source_words[-1]["end"])),
            "text": build_text_from_text_segments(
                source_text_segments,
                float(source_words[0]["start"]),
                float(source_words[-1]["end"]),
                fallback_text=build_text_from_word_list(source_words),
            ),
            "words": list(source_words),
        }]

    segments = []
    for index, timed_segment in enumerate(timed_segments):
        group_words = list(timed_segment.get("words") or [])
        if not group_words:
            continue

        if index == 0:
            clip_start = 0.0
        else:
            clip_start = round_timestamp_value(float(group_words[0]["start"]))
        clip_word_end = round_timestamp_value(float(group_words[-1]["end"]))
        clip_end = get_whisper_segment_clip_end(
            timed_segment,
            timed_segments[index + 1] if index + 1 < len(timed_segments) else None,
            media_duration,
        )
        timed_segment = timed_segments[index]
        clip_source_words = find_words_in_range(source_words, clip_start, clip_word_end)
        clip_display_words = find_words_in_range(display_words, clip_start, clip_word_end)
        display_text = clean_whisper_translated_text(
            timed_segment.get("text", "") or build_text_from_word_list(clip_display_words)
        )
        confidence = average_word_confidence(clip_source_words) or average_word_confidence(clip_display_words)
        confidence_status = get_confidence_status(confidence)
        raw_text = display_text
        subtitle_lines = [raw_text] if raw_text else []
        formatted_text = raw_text
        reading_speed = compute_reading_speed(formatted_text, clip_start, clip_end)
        words_per_second = get_words_per_second(group_words)
        display_start, display_end = compute_subtitle_display_timing(
            clip_start,
            clip_end,
            formatted_text,
            words_per_second,
        )

        segment_payload = {
            "clip_id": index,
            "start": clip_start,
            "end": clip_end,
            "subtitleStart": display_start,
            "subtitleEnd": display_end,
            "word_start": clip_start,
            "word_end": clip_word_end,
            "original_start": clip_start,
            "original_end": clip_end,
            "natural_start": clip_start,
            "natural_end": clip_end,
            "words_per_second": words_per_second,
            "silenceAdjustment": 0.0,
            "energyAdjustment": 0.0,
            "startPadding": 0.0,
            "endPadding": 0.0,
            "raw_text": raw_text,
            "text": formatted_text,
            "lines": subtitle_lines,
            "readingSpeed": reading_speed,
            "confidence": confidence,
            "confidenceStatus": confidence_status,
            "needsReview": confidence_status == "needs_review",
            "auto_aligned": True,
            "alignment_mode": normalize_alignment_mode(alignment_mode),
            "alignment_method": alignment_metadata.get("method", "whisper"),
            "forced_alignment_used": bool(alignment_metadata.get("used")),
            "text_corrected": False,
            "text_source": "whisper_segments",
            "confidence_expanded": False,
            "words": clip_display_words,
            "source_words": clip_source_words,
        }
        segments.append(segment_payload)

    return normalize_segments([segment for segment in segments if segment.get("text", "") != ""])


def build_segments_from_sentence_groups(
    sentence_groups,
    source_words,
    display_words,
    media_duration,
    alignment_metadata,
    alignment_mode=ALIGNMENT_MODE,
):
    if not sentence_groups:
        return []

    segments = []
    min_step = 1.0 / DEFAULT_AUDIO_SAMPLE_RATE
    min_duration_seconds = 0.3
    for index, sentence in enumerate(sentence_groups):
        word_indices = [
            int(word_index)
            for word_index in (sentence.get("word_indices") or [])
            if 0 <= int(word_index) < len(source_words)
        ]
        if not word_indices:
            continue

        clip_source_words = [source_words[word_index] for word_index in word_indices]
        clip_display_words = [
            display_words[word_index]
            for word_index in word_indices
            if 0 <= word_index < len(display_words)
        ] or clip_source_words

        if not clip_source_words:
            continue

        word_start = round_timestamp_value(float(clip_source_words[0]["start"]))
        word_end = round_timestamp_value(float(clip_source_words[-1]["end"]))
        raw_start = round_timestamp_value(float(sentence.get("start_time", word_start)))
        raw_end = round_timestamp_value(float(sentence.get("end_time", word_end)))
        new_start = round_timestamp_value(float(sentence.get("refined_start", raw_start)))
        new_end = round_timestamp_value(float(sentence.get("refined_end", raw_end)))

        new_start = round_timestamp_value(max(0.0, new_start))
        new_end = round_timestamp_value(max(new_end, word_end))
        new_end = round_timestamp_value(min(new_end, media_duration))

        if new_start >= new_end or (new_end - new_start) < min_duration_seconds:
            new_start = raw_start
            new_end = raw_end

        if new_start >= new_end or (new_end - new_start) < min_duration_seconds:
            new_start = round_timestamp_value(max(0.0, raw_start))
            new_end = round_timestamp_value(min(media_duration, max(raw_end, new_start + min_duration_seconds)))

        if (new_end - new_start) > 0.1:
            final_start = round_timestamp_value(max(0.0, new_start + CLIP_EDGE_TRIM_BUFFER_SECONDS))
            final_end = round_timestamp_value(min(media_duration, new_end - CLIP_EDGE_TRIM_BUFFER_SECONDS))
        else:
            final_start = new_start
            final_end = new_end

        if final_end < word_end:
            final_end = word_end
        if final_start >= final_end or (final_end - final_start) < min_duration_seconds:
            final_start = new_start
            final_end = new_end
        if final_start >= final_end:
            final_end = round_timestamp_value(min(media_duration, final_start + min_duration_seconds))
        if final_end <= final_start:
            final_end = round_timestamp_value(min(media_duration, final_start + min_step))
        if final_end <= final_start:
            continue

        raw_text = clean_whisper_translated_text(
            sentence.get("text", "") or build_text_from_word_list(clip_display_words)
        )
        if not raw_text:
            continue

        confidence = average_word_confidence(clip_source_words) or average_word_confidence(clip_display_words)
        confidence_status = get_confidence_status(confidence)
        subtitle_lines = [raw_text] if raw_text else []
        reading_speed = compute_reading_speed(raw_text, final_start, final_end)
        words_per_second = get_words_per_second(clip_source_words)
        display_start, display_end = compute_subtitle_display_timing(
            final_start,
            final_end,
            raw_text,
            words_per_second,
        )

        segment_payload = {
            "clip_id": index,
            "start": final_start,
            "end": final_end,
            "subtitleStart": display_start,
            "subtitleEnd": display_end,
            "word_start": word_start,
            "word_end": word_end,
            "original_start": raw_start,
            "original_end": raw_end,
            "natural_start": word_start,
            "natural_end": new_end,
            "words_per_second": words_per_second,
            "silenceAdjustment": round_timestamp_value(max(new_end - word_end, 0.0)),
            "energyAdjustment": 0.0,
            "startPadding": 0.0,
            "endPadding": 0.0,
            "raw_text": raw_text,
            "text": raw_text,
            "lines": subtitle_lines,
            "readingSpeed": reading_speed,
            "confidence": confidence,
            "confidenceStatus": confidence_status,
            "needsReview": confidence_status == "needs_review",
            "auto_aligned": True,
            "alignment_mode": normalize_alignment_mode(alignment_mode),
            "alignment_method": alignment_metadata.get("method", "word_sentences"),
            "forced_alignment_used": bool(alignment_metadata.get("used")),
            "text_corrected": False,
            "text_source": "word_sentence_grouping",
            "confidence_expanded": False,
            "words": clip_display_words,
            "source_words": clip_source_words,
        }
        segments.append(segment_payload)

    for segment_index in range(len(segments) - 1):
        current = segments[segment_index]
        next_segment = segments[segment_index + 1]
        if float(next_segment["start"]) < float(current["end"]):
            current["end"] = round_timestamp_value(float(next_segment["start"]))
            if float(current["end"]) <= float(current["start"]):
                current["end"] = round_timestamp_value(
                    min(media_duration, float(current["start"]) + min_step)
                )

    if segments:
        segments[-1]["end"] = round_timestamp_value(media_duration)
        segments[-1]["original_end"] = round_timestamp_value(media_duration)
        segments[-1]["natural_end"] = round_timestamp_value(media_duration)
        end = float(segments[-1]["end"])
        start = float(segments[-1]["start"])
        display_start, display_end = compute_subtitle_display_timing(
            start,
            end,
            segments[-1].get("text", ""),
            float(segments[-1].get("words_per_second") or 0.0),
        )
        segments[-1]["subtitleStart"] = display_start
        segments[-1]["subtitleEnd"] = display_end

    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        display_start, display_end = compute_subtitle_display_timing(
            start,
            end,
            segment.get("text", ""),
            float(segment.get("words_per_second") or 0.0),
        )
        segment["subtitleStart"] = display_start
        segment["subtitleEnd"] = display_end
        segment["readingSpeed"] = compute_reading_speed(segment.get("text", ""), start, end)

    print("=== DEBUG: SEGMENT BUILD ===")
    for segment in segments[:5]:
        print(
            f"SEGMENT: {round_timestamp_value(float(segment.get('start', 0.0)))} "
            f"-> {round_timestamp_value(float(segment.get('end', 0.0)))}"
        )
        print(f"TEXT: {segment.get('text', '')}")
    print("================================")

    for segment in segments[:5]:
        print("---- CLIP CHECK ----")
        print(f"TEXT: {segment.get('text', '')}")
        segment_start = float(segment.get("start", 0.0))
        segment_end = float(segment.get("end", segment_start))
        print(
            f"RANGE: {round_timestamp_value(segment_start)} "
            f"-> {round_timestamp_value(segment_end)}"
        )

        clip_words = [
            word for word in source_words
            if segment_start <= float(word.get("start", 0.0)) <= segment_end
        ]
        print("WORDS IN CLIP:")
        for word in clip_words:
            print(get_word_token(word), round_timestamp_value(float(word.get("start", 0.0))))
        print("---------------------")

    if not USE_ALIGNMENT_DEBUG_MODE:
        segments = enforce_continuity(segments, media_duration)
    normalized = normalize_segments(segments)
    validate_continuity(normalized)
    return normalized


def enforce_continuity(segments, media_duration=None, minimum_duration=0.3):
    if not segments:
        return segments

    safe_min_duration = max(float(minimum_duration), 1.0 / DEFAULT_AUDIO_SAMPLE_RATE)
    for index in range(1, len(segments)):
        previous = segments[index - 1]
        current = segments[index]

        chained_start = round_timestamp_value(float(previous.get("end", 0.0)))
        current["start"] = chained_start

        current_end = round_timestamp_value(float(current.get("end", chained_start)))
        if current["start"] >= current_end:
            current_end = round_timestamp_value(float(current["start"]) + safe_min_duration)

        if media_duration is not None:
            current_end = round_timestamp_value(min(float(media_duration), current_end))
            if current_end <= float(current["start"]):
                current_end = round_timestamp_value(min(float(media_duration), float(current["start"]) + safe_min_duration))

        current["end"] = current_end
        current["start_ms"] = seconds_to_milliseconds(current["start"])
        current["end_ms"] = seconds_to_milliseconds(current["end"])

    return segments


def log_alignment_debug_info(segments, silence_regions, energy_analysis):
    log(f"Alignment debug: silence_regions={json.dumps(silence_regions, ensure_ascii=True)}")
    log(
        "Alignment debug: energy="
        f"{json.dumps({'avg_energy': energy_analysis.get('avg_energy', 0.0), 'threshold': energy_analysis.get('threshold', 0.0), 'peaks': energy_analysis.get('peaks', [])}, ensure_ascii=True)}"
    )
    for segment in segments:
        first_word_start = float(segment.get("word_start", segment.get("start", 0.0)))
        final_start = float(segment.get("start", 0.0))
        if final_start > first_word_start:
            log(
                f"ERROR alignment clip {get_clip_number(segment.get('clip_id', 0))}: "
                f"final start {final_start:.3f} > firstWord.start {first_word_start:.3f}"
            )
        log(
            "Alignment debug clip "
            f"{get_clip_number(segment.get('clip_id', 0))}: "
            f"firstWord.start={segment.get('word_start')} "
            f"final.start={segment.get('start')} "
            f"energyAdjustment={segment.get('energyAdjustment', 0.0)} "
            f"silenceAdjustment={segment.get('silenceAdjustment', 0.0)} "
            f"original=({segment.get('original_start')}, {segment.get('original_end')}) "
            f"aligned=({segment.get('start')}, {segment.get('end')}) "
            f"confidence={segment.get('confidence')} "
            f"words={len(segment.get('words') or [])}"
        )


def segment_covers_time(segment, timestamp):
    start = float(segment.get("word_start", segment.get("start", 0.0)))
    end = float(segment.get("word_end", segment.get("end", start)))
    return start <= float(timestamp) <= end


def find_uncovered_transcript_words(segments, transcript_words):
    uncovered = []
    for word in transcript_words or []:
        word_start = float(word.get("start", 0.0))
        if any(segment_covers_time(segment, word_start) for segment in segments or []):
            continue
        uncovered.append(word)
    return uncovered


def group_words_by_coverage_gap(words, minimum_gap=UNCOVERED_SPEECH_MIN_GAP_SECONDS):
    if not words:
        return []

    groups = [[words[0]]]
    for word in words[1:]:
        previous_word = groups[-1][-1]
        if get_word_gap(previous_word, word) > float(minimum_gap):
            groups.append([word])
            continue
        groups[-1].append(word)

    return groups


def segment_overlaps_range(segment, start, end):
    segment_start = float(segment.get("start", 0.0))
    segment_end = float(segment.get("end", segment_start))
    return segment_end > float(start) and segment_start < float(end)


def build_gap_fill_segment(
    words,
    next_clip_start,
    media_duration,
    alignment_mode,
    alignment_metadata,
    display_text_segments=None,
    text_override=None,
    start_override=None,
    end_override=None,
):
    clip_words = list(words or [])
    word_start = float(start_override if start_override is not None else (clip_words[0]["start"] if clip_words else 0.0))
    word_end = float(end_override if end_override is not None else (clip_words[-1]["end"] if clip_words else word_start))
    gap_to_next = max(float(next_clip_start) - word_end, 0.0) if next_clip_start is not None else max(float(media_duration) - word_end, 0.0)
    clip_end = min(float(media_duration), word_end + min(gap_to_next, 0.5))
    raw_text = text_override or build_text_from_text_segments(
        display_text_segments,
        word_start,
        word_end,
        fallback_text=build_text_from_word_list(clip_words),
    )
    confidence = average_word_confidence(clip_words)
    if not raw_text or confidence is None or confidence < LOW_CONFIDENCE_THRESHOLD:
        raw_text = "⚠️ Low confidence audio"

    refined_text = refine_clip_text(raw_text, confidence)
    formatted_text, subtitle_lines, reading_speed = apply_subtitle_formatting(
        refined_text,
        word_start,
        clip_end,
        alignment_mode,
    )
    words_per_second = get_words_per_second(clip_words) if clip_words else 0.0
    display_start, display_end = compute_subtitle_display_timing(
        word_start,
        clip_end,
        formatted_text,
        words_per_second,
    )

    return {
        "start": round_timestamp_value(word_start),
        "end": round_timestamp_value(clip_end),
        "subtitleStart": display_start,
        "subtitleEnd": display_end,
        "word_start": round_timestamp_value(word_start),
        "word_end": round_timestamp_value(word_end),
        "original_start": round_timestamp_value(word_start),
        "original_end": round_timestamp_value(clip_end),
        "natural_start": round_timestamp_value(word_start),
        "natural_end": round_timestamp_value(clip_end),
        "words_per_second": round(words_per_second, 4),
        "silenceAdjustment": 0.0,
        "energyAdjustment": 0.0,
        "startPadding": 0.0,
        "endPadding": 0.0,
        "raw_text": raw_text,
        "text": formatted_text,
        "lines": subtitle_lines,
        "readingSpeed": reading_speed,
        "confidence": confidence,
        "confidenceStatus": get_confidence_status(confidence),
        "needsReview": confidence is None or float(confidence) < LOW_CONFIDENCE_THRESHOLD,
        "auto_aligned": True,
        "alignment_mode": normalize_alignment_mode(alignment_mode),
        "alignment_method": alignment_metadata.get("method", "whisper"),
        "forced_alignment_used": bool(alignment_metadata.get("used")),
        "text_corrected": False,
        "text_source": "speech_gap_fill",
        "confidence_expanded": False,
        "words": clip_words,
        "source_words": clip_words,
    }


def fill_missing_speech_coverage(
    segments,
    transcript_words,
    media_duration,
    alignment_mode,
    alignment_metadata,
    display_text_segments=None,
):
    if not segments:
        return segments

    working_segments = sorted((dict(segment) for segment in segments), key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))
    additional_segments = []

    uncovered_words = group_words_by_coverage_gap(
        find_uncovered_transcript_words(working_segments, transcript_words),
        UNCOVERED_SPEECH_MIN_GAP_SECONDS,
    )

    for group in uncovered_words:
        if not group:
            continue
        group_start = float(group[0]["start"])
        group_end = float(group[-1]["end"])
        if any(segment_overlaps_range(segment, group_start, group_end) for segment in working_segments + additional_segments):
            continue
        next_clip_start = next(
            (float(segment.get("start", 0.0)) for segment in working_segments if float(segment.get("start", 0.0)) > group_end),
            None,
        )
        additional_segments.append(
            build_gap_fill_segment(
                group,
                next_clip_start,
                media_duration,
                alignment_mode,
                alignment_metadata,
                display_text_segments,
            )
        )

    if not additional_segments:
        return working_segments

    combined = sorted(working_segments + additional_segments, key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))
    for index, segment in enumerate(combined):
        segment["clip_id"] = index
    return normalize_segments(combined)


def update_clip_timestamp_segments(segments, transcript_words, clip_id, start, end, fps, video_duration):
    if not segments:
        raise ValueError("No clips available to update")

    target_index = None
    working_segments = [dict(segment) for segment in segments]
    for index, segment in enumerate(working_segments):
        if int(segment.get("clip_id", index)) == int(clip_id):
            target_index = index
            break

    if target_index is None:
        raise ValueError(f"Clip {get_clip_number(clip_id)} not found")

    effective_fps = float(fps or DEFAULT_FPS)
    if effective_fps <= 0:
        effective_fps = DEFAULT_FPS
    frame_step = 1.0 / effective_fps
    video_limit = max(float(video_duration or 0.0), frame_step)
    current = working_segments[target_index]
    previous_clip = working_segments[target_index - 1] if target_index > 0 else None
    next_clip = working_segments[target_index + 1] if target_index < len(working_segments) - 1 else None

    snapped_start = snap_to_frame(float(start), effective_fps, method="nearest")
    snapped_end = snap_to_frame(float(end), effective_fps, method="nearest")

    min_start = 0.0
    max_end = video_limit
    if previous_clip:
        min_start = float(previous_clip["start"]) + frame_step
    if next_clip:
        max_end = float(next_clip["end"]) - frame_step

    if max_end <= min_start:
        raise ValueError(f"Clip {get_clip_number(clip_id)} has no room to resize")

    bounded_start = min(max(snapped_start, min_start), max_end - frame_step)
    bounded_end = max(min(snapped_end, max_end), bounded_start + frame_step)

    if bounded_start < 0:
        raise ValueError("Start time cannot be negative")
    if bounded_end > video_limit:
        raise ValueError("End time exceeds video duration")
    if bounded_start >= bounded_end:
        raise ValueError("Start time must be before end time")

    current["start"] = round_timestamp_value(bounded_start)
    current["end"] = round_timestamp_value(bounded_end)

    if previous_clip:
        previous_clip["end"] = round_timestamp_value(current["start"])

    if next_clip:
        next_clip["start"] = round_timestamp_value(current["end"])

    for index in range(len(working_segments) - 1):
        working_segments[index]["end"] = round_timestamp_value(
            float(working_segments[index + 1]["start"])
        )

    normalized_words = normalize_words(transcript_words or [])
    updated_clips = []
    for index, segment in enumerate(working_segments):
        start_value = float(segment.get("start", 0.0))
        end_value = float(segment.get("end", 0.0))

        if start_value < 0:
            raise ValueError(f"Clip {get_clip_number(segment.get('clip_id', index))} start cannot be negative")
        if end_value > video_limit:
            raise ValueError(f"Clip {get_clip_number(segment.get('clip_id', index))} exceeds video duration")
        if start_value >= end_value:
            raise ValueError(f"Clip {get_clip_number(segment.get('clip_id', index))} has invalid duration")

        if index < len(working_segments) - 1:
            expected_end = round_timestamp_value(float(working_segments[index + 1]["start"]))
            segment["end"] = expected_end

        fallback_text = clean_text(segment.get("text", ""))
        matched_words = find_words_in_range(normalized_words, start_value, end_value)
        rebuilt_text = rebuild_segment_text_from_words(normalized_words, segment, fallback_text)
        segment_alignment_mode = normalize_alignment_mode(segment.get("alignment_mode"))
        segment_confidence = average_word_confidence(matched_words)
        raw_text = rebuilt_text
        refined_text = refine_clip_text(raw_text, segment_confidence)
        formatted_text, subtitle_lines, reading_speed = apply_subtitle_formatting(
            refined_text,
            start_value,
            end_value,
            segment_alignment_mode,
        )
        display_start, display_end = compute_subtitle_display_timing(
            start_value,
            end_value,
            formatted_text,
            float(segment.get("words_per_second") or 0.0),
        )
        segment["raw_text"] = raw_text
        segment["text"] = formatted_text
        segment["lines"] = subtitle_lines
        segment["readingSpeed"] = reading_speed
        segment["subtitleStart"] = display_start
        segment["subtitleEnd"] = display_end
        segment["words"] = matched_words
        segment["confidence"] = segment_confidence
        segment["confidenceStatus"] = get_confidence_status(segment["confidence"])
        segment["needsReview"] = segment["confidenceStatus"] == "needs_review"
        segment["alignment_mode"] = segment_alignment_mode
        payload = build_segment_payload(segment, fallback_text)
        working_segments[index] = payload
        updated_clips.append(payload)

    return {
        "segments": working_segments,
        "updated_clips": updated_clips,
    }


def normalize_voice_style(style):
    return VOICE_STYLE_DEFAULT


def get_text_hash(text, style=VOICE_STYLE_DEFAULT):
    normalized_text = normalize_text_for_hash(text)
    return hashlib.sha256(
        f"{ENGLISH_AUDIO_PIPELINE_VERSION}::{normalized_text}".encode("utf-8")
    ).hexdigest()[:16]


def apply_voice_style(text, style):
    normalized_style = normalize_voice_style(style)
    cleaned = normalize_text_for_hash(text)
    if not cleaned:
        return ""

    if normalized_style == "formal":
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\.\.\.+", ".", cleaned)
        cleaned = cleaned.replace("!", ".").replace("?", ".")
    elif normalized_style == "friendly":
        cleaned = re.sub(r"\bplease\b", "please", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bnow\b", "Now", cleaned, flags=re.IGNORECASE)
    else:
        cleaned = re.sub(r"\b(step \d+)\b", r"\1,", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(after that|next|then)\b", lambda match: match.group(1).capitalize(), cleaned, flags=re.IGNORECASE)

    return cleaned.strip()


def infer_pause_duration(phrase, style):
    config = VOICE_STYLE_CONFIG[normalize_voice_style(style)]
    trimmed = phrase.strip()
    lowered = trimmed.lower().strip(",.!?;:")

    if not trimmed:
        return 0.0
    if trimmed.endswith("..."):
        return config["thinking_pause"]
    if trimmed.endswith(","):
        return config["comma_pause"]
    if trimmed.endswith((".", "!", "?", ";", ":")):
        return config["sentence_pause"]
    if lowered in PAUSE_KEYWORDS or lowered.startswith(("after that ", "next ", "now ", "then ", "and ")):
        return config["connector_pause"]
    return 0.0


def enhance_text_with_pauses(text, style=VOICE_STYLE_DEFAULT):
    styled_text = apply_voice_style(text, style)
    if not styled_text:
        return []

    marked_text = styled_text
    marked_text = re.sub(r"(\.\.\.)", r"\1|", marked_text)
    marked_text = re.sub(r"([,.;!?])", r"\1|", marked_text)
    marked_text = re.sub(
        r"\b(after that|next|now|then)\b\s+",
        lambda match: f"{match.group(1)}|",
        marked_text,
        flags=re.IGNORECASE,
    )
    if normalize_voice_style(style) != "formal":
        marked_text = re.sub(r"\s+\b(and)\b\s+", r"|and ", marked_text, flags=re.IGNORECASE)

    phrases = []
    for index, raw_part in enumerate(marked_text.split("|")):
        phrase = raw_part.strip()
        if not phrase:
            continue
        phrases.append({
            "index": index,
            "text": phrase,
            "pause_after": infer_pause_duration(phrase, style),
        })

    return phrases


def phrase_contains_emphasis_keyword(text):
    lowered = (text or "").lower()
    return any(re.search(rf"\b{re.escape(keyword)}\b", lowered) for keyword in EMPHASIS_KEYWORDS)


def derive_base_name(video_path):
    return sanitize_export_base_name(video_path)


def get_clip_number(clip_id):
    return int(clip_id) + 1


def get_clip_audio_file_path(video_path, clip_id):
    base_name = derive_base_name(video_path)
    clip_number = get_clip_number(clip_id)
    return os.path.join(get_audio_cache_dir(video_path), f"{base_name}_clip_{clip_number}.wav")


def get_clip_audio_meta_path(video_path, clip_id):
    base_name = derive_base_name(video_path)
    clip_number = get_clip_number(clip_id)
    return os.path.join(get_audio_cache_dir(video_path), f"{base_name}_clip_{clip_number}.json")


def build_audio_url(video_path, clip_id):
    return f"/audio/{clip_id}?video_path={quote(video_path, safe='')}"


def read_clip_audio_meta(video_path, clip_id):
    meta_path = get_clip_audio_meta_path(video_path, clip_id)
    if not os.path.exists(meta_path):
        return {}

    try:
        with open(meta_path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except (OSError, json.JSONDecodeError):
        return {}


def write_clip_audio_meta(video_path, clip_id, payload):
    meta_path = get_clip_audio_meta_path(video_path, clip_id)
    with open(meta_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=True, indent=2)


def invalidate_clip_audio_cache(video_path, clip_id):
    audio_path = get_clip_audio_file_path(video_path, clip_id)
    meta_path = get_clip_audio_meta_path(video_path, clip_id)

    for target_path in (audio_path, meta_path):
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except OSError:
                pass


def get_clip_audio_status(video_path, clip_id, text, voice_style=VOICE_STYLE_DEFAULT):
    normalized_style = normalize_voice_style(voice_style)
    audio_path = get_clip_audio_file_path(video_path, clip_id)
    meta = read_clip_audio_meta(video_path, clip_id)
    text_hash = get_text_hash(text, normalized_style)
    cached_hash = meta.get("text_hash")
    cached_style = normalize_voice_style(meta.get("voice_style"))
    job_state = get_audio_generation_state(video_path, clip_id, normalized_style)

    if os.path.exists(audio_path) and cached_hash == text_hash and cached_style == normalized_style:
        return {
            "clip_id": int(clip_id),
            "status": "ready",
            "error": "",
            "audio_url": build_audio_url(video_path, clip_id),
            "text_hash": text_hash,
            "voice_style": normalized_style,
        }

    if job_state.get("status") in {"queued", "generating"}:
        return {
            "clip_id": int(clip_id),
            "status": job_state["status"],
            "error": job_state.get("error", ""),
            "audio_url": "",
            "text_hash": text_hash,
            "voice_style": normalized_style,
        }

    if job_state.get("status") == "error":
        return {
            "clip_id": int(clip_id),
            "status": "error",
            "error": job_state.get("error", "Audio generation failed"),
            "audio_url": "",
            "text_hash": text_hash,
            "voice_style": normalized_style,
        }

    return {
        "clip_id": int(clip_id),
        "status": "dirty" if os.path.exists(audio_path) else "missing",
        "error": "",
        "audio_url": "",
        "text_hash": text_hash,
        "voice_style": normalized_style,
    }


def generate_audio(text, output_path, style=VOICE_STYLE_DEFAULT):
    del style
    raw_output_path = f"{output_path}.xtts.wav"
    try:
        cleaned_text = clean_text(text)
        print("TEXT:", cleaned_text)
        print("OUTPUT:", output_path)
        print("SPEAKER:", SPEAKER_PATH)
        print("EXISTS:", os.path.exists(SPEAKER_PATH))

        os.makedirs("audio_cache", exist_ok=True)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if not cleaned_text:
            raise Exception("Empty text")

        with tts_lock:
            print("XTTS LOCK: acquired")
            tts.tts_to_file(
                text=cleaned_text,
                file_path=raw_output_path,
                language="en",
                speaker_wav=SPEAKER_PATH,
                temperature=0.65,
                length_penalty=1.0,
                repetition_penalty=2.0,
            )
            print("XTTS LOCK: released")

        if not os.path.exists(raw_output_path):
            raise Exception("Audio file not created")

        run_command([
            "ffmpeg", "-y",
            "-i", raw_output_path,
            "-filter:a", "loudnorm",
            "-ar", str(DEFAULT_AUDIO_SAMPLE_RATE),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            output_path,
        ])

        if not os.path.exists(output_path):
            raise Exception("Audio file not created")
        print("SUCCESS:", output_path)
        return output_path
    except Exception as exc:
        print("XTTS ERROR:", str(exc))
        raise Exception(f"Audio generation failed: {str(exc)}")
    finally:
        if os.path.exists(raw_output_path):
            try:
                os.remove(raw_output_path)
            except OSError:
                pass


def finalize_clip_audio(raw_audio_path, final_audio_path):
    run_command([
        "ffmpeg", "-y",
        "-i", raw_audio_path,
        "-ar", str(DEFAULT_AUDIO_SAMPLE_RATE),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        final_audio_path
    ])


def generate_clip_english_audio(video_path, clip_id, text, start, end, invalidate=False, voice_style=VOICE_STYLE_DEFAULT):
    if not text or not text.strip():
        raise ValueError(f"Missing text for clip {get_clip_number(clip_id)}")

    normalized_style = normalize_voice_style(voice_style)
    text_hash = get_text_hash(text, normalized_style)
    audio_path = get_clip_audio_file_path(video_path, clip_id)
    meta = read_clip_audio_meta(video_path, clip_id)
    cached_hash = meta.get("text_hash")
    cached_style = normalize_voice_style(meta.get("voice_style"))

    if invalidate:
        invalidate_clip_audio_cache(video_path, clip_id)
        meta = {}
        cached_hash = None
        cached_style = None

    exists = os.path.exists(audio_path)
    log(
        f"Checking: {audio_path} exists={exists} hash_match={cached_hash == text_hash} "
        f"style_match={cached_style == normalized_style}"
    )
    if exists and cached_hash == text_hash and cached_style == normalized_style:
        return {
            "clip_id": clip_id,
            "text_hash": text_hash,
            "cached": True,
            "audio_path": audio_path,
            "audio_url": build_audio_url(video_path, clip_id),
            "duration_seconds": get_media_duration(audio_path),
            "voice_style": normalized_style,
        }

    raw_audio_path = f"{audio_path}.raw.wav"

    try:
        generate_audio(text=text, output_path=raw_audio_path, style=normalized_style)
        finalize_clip_audio(
            raw_audio_path=raw_audio_path,
            final_audio_path=audio_path,
        )
        if not os.path.exists(audio_path):
            raise Exception("Audio generation failed")
        actual_duration = get_media_duration(audio_path)
        write_clip_audio_meta(video_path, clip_id, {
            "clip_id": int(clip_id),
            "clip_number": get_clip_number(clip_id),
            "base_name": derive_base_name(video_path),
            "text_hash": text_hash,
            "text": normalize_text_for_hash(text),
            "start": float(start),
            "end": float(end),
            "duration_seconds": actual_duration,
            "audio_file": os.path.basename(audio_path),
            "voice_style": normalized_style,
        })
    finally:
        if os.path.exists(raw_audio_path):
            try:
                os.remove(raw_audio_path)
            except OSError:
                pass

    return {
        "clip_id": clip_id,
        "text_hash": text_hash,
        "cached": False,
        "audio_path": audio_path,
        "audio_url": build_audio_url(video_path, clip_id),
        "duration_seconds": get_media_duration(audio_path),
        "voice_style": normalized_style,
    }


def ensure_all_audio_generated(video_path, normalized_segments, voice_style=VOICE_STYLE_DEFAULT):
    results = [None] * len(normalized_segments)

    def generate_for_index(segment_index):
        segment = normalized_segments[segment_index]
        clip_id = int(segment["clip_id"])
        expected_path = get_clip_audio_file_path(video_path, clip_id)
        print("CHECK FILE:", expected_path, os.path.exists(expected_path))
        results[segment_index] = generate_clip_english_audio(
            video_path=video_path,
            clip_id=clip_id,
            text=segment["text"],
            start=segment["start"],
            end=segment["end"],
            invalidate=False,
            voice_style=voice_style,
        )
        print("CHECK FILE:", results[segment_index]["audio_path"], os.path.exists(results[segment_index]["audio_path"]))

    worker_count = min(4, max(1, len(normalized_segments)))
    if worker_count > 1:
        log(
            f"XTTS uses a shared global model; clip audio requests may queue through a lock. "
            f"Requested workers={worker_count}"
        )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(generate_for_index, segment_index)
            for segment_index in range(len(normalized_segments))
        ]
        for future in futures:
            future.result()

    for segment, result in zip(normalized_segments, results):
        if result is None or not os.path.exists(result["audio_path"]):
            raise FileNotFoundError(f"Missing audio for clip {get_clip_number(segment['clip_id'])}")

    return results


def generate_all_audio(video_path, normalized_segments, voice_style=VOICE_STYLE_DEFAULT, clip_ids=None, invalidate=False):
    normalized_style = normalize_voice_style(voice_style)
    target_clip_ids = None if clip_ids is None else {int(clip_id) for clip_id in clip_ids}

    for segment in normalized_segments:
        clip_id = int(segment["clip_id"])
        if target_clip_ids is not None and clip_id not in target_clip_ids:
            continue

        try:
            set_audio_generation_state(video_path, clip_id, normalized_style, status="generating", error="")
            generate_clip_english_audio(
                video_path=video_path,
                clip_id=clip_id,
                text=segment["text"],
                start=segment["start"],
                end=segment["end"],
                invalidate=invalidate,
                voice_style=normalized_style,
            )
            set_audio_generation_state(video_path, clip_id, normalized_style, status="ready", error="")
        except Exception as exc:
            log(f"Background audio generation failed for clip {get_clip_number(clip_id)}: {exc}")
            set_audio_generation_state(
                video_path,
                clip_id,
                normalized_style,
                status="error",
                error=str(exc),
            )


def _background_audio_worker(video_path, segments, voice_style, clip_ids=None, invalidate=False):
    normalized_segments = normalize_segments(segments)
    try:
        generate_all_audio(
            video_path=video_path,
            normalized_segments=normalized_segments,
            voice_style=voice_style,
            clip_ids=clip_ids,
            invalidate=invalidate,
        )
    finally:
        worker_key = f"{os.path.realpath(video_path)}::{normalize_voice_style(voice_style)}::{','.join(map(str, sorted(clip_ids or [])))}::{int(invalidate)}"
        with audio_generation_workers_lock:
            audio_generation_workers.pop(worker_key, None)


def start_background_audio_generation(video_path, segments, voice_style=VOICE_STYLE_DEFAULT, clip_ids=None, invalidate=False):
    normalized_style = normalize_voice_style(voice_style)
    selected_clip_ids = tuple(sorted({int(clip_id) for clip_id in (clip_ids or [])}))
    worker_key = f"{os.path.realpath(video_path)}::{normalized_style}::{','.join(map(str, selected_clip_ids))}::{int(invalidate)}"

    if clip_ids is None:
        for segment in segments:
            set_audio_generation_state(video_path, int(segment["clip_id"]), normalized_style, status="queued", error="")
    else:
        for clip_id in selected_clip_ids:
            set_audio_generation_state(video_path, clip_id, normalized_style, status="queued", error="")

    with audio_generation_workers_lock:
        worker = audio_generation_workers.get(worker_key)
        if worker and worker.is_alive():
            return False

        worker = threading.Thread(
            target=_background_audio_worker,
            args=(video_path, segments, normalized_style, selected_clip_ids or None, invalidate),
            daemon=True,
        )
        audio_generation_workers[worker_key] = worker
        worker.start()
        return True


def build_merge_signature(video_path, segments, voice_style=VOICE_STYLE_DEFAULT):
    signature_payload = [
        {
            "clip_id": segment["clip_id"],
            "start_ms": segment["start_ms"],
            "end_ms": segment["end_ms"],
            "hash": get_text_hash(segment["text"], voice_style),
        }
        for segment in segments
    ]
    raw_payload = json.dumps(
        {
            "video_path": os.path.basename(video_path),
            "voice_style": normalize_voice_style(voice_style),
            "segments": signature_payload,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()[:20]


def build_english_clip_paths(output_dir, clip_number, merge_signature):
    clip_prefix = f"clip_{clip_number}_{merge_signature}"
    return {
        "video_base": os.path.join(output_dir, f"{clip_prefix}_video.mp4"),
        "video_fixed": os.path.join(output_dir, f"{clip_prefix}_video_fixed.mp4"),
        "silence_audio": os.path.join(output_dir, f"{clip_prefix}_silence.wav"),
        "audio_fixed": os.path.join(output_dir, f"{clip_prefix}_audio_fixed.wav"),
        "processed_clip": os.path.join(output_dir, f"{clip_prefix}_english.mp4"),
    }


def create_video_only_clip(video_path, segment, output_path):
    stream_times = get_media_stream_start_times(video_path)
    stream_offset_seconds = float(stream_times.get("offset_seconds", 0.0))
    start, end = apply_stream_offset_to_clip_range(
        float(segment["start"]),
        float(segment["end"]),
        stream_offset_seconds,
    )
    log(
        f"Creating video-only clip start={format_seconds_for_ffmpeg(start)} "
        f"end={format_seconds_for_ffmpeg(end)}"
    )
    run_command([
        "ffmpeg", "-y",
        "-ss", format_seconds_for_ffmpeg(start),
        "-to", format_seconds_for_ffmpeg(end),
        "-i", video_path,
        "-map", "0:v:0",
        "-an",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-avoid_negative_ts", "make_zero",
        output_path,
    ])


def extend_video_with_frozen_frame(video_clip_path, extended_video_path, extra_duration_seconds):
    if extra_duration_seconds <= 0:
        shutil.copyfile(video_clip_path, extended_video_path)
        return extended_video_path

    run_command([
        "ffmpeg", "-y",
        "-i", video_clip_path,
        "-vf", f"tpad=stop_mode=clone:stop_duration={extra_duration_seconds:.3f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        extended_video_path,
    ])
    return extended_video_path


def extend_audio_with_silence(audio_path, extended_audio_path, extra_duration_seconds, clip_paths):
    if extra_duration_seconds <= 0:
        shutil.copyfile(audio_path, extended_audio_path)
        return extended_audio_path

    run_command([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-t", f"{extra_duration_seconds:.3f}",
        "-i", "anullsrc=r=16000:cl=mono",
        "-ar", str(DEFAULT_AUDIO_SAMPLE_RATE),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        clip_paths["silence_audio"],
    ])

    run_command([
        "ffmpeg", "-y",
        "-i", audio_path,
        "-i", clip_paths["silence_audio"],
        "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[a]",
        "-map", "[a]",
        "-ar", str(DEFAULT_AUDIO_SAMPLE_RATE),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        extended_audio_path,
    ])
    return extended_audio_path


def trim_audio_to_duration(audio_path, trimmed_audio_path, target_duration_seconds):
    run_command([
        "ffmpeg", "-y",
        "-i", audio_path,
        "-t", f"{target_duration_seconds:.3f}",
        "-ar", str(DEFAULT_AUDIO_SAMPLE_RATE),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        trimmed_audio_path,
    ])
    return trimmed_audio_path


def process_english_clip(video_path, segment, clip_audio_result, output_dir, merge_signature):
    clip_number = get_clip_number(segment["clip_id"])
    clip_paths = build_english_clip_paths(output_dir, clip_number, merge_signature)
    processed_clip_path = clip_paths["processed_clip"]

    if os.path.exists(processed_clip_path):
        log(f"English clip cache hit -> {processed_clip_path}")
        return processed_clip_path

    create_video_only_clip(video_path, segment, clip_paths["video_base"])

    video_duration = get_media_duration(clip_paths["video_base"])
    audio_duration = get_media_duration(clip_audio_result["audio_path"])
    log(
        f"Clip {clip_number} durations: "
        f"video={video_duration:.3f}s audio={audio_duration:.3f}s"
    )

    fixed_video_path = clip_paths["video_base"]
    fixed_audio_path = clip_audio_result["audio_path"]

    if audio_duration > video_duration:
        duration_gap = audio_duration - video_duration
        if duration_gap < SMALL_VIDEO_EXTENSION_THRESHOLD_SECONDS:
            log(f"Clip {clip_number}: extending video slightly by {duration_gap:.3f}s")
            fixed_video_path = extend_video_with_frozen_frame(
                video_clip_path=clip_paths["video_base"],
                extended_video_path=clip_paths["video_fixed"],
                extra_duration_seconds=duration_gap,
            )
        else:
            target_audio_duration = video_duration + SMALL_VIDEO_EXTENSION_THRESHOLD_SECONDS
            log(
                f"Clip {clip_number}: audio exceeds video by {duration_gap:.3f}s, "
                f"trimming audio to {target_audio_duration:.3f}s"
            )
            fixed_audio_path = trim_audio_to_duration(
                audio_path=clip_audio_result["audio_path"],
                trimmed_audio_path=clip_paths["audio_fixed"],
                target_duration_seconds=target_audio_duration,
            )
            fixed_video_path = extend_video_with_frozen_frame(
                video_clip_path=clip_paths["video_base"],
                extended_video_path=clip_paths["video_fixed"],
                extra_duration_seconds=SMALL_VIDEO_EXTENSION_THRESHOLD_SECONDS,
            )
    elif video_duration > audio_duration:
        extra_audio = video_duration - audio_duration
        log(f"Clip {clip_number}: extending audio with silence by {extra_audio:.3f}s")
        fixed_audio_path = extend_audio_with_silence(
            audio_path=clip_audio_result["audio_path"],
            extended_audio_path=clip_paths["audio_fixed"],
            extra_duration_seconds=extra_audio,
            clip_paths=clip_paths,
        )
    else:
        fixed_audio_path = clip_audio_result["audio_path"]

    run_command([
        "ffmpeg", "-y",
        "-i", fixed_video_path,
        "-i", fixed_audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        processed_clip_path,
    ])

    if not os.path.exists(processed_clip_path):
        raise FileNotFoundError(f"Failed to create processed clip {clip_number}")

    return processed_clip_path


def add_final_video_buffer(video_path, output_path, buffer_seconds):
    source_duration = get_media_duration(video_path)
    target_duration = source_duration + buffer_seconds
    run_command([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"tpad=stop_mode=clone:stop_duration={buffer_seconds:.3f}",
        "-af", f"apad=pad_dur={buffer_seconds:.3f}",
        "-t", f"{target_duration:.3f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        output_path,
    ])
    return output_path


def trim_final_video(video_path, output_path, target_duration_seconds):
    run_command([
        "ffmpeg", "-y",
        "-i", video_path,
        "-t", f"{target_duration_seconds:.3f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        output_path,
    ])
    return output_path


def balance_final_english_video(video_path, english_video_path, output_dir, merge_signature):
    original_duration = get_media_duration(video_path)
    final_duration = get_media_duration(english_video_path)
    log(
        f"Final duration check: original={original_duration:.3f}s "
        f"generated={final_duration:.3f}s"
    )

    if final_duration < original_duration:
        buffered_video_path = os.path.join(
            output_dir,
            f"{MERGED_VIDEO_FILE_PREFIX}_{merge_signature}_buffered.mp4",
        )
        log(f"Final video shorter than original, adding {FINAL_VIDEO_BUFFER_SECONDS:.3f}s end buffer")
        english_video_path = add_final_video_buffer(
            video_path=english_video_path,
            output_path=buffered_video_path,
            buffer_seconds=FINAL_VIDEO_BUFFER_SECONDS,
        )
        final_duration = get_media_duration(english_video_path)

    if final_duration > original_duration + FINAL_VIDEO_MAX_OVERAGE_SECONDS:
        trimmed_video_path = os.path.join(
            output_dir,
            f"{MERGED_VIDEO_FILE_PREFIX}_{merge_signature}_trimmed.mp4",
        )
        target_duration = original_duration + FINAL_VIDEO_TARGET_OVERAGE_SECONDS
        log(
            f"Final video too long ({final_duration:.3f}s), trimming to {target_duration:.3f}s"
        )
        english_video_path = trim_final_video(
            video_path=english_video_path,
            output_path=trimmed_video_path,
            target_duration_seconds=target_duration,
        )

    return english_video_path


def create_english_video_from_clips(video_path, normalized_segments, clip_audio_results, merge_signature):
    output_dir = get_english_output_dir(video_path)
    english_video_path = os.path.join(output_dir, f"{MERGED_VIDEO_FILE_PREFIX}_{merge_signature}.mp4")
    if os.path.exists(english_video_path):
        log(f"Merge video cache hit -> {english_video_path}")
        return english_video_path

    processed_clip_paths = []
    for segment, clip_result in zip(normalized_segments, clip_audio_results):
        if not os.path.exists(clip_result["audio_path"]):
            raise FileNotFoundError(f"Missing audio for clip {get_clip_number(clip_result['clip_id'])}")
        processed_clip_paths.append(
            process_english_clip(
                video_path=video_path,
                segment=segment,
                clip_audio_result=clip_result,
                output_dir=output_dir,
                merge_signature=merge_signature,
            )
        )

    concat_list_path = os.path.join(output_dir, f"{MERGED_VIDEO_FILE_PREFIX}_{merge_signature}.txt")
    with open(concat_list_path, "w", encoding="utf-8") as file_handle:
        for clip_path in processed_clip_paths:
            file_handle.write(f"file '{clip_path}'\n")

    log("Final video concat start")
    run_command([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        english_video_path,
    ])
    log(f"Final video concat end -> {english_video_path}")
    return balance_final_english_video(
        video_path=video_path,
        english_video_path=english_video_path,
        output_dir=output_dir,
        merge_signature=merge_signature,
    )


def generate_english_video(video_path, segments, voice_style=VOICE_STYLE_DEFAULT):
    normalized_segments = normalize_segments(segments)
    if not normalized_segments:
        raise ValueError("No clips available for English video generation")

    validate_continuity(normalized_segments)
    normalized_style = normalize_voice_style(voice_style)
    merge_signature = build_merge_signature(video_path, normalized_segments, normalized_style)
    clip_audio_results = ensure_all_audio_generated(video_path, normalized_segments, normalized_style)
    english_video_path = create_english_video_from_clips(
        video_path=video_path,
        normalized_segments=normalized_segments,
        clip_audio_results=clip_audio_results,
        merge_signature=merge_signature,
    )

    return {
        "video_path": english_video_path,
        "video_url": build_preview_url(english_video_path),
        "signature": merge_signature,
        "voice_style": normalized_style,
    }


def is_valid_drive_folder_url(folder_url):
    return bool(DRIVE_FOLDER_PATTERN.match(folder_url or ""))


def is_drive_folder_public(folder_url):
    request = urllib_request.Request(
        folder_url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    try:
        with urllib_request.urlopen(request, timeout=10) as response:
            page = response.read().decode("utf-8", errors="ignore")
    except urllib_error.HTTPError as exc:
        if exc.code in (401, 403):
            return False, "Please make the folder public to allow upload"
        raise

    restricted_markers = (
        "You need access",
        "Request access",
        "Access denied",
        "Sign in to continue",
    )

    if any(marker in page for marker in restricted_markers):
        return False, "Please make the folder public to allow upload"

    return True, None


def is_drive_upload_configured():
    return bool(os.getenv("GOOGLE_DRIVE_UPLOAD_ENDPOINT", "").strip())


def get_dev_reload_token():
    watched_files = [
        "app.py",
        "xtts_service.py",
        os.path.join("templates", "index.html"),
    ]
    mtimes = []

    for file_path in watched_files:
        if os.path.exists(file_path):
            mtimes.append(str(int(os.path.getmtime(file_path) * 1000)))

    return "-".join(mtimes) if mtimes else "0"


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "drive_upload_configured": is_drive_upload_configured(),
            "dev_auto_reload": DEV_AUTO_RELOAD,
            "dev_reload_token": get_dev_reload_token(),
        }
    )


@app.get("/__dev__/reload-token")
def dev_reload_token():
    return {
        "enabled": DEV_AUTO_RELOAD,
        "token": get_dev_reload_token(),
    }


@app.get("/drive-status")
def drive_status():
    return {
        "configured": is_drive_upload_configured()
    }


@app.post("/process")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("edit"),
    alignment_mode: str = Form(ALIGNMENT_MODE),
):
    resolved_alignment_mode = normalize_alignment_mode(alignment_mode)
    session_dir = create_session_dir()
    safe_name = os.path.basename(file.filename or "video.mp4")
    input_path = os.path.join(session_dir, safe_name)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    log("Extracting audio...")
    audio = extract_audio(input_path)

    fps = get_video_fps(input_path)
    log(f"Detected video FPS: {fps:.3f}")
    media_duration = get_media_duration(input_path)
    log(f"Detected video duration: {media_duration:.3f}s")

    log("Running transcription...")
    try:
        segments, transcript_words, alignment_metadata = transcribe(
            audio,
            fps,
            media_duration,
            input_path,
            resolved_alignment_mode,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    for index, segment in enumerate(segments):
        segment["clip_id"] = index
        segment["translation"] = segment.get("text") or ""

    background_tasks.add_task(
        start_background_audio_generation,
        input_path,
        segments,
        VOICE_STYLE_DEFAULT,
        None,
        False,
    )

    return {
        "mode": mode,
        "segments": segments,
        "transcript_words": transcript_words,
        "video_path": input_path,
        "preview_url": build_preview_url(input_path),
        "video_fps": fps,
        "alignment": alignment_metadata,
        "alignment_mode": resolved_alignment_mode,
        "warning": alignment_metadata.get("warning", ""),
        "background_audio_started": True,
        "background_voice_style": VOICE_STYLE_DEFAULT,
    }


@app.post("/restore-state")
async def restore_state(data: dict = Body(...)):
    video_path = (data.get("video_path") or "").strip()
    segments_count = int(data.get("segments_count") or 0)

    if not video_path or not os.path.exists(video_path):
        return {
            "ok": False,
            "exists": False,
            "message": "Saved video is no longer available"
        }

    summary = build_output_summary(video_path, segments_count)

    return {
        "ok": True,
        "exists": True,
        "preview_url": build_preview_url(video_path),
        "summary": {
            "clips": summary["clips"],
            "text_files": summary["text_files"],
            "segments": segments_count,
        }
    }


@app.post("/translate")
async def translate_endpoint(data: dict = Body(...)):
    try:
        text = data.get("text") or ""
        target_lang = data.get("target_lang") or "te"
        translated_text = translate_text(text, target_lang)
        return {
            "ok": translated_text is not None,
            "translated_text": translated_text or "",
        }
    except Exception as exc:
        log(f"ERROR translate: {exc}")
        return {
            "ok": False,
            "translated_text": "",
            "error": "",
        }


@app.post("/update-clip-timestamp")
async def update_clip_timestamp(data: dict = Body(...)):
    try:
        clip_id = int(data.get("clip_id"))
        start = float(data.get("start"))
        end = float(data.get("end"))
        fps = float(data.get("fps") or DEFAULT_FPS)
        video_duration = float(data.get("video_duration") or 0.0)
        segments = data.get("segments") or []
        transcript_words = data.get("transcript_words") or []

        if fps <= 0:
            fps = DEFAULT_FPS

        result = update_clip_timestamp_segments(
            segments=segments,
            transcript_words=transcript_words,
            clip_id=clip_id,
            start=start,
            end=end,
            fps=fps,
            video_duration=video_duration,
        )

        target_clip = next(
            (segment for segment in result["updated_clips"] if int(segment["clip_id"]) == clip_id),
            None,
        )
        if target_clip is None:
            raise ValueError(f"Clip {get_clip_number(clip_id)} was not updated")

        return {
            "ok": True,
            "clip_id": int(target_clip["clip_id"]),
            "start": target_clip["start"],
            "end": target_clip["end"],
            "text": target_clip["text"],
            "clips": [
                {
                    "clip_id": int(segment["clip_id"]),
                    "start": segment["start"],
                    "end": segment["end"],
                }
                for segment in result["segments"]
            ],
            "updated_clips": result["updated_clips"],
        }
    except Exception as exc:
        log(f"ERROR update-clip-timestamp: {exc}")
        return {
            "ok": False,
            "error": str(exc),
            "clip_id": data.get("clip_id"),
        }


@app.post("/transcribe-segment")
async def transcribe_segment(data: dict = Body(...)):
    try:
        video_path = resolve_safe_path(data.get("video_path") or "")
        clip_id = int(data.get("clip_id"))
        start_time = float(data.get("start_time"))
        end_time = float(data.get("end_time"))

        log(f"TRANSCRIBE SEGMENT clip_id={clip_id} start={start_time:.3f}s end={end_time:.3f}s")

        if end_time <= start_time:
            raise ValueError("Clip end must be greater than start")

        media_duration = get_media_duration(video_path)
        transcript_text = transcribe_clip_text(
            audio_path=video_path,
            clip_id=clip_id,
            start=start_time,
            end=end_time,
            alignment_mode="accurate",
            media_duration=media_duration,
        )
        transcript_text = refine_clip_text(transcript_text, 1.0)

        return {
            "ok": True,
            "clip_id": clip_id,
            "text": transcript_text,
        }
    except Exception as exc:
        log(f"ERROR transcribe-segment: {exc}")
        return {
            "ok": False,
            "error": str(exc),
            "clip_id": data.get("clip_id"),
        }


@app.post("/generate")
async def generate(data: dict = Body(...)):
    segments = data["segments"]
    video_path = resolve_safe_path(data["video_path"])
    mode = data.get("mode", "edit")

    log("Generating clips...")
    create_clips(video_path, segments)

    summary = build_output_summary(video_path, len(segments))

    return {
        "status": "done",
        "mode": mode,
        "summary": {
            "clips": summary["clips"],
            "text_files": summary["text_files"],
            "segments": summary["segments"],
        }
    }


@app.post("/generate-english-audio")
async def generate_english_audio(data: dict = Body(...)):
    try:
        clip_id = int(data.get("clip_id"))
        text = data.get("text") or ""
        video_path = resolve_safe_path(data.get("video_path") or "")
        start = float(data.get("start_time", data.get("start", 0.0)))
        end = float(data.get("end_time", data.get("end", 0.0)))
        invalidate = bool(data.get("invalidate", False))
        voice_style = normalize_voice_style(data.get("voice_style"))

        if end <= start:
            return {
                "ok": False,
                "error": f"Invalid timestamps for clip {get_clip_number(clip_id)}"
            }

        set_audio_generation_state(video_path, clip_id, voice_style, status="generating", error="")
        result = generate_clip_english_audio(
            video_path=video_path,
            clip_id=clip_id,
            text=text,
            start=start,
            end=end,
            invalidate=invalidate,
            voice_style=voice_style,
        )
        set_audio_generation_state(video_path, clip_id, voice_style, status="ready", error="")

        return {
            "ok": True,
            "clip_id": clip_id,
            "audio_url": result["audio_url"],
            "text_hash": result["text_hash"],
            "cached": result["cached"],
            "duration_seconds": result["duration_seconds"],
            "voice_style": result["voice_style"],
        }
    except Exception as exc:
        log(f"ERROR generate-english-audio: {exc}")
        try:
            set_audio_generation_state(video_path, clip_id, voice_style, status="error", error=str(exc))
        except Exception:
            pass
        return {
            "ok": False,
            "error": str(exc)
        }


@app.post("/generate-clip-audio")
async def generate_clip_audio(data: dict = Body(...)):
    return await generate_english_audio(data)


@app.post("/queue-english-audio")
async def queue_english_audio_generation(data: dict = Body(...)):
    try:
        video_path = resolve_safe_path(data.get("video_path") or "")
        segments = data.get("segments") or []
        voice_style = normalize_voice_style(data.get("voice_style"))
        clip_ids = data.get("clip_ids")
        clip_id = data.get("clip_id")
        invalidate = bool(data.get("invalidate", False))

        if clip_id is not None:
            clip_ids = [int(clip_id)]
        elif clip_ids is not None:
            clip_ids = [int(value) for value in clip_ids]

        started = start_background_audio_generation(
            video_path=video_path,
            segments=segments,
            voice_style=voice_style,
            clip_ids=clip_ids,
            invalidate=invalidate,
        )
        return {
            "ok": True,
            "queued": True,
            "started": started,
            "voice_style": voice_style,
            "clip_ids": clip_ids,
        }
    except Exception as exc:
        log(f"ERROR queue-english-audio: {exc}")
        return {
            "ok": False,
            "error": str(exc)
        }


@app.post("/english-audio-status")
async def english_audio_status(data: dict = Body(...)):
    try:
        video_path = resolve_safe_path(data.get("video_path") or "")
        segments = normalize_segments(data.get("segments") or [])
        voice_style = normalize_voice_style(data.get("voice_style"))
        statuses = [
            get_clip_audio_status(
                video_path=video_path,
                clip_id=int(segment["clip_id"]),
                text=segment.get("text") or "",
                voice_style=voice_style,
            )
            for segment in segments
        ]
        ready_count = sum(1 for item in statuses if item["status"] == "ready")
        generating_count = sum(1 for item in statuses if item["status"] in {"queued", "generating"})
        error_count = sum(1 for item in statuses if item["status"] == "error")
        return {
            "ok": True,
            "voice_style": voice_style,
            "clips": statuses,
            "summary": {
                "total": len(statuses),
                "ready": ready_count,
                "generating": generating_count,
                "errors": error_count,
            }
        }
    except Exception as exc:
        log(f"ERROR english-audio-status: {exc}")
        return {
            "ok": False,
            "error": str(exc)
        }


@app.get("/audio/{clip_id}")
async def get_generated_audio(
    clip_id: int,
    video_path: str = Query(...)
):
    resolved_video_path = resolve_safe_path(video_path)
    audio_path = get_clip_audio_file_path(resolved_video_path, clip_id)

    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="English clip audio not found")

    return FileResponse(audio_path, media_type="audio/wav", filename=os.path.basename(audio_path))


@app.post("/generate-english-video")
async def generate_english_video_endpoint(data: dict = Body(...)):
    try:
        video_path = resolve_safe_path(data.get("video_path") or "")
        segments = data.get("segments") or []
        voice_style = normalize_voice_style(data.get("voice_style"))
        result = generate_english_video(video_path=video_path, segments=segments, voice_style=voice_style)
        return {
            "ok": True,
            "signature": result["signature"],
            "video_url": result["video_url"],
            "voice_style": result["voice_style"],
        }
    except FileNotFoundError as exc:
        log(f"ERROR generate-english-video missing-file: {exc}")
        return {
            "ok": False,
            "error": str(exc)
        }
    except Exception as exc:
        log(f"ERROR generate-english-video: {exc}")
        return {
            "ok": False,
            "error": str(exc)
        }


@app.get("/test-tts")
async def test_tts():
    test_dir = os.path.join(TEMP_DIR, "test_tts")
    os.makedirs(test_dir, exist_ok=True)
    test_output_path = os.path.join(test_dir, "test.wav")

    try:
        generate_audio("Hello this is a test", test_output_path)
        return {
            "ok": True,
            "audio_path": test_output_path,
            "audio_url": build_preview_url(test_output_path),
        }
    except Exception as exc:
        log(f"ERROR test-tts: {exc}")
        return {
            "ok": False,
            "error": str(exc)
        }


@app.post("/upload-to-drive")
async def upload_to_drive(data: dict = Body(...)):
    folder_url = (data.get("folder_url") or "").strip()
    video_path = data.get("video_path") or ""
    segments_count = int(data.get("segments_count") or 0)

    if not folder_url:
        return {
            "ok": False,
            "message": "Enter Google Drive Folder URL"
        }

    if not is_valid_drive_folder_url(folder_url):
        return {
            "ok": False,
            "message": "Enter a valid Google Drive folder link"
        }

    if not video_path:
        return {
            "ok": False,
            "message": "No processed output is available for upload"
        }

    summary = build_output_summary(video_path, segments_count)
    if summary["clips"] == 0 and summary["text_files"] == 0:
        return {
            "ok": False,
            "message": "Generate clips before uploading to Google Drive"
        }

    try:
        is_public, message = is_drive_folder_public(folder_url)
    except Exception:
        return {
            "ok": False,
            "message": "Unable to verify the Google Drive folder right now"
        }

    if not is_public:
        return {
            "ok": False,
            "message": message or "Please make the folder public to allow upload"
        }

    target_folder_name = f"Processed_Video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    upload_endpoint = os.getenv("GOOGLE_DRIVE_UPLOAD_ENDPOINT", "").strip()

    if not upload_endpoint:
        return {
            "ok": False,
            "message": "Google Drive upload is not available in this app yet"
        }

    payload = {
        "folder_url": folder_url,
        "target_folder_name": target_folder_name,
        "files": summary["clip_files"] + summary["text_file_paths"],
    }

    request = urllib_request.Request(
        upload_endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "VideoTextEditor/1.0",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(request, timeout=30) as response:
            response_body = response.read().decode("utf-8", errors="ignore")
    except Exception:
        return {
            "ok": False,
            "message": "Failed to upload files to Google Drive"
        }

    try:
        upload_result = json.loads(response_body) if response_body else {}
    except json.JSONDecodeError:
        upload_result = {}

    if response.status >= 400 or upload_result.get("ok") is False:
        return {
            "ok": False,
            "message": upload_result.get("message") or "Failed to upload files to Google Drive"
        }

    return {
        "ok": True,
        "message": upload_result.get("message") or "Files successfully uploaded to Google Drive",
        "folder_name": target_folder_name
    }


@app.get("/download")
def download(video_path: str):
    zip_path = build_zip_archive(video_path)

    response = FileResponse(
        zip_path,
        media_type="application/zip",
        filename=os.path.basename(zip_path)
    )

    def cleanup():
        time.sleep(3)
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass

    threading.Thread(target=cleanup).start()

    return response
