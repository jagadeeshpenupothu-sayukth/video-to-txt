import os
import json
import re
import subprocess
import time
import shutil
import zipfile
import threading
from datetime import datetime
from urllib import error as urllib_error
from urllib import request as urllib_request

from fastapi import FastAPI, UploadFile, File, Request, Body, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from faster_whisper import WhisperModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)
app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")
DEFAULT_FPS = 30.0
DEFAULT_AUDIO_SAMPLE_RATE = 16000
CLIP_PADDING_SECONDS = 0.08
TIMESTAMP_PRECISION_DECIMALS = 3
DRIVE_FOLDER_PATTERN = re.compile(
    r"^https://drive\.google\.com/drive/folders/([A-Za-z0-9_-]+)(?:\?.*)?$"
)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ======================
# EXISTING PIPELINE (UNCHANGED)
# ======================

def extract_audio(input_video):
    audio_path = os.path.join(TEMP_DIR, "audio.wav")

    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_video,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ], check=True)

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

def normalize_segments(segments, sample_rate=DEFAULT_AUDIO_SAMPLE_RATE):
    normalized = []
    previous_end_ms = None

    for segment in segments:
        start = snap_to_audio_frame(float(segment["start"]), sample_rate)
        end = snap_to_audio_frame(float(segment["end"]), sample_rate)
        start_ms = seconds_to_milliseconds(start)
        end_ms = seconds_to_milliseconds(end)

        if previous_end_ms is not None:
            start_ms = previous_end_ms

        if end_ms <= start_ms:
            log(
                f"Skipping segment with non-positive duration: "
                f"start={format_timestamp_for_ffmpeg(start_ms)} "
                f"end={format_timestamp_for_ffmpeg(end_ms)} "
                f"text={segment['text'][:40]!r}"
            )
            continue

        normalized_segment = {
            "start": milliseconds_to_seconds(start_ms),
            "end": milliseconds_to_seconds(end_ms),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": segment["text"]
        }
        for key, value in segment.items():
            if key not in normalized_segment:
                normalized_segment[key] = value

        normalized.append(normalized_segment)
        previous_end_ms = end_ms

    return normalized

def normalize_words(words, sample_rate=DEFAULT_AUDIO_SAMPLE_RATE):
    normalized = []

    for word in words:
        text = (word.get("text") or "").strip()
        if not text:
            continue

        start = snap_to_audio_frame(float(word["start"]), sample_rate)
        end = snap_to_audio_frame(float(word["end"]), sample_rate)
        start_ms = seconds_to_milliseconds(start)
        end_ms = seconds_to_milliseconds(end)

        if end_ms <= start_ms:
            end_ms = start_ms + 1

        normalized.append({
            "text": text,
            "start": milliseconds_to_seconds(start_ms),
            "end": milliseconds_to_seconds(end_ms),
            "start_ms": start_ms,
            "end_ms": end_ms,
        })

    return normalized

def build_segment_text_from_words(segment, words, fallback_text=""):
    start = float(segment["start"])
    end = float(segment["end"])
    selected_words = []

    for word in words:
        word_start = float(word["start"])
        word_end = float(word["end"])
        if word_start >= start and word_end <= end:
            selected_words.append(word["text"])

    if not selected_words:
        return fallback_text.strip()

    text = " ".join(selected_words)
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return re.sub(r"\s{2,}", " ", text).strip()

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

        if previous_end_ms is not None and start_ms != previous_end_ms:
            raise ValueError(
                f"Segment continuity error at {index}: "
                f"previous end {format_timestamp_for_ffmpeg(previous_end_ms)} != "
                f"current start {format_timestamp_for_ffmpeg(start_ms)}"
            )

        if end_ms <= start_ms:
            raise ValueError(
                f"Segment duration error at {index}: "
                f"start {format_timestamp_for_ffmpeg(start_ms)} "
                f"end {format_timestamp_for_ffmpeg(end_ms)}"
            )

        previous_end_ms = end_ms

def transcribe(audio, fps):
    log("Loading Whisper model...")
    model = WhisperModel("medium", compute_type="int8")

    log("Translating Telugu to English...")
    segments, _ = model.transcribe(
        audio,
        task="translate",
        language="te",
        word_timestamps=True
    )

    results = []
    transcript_words = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue

        words = [word for word in (seg.words or []) if word.word.strip()]
        transcript_words.extend(
            {
                "text": word.word.strip(),
                "start": float(word.start),
                "end": float(word.end),
            }
            for word in words
        )

        if words:
            raw_start = max(float(words[0].start) - CLIP_PADDING_SECONDS, 0.0)
            raw_end = float(words[-1].end) + CLIP_PADDING_SECONDS
        else:
            raw_start = max(float(seg.start) - CLIP_PADDING_SECONDS, 0.0)
            raw_end = float(seg.end) + CLIP_PADDING_SECONDS

        snapped_start = snap_to_audio_frame(raw_start)
        snapped_end = snap_to_audio_frame(raw_end)

        if snapped_end <= snapped_start:
            snapped_end = snap_to_audio_frame(snapped_start + (1.0 / DEFAULT_AUDIO_SAMPLE_RATE))

        results.append({
            "start": snapped_start,
            "end": snapped_end,
            "text": text
        })

    return normalize_segments(results), normalize_words(transcript_words)

def create_clips(input_video, segments):
    video_name = os.path.splitext(os.path.basename(input_video))[0]
    base_dir = os.path.dirname(input_video)
    normalized_segments = normalize_segments(segments)
    validate_continuity(normalized_segments)

    for i, seg in enumerate(normalized_segments):
        start_ms = int(seg["start_ms"])
        end_ms = int(seg["end_ms"])
        duration_ms = end_ms - start_ms

        if duration_ms <= 0:
            log(f"Skipping segment {i+1} because duration is not positive")
            continue

        start = format_timestamp_for_ffmpeg(start_ms)
        end = format_timestamp_for_ffmpeg(end_ms)

        log(f"Creating clip {i+1}")

        clip_path = os.path.join(base_dir, f"{video_name}_clip{i+1}.mp4")

        subprocess.run([
            "ffmpeg", "-y",
            "-i", input_video,
            "-vf", f"trim=start={start}:end={end},setpts=PTS-STARTPTS",
            "-af", f"atrim=start={start}:end={end},asetpts=PTS-STARTPTS",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "fast",
            "-crf", "23",
            clip_path
        ], check=True)

        txt_path = os.path.join(base_dir, f"{video_name}_en_clip{i+1}.txt")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(seg.get("text") or "")

def get_generated_files(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    base_dir = os.path.dirname(video_path)
    clip_files = []
    text_files = []

    for file_name in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, file_name)
        if not os.path.isfile(full_path):
            continue

        if file_name.startswith(f"{video_name}_clip") and file_name.endswith(".mp4"):
            clip_files.append(full_path)
        elif file_name.startswith(f"{video_name}_en_clip") and file_name.endswith(".txt"):
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
    file_name = os.path.basename(video_path or "")
    if not file_name:
        return ""
    return f"/temp/{file_name}"

def build_zip_archive(video_path):
    clip_files, text_files = get_generated_files(video_path)
    archive_files = clip_files + text_files

    if not archive_files:
        raise FileNotFoundError("No generated output files were found.")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    zip_path = os.path.join(TEMP_DIR, f"{video_name}_clips.zip")

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file_path in archive_files:
            zipf.write(file_path, os.path.basename(file_path))

    return zip_path, archive_files

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

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "drive_upload_configured": is_drive_upload_configured()
        }
    )

@app.get("/drive-status")
def drive_status():
    return {
        "configured": is_drive_upload_configured()
    }

@app.post("/process")
async def process_video(
    file: UploadFile = File(...),
    mode: str = Form("edit")
):
    input_path = os.path.join(TEMP_DIR, file.filename)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    log("Extracting audio...")
    audio = extract_audio(input_path)

    fps = get_video_fps(input_path)
    log(f"Detected video FPS: {fps:.3f}")

    log("Running transcription...")
    segments, transcript_words = transcribe(audio, fps)

    return {
        "mode": mode,
        "segments": segments,
        "transcript_words": transcript_words,
        "video_path": input_path,
        "preview_url": build_preview_url(input_path),
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

@app.post("/generate")
async def generate(data: dict = Body(...)):
    segments = data["segments"]
    video_path = data["video_path"]
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

# ======================
# DOWNLOAD + CLEANUP
# ======================

@app.get("/download")
def download(video_path: str):
    zip_path, archive_files = build_zip_archive(video_path)

    response = FileResponse(
        zip_path,
        media_type='application/zip',
        filename=os.path.basename(zip_path)
    )

    def cleanup():
        time.sleep(3)
        for file in os.listdir(TEMP_DIR):
            try:
                file_path = os.path.join(TEMP_DIR, file)
                if file_path == zip_path:
                    os.remove(file_path)
            except:
                pass

    threading.Thread(target=cleanup).start()

    return response
