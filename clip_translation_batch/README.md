# Clip Translation Batch

This is a separate workflow for your first method. It does not change the current UI flow.

It takes a folder of short segment clips, runs Whisper in translate mode, and creates one English `.txt` file per clip.

## Run

Use the XTTS environment from this repo:

```bash
cd /Users/rupid/Desktop/video-text-ui-copy-copy
./xtts-env/bin/python clip_translation_batch/export_segment_translations.py
```

The script will ask for:

1. The folder containing your small segment clips.
2. The output folder for the English text files.

## Notes

- Supported inputs: `.mp4`, `.mov`, `.m4v`, `.avi`, `.mkv`, `.webm`, `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`
- Default model: `large-v3`
- Default source language: `te`

## Optional CLI Usage

```bash
./xtts-env/bin/python clip_translation_batch/export_segment_translations.py \
  --input-dir /path/to/segment_clips \
  --output-dir /path/to/segment_clips/translated_text
```
