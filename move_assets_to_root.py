#!/usr/bin/env python3
"""
Move .mp4 and .txt files from subfolders into a root directory safely.

Expected layout:
root/
  text/
  videos/

Actions:
1) Move all .mp4 files from videos/ -> root/
2) Move all .txt files from text/ -> root/, removing "_en" from filename
3) Avoid overwrite collisions by creating unique names: "name (1).ext", etc.
4) Optionally delete empty source folders with --delete-empty
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def unique_destination(dest: Path) -> Path:
    """
    Return a non-conflicting destination path.
    If destination exists, append " (n)" before the extension.
    """
    if not dest.exists():
        return dest

    stem, suffix = dest.stem, dest.suffix
    i = 1
    while True:
        candidate = dest.with_name(f"{stem} ({i}){suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def move_files(source_dir: Path, pattern: str, root_dir: Path, rename_txt: bool = False) -> int:
    """
    Move matching files from source_dir to root_dir.

    rename_txt=True:
      Removes '_en' from the filename stem for .txt files.
    """
    moved = 0

    # rglob handles nested directories and scales well for larger trees.
    for src in source_dir.rglob(pattern):
        if not src.is_file():
            continue

        target_name = src.name
        if rename_txt:
            target_name = src.stem.replace("_en", "") + src.suffix

        dest = unique_destination(root_dir / target_name)
        shutil.move(str(src), str(dest))
        moved += 1

        if src.name != dest.name:
            print(f"MOVED+RENAMED: {src} -> {dest}")
        else:
            print(f"MOVED:         {src} -> {dest}")

    return moved


def remove_empty_dirs(root_dir: Path, include_root: bool = False) -> None:
    """Delete empty directories under root_dir, deepest first."""
    for path in sorted(root_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
                print(f"DELETED EMPTY DIR: {path}")
            except OSError:
                # Not empty or not removable; leave it.
                pass

    if include_root:
        try:
            root_dir.rmdir()
            print(f"DELETED EMPTY DIR: {root_dir}")
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move .mp4/.txt files to root safely and optionally clean empty dirs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root containing 'videos/' and 'text/' (default: current directory).",
    )
    parser.add_argument(
        "--delete-empty",
        action="store_true",
        help="Delete empty directories under videos/ and text/ after moving.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    videos_dir = root / "videos"
    text_dir = root / "text"

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Invalid root directory: {root}")

    total_moved = 0

    if videos_dir.exists() and videos_dir.is_dir():
        total_moved += move_files(videos_dir, "*.mp4", root, rename_txt=False)
    else:
        print(f"SKIP: videos directory not found: {videos_dir}")

    if text_dir.exists() and text_dir.is_dir():
        total_moved += move_files(text_dir, "*.txt", root, rename_txt=True)
    else:
        print(f"SKIP: text directory not found: {text_dir}")

    if args.delete_empty:
        if videos_dir.exists():
            remove_empty_dirs(videos_dir, include_root=True)
        if text_dir.exists():
            remove_empty_dirs(text_dir, include_root=True)

    print(f"\nDone. Total files moved: {total_moved}")


if __name__ == "__main__":
    main()
