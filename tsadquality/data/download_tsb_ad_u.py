#!/usr/bin/env python3
"""Download and extract the TSB-AD-U dataset archive into this directory."""

import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.thedatum.org/datasets/TSB-AD-U.zip"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEST = DATA_DIR / "TSB-AD-U.zip"


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest}")

    def report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        percent = min(downloaded / total_size * 100, 100)
        sys.stdout.write(f"\r{percent:6.2f}% ({downloaded}/{total_size} bytes)")
        sys.stdout.flush()

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, dest, reporthook=report)
    print(f"\nSaved to {dest}")


def extract(archive: Path, dest_dir: Path) -> None:
    print(f"Extracting {archive} -> {dest_dir}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir)
    archive.unlink()
    print(f"Removed {archive}")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download(URL, DEST)
    extract(DEST, DATA_DIR)
