"""Download the Extended YaleB cropped face dataset.

Primary source: UCSD vision lab mirror.
Fallback:       the ``yaleb.mat`` redistributed by the sparse subspace
                 clustering community on GitHub (a much smaller 32x32
                 preprocessed version, useful if the big zip is
                 unreachable).

Artifacts placed under ``data/``.
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
import zipfile

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

SOURCES = [
    # Extended YaleB, 38 subjects x 64 illuminations, cropped 48x42 images.
    # Redistributed by the Deep-Subspace-Clustering-Networks codebase.
    ("YaleBCrop025.mat",
     "https://github.com/panji1990/Deep-subspace-clustering-networks/raw/master/Data/YaleBCrop025.mat"),
]


def _download(url: str, dest: str) -> bool:
    print(f"→ fetching {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())
        size = os.path.getsize(dest)
        print(f"  saved {dest} ({size/1e6:.2f} MB)")
        return size > 1024
    except Exception as e:
        print(f"  FAILED: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def main():
    for name, url in SOURCES:
        dest = os.path.join(DATA_DIR, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            print(f"already have {dest}")
            return 0
        if _download(url, dest):
            return 0
    print("ERROR: could not fetch YaleB from any mirror.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
