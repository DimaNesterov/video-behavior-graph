"""Parse the MEVA S3 file listing into a per-camera summary.

Clip name pattern: YYYY-MM-DD.HH-MM-SS.HH-MM-SS.location.camera.r13.avi
Usage: python scripts/meva_camera_stats.py
"""
import re
from pathlib import Path

import pandas as pd

LISTING = Path("data/raw/meva/file_list.txt")

# Example key: drops-123-r13/2018-03-07/16/2018-03-07.16-50-00.16-55-00.admin.G329.r13.avi
PATTERN = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})\.\d{2}-\d{2}-\d{2}\.\d{2}-\d{2}-\d{2}"
    r"\.(?P<location>[\w-]+)\.(?P<camera>[GU]\d+)\."
)

def main():
    rows = []
    for line in LISTING.read_text(encoding="utf-16").splitlines():
        m = PATTERN.search(line)
        if not m:
            continue
        size_bytes = int(line.split()[2])  # ls format: date time size key
        rows.append(dict(**m.groupdict(), size_mb=size_bytes / 1e6))
    df = pd.DataFrame(rows)
    print(f"Total clips parsed: {len(df)}\n")

    summary = (df.groupby(["camera", "location"])
                 .agg(clips=("date", "size"),
                      days=("date", "nunique"),
                      total_gb=("size_mb", lambda s: round(s.sum() / 1000, 1)))
                 .sort_values("clips", ascending=False))
    print(summary.to_string())
    out = Path("data/raw/meva/camera_stats.csv")
    summary.to_csv(out)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()