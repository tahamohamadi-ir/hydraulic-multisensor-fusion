"""Fetch official UCI 447 ZIP or import previously downloaded archive.

Use the downloaded ZIP directly in the experiment: no full extraction necessary.
"""
import argparse
from pathlib import Path
from shutil import copyfile
from urllib.request import urlretrieve
from zipfile import ZipFile

# Official UCI legacy archive often contains data.zip with raw txt files.
URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00447/data.zip",
    "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, help="Locally downloaded UCI ZIP")
    p.add_argument("--output", type=Path, default=Path("data/hydraulic_uci.zip"))
    a = p.parse_args()
    a.output.parent.mkdir(exist_ok=True, parents=True)
    if a.archive:
        if a.archive.resolve() != a.output.resolve():
            copyfile(a.archive, a.output)
    elif not a.output.exists():
        for url in URLS:
            try:
                print("Downloading from", url)
                urlretrieve(url, a.output)
                break
            except Exception as exc:
                print("This URL failed:", type(exc).__name__)
                a.output.unlink(missing_ok=True)
        else:
            raise SystemExit("Could not reach UCI. Download archive in a browser and "
                             "run --archive path/to/downloaded.zip")
    with ZipFile(a.output) as f:
        if not f.namelist():
            raise ValueError("Empty ZIP archive")
        if not any(x.endswith("profile.txt") or x.endswith("data.zip") for x in f.namelist()):
            raise ValueError("No profile.txt or data.zip in archive; check UCI download")
    print("Ready:", a.output)


if __name__ == "__main__":
    main()
