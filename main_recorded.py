import sys
from pathlib import Path

from main import main

if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).resolve().parent

    video_path = base_path / "data" / "input" / "original - Trim.mp4"

    sys.argv = [
        "main.py",
        "--input",
        str(video_path),
    ]

    sys.exit(main())