"""
Visualization demo for Module 3 — Person Detection.

Usage::

    python examples/visualize_person_detection.py

Opens a file-explorer dialog, then plays the video with bounding boxes,
confidence scores, detection count, and inference FPS drawn on each frame.

Press **q** to quit.

This is a development tool — NOT part of the production pipeline.
The detector returns data; this script just draws it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np

from src.video import FileVideoSource
from src.detection import PersonDetector, DetectionConfig

_VIDEO_EXTENSIONS = (
    ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
    ("All files", "*.*"),
)

# Visual constants
_BOX_COLOR = (0, 255, 120)       # bright green (BGR)
_TEXT_COLOR = (255, 255, 255)     # white
_BAR_COLOR = (30, 30, 30)        # dark bar background
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_THICKNESS = 2


def _select_video_file() -> str | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    fp = filedialog.askopenfilename(title="Select a video file", filetypes=_VIDEO_EXTENSIONS)
    root.destroy()
    return fp if fp else None


def _draw_detections(image: np.ndarray, result, fps: float) -> np.ndarray:
    """Draw bounding boxes, confidence, count, and FPS on a copy of the image."""
    vis = image.copy()
    h, w = vis.shape[:2]

    for det in result.detections:
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)

        # Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), _BOX_COLOR, _THICKNESS)

        # Label background
        label = f"Person {det.confidence:.2f}"
        (lw, lh), baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, 1)
        cv2.rectangle(vis, (x1, y1 - lh - 8), (x1 + lw + 4, y1), _BOX_COLOR, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4), _FONT, _FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)

    # Bottom status bar
    bar_h = 36
    cv2.rectangle(vis, (0, h - bar_h), (w, h), _BAR_COLOR, -1)
    status = f"People: {result.count}    Detection: {fps:.1f} FPS    Frame: {result.frame_id}"
    cv2.putText(vis, status, (10, h - 10), _FONT, 0.6, _TEXT_COLOR, 1, cv2.LINE_AA)

    return vis


def main() -> None:
    if len(sys.argv) >= 2:
        video_path = sys.argv[1]
    else:
        print("  Opening file explorer — please select a video file …")
        video_path = _select_video_file()
        if not video_path:
            print("  No file selected. Exiting.")
            sys.exit(0)

    print(f"\n  Visualizing person detection on: {video_path}")
    print(f"  Press 'q' to quit.\n")

    config = DetectionConfig(
        model_path="yolo11n.pt",
        confidence_threshold=0.35,
        device="cpu",
    )
    detector = PersonDetector(config)
    detector.warmup()

    source = FileVideoSource(video_path)
    source.open()
    meta = source.get_metadata()

    window_name = "Person Detection — Module 3"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Reasonable initial window size.
    display_w = min(meta.width or 1280, 1280)
    display_h = int(display_w * (meta.height or 720) / (meta.width or 1280))
    cv2.resizeWindow(window_name, display_w, display_h)

    fps_smooth = 0.0

    while True:
        frame = source.read()
        if frame is None:
            break

        result = detector.detect(frame)

        # Smoothed FPS from total_time_ms.
        if result.total_time_ms > 0:
            instant_fps = 1000.0 / result.total_time_ms
            fps_smooth = 0.8 * fps_smooth + 0.2 * instant_fps

        vis = _draw_detections(frame.image, result, fps_smooth)
        cv2.imshow(window_name, vis)

        # Wait ~1 ms; exit on 'q'.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    source.close()
    cv2.destroyAllWindows()
    print("  Done.")


if __name__ == "__main__":
    main()

