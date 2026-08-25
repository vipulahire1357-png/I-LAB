"""
Visualization demo for Module 3+4 — Classroom-optimised Person Tracking.

Usage::

    python examples/visualize_tracking.py                  # general mode
    python examples/visualize_tracking.py --classroom      # classroom mode

Opens a file-explorer dialog, then plays the video with:
- Colour-coded bounding boxes per Track ID
- Track ID + confidence labels
- Person count, active tracks, total tracks
- Detection + tracking latency
- Scene profile indicator

Press **q** to quit.
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
from src.tracking import PersonTracker, TrackerConfig, TrackState

_VIDEO_EXTENSIONS = (
    ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
    ("All files", "*.*"),
)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BAR_BG = (25, 25, 25)
_TEXT_WHITE = (255, 255, 255)
_TEXT_DARK = (0, 0, 0)
_ROI_COLOUR = (0, 200, 255)  # yellow-orange for ROI outline


def _select_video_file() -> str | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    fp = filedialog.askopenfilename(title="Select a video file", filetypes=_VIDEO_EXTENSIONS)
    root.destroy()
    return fp if fp else None


def _id_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR colour derived from track ID."""
    hue = int((track_id * 47) % 180)
    hsv = np.uint8([[[hue, 220, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_tracks(
    image: np.ndarray,
    tracks,
    det_count: int,
    det_ms: float,
    trk_ms: float,
    total_created: int,
    scene: str,
    roi_norm: tuple | None = None,
) -> np.ndarray:
    """Draw bounding boxes with Track IDs and a status bar."""
    vis = image.copy()
    h, w = vis.shape[:2]

    # Draw ROI boundary if classroom
    if roi_norm is not None:
        rx1 = int(roi_norm[0] * w)
        ry1 = int(roi_norm[1] * h)
        rx2 = int(roi_norm[2] * w)
        ry2 = int(roi_norm[3] * h)
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), _ROI_COLOUR, 1)
        cv2.putText(vis, "FAR ROI", (rx1 + 5, ry1 + 18), _FONT, 0.45, _ROI_COLOUR, 1, cv2.LINE_AA)

    active_count = 0
    for t in tracks:
        if t.state != TrackState.CONFIRMED:
            continue
        active_count += 1

        x1, y1, x2, y2 = int(t.x1), int(t.y1), int(t.x2), int(t.y2)
        colour = _id_color(t.track_id)

        # Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)

        # Label: "ID <n> <conf>"
        label = f"ID {t.track_id} {t.confidence:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, _FONT, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - lh - 8), (x1 + lw + 6, y1), colour, -1)
        cv2.putText(vis, label, (x1 + 3, y1 - 4), _FONT, 0.5, _TEXT_DARK, 1, cv2.LINE_AA)

    # --- status bar (two lines) ------------------------------------------
    bar_h = 60
    cv2.rectangle(vis, (0, h - bar_h), (w, h), _BAR_BG, -1)

    line1 = (
        f"Detected: {det_count}    Active tracks: {active_count}    "
        f"Total tracks: {total_created}    Scene: {scene.upper()}"
    )
    cv2.putText(vis, line1, (10, h - 35), _FONT, 0.50, _TEXT_WHITE, 1, cv2.LINE_AA)

    line2 = (
        f"Detection: {det_ms:.0f} ms    "
        f"Tracking: {trk_ms:.1f} ms    "
        f"Total: {det_ms + trk_ms:.0f} ms    "
        f"FPS: {1000.0 / max(det_ms + trk_ms, 1):.1f}"
    )
    cv2.putText(vis, line2, (10, h - 12), _FONT, 0.50, _TEXT_WHITE, 1, cv2.LINE_AA)

    return vis


def main() -> None:
    classroom_mode = "--classroom" in sys.argv

    # Select video
    args_paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args_paths:
        video_path = args_paths[0]
    else:
        print("  Opening file explorer — please select a video file …")
        video_path = _select_video_file()
        if not video_path:
            print("  No file selected. Exiting.")
            sys.exit(0)

    scene_label = "classroom" if classroom_mode else "general"
    print(f"\n  Mode: {scene_label.upper()}")
    print(f"  Tracking persons in: {video_path}")
    print(f"  Press 'q' to quit.\n")

    # --- initialise pipeline ---------------------------------------------
    if classroom_mode:
        det_config = DetectionConfig.classroom_default()
        trk_config = TrackerConfig.classroom_default()
    else:
        det_config = DetectionConfig(
            model_path="yolo11n.pt",
            confidence_threshold=0.35,
            device="cpu",
        )
        trk_config = TrackerConfig(
            max_age=30,
            min_hits=3,
            track_high_threshold=0.25,
            match_threshold=0.8,
        )

    detector = PersonDetector(det_config)
    detector.warmup()

    tracker = PersonTracker(trk_config)

    source = FileVideoSource(video_path)
    source.open()
    meta = source.get_metadata()

    window_name = f"Person Tracking — {scene_label.upper()}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    display_w = min(meta.width or 1280, 1280)
    display_h = int(display_w * (meta.height or 720) / (meta.width or 1280))
    cv2.resizeWindow(window_name, display_w, display_h)

    while True:
        frame = source.read()
        if frame is None:
            break

        # Detection
        det_result = detector.detect(frame)

        # Tracking
        trk_result = tracker.update(det_result)

        metrics = tracker.get_metrics()
        vis = _draw_tracks(
            frame.image,
            trk_result.tracks,
            det_count=det_result.count,
            det_ms=det_result.total_time_ms,
            trk_ms=trk_result.tracking_time_ms,
            total_created=metrics.total_tracks_created,
            scene=scene_label,
            roi_norm=det_config.classroom_far_roi if classroom_mode else None,
        )

        cv2.imshow(window_name, vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    source.close()
    cv2.destroyAllWindows()

    # Final summary
    m = tracker.get_metrics()
    print(f"\n  {'─' * 50}")
    print(f"  Scene profile        : {scene_label.upper()}")
    print(f"  Model                : {det_config.model_path}")
    print(f"  Input size           : {det_config.input_size}")
    print(f"  Confidence threshold : {det_config.confidence_threshold}")
    if classroom_mode and det_config.classroom_far_roi:
        print(f"  Far ROI              : {det_config.classroom_far_roi}")
        print(f"  ROI confidence       : {det_config.classroom_roi_confidence}")
    print(f"  Frames processed     : {m.frames_processed}")
    print(f"  Total tracks created : {m.total_tracks_created}")
    print(f"  Max active tracks    : {m.max_active_tracks}")
    if m.average_tracking_time is not None:
        print(f"  Avg tracking time    : {m.average_tracking_time:.2f} ms")
    print(f"  {'─' * 50}")
    print("  Done.")


if __name__ == "__main__":
    main()

