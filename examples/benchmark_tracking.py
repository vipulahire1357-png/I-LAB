"""
Benchmark script for Module 4 — Person Tracking.

Usage::

    python examples/benchmark_tracking.py

Opens a file-explorer dialog, runs the full detection + tracking pipeline,
and reports timing and track statistics.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.video import FileVideoSource
from src.detection import PersonDetector, DetectionConfig
from src.tracking import PersonTracker, TrackerConfig, TrackState

_VIDEO_EXTENSIONS = (
    ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
    ("All files", "*.*"),
)


def _select_video_file() -> str | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    fp = filedialog.askopenfilename(title="Select a video file", filetypes=_VIDEO_EXTENSIONS)
    root.destroy()
    return fp if fp else None


def main() -> None:
    if len(sys.argv) >= 2:
        video_path = sys.argv[1]
    else:
        print("  Opening file explorer — please select a video file …")
        video_path = _select_video_file()
        if not video_path:
            print("  No file selected. Exiting.")
            sys.exit(0)

    print(f"\n{'=' * 60}")
    print(f"  Module 4 — Tracking Benchmark")
    print(f"{'=' * 60}")

    # --- Detector --------------------------------------------------------
    det_config = DetectionConfig(
        model_path="yolo11n.pt",
        confidence_threshold=0.35,
        device="cpu",
    )
    print("\n  Loading detector …")
    detector = PersonDetector(det_config)
    detector.warmup(iterations=3)

    # --- Tracker ---------------------------------------------------------
    trk_config = TrackerConfig(
        max_age=30,
        min_hits=3,
        track_high_threshold=0.25,
        match_threshold=0.8,
    )
    tracker = PersonTracker(trk_config)

    # --- Video -----------------------------------------------------------
    source = FileVideoSource(video_path)
    source.open()
    meta = source.get_metadata()

    print(f"\n  {'─' * 45}")
    print(f"  Video       : {meta.source_path}")
    print(f"  Resolution  : {meta.width}×{meta.height}")
    print(f"  FPS         : {meta.fps}")
    print(f"  Frames      : {meta.frame_count}")
    print(f"  Detector    : {det_config.model_path} (conf={det_config.confidence_threshold})")
    print(f"  Tracker     : {trk_config.tracker_type} (max_age={trk_config.max_age}, min_hits={trk_config.min_hits})")
    print(f"  {'─' * 45}\n")

    # --- Benchmark -------------------------------------------------------
    det_times: list[float] = []
    trk_times: list[float] = []
    active_counts: list[int] = []
    frames_processed = 0

    bench_t0 = time.monotonic()

    while True:
        frame = source.read()
        if frame is None:
            break

        det_result = detector.detect(frame)
        trk_result = tracker.update(det_result)

        det_times.append(det_result.total_time_ms)
        trk_times.append(trk_result.tracking_time_ms)
        active_counts.append(trk_result.active_count)
        frames_processed += 1

        if frames_processed % 50 == 0:
            print(
                f"  Frame {frames_processed:>5d}  |  "
                f"det: {det_result.total_time_ms:>6.1f} ms  |  "
                f"trk: {trk_result.tracking_time_ms:>5.2f} ms  |  "
                f"active: {trk_result.active_count}"
            )

    bench_elapsed = time.monotonic() - bench_t0
    source.close()

    if not trk_times:
        print("  No frames processed.")
        return

    # --- Report ----------------------------------------------------------
    m = tracker.get_metrics()

    print(f"\n  {'=' * 50}")
    print(f"  BENCHMARK RESULTS")
    print(f"  {'=' * 50}")
    print(f"  Frames processed     : {frames_processed}")
    print(f"  Total wall time      : {bench_elapsed:.2f} s")
    print()
    print(f"  Detector")
    print(f"    Avg inference       : {statistics.mean(det_times):.1f} ms")
    print(f"    Median              : {statistics.median(det_times):.1f} ms")
    print()
    print(f"  Tracker")
    print(f"    Avg tracking time   : {statistics.mean(trk_times):.2f} ms")
    print(f"    Median              : {statistics.median(trk_times):.2f} ms")
    if len(trk_times) >= 20:
        p95 = sorted(trk_times)[int(len(trk_times) * 0.95)]
        print(f"    P95                 : {p95:.2f} ms")
    print()
    print(f"  Track statistics")
    print(f"    Total created       : {m.total_tracks_created}")
    print(f"    Total removed       : {m.total_tracks_removed}")
    print(f"    Max active          : {m.max_active_tracks}")
    print(f"    Avg active/frame    : {statistics.mean(active_counts):.1f}")
    print()

    total_avg = statistics.mean(det_times) + statistics.mean(trk_times)
    print(f"  Pipeline")
    print(f"    Avg total/frame     : {total_avg:.1f} ms")
    print(f"    Approximate FPS     : {1000.0 / total_avg:.1f}")
    print(f"  {'=' * 50}\n")


if __name__ == "__main__":
    main()

