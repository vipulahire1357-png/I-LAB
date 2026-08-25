"""
Benchmark script for Module 3 — Person Detection.

Usage::

    python examples/benchmark_person_detection.py

Opens a file-explorer dialog to select a video, then benchmarks the
person detector over all frames and reports statistics.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.video import FileVideoSource, VideoSourceError
from src.detection import PersonDetector, DetectionConfig

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

    config = DetectionConfig(
        model_path="yolo11n.pt",
        confidence_threshold=0.35,
        device="cpu",
    )

    print(f"\n{'=' * 60}")
    print(f"  Module 3 — Person Detection Benchmark")
    print(f"{'=' * 60}")

    # --- Load model ------------------------------------------------------
    print("\n  Loading model …")
    load_t0 = time.monotonic()
    detector = PersonDetector(config)
    load_ms = (time.monotonic() - load_t0) * 1000
    print(f"  Model loaded in {load_ms:.0f} ms")

    # --- Warm-up ---------------------------------------------------------
    print("  Warming up …")
    warmup_ms = detector.warmup(iterations=3)
    print(f"  Warm-up done in {warmup_ms:.0f} ms")

    # --- Open video ------------------------------------------------------
    source = FileVideoSource(video_path)
    source.open()
    meta = source.get_metadata()

    print(f"\n  {'─' * 40}")
    print(f"  Model       : {config.model_path}")
    print(f"  Device      : {config.device}")
    print(f"  Input size  : {config.input_size}")
    print(f"  Confidence  : {config.confidence_threshold}")
    print(f"  IoU         : {config.iou_threshold}")
    print(f"  Video       : {meta.source_path}")
    print(f"  Resolution  : {meta.width}×{meta.height}")
    print(f"  Video FPS   : {meta.fps}")
    print(f"  Frame count : {meta.frame_count}")
    print(f"  {'─' * 40}\n")

    # --- Benchmark -------------------------------------------------------
    infer_times: list[float] = []
    total_times: list[float] = []
    det_counts: list[int] = []
    frames_processed = 0

    bench_t0 = time.monotonic()

    while True:
        frame = source.read()
        if frame is None:
            break

        result = detector.detect(frame)
        infer_times.append(result.inference_time_ms)
        total_times.append(result.total_time_ms)
        det_counts.append(result.count)
        frames_processed += 1

        if frames_processed % 50 == 0:
            avg = statistics.mean(infer_times[-50:])
            print(f"  Frame {frames_processed:>5d}  |  last-50 avg infer: {avg:>6.1f} ms  |  detections: {result.count}")

    bench_elapsed = (time.monotonic() - bench_t0) * 1000
    source.close()

    # --- Report ----------------------------------------------------------
    if not infer_times:
        print("  No frames processed.")
        return

    print(f"\n  {'=' * 50}")
    print(f"  BENCHMARK RESULTS")
    print(f"  {'=' * 50}")
    print(f"  Frames processed     : {frames_processed}")
    print(f"  Total wall time      : {bench_elapsed:.0f} ms ({bench_elapsed / 1000:.2f} s)")
    print()
    print(f"  Inference time (ms)")
    print(f"    Average            : {statistics.mean(infer_times):.1f}")
    print(f"    Median             : {statistics.median(infer_times):.1f}")
    if len(infer_times) >= 20:
        p95 = sorted(infer_times)[int(len(infer_times) * 0.95)]
        print(f"    P95                : {p95:.1f}")
    print(f"    Min                : {min(infer_times):.1f}")
    print(f"    Max                : {max(infer_times):.1f}")
    print()
    print(f"  Total detect time (ms)")
    print(f"    Average            : {statistics.mean(total_times):.1f}")
    print()
    approx_fps = 1000.0 / statistics.mean(total_times)
    print(f"  Approximate FPS      : {approx_fps:.1f}")
    print(f"  Avg detections/frame : {statistics.mean(det_counts):.2f}")
    print(f"  {'=' * 50}\n")


if __name__ == "__main__":
    main()

