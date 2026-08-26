"""
Pipeline Orchestrator (Module 7).

Coordinates the sequential processing of video frames through VideoSource,
PersonDetector, PersonTracker, EntryExitCounter, OccupancyManager, and Visualizer,
while managing video encoding and metrics collection.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2

if TYPE_CHECKING:
    from src.config.models import AppConfig

from src.counting.entry_exit_counter import EntryExitCounter
from src.detection.person_detector import PersonDetector
from src.occupancy.occupancy_manager import OccupancyManager
from src.pipeline.models import PipelineMetrics
from src.tracking.person_tracker import PersonTracker
from src.video.file_video_source import FileVideoSource
from src.visualization.visualizer import Visualizer

logger = logging.getLogger(__name__)


class Pipeline:
    """End-to-end computer vision pipeline orchestrator.

    Parameters
    ----------
    config : AppConfig
        Complete application configuration.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config

        # Instantiate modules
        self._video_source = FileVideoSource(config.pipeline.input_path)
        self._detector = PersonDetector(config.detection)
        self._tracker = PersonTracker(config.tracking)
        self._counter = EntryExitCounter(config.counting)
        self._occupancy = OccupancyManager()
        self._visualizer = Visualizer(config.visualization)

    def run(self) -> PipelineMetrics:
        """Execute the pipeline on the input video until completion or interruption.

        Returns
        -------
        PipelineMetrics
            Execution and domain summary statistics.
        """
        logger.info("Starting CampusNetra product0.3 Pipeline...")
        self._video_source.open()
        metadata = self._video_source.get_metadata()

        frame_w = metadata.width or 1024
        frame_h = metadata.height or 576
        fps = metadata.fps or 25.0
        total_frames = metadata.frame_count or 0

        # Ensure output directory exists
        out_path = Path(self._config.pipeline.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, frame_h))

        # Timing accumulators
        t_read_list: list[float] = []
        t_detect_list: list[float] = []
        t_track_list: list[float] = []
        t_count_list: list[float] = []
        t_render_list: list[float] = []
        t_write_list: list[float] = []

        processed_count = 0
        pipeline_start_t = time.monotonic()
        recent_fps = fps

        print("=" * 80)
        print("CampusNetra product0.3 — Vision-Based Classroom Occupancy & Counting Pipeline")
        print("=" * 80)
        print(f"Input Video       : {self._config.pipeline.input_path}")
        print(f"Output Video      : {self._config.pipeline.output_path}")
        print(f"Resolution        : {frame_w}x{frame_h} @ {fps:.2f} FPS ({total_frames} total frames)")
        print(f"Detector Model    : {self._config.detection.model_path} (device={self._config.detection.device})")
        print(f"Tracker           : ByteTrack (max_age={self._config.tracking.max_age}, min_hits={self._config.tracking.min_hits})")
        print(f"Counting Line     : {self._config.counting.line_start} -> {self._config.counting.line_end} [inside={self._config.counting.inside_direction}]")
        print("-" * 80)

        max_frames = self._config.pipeline.max_frames

        try:
            while True:
                if max_frames is not None and processed_count >= max_frames:
                    print(f"\n[INFO] Reached max_frames limit ({max_frames}). Stopping.")
                    break

                # 1. Read
                t0 = time.monotonic()
                frame = self._video_source.read()
                if frame is None:
                    break
                t_read = (time.monotonic() - t0) * 1000.0

                frame_start_t = time.monotonic()

                # 2. Detect
                t0 = time.monotonic()
                det_result = self._detector.detect(frame)
                t_detect = (time.monotonic() - t0) * 1000.0

                # 3. Track
                t0 = time.monotonic()
                track_result = self._tracker.update(det_result)
                t_track = (time.monotonic() - t0) * 1000.0

                # 4. Count (Line Crossing & Zone Occupancy Events)
                t0 = time.monotonic()
                events = self._counter.update(track_result)
                approx = self._counter.compute_occlusion_approximation(track_result)
                left_area_cnt, right_area_cnt = self._counter.detect_area_occupants(track_result)
                t_count = (time.monotonic() - t0) * 1000.0

                # 5. Persistent Occupancy Update (Pure Counting Line Room Count + Decoupled Area States)
                occupancy_state = self._occupancy.apply(
                    events,
                    approximation=approx,
                    left_area_count=left_area_cnt,
                    right_area_count=right_area_cnt,
                )

                # 6. Visualize
                t0 = time.monotonic()
                annotated = self._visualizer.annotate(
                    frame=frame,
                    tracking_result=track_result,
                    occupancy_state=occupancy_state,
                    events=events,
                    counting_config=self._config.counting,
                    fps=recent_fps,
                )
                t_render = (time.monotonic() - t0) * 1000.0

                # 7. Write Frame
                t0 = time.monotonic()
                writer.write(annotated)
                t_write = (time.monotonic() - t0) * 1000.0

                # Accumulate timings
                t_read_list.append(t_read)
                t_detect_list.append(t_detect)
                t_track_list.append(t_track)
                t_count_list.append(t_count)
                t_render_list.append(t_render)
                t_write_list.append(t_write)

                processed_count += 1
                frame_duration = time.monotonic() - frame_start_t
                if frame_duration > 0:
                    recent_fps = 1.0 / frame_duration

                # Live preview window if enabled
                if self._config.pipeline.show_live_window:
                    cv2.imshow("CampusNetra Preview", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        print("\n[INFO] User interrupted live preview.")
                        break

                # Console progress
                interval = self._config.metrics.console_every_n_frames
                if processed_count % interval == 0 or processed_count == total_frames:
                    total_str = f"/{total_frames}" if total_frames > 0 else ""
                    approx_str = f" (±{occupancy_state.occupancy_approximation})" if occupancy_state.occupancy_approximation > 0 else ""
                    print(
                        f"Frame: {processed_count:5d}{total_str} | "
                        f"FPS: {recent_fps:4.1f} | "
                        f"Active: {track_result.active_count:2d} | "
                        f"Left Area: {occupancy_state.left_area_state.value} | "
                        f"Right Area: {occupancy_state.right_area_state.value} | "
                        f"In: {occupancy_state.total_entries:2d} | "
                        f"Out: {occupancy_state.total_exits:2d} | "
                        f"Occupancy: {occupancy_state.occupancy_count:2d}{approx_str} | "
                        f"State: {occupancy_state.classroom_state.value}"
                    )


        finally:
            self._video_source.close()
            writer.release()
            if self._config.pipeline.show_live_window:
                cv2.destroyAllWindows()

        total_wall_time = time.monotonic() - pipeline_start_t
        avg_fps = processed_count / total_wall_time if total_wall_time > 0 else 0.0

        def _mean(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        final_occ = self._occupancy.get_state()

        metrics = PipelineMetrics(
            total_frames=total_frames,
            processed_frames=processed_count,
            total_wall_time_s=total_wall_time,
            average_fps=avg_fps,
            avg_read_ms=_mean(t_read_list),
            avg_detect_ms=_mean(t_detect_list),
            avg_track_ms=_mean(t_track_list),
            avg_count_ms=_mean(t_count_list),
            avg_render_ms=_mean(t_render_list),
            avg_write_ms=_mean(t_write_list),
            total_entries=final_occ.total_entries,
            total_exits=final_occ.total_exits,
            final_occupancy=final_occ.occupancy_count,
            final_state=final_occ.classroom_state,
        )

        print("-" * 80)
        print("Pipeline Completed Successfully!")
        print(f"Total Frames Processed : {metrics.processed_frames}")
        print(f"Total Wall Time        : {metrics.total_wall_time_s:.2f} s")
        print(f"Average FPS            : {metrics.average_fps:.2f}")
        print(f"Stage Breakdown (avg)  : Read={metrics.avg_read_ms:.1f}ms | Detect={metrics.avg_detect_ms:.1f}ms | Track={metrics.avg_track_ms:.1f}ms | Render={metrics.avg_render_ms:.1f}ms | Write={metrics.avg_write_ms:.1f}ms")
        print(f"Total Entered          : {metrics.total_entries}")
        print(f"Total Exited           : {metrics.total_exits}")
        print(f"Final Occupancy        : {metrics.final_occupancy}")
        print(f"Final Classroom State  : {metrics.final_state.value}")
        print(f"Saved Processed Video  : {self._config.pipeline.output_path}")
        print("=" * 80)

        return metrics
