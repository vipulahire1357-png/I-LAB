"""
Person Tracker — ByteTrack-based multi-object person tracking.

Wraps the ``supervision`` library's ByteTrack behind a clean API so
downstream modules never import supervision or tracking internals.

Architecture::

    DetectionResult  →  convert to sv.Detections
                     →  ByteTrack association
                     →  update internal track registry
                     →  TrackingResult (frozen snapshots)

The tracker is **stateful**: it maintains track state across frames and
must be called sequentially.  It is single-threaded.

Key design principle:
    Module 3 determines WHERE people are (per-frame).
    Module 4 determines WHICH detection belongs to WHICH track (temporal).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import supervision as sv
except ImportError as _exc:
    raise ImportError(
        "The 'supervision' package is required for tracking. "
        "Install it with: pip install supervision"
    ) from _exc

from src.detection.models import DetectionResult
from src.tracking.exceptions import TrackerConfigError, TrackerError, TrackerUpdateError
from src.tracking.models import (
    Track,
    TrackerConfig,
    TrackingMetrics,
    TrackingResult,
    TrackState,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Internal mutable track record (NOT exposed publicly)
# ======================================================================

@dataclass
class _TrackRecord:
    """Mutable internal state for one tracked person."""

    track_id: int
    state: TrackState
    first_frame: int
    first_timestamp: float
    last_frame: int
    last_timestamp: float
    last_bbox: tuple[float, float, float, float]
    last_confidence: float
    hits: int = 1
    consecutive_misses: int = 0
    _age_frames: int = 1  # frames since creation

    def snapshot(self, current_frame: int) -> Track:
        """Create a frozen :class:`Track` snapshot."""
        return Track(
            track_id=self.track_id,
            bbox=self.last_bbox,
            confidence=self.last_confidence,
            state=self.state,
            age=self._age_frames,
            hits=self.hits,
            misses=self.consecutive_misses,
            first_frame=self.first_frame,
            last_frame=self.last_frame,
            first_timestamp=self.first_timestamp,
            last_timestamp=self.last_timestamp,
        )


# ======================================================================
# PersonTracker
# ======================================================================

class PersonTracker:
    """ByteTrack-based multi-object person tracker.

    Parameters
    ----------
    config : TrackerConfig | None
        Tracking configuration.  Defaults to ``TrackerConfig()``.

    Example
    -------
    ::

        tracker = PersonTracker()

        for frame in frames:
            det_result = detector.detect(frame)
            tracking_result = tracker.update(det_result)
            for t in tracking_result.tracks:
                print(t.track_id, t.state, t.bbox)
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self._config = config or TrackerConfig()
        self._byte_track: sv.ByteTrack | None = None
        self._registry: dict[int, _TrackRecord] = {}
        self._metrics = TrackingMetrics()
        self._frame_count = 0
        self._init_byte_track()

    def _init_byte_track(self) -> None:
        """Instantiate the underlying supervision ByteTrack."""
        try:
            self._byte_track = sv.ByteTrack(
                track_activation_threshold=self._config.track_high_threshold,
                lost_track_buffer=self._config.max_age,
                minimum_matching_threshold=self._config.match_threshold,
                frame_rate=self._config.frame_rate,
                minimum_consecutive_frames=self._config.min_hits,
            )
        except Exception as exc:
            raise TrackerError(
                f"Failed to initialise ByteTrack: {exc}", operation="init"
            ) from exc

        logger.info(
            "Tracker initialised: type=%s, max_age=%d, min_hits=%d, "
            "high_thresh=%.2f, match_thresh=%.2f",
            self._config.tracker_type,
            self._config.max_age,
            self._config.min_hits,
            self._config.track_high_threshold,
            self._config.match_threshold,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> TrackerConfig:
        return self._config

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update(self, detection_result: DetectionResult) -> TrackingResult:
        """Process one frame's detections and return tracked persons.

        Parameters
        ----------
        detection_result : DetectionResult
            Output from Module 3 for a single frame.

        Returns
        -------
        TrackingResult
            Tracked persons with stable IDs.
        """
        t0 = time.monotonic()
        self._frame_count += 1
        frame_id = detection_result.frame_id
        timestamp = detection_result.timestamp

        # --- convert to supervision Detections ---------------------------
        sv_dets = self._to_sv_detections(detection_result)

        # --- run ByteTrack association -----------------------------------
        try:
            tracked = self._byte_track.update_with_detections(sv_dets)
        except Exception as exc:
            raise TrackerUpdateError(
                f"ByteTrack update failed: {exc}", frame_id=frame_id
            ) from exc

        # --- update internal registry ------------------------------------
        matched_ids: set[int] = set()
        new_count = 0

        if tracked.tracker_id is not None and len(tracked.tracker_id) > 0:
            for i in range(len(tracked.tracker_id)):
                tid = int(tracked.tracker_id[i])
                bbox = tuple(float(v) for v in tracked.xyxy[i])
                conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0

                matched_ids.add(tid)

                if tid not in self._registry:
                    # New track
                    self._registry[tid] = _TrackRecord(
                        track_id=tid,
                        state=TrackState.CONFIRMED,
                        first_frame=frame_id,
                        first_timestamp=timestamp,
                        last_frame=frame_id,
                        last_timestamp=timestamp,
                        last_bbox=bbox,
                        last_confidence=conf,
                        hits=1,
                        consecutive_misses=0,
                        _age_frames=1,
                    )
                    new_count += 1
                else:
                    # Existing track — update
                    rec = self._registry[tid]
                    rec.state = TrackState.CONFIRMED
                    rec.last_frame = frame_id
                    rec.last_timestamp = timestamp
                    rec.last_bbox = bbox
                    rec.last_confidence = conf
                    rec.hits += 1
                    rec.consecutive_misses = 0
                    rec._age_frames += 1

        # --- handle tracks NOT matched this frame ------------------------
        removed_count = 0
        for tid, rec in list(self._registry.items()):
            if tid in matched_ids:
                continue
            rec.consecutive_misses += 1
            rec._age_frames += 1
            if rec.consecutive_misses > self._config.max_age:
                rec.state = TrackState.REMOVED
                removed_count += 1
            else:
                rec.state = TrackState.LOST

        # --- purge REMOVED tracks ----------------------------------------
        for tid in [t for t, r in self._registry.items() if r.state is TrackState.REMOVED]:
            del self._registry[tid]

        # --- build output ------------------------------------------------
        active_tracks: list[Track] = []
        lost_tracks: list[Track] = []
        for rec in self._registry.values():
            snap = rec.snapshot(frame_id)
            if rec.state is TrackState.CONFIRMED:
                active_tracks.append(snap)
            elif rec.state is TrackState.LOST:
                lost_tracks.append(snap)

        all_tracks = tuple(active_tracks + lost_tracks)
        tracking_ms = (time.monotonic() - t0) * 1000

        # --- metrics -----------------------------------------------------
        self._metrics._record_frame(
            tracking_time_ms=tracking_ms,
            active=len(active_tracks),
            created=new_count,
            removed=removed_count,
        )

        return TrackingResult(
            frame_id=frame_id,
            timestamp=timestamp,
            tracks=all_tracks,
            tracking_time_ms=tracking_ms,
            active_count=len(active_tracks),
            lost_count=len(lost_tracks),
        )

    def reset(self) -> None:
        """Clear all tracks and reset to initial state.

        After reset, Track IDs start from scratch (new session).
        """
        self._registry.clear()
        self._frame_count = 0
        self._metrics._reset()
        self._init_byte_track()  # fresh ByteTrack instance
        logger.info("Tracker reset — new session")

    def get_metrics(self) -> TrackingMetrics:
        """Return a reference to the live metrics object."""
        return self._metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_sv_detections(det_result: DetectionResult) -> sv.Detections:
        """Convert a Module 3 :class:`DetectionResult` to supervision format."""
        if det_result.count == 0:
            return sv.Detections.empty()

        xyxy = np.array(
            [d.bbox for d in det_result.detections], dtype=np.float32
        )
        confidence = np.array(
            [d.confidence for d in det_result.detections], dtype=np.float32
        )
        class_id = np.array(
            [d.class_id for d in det_result.detections], dtype=int
        )

        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PersonTracker("
            f"type={self._config.tracker_type!r}, "
            f"active={self._metrics.current_active_tracks}, "
            f"total_created={self._metrics.total_tracks_created})"
        )

