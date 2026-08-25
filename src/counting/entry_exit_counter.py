"""
Entry/Exit Counter (Module 4).

Evaluates tracked person positions against a configurable virtual counting line,
detects complete directional crossing transitions, and updates occupancy.
Allows any person to cross in (+1) and out (-1) multiple times with hysteresis
boundary filtering to guarantee reliable counting and zero jitter.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from src.counting.models import CountingConfig, CrossingEvent, EventType, TrackSideState
from src.tracking.models import Track, TrackingResult, TrackState

logger = logging.getLogger(__name__)


class EntryExitCounter:
    """Detects student entries and exits across a configurable 2D virtual line.

    Parameters
    ----------
    config : CountingConfig | None
        Line geometry, orientation, and hysteresis configuration.
    """

    def __init__(self, config: Optional[CountingConfig] = None) -> None:
        self._config = config or CountingConfig()
        # Maps track_id -> TrackSideState (OUTSIDE or INSIDE)
        self._track_states: Dict[int, TrackSideState] = {}
        # Precomputed line geometry for fast perpendicular distance calculation
        self._x1, self._y1 = self._config.line_start
        self._x2, self._y2 = self._config.line_end
        self._dx = self._x2 - self._x1
        self._dy = self._y2 - self._y1
        self._line_length = math.hypot(self._dx, self._dy) or 1.0
        self._hysteresis = float(self._config.hysteresis_distance)

    @property
    def config(self) -> CountingConfig:
        """Active counting configuration."""
        return self._config

    def update(self, tracking_result: TrackingResult) -> List[CrossingEvent]:
        """Evaluate tracks for the current frame and return any confirmed crossing events.

        Parameters
        ----------
        tracking_result : TrackingResult
            Tracking snapshot containing all active tracks for the frame.

        Returns
        -------
        list[CrossingEvent]
            List of confirmed ENTER or EXIT events in this frame.
        """
        events: List[CrossingEvent] = []
        active_track_ids: set[int] = set()

        for track in tracking_result.tracks:
            # Evaluate all confirmed tracks
            if track.state is not TrackState.CONFIRMED:
                continue

            tid = track.track_id
            active_track_ids.add(tid)

            # Foot-point: bottom center of bounding box (best for floor ground-plane contact)
            foot_x = (track.x1 + track.x2) / 2.0
            foot_y = track.y2

            # Compute signed perpendicular distance from the line (in pixels)
            dist = self._signed_distance(foot_x, foot_y)
            prev_state = self._track_states.get(tid, TrackSideState.UNKNOWN)

            if prev_state is TrackSideState.UNKNOWN:
                # Initialize state based on which side the person first appears
                if dist > self._hysteresis:
                    self._track_states[tid] = TrackSideState.INSIDE
                elif dist < -self._hysteresis:
                    self._track_states[tid] = TrackSideState.OUTSIDE
                else:
                    # Near the boundary: assign side based on raw sign
                    self._track_states[tid] = TrackSideState.INSIDE if dist >= 0 else TrackSideState.OUTSIDE

            elif prev_state is TrackSideState.OUTSIDE and dist > self._hysteresis:
                # Person has completely crossed the counting line from OUTSIDE -> INSIDE: ENTER (+1)
                event = CrossingEvent(
                    event_type=EventType.ENTER,
                    track_id=tid,
                    frame_id=tracking_result.frame_id,
                    timestamp=tracking_result.timestamp,
                )
                events.append(event)
                self._track_states[tid] = TrackSideState.INSIDE
                logger.info(
                    "ENTER (+1) detected: track_id=%d, frame=%d, timestamp=%.2fs",
                    tid,
                    tracking_result.frame_id,
                    tracking_result.timestamp,
                )

            elif prev_state is TrackSideState.INSIDE and dist < -self._hysteresis:
                # Person has completely crossed the counting line from INSIDE -> OUTSIDE: EXIT (-1)
                event = CrossingEvent(
                    event_type=EventType.EXIT,
                    track_id=tid,
                    frame_id=tracking_result.frame_id,
                    timestamp=tracking_result.timestamp,
                )
                events.append(event)
                self._track_states[tid] = TrackSideState.OUTSIDE
                logger.info(
                    "EXIT (-1) detected: track_id=%d, frame=%d, timestamp=%.2fs",
                    tid,
                    tracking_result.frame_id,
                    tracking_result.timestamp,
                )

        return events

    def reset(self) -> None:
        """Reset internal tracking state machines for a new session."""
        self._track_states.clear()
        logger.info("EntryExitCounter reset.")

    def _signed_distance(self, px: float, py: float) -> float:
        """Compute signed perpendicular distance (in pixels) from line to point.

        Positive means INSIDE, Negative means OUTSIDE (adjusted by inside_direction).
        """
        # Cross product = (px - x1)*dy - (py - y1)*dx
        cross = (px - self._x1) * self._dy - (py - self._y1) * self._dx
        # Normalize to pixel distance
        perp_dist = cross / self._line_length

        if self._config.inside_direction == "positive":
            return perp_dist
        else:
            return -perp_dist

    def __repr__(self) -> str:
        return (
            f"EntryExitCounter(line={self._config.line_start}->{self._config.line_end}, "
            f"direction={self._config.inside_direction}, "
            f"hysteresis={self._hysteresis}px, "
            f"active_records={len(self._track_states)})"
        )
