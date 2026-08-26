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

from src.counting.models import CountingConfig, CrossingEvent, DividerSideState, EventType, TrackSideState
from src.tracking.models import Track, TrackingResult, TrackState

logger = logging.getLogger(__name__)


class EntryExitCounter:
    """Detects student entries and exits across a configurable 2D virtual counting line,
    and accurately tracks directional transitions across dual parallel walkway divider lines.

    Key Features:
    - Independent entry/exit counting on the primary boundary.
    - Walkway/Aisle model with dual parallel divider lines (Left Divider & Right Divider).
    - Perpendicular divider counters are strictly activated ONLY after student crosses the counting line.
    - Accurately detects students entering and exiting the Left and Right seating areas.
    - Occlusion resilience via per-track trajectory history and cross-frame interpolation.
    - Complete crossing requirement with hysteresis deadband.

    Parameters
    ----------
    config : CountingConfig | None
        Line geometry, walkway configuration, and hysteresis parameters.
    """

    def __init__(self, config: Optional[CountingConfig] = None) -> None:
        self._config = config or CountingConfig()
        # Maps track_id -> TrackSideState (OUTSIDE or INSIDE) for counting line
        self._track_states: Dict[int, TrackSideState] = {}
        # Maps track_id -> DividerSideState (LEFT, WALKWAY, RIGHT) for perpendicular walkway dividers
        self._divider_states: Dict[int, DividerSideState] = {}
        # Trajectory history per track: list of (foot_x, foot_y, head_x, head_y, frame_id, line_dist, left_div_dist, right_div_dist)
        self._track_history: Dict[int, List[Tuple[float, float, float, float, int, float, float, float]]] = {}
        # Candidate confirmation counter: track_id -> (target_state, consecutive_frames)
        self._divider_candidates: Dict[int, Tuple[DividerSideState, int]] = {}

        # Precomputed counting line geometry
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

        Generates:
        - ENTER (+1 total occupancy, assigned to initial entry zone)
        - EXIT (-1 total occupancy, deducted from last occupant zone)
        - ZONE_ENTER_LEFT / ZONE_EXIT_LEFT (entering/exiting Left Seating)
        - ZONE_ENTER_RIGHT / ZONE_EXIT_RIGHT (entering/exiting Right Seating)
        - ZONE_CROSS_* for backward compatibility

        Parameters
        ----------
        tracking_result : TrackingResult
            Tracking snapshot containing all active and lost tracks for the frame.

        Returns
        -------
        list[CrossingEvent]
            List of confirmed crossing events in this frame.
        """
        events: List[CrossingEvent] = []
        active_track_ids: set[int] = set()

        for track in tracking_result.tracks:
            if track.state is not TrackState.CONFIRMED:
                continue

            tid = track.track_id
            active_track_ids.add(tid)

            # Dual tracking reference dots:
            # 1. Lower dot (foot-point): ground contact at bottom center of bounding box
            foot_x = (track.x1 + track.x2) / 2.0
            foot_y = track.y2

            # 2. Upper dot (head-point): upper body anchor at top center of bounding box
            head_x = (track.x1 + track.x2) / 2.0
            head_y = track.y1

            # Compute signed perpendicular distances from counting line (pixels)
            # Positive = INSIDE, Negative = OUTSIDE
            dist_foot = self._signed_distance(foot_x, foot_y)
            dist_head = self._signed_distance(head_x, head_y)

            # Compute signed distances to Left and Right Walkway Divider lines
            div_left_dist = self._signed_distance_left_divider(foot_x, foot_y)
            div_right_dist = self._signed_distance_right_divider(foot_x, foot_y)

            # Record trajectory history
            if tid not in self._track_history:
                self._track_history[tid] = []
            self._track_history[tid].append(
                (foot_x, foot_y, head_x, head_y, tracking_result.frame_id, dist_foot, div_left_dist, div_right_dist)
            )
            if len(self._track_history[tid]) > 20:
                self._track_history[tid].pop(0)

            current_zone_str = self.classify_track_zone(track)
            current_zone = (
                DividerSideState.LEFT
                if current_zone_str == "LEFT"
                else (DividerSideState.RIGHT if current_zone_str == "RIGHT" else DividerSideState.WALKWAY)
            )

            prev_state = self._track_states.get(tid, TrackSideState.UNKNOWN)

            # ------------------------------------------------------------------
            # 1. First time seeing track: initialize state
            # ------------------------------------------------------------------
            if prev_state is TrackSideState.UNKNOWN:
                if dist_foot > self._hysteresis:
                    self._track_states[tid] = TrackSideState.INSIDE
                    self._divider_states[tid] = current_zone
                elif dist_foot < -self._hysteresis:
                    self._track_states[tid] = TrackSideState.OUTSIDE
                    self._divider_states[tid] = DividerSideState.OUTSIDE
                else:
                    is_in = dist_foot >= 0
                    self._track_states[tid] = TrackSideState.INSIDE if is_in else TrackSideState.OUTSIDE
                    self._divider_states[tid] = current_zone if is_in else DividerSideState.OUTSIDE

            # ------------------------------------------------------------------
            # 2. Track previously OUTSIDE: Check for ENTER crossing
            # ------------------------------------------------------------------
            elif prev_state is TrackSideState.OUTSIDE:
                # Primary condition: Foot-point crosses into positive inside zone past hysteresis
                is_foot_crossed = dist_foot > self._hysteresis
                # Validation anchor: Head-point validates inward orientation
                is_head_validated = dist_head > -(track.height * 0.8)

                # Occlusion handling: check if trajectory segment crossed counting line
                trajectory_crossed = False
                if not is_foot_crossed and len(self._track_history[tid]) >= 2:
                    prev_hist = self._track_history[tid][-2]
                    prev_dist = prev_hist[5]
                    if prev_dist < 0 and dist_foot > 0:
                        trajectory_crossed = True

                if (is_foot_crossed and is_head_validated) or (trajectory_crossed and dist_foot >= self._hysteresis * 0.5):
                    # Classify entry zone using walkway divider geometry
                    entry_zone_str = self._determine_crossing_zone(foot_x, foot_y, track)
                    entry_zone = (
                        DividerSideState.LEFT
                        if entry_zone_str == "LEFT"
                        else (DividerSideState.RIGHT if entry_zone_str == "RIGHT" else DividerSideState.WALKWAY)
                    )

                    event = CrossingEvent(
                        event_type=EventType.ENTER,
                        track_id=tid,
                        frame_id=tracking_result.frame_id,
                        timestamp=tracking_result.timestamp,
                        side=entry_zone_str,
                    )
                    events.append(event)
                    self._track_states[tid] = TrackSideState.INSIDE
                    self._divider_states[tid] = entry_zone
                    self._divider_candidates.pop(tid, None)

                    logger.info(
                        "ENTER (+1) verified in %s zone (foot_dist=%.1f, head_dist=%.1f): track_id=%d, frame=%d",
                        entry_zone_str,
                        dist_foot,
                        dist_head,
                        tid,
                        tracking_result.frame_id,
                    )

            # ------------------------------------------------------------------
            # 3. Track previously INSIDE: Check for EXIT crossing or Divider Zone Crossing
            # ------------------------------------------------------------------
            elif prev_state is TrackSideState.INSIDE:
                # Check if person EXITS through the counting line to outside
                is_foot_exited = dist_foot < -self._hysteresis
                is_head_validated = dist_head < (track.height * 0.8)

                if is_foot_exited and is_head_validated:
                    last_zone = self._divider_states.get(tid, current_zone)
                    event = CrossingEvent(
                        event_type=EventType.EXIT,
                        track_id=tid,
                        frame_id=tracking_result.frame_id,
                        timestamp=tracking_result.timestamp,
                        side=last_zone.value,
                    )
                    events.append(event)

                    self._track_states[tid] = TrackSideState.OUTSIDE
                    self._divider_states[tid] = DividerSideState.OUTSIDE
                    self._divider_candidates.pop(tid, None)
                    logger.info(
                        "EXIT (-1) verified from %s zone (foot_dist=%.1f, head_dist=%.1f): track_id=%d, frame=%d",
                        last_zone.value,
                        dist_foot,
                        dist_head,
                        tid,
                        tracking_result.frame_id,
                    )
                else:
                    # ----------------------------------------------------------
                    # Person remains INSIDE: Evaluate Walkway & Perpendicular Divider Transitions
                    # Perpendicular divider detection is strictly active ONLY for inside occupants.
                    # Requires COMPLETE crossing with hysteresis clearance.
                    # ----------------------------------------------------------
                    prev_zone = self._divider_states.get(tid, current_zone)

                    # Case A: Person currently in WALKWAY
                    if prev_zone is DividerSideState.WALKWAY:
                        # Entering LEFT seating: crosses Left Divider (left_dist < -hysteresis)
                        if div_left_dist < -self._hysteresis:
                            cand_state, cand_frames = self._divider_candidates.get(
                                tid, (DividerSideState.LEFT, 0)
                            )
                            cand_frames = (cand_frames + 1) if cand_state is DividerSideState.LEFT else 1
                            self._divider_candidates[tid] = (DividerSideState.LEFT, cand_frames)

                            if cand_frames >= 2 or div_left_dist <= -(self._hysteresis * 1.25):
                                events.append(
                                    CrossingEvent(
                                        event_type=EventType.ZONE_ENTER_LEFT,
                                        track_id=tid,
                                        frame_id=tracking_result.frame_id,
                                        timestamp=tracking_result.timestamp,
                                        side="LEFT",
                                    )
                                )
                                self._divider_states[tid] = DividerSideState.LEFT
                                self._divider_candidates.pop(tid, None)
                                logger.info(
                                    "ZONE ENTER LEFT: student entered Left Seating from Walkway (track_id=%d, frame=%d)",
                                    tid,
                                    tracking_result.frame_id,
                                )

                        # Entering RIGHT seating: crosses Right Divider (right_dist > +hysteresis)
                        elif div_right_dist > self._hysteresis:
                            cand_state, cand_frames = self._divider_candidates.get(
                                tid, (DividerSideState.RIGHT, 0)
                            )
                            cand_frames = (cand_frames + 1) if cand_state is DividerSideState.RIGHT else 1
                            self._divider_candidates[tid] = (DividerSideState.RIGHT, cand_frames)

                            if cand_frames >= 2 or div_right_dist >= (self._hysteresis * 1.25):
                                events.append(
                                    CrossingEvent(
                                        event_type=EventType.ZONE_ENTER_RIGHT,
                                        track_id=tid,
                                        frame_id=tracking_result.frame_id,
                                        timestamp=tracking_result.timestamp,
                                        side="RIGHT",
                                    )
                                )
                                self._divider_states[tid] = DividerSideState.RIGHT
                                self._divider_candidates.pop(tid, None)
                                logger.info(
                                    "ZONE ENTER RIGHT: student entered Right Seating from Walkway (track_id=%d, frame=%d)",
                                    tid,
                                    tracking_result.frame_id,
                                )
                        else:
                            self._divider_candidates.pop(tid, None)

                    # Case B: Person currently in LEFT Seating
                    elif prev_zone is DividerSideState.LEFT:
                        # Exiting LEFT seating back into Walkway (left_dist > +hysteresis)
                        if div_left_dist > self._hysteresis:
                            cand_state, cand_frames = self._divider_candidates.get(
                                tid, (DividerSideState.WALKWAY, 0)
                            )
                            cand_frames = (cand_frames + 1) if cand_state is DividerSideState.WALKWAY else 1
                            self._divider_candidates[tid] = (DividerSideState.WALKWAY, cand_frames)

                            if cand_frames >= 2 or div_left_dist >= (self._hysteresis * 1.25):
                                events.append(
                                    CrossingEvent(
                                        event_type=EventType.ZONE_EXIT_LEFT,
                                        track_id=tid,
                                        frame_id=tracking_result.frame_id,
                                        timestamp=tracking_result.timestamp,
                                        side="WALKWAY",
                                    )
                                )
                                self._divider_states[tid] = DividerSideState.WALKWAY
                                self._divider_candidates.pop(tid, None)
                                logger.info(
                                    "ZONE EXIT LEFT: student moved from Left Seating to Walkway (track_id=%d, frame=%d)",
                                    tid,
                                    tracking_result.frame_id,
                                )
                        else:
                            self._divider_candidates.pop(tid, None)

                    # Case C: Person currently in RIGHT Seating
                    elif prev_zone is DividerSideState.RIGHT:
                        # Exiting RIGHT seating back into Walkway (right_dist < -hysteresis)
                        if div_right_dist < -self._hysteresis:
                            cand_state, cand_frames = self._divider_candidates.get(
                                tid, (DividerSideState.WALKWAY, 0)
                            )
                            cand_frames = (cand_frames + 1) if cand_state is DividerSideState.WALKWAY else 1
                            self._divider_candidates[tid] = (DividerSideState.WALKWAY, cand_frames)

                            if cand_frames >= 2 or div_right_dist <= -(self._hysteresis * 1.25):
                                events.append(
                                    CrossingEvent(
                                        event_type=EventType.ZONE_EXIT_RIGHT,
                                        track_id=tid,
                                        frame_id=tracking_result.frame_id,
                                        timestamp=tracking_result.timestamp,
                                        side="WALKWAY",
                                    )
                                )
                                self._divider_states[tid] = DividerSideState.WALKWAY
                                self._divider_candidates.pop(tid, None)
                                logger.info(
                                    "ZONE EXIT RIGHT: student moved from Right Seating to Walkway (track_id=%d, frame=%d)",
                                    tid,
                                    tracking_result.frame_id,
                                )
                        else:
                            self._divider_candidates.pop(tid, None)

        return events

    def _determine_crossing_zone(self, px: float, py: float, track: Track) -> str:
        """Accurately determine entry zone ('LEFT', 'WALKWAY', or 'RIGHT') at the boundary."""
        zone = self.classify_track_zone(track)
        return zone

    def classify_point_side(self, px: float, py: float) -> str:
        """Classify a 2D point as 'LEFT' or 'RIGHT' relative to the center perpendicular divider."""
        return self._config.classify_point_divider_side(px, py)

    def classify_point_zone(self, px: float, py: float) -> str:
        """Classify a 2D point as 'LEFT', 'WALKWAY', or 'RIGHT' based on dual parallel divider lines."""
        return self._config.classify_point_zone(px, py)

    def classify_track_side(self, track: Track) -> str:
        """Classify a tracked person's location as 'LEFT' or 'RIGHT' of the central perpendicular divider."""
        foot_x = (track.x1 + track.x2) / 2.0
        foot_y = track.y2
        return self._config.classify_point_divider_side(foot_x, foot_y)

    def classify_track_zone(self, track: Track) -> str:
        """Classify a tracked person's location as 'LEFT', 'WALKWAY', or 'RIGHT'."""
        foot_x = (track.x1 + track.x2) / 2.0
        foot_y = track.y2
        head_x = (track.x1 + track.x2) / 2.0
        head_y = track.y1

        zone_foot = self._config.classify_point_zone(foot_x, foot_y)
        zone_head = self._config.classify_point_zone(head_x, head_y)

        if zone_foot == zone_head:
            return zone_foot

        mid_y = (track.y1 + track.y2) / 2.0
        return self._config.classify_point_zone(foot_x, mid_y)

    def detect_area_occupants(self, tracking_result: TrackingResult) -> Tuple[int, int]:
        """Detect visible occupants (both moving and static/seated) situated in the Left and Right seating areas.
        This spatial area detection operates independently of divider crossing lines, detecting people
        who are physically in the area even if divider lines were not crossed.

        Parameters
        ----------
        tracking_result : TrackingResult
            Tracking snapshot containing all tracks for the frame.

        Returns
        -------
        tuple[int, int]
            (left_area_count, right_area_count)
        """
        left_count = 0
        right_count = 0

        for track in tracking_result.tracks:
            if track.state is not TrackState.CONFIRMED:
                continue

            foot_x = (track.x1 + track.x2) / 2.0
            foot_y = track.y2
            is_inside = self._track_states.get(track.track_id) == TrackSideState.INSIDE or (
                track.track_id not in self._track_states and self._signed_distance(foot_x, foot_y) >= 0
            )
            if not is_inside:
                continue

            zone = self.classify_track_zone(track)
            if zone == "LEFT":
                left_count += 1
            elif zone == "RIGHT":
                right_count += 1

        return left_count, right_count

    def compute_occlusion_approximation(self, tracking_result: TrackingResult) -> int:
        """Estimate the occupancy uncertainty approximation (0 for ±0, 1 for ±1, 2 for ±2)
        when occlusion or track clustering occurs near the counting line boundary.

        Parameters
        ----------
        tracking_result : TrackingResult
            Current frame tracking snapshot.

        Returns
        -------
        int
            0 (no occlusion), 1 (moderate occlusion / 2 overlapping tracks), or 2 (heavy occlusion / 3+ overlapping tracks).
        """
        near_line_tracks: List[Track] = []
        corridor_width = self._hysteresis * 3.5

        for track in tracking_result.tracks:
            if track.state is not TrackState.CONFIRMED:
                continue
            foot_x = (track.x1 + track.x2) / 2.0
            foot_y = track.y2
            dist = abs(self._signed_distance(foot_x, foot_y))
            if dist <= corridor_width:
                near_line_tracks.append(track)

        if len(near_line_tracks) < 2:
            return 0

        # Check for overlapping bounding boxes or close proximity between near-line tracks
        overlapping_pairs = 0
        n = len(near_line_tracks)
        for i in range(n):
            for j in range(i + 1, n):
                t1 = near_line_tracks[i]
                t2 = near_line_tracks[j]
                # Compute IoU between t1 and t2
                ix1 = max(t1.x1, t2.x1)
                iy1 = max(t1.y1, t2.y1)
                ix2 = min(t1.x2, t2.x2)
                iy2 = min(t1.y2, t2.y2)
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                intersection = iw * ih
                if intersection > 0:
                    area1 = t1.width * t1.height
                    area2 = t2.width * t2.height
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0.0
                    if iou > 0.05:
                        overlapping_pairs += 1
                else:
                    # Check center distance proximity
                    c1x = (t1.x1 + t1.x2) / 2.0
                    c1y = (t1.y1 + t1.y2) / 2.0
                    c2x = (t2.x1 + t2.x2) / 2.0
                    c2y = (t2.y1 + t2.y2) / 2.0
                    if math.hypot(c1x - c2x, c1y - c2y) < (t1.width + t2.width) * 0.6:
                        overlapping_pairs += 1

        if overlapping_pairs >= 2 or len(near_line_tracks) >= 3:
            return 2
        elif overlapping_pairs == 1:
            return 1
        return 0

    def compute_side_occupancy(self, tracking_result: TrackingResult) -> Tuple[int, int]:
        """Compute real-time visible occupant counts in LEFT and RIGHT seating areas (alias for detect_area_occupants)."""
        return self.detect_area_occupants(tracking_result)

    def reset(self) -> None:
        """Reset internal tracking state machines for a new session."""
        self._track_states.clear()
        self._divider_states.clear()
        self._track_history.clear()
        self._divider_candidates.clear()
        logger.info("EntryExitCounter reset.")

    def _signed_distance(self, px: float, py: float) -> float:
        """Compute signed perpendicular distance (in pixels) from counting line to point.

        Positive means INSIDE, Negative means OUTSIDE (adjusted by inside_direction).
        """
        cross = (px - self._x1) * self._dy - (py - self._y1) * self._dx
        perp_dist = cross / self._line_length

        if self._config.inside_direction == "positive":
            return perp_dist
        else:
            return -perp_dist

    def _signed_distance_left_divider(self, px: float, py: float) -> float:
        """Signed perpendicular distance from Left Divider line.

        Negative means situated in LEFT seating, Positive means inside Walkway/Right.
        """
        (p1x, p1y), (p2x, p2y) = self._config.get_left_divider_line()
        pdx = p2x - p1x
        pdy = p2y - p1y
        div_len = math.hypot(pdx, pdy) or 1.0

        cross = (px - p1x) * pdy - (py - p1y) * pdx
        return cross / div_len

    def _signed_distance_right_divider(self, px: float, py: float) -> float:
        """Signed perpendicular distance from Right Divider line.

        Negative means situated in Walkway/Left, Positive means situated in RIGHT seating.
        """
        (p1x, p1y), (p2x, p2y) = self._config.get_right_divider_line()
        pdx = p2x - p1x
        pdy = p2y - p1y
        div_len = math.hypot(pdx, pdy) or 1.0

        cross = (px - p1x) * pdy - (py - p1y) * pdx
        return cross / div_len

    def __repr__(self) -> str:
        return (
            f"EntryExitCounter(line={self._config.line_start}->{self._config.line_end}, "
            f"walkway_width={self._config.walkway_width}px, "
            f"active_records={len(self._track_states)})"
        )



