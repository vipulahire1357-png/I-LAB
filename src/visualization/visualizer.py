"""
Visualizer (Module 6).

Renders person bounding boxes, persistent track badges, the virtual counting boundary,
event flash alerts, and a real-time HUD stats card onto frame images.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.counting.models import CountingConfig, CrossingEvent, EventType
from src.occupancy.models import ClassroomState, OccupancyState
from src.tracking.models import Track, TrackingResult, TrackState
from src.video.models import Frame
from src.visualization.models import VisualizationConfig


class Visualizer:
    """Renders real-time computer vision annotations onto frame images.

    Parameters
    ----------
    config : VisualizationConfig | None
        Overlay appearance settings.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None) -> None:
        self._config = config or VisualizationConfig()
        self._flash_counter: int = 0
        self._last_event_text: str = ""
        self._last_event_color: Tuple[int, int, int] = (0, 255, 0)

    @property
    def config(self) -> VisualizationConfig:
        return self._config

    def annotate(
        self,
        frame: Frame,
        tracking_result: TrackingResult,
        occupancy_state: OccupancyState,
        events: List[CrossingEvent],
        counting_config: Optional[CountingConfig] = None,
        fps: float = 0.0,
    ) -> np.ndarray:
        """Draw bounding boxes, track IDs, counting line, and HUD overlay on a frame.

        Parameters
        ----------
        frame : Frame
            Raw decoded frame.
        tracking_result : TrackingResult
            Tracking snapshot containing active/lost tracks.
        occupancy_state : OccupancyState
            Current occupancy counts and classroom state.
        events : list[CrossingEvent]
            Any crossing events confirmed in this frame.
        counting_config : CountingConfig | None
            Virtual line geometry for rendering.
        fps : float
            Current processing frame rate for HUD display.

        Returns
        -------
        np.ndarray
            BGR image with annotations applied.
        """
        # Create a copy so original frame image remains pristine
        canvas = frame.image.copy()

        # 1. Update event flash state if new events occurred
        if events:
            # Take the most recent event
            latest = events[-1]
            if latest.event_type == EventType.ENTER:
                side_info = f" ({latest.side})" if latest.side else ""
                self._last_event_text = f"ENTER{side_info}: Track {latest.track_id}"
                self._last_event_color = (0, 255, 0)  # Green
            elif latest.event_type == EventType.EXIT:
                side_info = f" ({latest.side})" if latest.side else ""
                self._last_event_text = f"EXIT{side_info}: Track {latest.track_id}"
                self._last_event_color = (0, 0, 255)  # Red
            elif latest.event_type == EventType.ZONE_ENTER_LEFT:
                self._last_event_text = f"ENTER LEFT: Track {latest.track_id}"
                self._last_event_color = (255, 200, 0)  # Cyan
            elif latest.event_type == EventType.ZONE_EXIT_LEFT:
                self._last_event_text = f"EXIT LEFT: Track {latest.track_id}"
                self._last_event_color = (180, 180, 180)  # Gray
            elif latest.event_type == EventType.ZONE_ENTER_RIGHT:
                self._last_event_text = f"ENTER RIGHT: Track {latest.track_id}"
                self._last_event_color = (255, 200, 0)  # Cyan
            elif latest.event_type == EventType.ZONE_EXIT_RIGHT:
                self._last_event_text = f"EXIT RIGHT: Track {latest.track_id}"
                self._last_event_color = (180, 180, 180)  # Gray
            elif latest.event_type == EventType.ZONE_CROSS_LEFT_TO_RIGHT:
                self._last_event_text = f"CROSS L->R: Track {latest.track_id}"
                self._last_event_color = (255, 200, 0)
            elif latest.event_type == EventType.ZONE_CROSS_RIGHT_TO_LEFT:
                self._last_event_text = f"CROSS R->L: Track {latest.track_id}"
                self._last_event_color = (255, 200, 0)
            self._flash_counter = self._config.event_flash_frames
        elif self._flash_counter > 0:
            self._flash_counter -= 1


        # 2. Draw Virtual Counting Line and Dual Parallel Walkway Dividers
        if counting_config is not None:
            self._draw_counting_line(canvas, counting_config, occupancy_state)

        # 3. Draw Track Bounding Boxes and ID Badges
        self._draw_tracks(canvas, tracking_result.tracks)

        # 4. Draw HUD Stats Overlay
        self._draw_hud(canvas, occupancy_state, fps)

        return canvas

    def _draw_counting_line(
        self,
        canvas: np.ndarray,
        counting_config: CountingConfig,
        occupancy_state: Optional[OccupancyState] = None,
    ) -> None:
        p1 = (int(counting_config.line_start[0]), int(counting_config.line_start[1]))
        p2 = (int(counting_config.line_end[0]), int(counting_config.line_end[1]))
        color = self._config.line_color
        thickness = self._config.line_thickness

        # 1. Primary Counting Line
        cv2.line(canvas, p1, p2, color, thickness)
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        cv2.putText(
            canvas,
            "COUNTING LINE",
            (max(10, mid_x - 60), max(20, mid_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        # 2. Dual Parallel Walkway Divider Lines
        perp_color = self._config.perpendicular_line_color
        perp_thick = self._config.perpendicular_line_thickness

        # Left Divider Line
        lp1_f, lp2_f = counting_config.get_left_divider_line()
        lp1 = (int(lp1_f[0]), int(lp1_f[1]))
        lp2 = (int(lp2_f[0]), int(lp2_f[1]))
        cv2.line(canvas, lp1, lp2, perp_color, perp_thick)
        cv2.putText(
            canvas,
            "LEFT DIVIDER",
            (max(10, lp1[0] + 10), max(25, lp1[1] + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            perp_color,
            1,
            cv2.LINE_AA,
        )

        # Right Divider Line
        rp1_f, rp2_f = counting_config.get_right_divider_line()
        rp1 = (int(rp1_f[0]), int(rp1_f[1]))
        rp2 = (int(rp2_f[0]), int(rp2_f[1]))
        cv2.line(canvas, rp1, rp2, perp_color, perp_thick)
        cv2.putText(
            canvas,
            "RIGHT DIVIDER",
            (max(10, rp1[0] + 10), max(25, rp1[1] + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            perp_color,
            1,
            cv2.LINE_AA,
        )

        # Walkway Center Label
        walk_x = (lp1[0] + rp1[0]) // 2
        walk_y = (lp1[1] + rp1[1]) // 2 + 50
        cv2.putText(
            canvas,
            "WALKWAY",
            (max(10, walk_x - 35), max(30, walk_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # 3. Floating Side Occupancy Badges (if enabled and state available)
        if self._config.show_side_occupancy and occupancy_state is not None:
            left_badge_x = int(p1[0] * 0.75 + p2[0] * 0.25)
            left_badge_y = int(p1[1] * 0.75 + p2[1] * 0.25) + 30

            right_badge_x = int(p1[0] * 0.25 + p2[0] * 0.75)
            right_badge_y = int(p1[1] * 0.25 + p2[1] * 0.75) + 30

            left_text = f"LEFT AREA: {occupancy_state.left_area_state.value}"
            right_text = f"RIGHT AREA: {occupancy_state.right_area_state.value}"

            self._draw_side_badge(
                canvas,
                left_text,
                (left_badge_x, left_badge_y),
                is_occupied=occupancy_state.is_left_area_occupied,
            )
            self._draw_side_badge(
                canvas,
                right_text,
                (right_badge_x, right_badge_y),
                is_occupied=occupancy_state.is_right_area_occupied,
            )

    def _draw_side_badge(
        self,
        canvas: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        is_occupied: bool,
    ) -> None:
        """Render a styled zone occupancy badge onto the canvas."""
        x, y = pos
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thick = 1
        (w, h), baseline = cv2.getTextSize(text, font, scale, thick)

        # Keep within canvas bounds
        img_h, img_w = canvas.shape[:2]
        x = max(10, min(x, img_w - w - 16))
        y = max(h + 10, min(y, img_h - 10))

        bg_color = (0, 140, 0) if is_occupied else (50, 50, 50)
        border_color = (0, 255, 0) if is_occupied else (120, 120, 120)

        # Background card
        cv2.rectangle(canvas, (x - 4, y - h - 6), (x + w + 8, y + 4), bg_color, -1)
        cv2.rectangle(canvas, (x - 4, y - h - 6), (x + w + 8, y + 4), border_color, 1)
        cv2.putText(canvas, text, (x + 2, y - 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    def _draw_tracks(self, canvas: np.ndarray, tracks: Tuple[Track, ...]) -> None:
        for track in tracks:
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            is_confirmed = track.state == TrackState.CONFIRMED
            color = self._config.bbox_color_confirmed if is_confirmed else self._config.bbox_color_lost

            # Bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            center_x = (x1 + x2) // 2

            # 1. Lower dot: Foot-point / Ground contact (Orange / Amber)
            foot_y = y2
            cv2.circle(canvas, (center_x, foot_y), 4, (0, 165, 255), -1)
            cv2.circle(canvas, (center_x, foot_y), 5, (0, 0, 0), 1)

            # 2. Upper dot: Head-point / Validation anchor (Cyan / Sky Blue)
            head_y = y1
            cv2.circle(canvas, (center_x, head_y), 4, (255, 200, 0), -1)
            cv2.circle(canvas, (center_x, head_y), 5, (0, 0, 0), 1)

            # 3. Vertical body axis line connecting upper and lower dots
            cv2.line(canvas, (center_x, head_y + 4), (center_x, foot_y - 4), (180, 180, 180), 1)

            # Track ID badge
            label = f"ID:{track.track_id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, y1 - h - 4), (x1 + w + 4, y1), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 2, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )


    def _draw_hud(self, canvas: np.ndarray, occupancy_state: OccupancyState, fps: float) -> None:
        ox, oy = self._config.overlay_position
        box_w, box_h = 270, 200

        # Draw semi-transparent background card
        overlay = canvas.copy()
        cv2.rectangle(overlay, (ox, oy), (ox + box_w, oy + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.rectangle(canvas, (ox, oy), (ox + box_w, oy + box_h), (80, 80, 80), 1)

        # State text colors
        state_color = (0, 255, 0) if occupancy_state.is_occupied else (180, 180, 180)
        left_color = (0, 255, 0) if occupancy_state.is_left_area_occupied else (180, 180, 180)
        right_color = (0, 255, 0) if occupancy_state.is_right_area_occupied else (180, 180, 180)
        approx_color = (0, 200, 255) if occupancy_state.occupancy_approximation > 0 else (160, 160, 160)

        lines = [
            (f"OCCUPANCY:  {occupancy_state.occupancy_count}", (255, 255, 255), 0.60, 2),
            (f"APPROX:     +-{occupancy_state.occupancy_approximation}", approx_color, 0.46, 1),
            (f"LEFT AREA:  {occupancy_state.left_area_state.value}", left_color, 0.46, 1),
            (f"RIGHT AREA: {occupancy_state.right_area_state.value}", right_color, 0.46, 1),
            (f"ENTRIES:    {occupancy_state.total_entries}", (0, 255, 200), 0.44, 1),
            (f"EXITS:      {occupancy_state.total_exits}", (0, 150, 255), 0.44, 1),
            (f"STATE:      {occupancy_state.classroom_state.value}", state_color, 0.48, 2),
            (f"FPS:        {fps:.1f}", (200, 200, 200), 0.40, 1),
        ]

        text_y = oy + 21
        for text, color, scale, thick in lines:
            cv2.putText(
                canvas,
                text,
                (ox + 10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thick,
                cv2.LINE_AA,
            )
            text_y += 22

        # Draw event flash banner if active
        if self._flash_counter > 0 and self._last_event_text:
            banner_w = box_w
            banner_h = 30
            bx = ox
            by = oy + box_h + 6
            cv2.rectangle(canvas, (bx, by), (bx + banner_w, by + banner_h), self._last_event_color, -1)
            cv2.putText(
                canvas,
                self._last_event_text,
                (bx + 10, by + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # Draw event flash banner if active
        if self._flash_counter > 0 and self._last_event_text:
            banner_w = box_w
            banner_h = 30
            bx = ox
            by = oy + box_h + 6
            cv2.rectangle(canvas, (bx, by), (bx + banner_w, by + banner_h), self._last_event_color, -1)
            cv2.putText(
                canvas,
                self._last_event_text,
                (bx + 10, by + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

