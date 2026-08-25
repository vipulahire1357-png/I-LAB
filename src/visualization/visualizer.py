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
                self._last_event_text = f"ENTER: Track {latest.track_id}"
                self._last_event_color = (0, 255, 0)  # Green
            else:
                self._last_event_text = f"EXIT: Track {latest.track_id}"
                self._last_event_color = (0, 0, 255)  # Red
            self._flash_counter = self._config.event_flash_frames
        elif self._flash_counter > 0:
            self._flash_counter -= 1

        # 2. Draw Virtual Counting Line
        if counting_config is not None:
            self._draw_counting_line(canvas, counting_config)

        # 3. Draw Track Bounding Boxes and ID Badges
        self._draw_tracks(canvas, tracking_result.tracks)

        # 4. Draw HUD Stats Overlay
        self._draw_hud(canvas, occupancy_state, fps)

        return canvas

    def _draw_counting_line(self, canvas: np.ndarray, counting_config: CountingConfig) -> None:
        p1 = (int(counting_config.line_start[0]), int(counting_config.line_start[1]))
        p2 = (int(counting_config.line_end[0]), int(counting_config.line_end[1]))
        color = self._config.line_color
        thickness = self._config.line_thickness

        cv2.line(canvas, p1, p2, color, thickness)
        # Line label
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

    def _draw_tracks(self, canvas: np.ndarray, tracks: Tuple[Track, ...]) -> None:
        for track in tracks:
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            is_confirmed = track.state == TrackState.CONFIRMED
            color = self._config.bbox_color_confirmed if is_confirmed else self._config.bbox_color_lost

            # Bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Foot-point circle
            foot_x = (x1 + x2) // 2
            foot_y = y2
            cv2.circle(canvas, (foot_x, foot_y), 4, (0, 165, 255), -1)

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
        box_w, box_h = 240, 135

        # Draw semi-transparent background card
        overlay = canvas.copy()
        cv2.rectangle(overlay, (ox, oy), (ox + box_w, oy + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.rectangle(canvas, (ox, oy), (ox + box_w, oy + box_h), (80, 80, 80), 1)

        # State text color
        state_color = (0, 255, 0) if occupancy_state.is_occupied else (180, 180, 180)

        lines = [
            (f"OCCUPANCY: {occupancy_state.occupancy_count}", (255, 255, 255), 0.65, 2),
            (f"ENTRIES:   {occupancy_state.total_entries}", (0, 255, 200), 0.5, 1),
            (f"EXITS:     {occupancy_state.total_exits}", (0, 150, 255), 0.5, 1),
            (f"STATE:     {occupancy_state.classroom_state.value}", state_color, 0.55, 2),
            (f"FPS:       {fps:.1f}", (200, 200, 200), 0.45, 1),
        ]

        text_y = oy + 24
        for text, color, scale, thick in lines:
            cv2.putText(
                canvas,
                text,
                (ox + 12, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thick,
                cv2.LINE_AA,
            )
            text_y += 22

        # Draw event flash banner if active
        if self._flash_counter > 0 and self._last_event_text:
            banner_w = 240
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
