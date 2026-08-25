"""
Data models for the Visualizer module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    """Configuration for video overlay and visual annotations.

    Attributes
    ----------
    line_color : tuple[int, int, int]
        BGR color for virtual counting line (default yellow: 0, 255, 255).
    line_thickness : int
        Thickness of counting line in pixels.
    bbox_color_confirmed : tuple[int, int, int]
        BGR color for active/confirmed student bounding boxes (default green).
    bbox_color_lost : tuple[int, int, int]
        BGR color for lost student tracks (default grey).
    event_flash_frames : int
        Number of frames an ENTER/EXIT event notification is visibly flashed.
    font_scale : float
        OpenCV font scaling factor.
    font_thickness : int
        OpenCV text thickness in pixels.
    overlay_position : tuple[int, int]
        Top-left (x, y) coordinates of the HUD statistics overlay box.
    """

    line_color: Tuple[int, int, int] = (0, 255, 255)
    line_thickness: int = 2
    bbox_color_confirmed: Tuple[int, int, int] = (0, 255, 0)
    bbox_color_lost: Tuple[int, int, int] = (128, 128, 128)
    event_flash_frames: int = 10
    font_scale: float = 0.65
    font_thickness: int = 2
    overlay_position: Tuple[int, int] = (10, 10)
