"""
Concrete VideoSource backed by the RTSP Integration module.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.video.exceptions import VideoOpenError, VideoReadError, VideoSourceError
from src.video.models import Frame, VideoMetadata
from src.video.video_source import VideoSource

from src.rtsp.camera_manager import CameraManager
from src.rtsp.rtsp_client import RTSPClient
from src.rtsp.frame_extractor import FrameExtractor

logger = logging.getLogger(__name__)


class RTSPVideoSource(VideoSource):
    """Read frames sequentially from a live RTSP camera feed.

    Parameters
    ----------
    camera_id : str
        The camera identifier (e.g., 'A14-419') defined in cameras.yaml.
    config_path : str
        Path to the cameras configuration file (default: config/cameras.yaml).
    """

    def __init__(self, camera_id: str, config_path: str = "config/cameras.yaml") -> None:
        self.camera_id = camera_id
        self.config_path = config_path
        self._camera_manager: Optional[CameraManager] = None
        self._rtsp_client: Optional[RTSPClient] = None
        self._extractor: Optional[FrameExtractor] = None
        
        self._frame_counter: int = 0
        self._metadata: Optional[VideoMetadata] = None

    def open(self) -> None:
        """Open the RTSP stream and start background extraction."""
        if self.is_open():
            return
            
        try:
            self._camera_manager = CameraManager(config_path=self.config_path)
            camera_config = self._camera_manager.get_camera_config(self.camera_id)
        except Exception as e:
            raise VideoOpenError(f"Failed to load configuration for camera {self.camera_id}: {e}")

        extract_config = self._camera_manager.get_global_config("frame_extraction")
        output_config = self._camera_manager.get_global_config("frame_output")
        connection_config = self._camera_manager.get_global_config("connection")

        self._rtsp_client = RTSPClient(self.camera_id, camera_config, connection_config)
        self._extractor = FrameExtractor(
            self.camera_id, 
            self._rtsp_client, 
            extract_config, 
            output_config
        )
        
        # Determine metadata from config
        self._metadata = VideoMetadata(
            source_path=camera_config["rtsp_url"],
            fps=extract_config.get("fps", 5)
        )

        logger.info(f"Starting live RTSP extraction for camera {self.camera_id}")
        self._extractor.start()

    def read(self) -> Optional[Frame]:
        """Return the next frame from the RTSP queue.

        Blocks until a frame is available.
        """
        if not self.is_open():
            raise VideoSourceError("Cannot read: video source is not open.", operation="read")

        # Get frame from extractor (blocks indefinitely if no timeout, 
        # but let's give a small timeout to allow interruptions)
        frame_data = None
        try:
            # Using timeout to avoid blocking forever if thread dies
            frame_data = self._extractor.get_frame(timeout=5.0)
        except Exception as e:
            raise VideoReadError(f"Error retrieving frame from RTSP stream: {e}", frame_id=self._frame_counter)

        if frame_data is None:
            # If empty after timeout or stream disconnected
            return None

        # Convert FrameData to standard Pipeline Frame
        # RTSP frame timestamp is currently a datetime object, convert to float relative to start or just posix
        timestamp_sec = frame_data.timestamp.timestamp()
        
        frame = Frame(
            frame_id=self._frame_counter,
            timestamp=timestamp_sec,
            image=frame_data.frame,
            source_type="rtsp",
            camera_id=self.camera_id
        )
        
        self._frame_counter += 1
        return frame

    def get_metadata(self) -> VideoMetadata:
        if self._metadata is not None:
            return self._metadata
        return VideoMetadata(source_path=f"rtsp://camera/{self.camera_id}")

    def is_open(self) -> bool:
        return self._extractor is not None and self._extractor.is_running

    def close(self) -> None:
        if self._extractor is not None:
            self._extractor.stop()
            self._extractor = None
        if self._rtsp_client is not None:
            self._rtsp_client.release()
            self._rtsp_client = None
