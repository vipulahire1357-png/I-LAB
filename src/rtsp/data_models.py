from dataclasses import dataclass
from datetime import datetime
import numpy as np

@dataclass
class FrameData:
    """
    Data structure representing an extracted frame and its metadata.
    """
    camera_id: str
    timestamp: datetime
    frame: np.ndarray

    def __repr__(self):
        shape = self.frame.shape if self.frame is not None else None
        return f"FrameData(camera_id='{self.camera_id}', timestamp={self.timestamp}, frame_shape={shape})"
