import threading
import time
import queue
import logging
import os
import cv2
from datetime import datetime

try:
    from .rtsp_client import RTSPClient
    from .data_models import FrameData
except ModuleNotFoundError:  # pragma: no cover
    from rtsp_client import RTSPClient
    from data_models import FrameData

logger = logging.getLogger(__name__)

class FrameExtractor:
    """
    Continuously pulls frames from the RTSPClient in a background thread,
    enforces FPS limits, and provides them via a thread-safe bounded queue.
    """
    def __init__(self, camera_id: str, client: RTSPClient, extract_config: dict, output_config: dict, queue_size: int = 10):
        self.camera_id = camera_id
        self.client = client
        self.extract_config = extract_config
        self.output_config = output_config
        
        self.target_fps = self.extract_config.get("fps", 5)
        self.frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0
        
        # Bounded queue to avoid memory leaks
        self.frame_queue = queue.Queue(maxsize=queue_size)
        
        self.is_running = False
        self._thread = None
        
        # Disk saving setup
        self.save_to_disk = self.output_config.get("save_to_disk", False)
        self.output_dir = self.output_config.get("directory", "output/frames")
        if self.save_to_disk:
            os.makedirs(self.output_dir, exist_ok=True)

    def start(self):
        """
        Starts the background extraction thread.
        """
        if self.is_running:
            return
            
        logger.info(f"Starting frame extraction for {self.camera_id} at {self.target_fps} FPS.")
        self.is_running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name=f"Extractor-{self.camera_id}")
        self._thread.start()

    def stop(self):
        """
        Signals the extraction thread to stop and waits for it.
        """
        logger.info(f"Stopping frame extraction for {self.camera_id}.")
        self.is_running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.client.release()

    def get_frame(self, timeout=None) -> FrameData:
        """
        Interface for the CV pipeline to get the latest frame.
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _worker(self):
        """
        Background loop to read frames continuously, keeping buffer clear,
        and forwarding frames at the requested target FPS.
        """
        if not self.client.is_connected:
            if not self.client.connect():
                self.client.reconnect_with_backoff()

        last_yield_time = 0

        while self.is_running:
            if not self.client.is_connected:
                # Need to reconnect
                logger.info(f"Stream disconnected. Attempting to reconnect {self.camera_id}...")
                if not self.client.reconnect_with_backoff():
                    # If reconnect fails completely (e.g. max retries), we might just sleep and try later
                    # or exit depending on requirements. For now, reconnect_with_backoff loops forever.
                    continue

            # Read frame as fast as possible to keep RTSP buffer clear
            ret, frame = self.client.read()
            
            if not ret:
                # Read failed, connection might be broken. Next loop iteration will handle reconnect.
                continue

            current_time = time.time()
            # Enforce FPS
            if (current_time - last_yield_time) >= self.frame_interval:
                last_yield_time = current_time
                
                # Check if resizing is requested
                resize_config = self.extract_config.get("resize", {})
                if resize_config.get("enabled", False):
                    width = resize_config.get("width", 1280)
                    height = resize_config.get("height", 720)
                    frame = cv2.resize(frame, (width, height))

                frame_data = FrameData(
                    camera_id=self.camera_id,
                    timestamp=datetime.now(),
                    frame=frame
                )
                
                # If queue is full, drop the oldest frame to insert the new one
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(frame_data)
                
                if self.save_to_disk:
                    self._save_frame_to_disk(frame_data)
                    
            # Small sleep to prevent 100% CPU usage if stream goes too fast
            # but usually cv2.read() blocks until a frame is ready.
            # time.sleep(0.001)

    def _save_frame_to_disk(self, frame_data: FrameData):
        """
        Saves a frame to disk with a timestamped filename.
        Example: A14-419_20260915_094512_123.jpg
        """
        try:
            ts_str = frame_data.timestamp.strftime("%Y%m%d_%H%M%S_%f")[:19] # Truncate microseconds to 3 digits (milli)
            filename = f"{self.camera_id}_{ts_str}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, frame_data.frame)
        except Exception as e:
            logger.error(f"Failed to save frame to disk: {e}")
