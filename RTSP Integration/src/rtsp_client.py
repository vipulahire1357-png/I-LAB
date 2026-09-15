import cv2
import logging
import time
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class RTSPClient:
    """
    Handles the raw OpenCV connection to an RTSP stream.
    """
    def __init__(self, camera_id: str, camera_config: dict, connection_config: dict):
        self.camera_id = camera_id
        self.camera_config = camera_config
        self.connection_config = connection_config
        self.cap = None
        self.is_connected = False

    def _build_auth_url(self) -> str:
        """
        Injects username and password into the RTSP URL.
        """
        original_url = self.camera_config["rtsp_url"]
        username = self.camera_config["username"]
        password = self.camera_config["password"]
        
        parsed = urlparse(original_url)
        
        # If the URL already has credentials, return as is (or handle it, but let's assume it doesn't)
        if parsed.username or not username or not password:
            return original_url
            
        auth_url = f"{parsed.scheme}://{username}:{password}@{parsed.netloc}{parsed.path}"
        if parsed.params:
            auth_url += f";{parsed.params}"
        if parsed.query:
            auth_url += f"?{parsed.query}"
        if parsed.fragment:
            auth_url += f"#{parsed.fragment}"
            
        return auth_url

    def connect(self) -> bool:
        """
        Attempts to connect to the RTSP stream.
        """
        if self.cap is not None:
            self.release()

        logger.info(f"Connecting to camera {self.camera_id}...")
        
        # NOTE: Do NOT log the authenticated URL as it contains credentials!
        auth_url = self._build_auth_url()
        
        # Open the capture
        # Setting some environment variables for ffmpeg can reduce latency, but we keep it simple here.
        os_env_bak = None
        # In OpenCV, cv2.CAP_FFMPEG is generally used for RTSP.
        self.cap = cv2.VideoCapture(auth_url, cv2.CAP_FFMPEG)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open RTSP stream for camera {self.camera_id}.")
            self.is_connected = False
            return False

        logger.info(f"RTSP connection established for camera {self.camera_id}.")
        self.is_connected = True
        return True

    def read(self):
        """
        Reads a single frame from the stream.
        Returns:
            (success_boolean, frame)
        """
        if not self.is_connected or self.cap is None:
            return False, None
            
        ret, frame = self.cap.read()
        if not ret:
            logger.warning(f"Failed to read frame from camera {self.camera_id}.")
            self.is_connected = False
            
        return ret, frame

    def release(self):
        """
        Releases the OpenCV VideoCapture resource.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        logger.info(f"Released connection for camera {self.camera_id}.")

    def reconnect_with_backoff(self):
        """
        Attempts to reconnect using exponential backoff.
        """
        if not self.connection_config.get("retry_enabled", True):
            logger.warning(f"Retry is disabled. Not reconnecting to camera {self.camera_id}.")
            return False

        delay = self.connection_config.get("initial_retry_delay", 2)
        max_delay = self.connection_config.get("max_retry_delay", 30)

        while True:
            logger.info(f"Retrying connection to {self.camera_id} in {delay} seconds...")
            time.sleep(delay)
            
            if self.connect():
                return True
                
            delay = min(delay * 2, max_delay)
