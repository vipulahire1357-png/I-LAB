import os
import yaml
import logging

logger = logging.getLogger(__name__)

class CameraManager:
    """
    Loads and validates the camera configuration.
    """
    def __init__(self, config_path: str = "config/cameras.yaml"):
        self.config_path = config_path
        self.config = {}
        self.cameras = {}
        self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML configuration: {e}")

        self._validate_global_config()
        self._validate_cameras()

    def _validate_global_config(self):
        # Set defaults if not present
        if "frame_extraction" not in self.config:
            self.config["frame_extraction"] = {"fps": 5}
        if "fps" not in self.config["frame_extraction"]:
            self.config["frame_extraction"]["fps"] = 5

        if "connection" not in self.config:
            self.config["connection"] = {
                "retry_enabled": True,
                "initial_retry_delay": 2,
                "max_retry_delay": 30
            }

        if "frame_output" not in self.config:
            self.config["frame_output"] = {
                "save_to_disk": False,
                "directory": "output/frames"
            }

    def _validate_cameras(self):
        if "cameras" not in self.config or not self.config["cameras"]:
            raise ValueError("No cameras defined in the configuration.")

        for cam_id, cam_config in self.config["cameras"].items():
            required_keys = ["name", "username", "password", "rtsp_url"]
            missing_keys = [key for key in required_keys if key not in cam_config]
            
            if missing_keys:
                raise ValueError(f"Camera '{cam_id}' is missing required fields: {', '.join(missing_keys)}")
            
            # Simple RTSP URL validation
            if not cam_config["rtsp_url"].startswith("rtsp://"):
                raise ValueError(f"Camera '{cam_id}' has an invalid RTSP URL. Must start with 'rtsp://'")

            self.cameras[cam_id] = cam_config

    def get_camera_config(self, camera_id: str) -> dict:
        """
        Retrieve configuration for a specific camera ID.
        """
        if camera_id not in self.cameras:
            raise ValueError(f"Camera '{camera_id}' does not exist.")
        return self.cameras[camera_id]

    def list_cameras(self):
        """
        Returns a list of tuples containing (camera_id, camera_name).
        """
        return [(cam_id, cam_data["name"]) for cam_id, cam_data in self.cameras.items()]

    def get_global_config(self, key: str, default=None):
        return self.config.get(key, default)
