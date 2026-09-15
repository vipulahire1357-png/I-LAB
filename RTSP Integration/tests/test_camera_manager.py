import pytest
import os
import yaml
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from camera_manager import CameraManager

@pytest.fixture
def temp_config_file(tmp_path):
    config = {
        "cameras": {
            "test_cam": {
                "name": "Test Camera",
                "username": "user",
                "password": "pwd",
                "rtsp_url": "rtsp://192.168.1.100/stream"
            }
        }
    }
    file_path = tmp_path / "cameras.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config, f)
    return str(file_path)

def test_load_valid_config(temp_config_file):
    manager = CameraManager(config_path=temp_config_file)
    assert manager.get_camera_config("test_cam")["name"] == "Test Camera"

def test_list_cameras(temp_config_file):
    manager = CameraManager(config_path=temp_config_file)
    cameras = manager.list_cameras()
    assert len(cameras) == 1
    assert cameras[0] == ("test_cam", "Test Camera")

def test_invalid_camera_id(temp_config_file):
    manager = CameraManager(config_path=temp_config_file)
    with pytest.raises(ValueError):
        manager.get_camera_config("invalid_cam")
