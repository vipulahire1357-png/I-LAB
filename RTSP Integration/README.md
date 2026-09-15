# RTSP Camera Frame Extraction Module

A production-ready RTSP frame extraction module that connects to IP cameras, reads the stream continuously to prevent lag, and exposes a bounded queue of `FrameData` objects to a downstream Computer Vision pipeline.

## Features
- **Independent Frame Extraction**: A dedicated background thread manages RTSP reading, leaving your CV algorithms unblocked.
- **Configurable Target FPS**: Throttle the extraction FPS without modifying the actual code, while keeping the underlying OpenCV stream buffer clear.
- **Robust Reconnection**: Implements exponential backoff to recover automatically from RTSP timeouts, network drops, or camera reboots.
- **Secure Configuration**: Uses `yaml` configuration for multiple cameras. No hard-coded passwords.
- **Optional Disk Output**: Easily test by saving frames to disk.

## Setup

### 1. Installation
Requires Python 3.8+.
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configuration
Copy the example config and fill in your actual credentials.
```bash
cp config/cameras.example.yaml config/cameras.yaml
```
*Note: `cameras.yaml` is excluded from git to prevent leaking passwords.*

### 3. Usage
**List configured cameras:**
```bash
python src/main.py --list-cameras
```

**Run extraction for a specific camera:**
```bash
python src/main.py --camera A14-419
```

**Override FPS limit via CLI:**
```bash
python src/main.py --camera A14-419 --fps 1
```

## Integration with a Computer Vision Pipeline
To use this module in your own project, instantiate `CameraManager`, `RTSPClient`, and `FrameExtractor`:

```python
from src.camera_manager import CameraManager
from src.rtsp_client import RTSPClient
from src.frame_extractor import FrameExtractor

manager = CameraManager("config/cameras.yaml")
cam_config = manager.get_camera_config("A14-419")
global_extract = manager.get_global_config("frame_extraction")
global_conn = manager.get_global_config("connection")
global_out = manager.get_global_config("frame_output")

client = RTSPClient("A14-419", cam_config, global_conn)
extractor = FrameExtractor("A14-419", client, global_extract, global_out)

extractor.start()
try:
    while True:
        frame_data = extractor.get_frame(timeout=1.0)
        if frame_data:
            print(f"Got frame from {frame_data.camera_id}")
            # => Pass frame_data.frame (numpy array) to CV pipeline
except KeyboardInterrupt:
    extractor.stop()
```
