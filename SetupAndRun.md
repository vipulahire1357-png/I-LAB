# CampusNetra product0.3 — Setup & Run Guide
## Vision-Based Classroom Occupancy & Entry/Exit Counting System

---

## 1. Prerequisites

### 1.1 Operating System & Python Environment
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
- **Python Version:** Python `3.10` or `3.11` (64-bit recommended)
- **Hardware Requirements:**
  - **CPU:** Quad-core Intel/AMD x86_64 processor (Baseline MVP runs CPU-only)
  - **RAM:** Minimum 4 GB (8 GB+ recommended)
  - **GPU (Optional):** NVIDIA CUDA-compatible GPU (if PyTorch CUDA is installed, set `device: "0"` in configuration)
  - **Storage:** ~500 MB free space for dependencies, models, and output video

---

## 2. Virtual Environment Setup

Always isolate project dependencies within a dedicated Python virtual environment.

### 2.1 Windows (PowerShell)
```powershell
# Navigate to the product0.3 project directory
cd d:\CampusNetra\product0.3

# Create a virtual environment named .venv
python -m venv .venv

# Enable script execution for current session if execution policy is restricted
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

### 2.2 Windows (Command Prompt)
```cmd
cd d:\CampusNetra\product0.3
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### 2.3 Linux / macOS
```bash
cd /path/to/product0.3
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Dependency Installation

Install all required packages from `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.1 Key Dependencies Overview
| Package | Version | Purpose |
| :--- | :--- | :--- |
| `opencv-python` | `>=4.8.0` | Video file decoding, frame manipulation, line/box annotation, video encoding |
| `numpy` | `>=1.24.0` | Vector geometry, array manipulation, signed distance calculation |
| `ultralytics` | `>=8.3.0` | YOLO11 person detection runtime (PyTorch-backed) |
| `supervision` | `>=0.28.0` | ByteTrack multi-object tracking implementation |
| `pyyaml` | `>=6.0.0` | Typed runtime configuration loading (`config/config.yaml`) |
| `pytest` | `>=7.4.0` | Unit and integration test suite execution |
| `pytest-cov` | `>=4.1.0` | Test coverage reporting |

---

## 4. Model Download & Installation

The system uses standard Ultralytics YOLO11 weights.

### 4.1 Automatic Model Download
On the first run of the pipeline or test suite, Ultralytics will automatically download the requested model weights to the working directory or cache:
- **Default MVP Model:** `yolo11n.pt` (~5.4 MB) — Lightweight, fast, CPU-friendly
- **Classroom Profile Model (Optional):** `yolo11s.pt` (~18.4 MB) — Higher recall for small/distant seated students

### 4.2 Manual / Offline Download
If running in an air-gapped or restricted network environment:
1. Download `yolo11n.pt` from the official Ultralytics releases.
2. Place `yolo11n.pt` in the project root `d:\CampusNetra\product0.3\` or `models/` directory.
3. Verify the path matches `model_path` in `config/config.yaml`.

---

## 5. Input Video Setup

The system is designed to process prerecorded classroom CCTV footage.

### 5.1 Video Directory Placement
Place your test video in `data/input/`:
```
d:\CampusNetra\product0.3\data\input\classroom.mp4
```

### 5.2 Supported Video Formats
- Container formats: `.mp4`, `.avi`, `.mkv`, `.mov`
- Codecs: H.264 (AVC), H.265 (HEVC), MPEG-4, MJPEG
- Note: Video is decoded sequentially frame-by-frame via OpenCV `VideoCapture`.

---

## 6. Configuration Management

All runtime behaviors are governed by `config/config.yaml`. **No Python code modification is needed to adjust runtime parameters.**

```yaml
# CampusNetra product0.3 — Runtime Configuration

# ── Video Input / Output ─────────────────────────────────────────────────────
video:
  input_path: "data/input/classroom.mp4"
  output_path: "data/output/classroom_processed.mp4"
  show_live_window: false          # Set to true for live OpenCV preview window

# ── Person Detection ─────────────────────────────────────────────────────────
detection:
  model_path: "yolo11n.pt"         # "yolo11n.pt" (fast) or "yolo11s.pt" (high-recall)
  confidence_threshold: 0.35       # Minimum confidence to accept a detection
  iou_threshold: 0.45              # NMS IoU threshold
  input_size: 640                  # Inference resolution (640 or 1280)
  device: "cpu"                    # "cpu" or "0" (for CUDA GPU)
  scene_profile: "general"         # "general" or "classroom" (enables far-ROI pass)

# ── Multi-Object Tracking ────────────────────────────────────────────────────
tracking:
  tracker_type: "bytetrack"
  track_high_threshold: 0.25       # Detection confidence to initialize/match track
  track_low_threshold: 0.10        # Secondary association threshold for lost/weak tracks
  match_threshold: 0.80            # IoU matching threshold
  max_age: 30                      # Max frames to retain a lost track before removal
  min_hits: 3                      # Consecutive hits required to confirm a track

# ── Counting Line & Perpendicular Divider ────────────────────────────────────
# Coordinates are [X, Y] in original video frame pixels.
# Primary counting line strictly manages total classroom entries, exits, and occupancy.
# Perpendicular divider operates an independent state-machine exclusively for occupants
# INSIDE the classroom to maintain exact Left/Right zone occupancy conservation.
counting:
  line_start: [0, 0]               # [x, y] start point of virtual counting line (upper-left corner)
  line_end: [1520, 1080]           # [x, y] end point of virtual counting line
  inside_direction: "positive"     # "positive" or "negative" (flips INSIDE/OUTSIDE side)
  hysteresis_distance: 12.0        # Perpendicular deadband to prevent oscillation
  perpendicular_line_start: [1460, 0]   # Top edge toward upper-right [x, y] (shifted 1cm left)
  perpendicular_line_end: [130, 1080]   # Bottom edge of screen [x, y] (shifted 1cm left)
  perpendicular_line_length: 1700.0     # Length of perpendicular divider (px)






# ── Visualization ────────────────────────────────────────────────────────────
visualization:
  line_color: [0, 255, 255]        # BGR Yellow for primary counting line
  line_thickness: 2
  perpendicular_line_color: [255, 200, 0] # BGR Cyan/Sky Blue for perpendicular divider
  perpendicular_line_thickness: 2
  show_side_occupancy: true        # Show floating LEFT and RIGHT zone occupancy badges
  bbox_color_confirmed: [0, 255, 0] # BGR Green for active tracked students
  bbox_color_lost: [128, 128, 128]  # BGR Grey for lost tracks
  event_flash_frames: 10           # Duration (in frames) to flash ENTER (green) / EXIT (red)
  font_scale: 0.65
  font_thickness: 2
  overlay_position: [20, 20]       # Top-left corner [X, Y] of the HUD stats box

# ── Console Metrics ──────────────────────────────────────────────────────────
metrics:
  console_every_n_frames: 30       # Print progress and metrics every N frames
```

---

## 7. How to Run the System

### 7.1 Running with Default Configuration
```bash
python main.py
```

### 7.2 Overriding Configuration File
```bash
python main.py --config config/config.yaml
```

### 7.3 Overriding Input/Output Paths via CLI
```bash
python main.py --input data/input/classroom.mp4 --output data/output/classroom_processed.mp4
```

### 7.4 Enabling Live Visualization Preview
```bash
python main.py --show
```
*(Press `q` or `ESC` in the preview window to stop processing early).*

---

## 8. Expected Output

### 8.1 Output Video File
Processed annotated video is saved to:
```
data/output/classroom_processed.mp4
```
Each frame contains:
- Bounding boxes around detected/tracked persons with persistent Track IDs.
- Primary virtual counting line (Yellow) and perpendicular dividing line (Cyan/Sky Blue).
- On-canvas floating zone occupancy badges:
  - **LEFT SIDE:** `LEFT: <count> (OCCUPIED / EMPTY)`
  - **RIGHT SIDE:** `RIGHT: <count> (OCCUPIED / EMPTY)`
- Real-time HUD overlay card:
  - **OCCUPANCY:** Current estimated total student count inside classroom
  - **LEFT SIDE:** Real-time occupant count & state on left side of divider
  - **RIGHT SIDE:** Real-time occupant count & state on right side of divider
  - **ENTRIES:** Total cumulative entrance count
  - **EXITS:** Total cumulative exit count
  - **STATE:** `OCCUPIED` (if occupancy > 0) or `EMPTY` (if occupancy == 0)
  - **FPS:** Instantaneous and average frame processing rate

### 8.2 Console Output
During execution, structured metrics are printed to stdout:
```text
================================================================================
CampusNetra product0.3 — Vision-Based Classroom Occupancy & Counting Pipeline
================================================================================
Input Video       : data/input/classroom.mp4
Output Video      : data/output/classroom_processed.mp4
Resolution        : 1920x1080 @ 25.00 FPS (750 total frames)
Detector Model    : yolo11n.pt (device=cpu, conf=0.25)
Tracker           : ByteTrack (max_age=45, min_hits=2)
Counting Line     : (0, 195) -> (1520, 1080) [inside=positive]
--------------------------------------------------------------------------------
Frame:   30/750 | FPS: 11.2 | Active: 2 | Left:  1 (OCCUPIED) | Right:  1 (OCCUPIED) | In: 2 | Out: 0 | Occupancy: 2 | State: OCCUPIED
Frame:   60/750 | FPS: 11.5 | Active: 3 | Left:  2 (OCCUPIED) | Right:  1 (OCCUPIED) | In: 3 | Out: 0 | Occupancy: 3 | State: OCCUPIED
Frame:   90/750 | FPS: 11.4 | Active: 4 | Left:  3 (OCCUPIED) | Right:  1 (OCCUPIED) | In: 4 | Out: 0 | Occupancy: 4 | State: OCCUPIED
...
Frame:  750/750 | FPS: 11.8 | Active: 1 | Left:  1 (OCCUPIED) | Right:  0 (EMPTY)    | In: 12 | Out: 3 | Occupancy: 9 | State: OCCUPIED
--------------------------------------------------------------------------------
Pipeline Completed Successfully!
Total Frames Processed : 750
Total Wall Time        : 63.56 s
Average FPS            : 11.80
Total Entered          : 12
Total Exited           : 3
Final Occupancy        : 9
Final Classroom State  : OCCUPIED
Saved Video            : data/output/classroom_processed.mp4
================================================================================
```

---

## 9. Running Tests

Execute the automated test suite to verify all components and invariants:

```bash
# Run all unit tests
pytest tests/ -v

# Run with test coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run specific module test suites
pytest tests/test_video_source.py -v
pytest tests/test_person_detector.py -v
pytest tests/test_person_tracker.py -v
pytest tests/test_entry_exit_counter.py -v
pytest tests/test_occupancy_manager.py -v
pytest tests/test_visualizer.py -v
pytest tests/test_config_loader.py -v
```
```

---

## 10. Troubleshooting & FAQ

| Issue / Error | Root Cause | Solution |
| :--- | :--- | :--- |
| `VideoOpenError: File does not exist` | Invalid input file path specified in config or CLI | Verify the video file exists at `data/input/classroom.mp4` or provide absolute path. |
| `VideoOpenError: OpenCV could not open the video file` | Unsupported video codec or missing ffmpeg backend | Re-encode video to standard H.264 MP4 format using ffmpeg: `ffmpeg -i input.mov -c:v libx264 -crf 23 output.mp4`. |
| `ModelLoadError` | PyTorch / Ultralytics weights download interrupted | Delete corrupt `.pt` file from root/cache directory and re-run to allow clean re-download. |
| **Entries and Exits are reversed** | `inside_direction` configuration inverted for camera view | Change `inside_direction: "positive"` to `inside_direction: "negative"` in `config/config.yaml`. |
| **No detections appear on video** | `confidence_threshold` is set too high for camera lighting/angle | Lower `confidence_threshold` to `0.20` or `0.25` in `config/config.yaml`. |
| **Track ID switches frequently** | Occlusion or fast motion exceeding tracker thresholds | Increase `max_age` to `45` or `60` in `config/config.yaml` to tolerate longer occlusion. |
| **Processing is too slow on CPU** | High inference resolution or large model | Ensure `model_path: "yolo11n.pt"` and `input_size: 640` are configured. |

---

## 11. Performance Measurement & Benchmarking

To benchmark individual pipeline stages without rendering overhead:

```bash
# Run detection benchmark
python examples/benchmark_person_detection.py

# Run tracking benchmark
python examples/benchmark_tracking.py
```

The pipeline exposes fine-grained per-frame stage timings (`t_read`, `t_detect`, `t_track`, `t_count`, `t_render`, `t_write`) in `PipelineMetrics` for exact latency profiling.
