# CampusNetra product0.3 — System Documentation & Audit Report
## Vision-Based Classroom Occupancy & Entry/Exit Counting System

---

## 1. Project Overview

| Attribute | Description |
| :--- | :--- |
| **System Name** | CampusNetra product0.3 |
| **Domain** | Computer Vision / Smart Campus Infrastructure |
| **Application** | Vision-Based Classroom Occupancy Estimation and Entry/Exit Tracking |
| **Core Paradigm** | Detection $\rightarrow$ Tracking $\rightarrow$ Line Crossing $\rightarrow$ Occupancy State |
| **Runtime Model** | Synchronous, deterministic, single-threaded, CPU-first (CUDA optional) |
| **Language & Stack** | Python 3.10+, OpenCV, NumPy, Ultralytics YOLO11, Supervision ByteTrack, PyYAML |

CampusNetra product0.3 is a standalone software system designed to analyze elevated CCTV video feeds of classroom entrances. It tracks individuals crossing a configurable virtual boundary, categorizes movements as **ENTER** or **EXIT** events, maintains real-time occupancy counts, and determines the classroom's high-level operational state (`OCCUPIED` vs. `EMPTY`).

---

## 2. Current MVP Scope

The product0.3 MVP is a **strictly software-only** vision pipeline. It operates on downloaded/prerecorded classroom CCTV video files.

### 2.1 Inputs
- Prerecorded CCTV video file: `data/input/classroom.mp4` (elevated angle overlooking classroom entrance and desks).
- YAML configuration file: `config/config.yaml`.

### 2.2 Processing
- Sequential frame decoding.
- YOLO11 person detection (class `0` only).
- ByteTrack multi-object tracking for temporary identity preservation.
- Virtual counting line signed-distance assessment using track foot-points.
- Per-track finite state machine ensuring single-event emission per crossing.
- Occupancy arithmetic with non-negative lower bound clamping.
- Visual frame annotation and console metric telemetry.

### 2.3 Outputs
- Processed annotated video file: `data/output/classroom_processed.mp4`.
- Standard output stream of real-time metrics and summary performance statistics.

---

## 3. Objective

The primary technical objective of the system is to answer fundamental questions with high reliability from video footage:

1. **How many students entered the classroom?** (`total_entries`)
2. **How many students exited the classroom?** (`total_exits`)
3. **What is the current estimated classroom occupancy?** (`occupancy_count = entries - exits`)
4. **When did an ENTER event occur?** (`timestamp`, `frame_id`, `track_id`)
5. **When did an EXIT event occur?** (`timestamp`, `frame_id`, `track_id`)
6. **Is the classroom currently considered OCCUPIED or EMPTY?** (`classroom_state: OCCUPIED | EMPTY`)
7. **Is the LEFT side or RIGHT side of the entrance/room occupied, and what is its occupancy number?** (`left_occupancy`, `right_occupancy`, `left_state`, `right_state`)

---

## 4. System Architecture

The architecture is composed of strictly decoupled, single-responsibility modules operating in a synchronous linear pipeline.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VIDEO SOURCE (Module 1)                          │
│               FileVideoSource (OpenCV VideoCapture backend)             │
│            Validates container, decodes frame, produces Frame           │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Frame(frame_id, timestamp, image)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       PERSON DETECTOR (Module 2)                        │
│                     PersonDetector (YOLO11n / YOLO11s)                  │
│             Filters to person class (id=0), applies confidence NMS      │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ DetectionResult(detections=[...])
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        PERSON TRACKER (Module 3)                        │
│                     PersonTracker (ByteTrack backend)                   │
│        Associates detections across frames, assigns monotonic track_id  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ TrackingResult(tracks=[...])
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     ENTRY/EXIT COUNTER (Module 4)                       │
│                     EntryExitCounter (Geometry + FSM)                   │
│     Evaluates foot-points against counting line & perpendicular divider │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ list[CrossingEvent] + (left_occ, right_occ)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       OCCUPANCY MANAGER (Module 5)                      │
│                        OccupancyManager (State)                         │
│       Maintains total, left, right occupancy; sets OCCUPIED / EMPTY     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ OccupancyState
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          VISUALIZER (Module 6)                          │
│                         Visualizer (OpenCV HUD)                         │
│        Renders boxes, counting line, perpendicular divider, HUD overlay │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Annotated np.ndarray frame
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          PIPELINE (Module 7)                            │
│                 Pipeline Orchestrator & CLI Entrypoint                  │
│       Encodes output video (VideoWriter), streams console metrics       │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
                         [Future Hardware Boundary]
                 OccupancyState -> (ESP32 Controller - Future)
```

---

## 5. Module Responsibilities & Contracts

| Module | Package / Class | Input Data Contract | Output Data Contract | Key Responsibility |
| :--- | :--- | :--- | :--- | :--- |
| **Module 1: Video Source** | `src.video` <br>`FileVideoSource` | Video file path (`str` / `Path`) | `Frame(frame_id, timestamp, image)` | Safe frame-by-frame decoding, metadata extraction, error isolation. |
| **Module 2: Person Detector** | `src.detection` <br>`PersonDetector` | `Frame` | `DetectionResult(detections, inference_ms)` | Person class filtering (class 0), confidence thresholding, boundary clamping. |
| **Module 3: Person Tracker** | `src.tracking` <br>`PersonTracker` | `DetectionResult` | `TrackingResult(tracks, tracking_ms)` | Temporal association via Kalman filter & ByteTrack; assigns persistent `track_id`. |
| **Module 4: Entry/Exit Counter** | `src.counting` <br>`EntryExitCounter` | `TrackingResult` | `list[CrossingEvent]` | Vector geometry line testing, per-track FSM crossing locking, occlusion trajectory interpolation, complete perpendicular divider crossing detection. |
| **Module 5: Occupancy Manager** | `src.occupancy` <br>`OccupancyManager` | `list[CrossingEvent]` | `OccupancyState(occupancy, entries, exits, state, left, right)` | Event-driven cumulative aggregation, conservation invariant enforcement (`left + right == total`), persistent occupancy maintenance even when tracks leave frame. |
| **Module 6: Visualizer** | `src.visualization` <br>`Visualizer` | `Frame`, `TrackingResult`, `OccupancyState`, `list[CrossingEvent]` | Annotated `np.ndarray` image | Rendering bounding boxes, IDs, virtual line, perpendicular divider, side badges, HUD. |
| **Module 7: Pipeline** | `src.pipeline` <br>`Pipeline` | `AppConfig` | Output Video & `PipelineMetrics` | Orchestrates execution, manages file writing, tracks latency profiles, outputs telemetry. |
| **Config System** | `src.config` <br>`ConfigLoader` | YAML file path | `AppConfig` (Typed Dataclasses) | Loads and validates configuration into strongly-typed dataclasses. |


---

## 6. Data Flow & Lifecycles

### 6.1 End-to-End Frame Execution Lifecycle
```
1. FileVideoSource.read()
   └── Produces Frame(frame_id=102, timestamp=4.08, image=ndarray(576, 1024, 3))

2. PersonDetector.detect(frame)
   └── Executes YOLO11n inference on frame.image
   └── Filters results: class_id == 0, confidence >= 0.35
   └── Produces DetectionResult with 3 PersonDetection instances

3. PersonTracker.update(detection_result)
   └── Formats detections for ByteTrack association
   └── Updates Kalman states, resolves matches against existing tracks
   └── Produces TrackingResult with Track(id=14), Track(id=18), Track(id=21)

4. EntryExitCounter.update(tracking_result)
   └── Extracts foot-point (x_mid, y_bottom) for Track(id=14)
   └── Computes signed distance relative to counting line vector
   └── Identifies state transition: OUTSIDE -> INSIDE for Track(id=14)
   └── Emits CrossingEvent(event_type=ENTER, track_id=14, frame_id=102, timestamp=4.08)
   └── Transitions Track(id=14) crossing state to ENTERED (locked)

5. OccupancyManager.apply(events)
   └── Receives CrossingEvent(ENTER)
   └── Increments total_entries: 4 -> 5
   └── Increments occupancy_count: 3 -> 4
   └── Evaluates state: occupancy_count > 0 -> ClassroomState.OCCUPIED

6. Visualizer.annotate(frame, tracking_result, occupancy_state, events)
   └── Draws yellow counting line across frame
   └── Draws green bounding boxes for confirmed tracks with ID badges
   └── Draws green ENTER flash indicator on HUD
   └── Renders semi-transparent statistics HUD overlay in corner

7. Pipeline.step()
   └── Writes annotated frame to cv2.VideoWriter
   └── Accumulates execution metrics (read_time, detect_time, track_time, render_time)
   └── Emits console progress update
```

---

## 7. Entry/Exit Algorithm & State Machine

### 7.1 Virtual Counting Line Vector Geometry
The virtual line is defined by two Cartesian coordinates in frame pixel space:
$$P_{\text{start}} = (x_1, y_1), \quad P_{\text{end}} = (x_2, y_2)$$

The directed line vector $\vec{L}$ and normal vector $\vec{N}$ are defined as:
$$\vec{L} = (x_2 - x_1, \, y_2 - y_1)$$
$$\vec{N} = (-(y_2 - y_1), \, x_2 - x_1)$$

For any tracked person with bounding box $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$, the **foot-point** $F$ is used:
$$F = \left( \frac{x_{\min} + x_{\max}}{2}, \, y_{\max} \right)$$

The signed scalar distance $d_s$ from the foot-point $F$ to the line is computed via the 2D cross product:
$$d_s = (F_x - x_1) \cdot (- (y_2 - y_1)) + (F_y - y_1) \cdot (x_2 - x_1)$$

- If $d_s > 0$: Foot-point lies on the **Positive Half-Plane**.
- If $d_s < 0$: Foot-point lies on the **Negative Half-Plane**.
- If $d_s = 0$: Foot-point lies exactly on the line (retains previous state).

The configuration parameter `inside_direction` maps half-planes to semantic regions:
- If `inside_direction: "positive"`: Positive Half-Plane = `INSIDE`, Negative Half-Plane = `OUTSIDE`.
- If `inside_direction: "negative"`: Negative Half-Plane = `INSIDE`, Positive Half-Plane = `OUTSIDE`.

### 7.2 Per-Track Finite State Machine (FSM)

To prevent duplicate events when a student crosses, pauses on, or oscillates across the counting line, each `track_id` is tracked by a state machine:

```
                      [ Initial Track Creation ]
                                   │
                                   ▼
                            ┌──────────────┐
                            │   UNKNOWN    │
                            └──────┬───────┘
                     Determined by │ initial foot-point
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
              ┌──────────────┐            ┌──────────────┐
              │   OUTSIDE    │            │    INSIDE    │
              └──────┬───────┘            └──────┬───────┘
                     │                           │
          Crosses to │ INSIDE         Crosses to │ OUTSIDE
                     ▼                           ▼
              ┌──────────────┐            ┌──────────────┐
              │  [ENTER=1]   │            │   [EXIT=1]   │
              │   ENTERED    │            │    EXITED    │
              │   (LOCKED)   │            │   (LOCKED)   │
              └──────────────┘            └──────────────┘
```

### 7.3 Duplicate Event Prevention Invariants
1. **Monotonic Event Emission:** Once a track achieves `ENTERED` or `EXITED` status, no additional crossing events will be emitted for that `track_id`.
2. **ByteTrack Monotonic IDs:** ByteTrack never reuses a `track_id` within the same video session.
3. **Memory Bounding:** When ByteTrack purges a dead track (exceeding `max_age`), the corresponding entry in `_track_states` is deleted.

### 7.4 Occupancy Arithmetic Invariants
$$\text{occupancy}(t) = \max\left(0, \, \text{occupancy}(t-1) + \Delta \text{entries}(t) - \Delta \text{exits}(t)\right)$$

$$\text{classroom\_state}(t) = \begin{cases} \text{OCCUPIED}, & \text{if } \text{occupancy}(t) > 0 \\ \text{EMPTY}, & \text{if } \text{occupancy}(t) == 0 \end{cases}$$

### 7.5 Separated Counting Line & Perpendicular Divider State Machines

The counting line and perpendicular divider operate dedicated, coupled finite-state machines (FSMs) with hysteresis filtering and dual-dot body validation:

1. **Counting Line FSM (Pure Total Classroom Occupancy):**
   - Strictly determines classroom total occupancy: $\text{occupancy}(t) = \max(0, \text{entries} - \text{exits})$.
   - Transitions between `OUTSIDE` and `INSIDE` based on foot-point ground contact verified by head-point orientation.
   - **Occlusion Approximation ($\pm 0, \pm 1, \pm 2$):** Quantifies track overlap/clustering near the counting line as a separate uncertainty approximation.

2. **Decoupled Area vs. Divider Detection:**
   - **Divider Detection (FSM Line Crossings):**
     - Detects directional boundary crossings across Left Divider (`[1380, 0] -> [-412, 1080]`) and Right Divider (`[1540, 0] -> [58, 1080]`).
     - Emits `ZONE_ENTER_LEFT`, `ZONE_EXIT_LEFT`, `ZONE_ENTER_RIGHT`, `ZONE_EXIT_RIGHT`.
   - **Area Detection (Spatial Presence):**
     - Evaluates all confirmed tracks in the classroom (both static seated and moving students).
     - Directly captures occupants located in Left Seating Area (`left_area_state: OCCUPIED/EMPTY`) and Right Seating Area (`right_area_state: OCCUPIED/EMPTY`) even if missed by divider lines.
   - **Display Invariant:** Left and right occupant counts are removed from the main HUD; only room `OCCUPANCY` (with separate `APPROX ±X`) and area operational states (`LEFT AREA`, `RIGHT AREA`) are displayed.


---

## 8. Configuration Reference

All settings in `config/config.yaml`:

```yaml
video:
  input_path: "data/input/classroom.mp4"       # Path to input video
  output_path: "data/output/classroom_processed.mp4" # Path for processed video
  show_live_window: false                     # Toggle live OpenCV display window

detection:
  model_path: "yolo11n.pt"                    # Model weights filename
  confidence_threshold: 0.25                  # Min confidence score [0.0 - 1.0]
  iou_threshold: 0.45                         # NMS IoU threshold [0.0 - 1.0]
  input_size: 640                             # Model input size (pixels)
  device: "cpu"                               # Inference device ("cpu" or "0")
  scene_profile: "general"                    # Profile: "general" or "classroom"

tracking:
  tracker_type: "bytetrack"                   # Tracker algorithm
  track_high_threshold: 0.20                  # Primary detection match threshold
  track_low_threshold: 0.05                   # Secondary detection match threshold
  match_threshold: 0.70                       # Tracking IoU match threshold
  max_age: 45                                 # Frames to hold lost track
  min_hits: 2                                 # Frames required to confirm track

counting:
  line_start: [0, 80]                         # Virtual line start [X, Y] (shifted 2cm to lower left)
  line_end: [1440, 1080]                      # Virtual line end [X, Y]
  inside_direction: "positive"                # "positive" or "negative"
  hysteresis_distance: 12.0                   # Perpendicular crossing deadband (pixels)
  perpendicular_line_start: [1460, 0]         # Center divider reference start [X, Y]
  perpendicular_line_end: [130, 1080]         # Center divider reference end [X, Y]
  perpendicular_line_length: 1700.0           # Length of perpendicular divider (pixels)
  walkway_width: 160.0                        # Central walkway corridor width (pixels, 4cm)
  divider_left_start: [1380, 0]               # Left Walkway Divider start [X, Y] (fixed upper end)
  divider_left_end: [-412, 1080]              # Left Walkway Divider end [X, Y] (rotated 8 deg left)
  divider_right_start: [1540, 0]              # Right Walkway Divider start [X, Y] (fixed upper end)
  divider_right_end: [58, 1080]               # Right Walkway Divider end [X, Y] (rotated 3 deg left)

visualization:
  line_color: [0, 255, 255]                   # Counting line color BGR (Yellow)
  line_thickness: 2                           # Counting line thickness in pixels
  perpendicular_line_color: [255, 200, 0]     # Perpendicular line color BGR (Cyan / Sky Blue)
  perpendicular_line_thickness: 2             # Perpendicular line thickness in pixels
  show_side_occupancy: true                   # Render on-canvas LEFT / RIGHT zone badges
  bbox_color_confirmed: [0, 255, 0]           # Active box color BGR (Green)
  bbox_color_lost: [128, 128, 128]            # Lost box color BGR (Grey)
  event_flash_frames: 10                      # Alert flash duration in frames
  font_scale: 0.65                            # HUD font scale
  font_thickness: 2                           # HUD font thickness
  overlay_position: [20, 20]                  # HUD top-left coordinate [X, Y]

metrics:
  console_every_n_frames: 30                  # Telemetry reporting interval
```

---

## 9. Testing Strategy & Test Suite

The test suite contains automated unit and integration tests using `pytest`.

### 9.1 Test Suite Breakdown (144 Tests Total — All Passing)
| Test File | Test Count | Target Module | Scope |
| :--- | :--- | :--- | :--- |
| `tests/test_video_source.py` | 26 tests | Module 1: Video Source | File validation, frame reading, metadata, error handling. |
| `tests/test_person_detector.py` | 30 tests | Module 2: Person Detector | Model loading, person class filtering, confidence NMS, bounding box clamping. |
| `tests/test_classroom.py` | 24 tests | Module 2 & 3: Profiles | Classroom far-ROI detection fusion, IoU math, specialized tracker configurations. |
| `tests/test_person_tracker.py` | 27 tests | Module 3: Person Tracker | Track association, ID persistence, lost track recovery, state pruning. |
| `tests/test_entry_exit_counter.py` | 15 tests | Module 4: Counter | Line geometry, dual parallel divider lines, occlusion approximation, static/moving area presence, zone transitions. |
| `tests/test_occupancy_manager.py` | 14 tests | Module 5: Occupancy | Pure counting line occupancy, occlusion approximation separation, decoupled area vs divider detection. |
| `tests/test_visualizer.py` | 3 tests | Module 6: Visualizer | Annotation correctness, approximation rendering, area badges, empty frame safety. |
| `tests/test_config_loader.py` | 4 tests | Config System | YAML deserialization, walkway divider parsing, missing parameter catching. |
| `tests/test_pipeline_integration.py`| 1 test | Module 7: Pipeline | End-to-end processing of synthetic video streams, video file creation. |

---

## 10. Performance Baseline & Profiling

### 10.1 Latency Distribution per Frame (CPU Baseline on YOLO11n @ 640x640)
```
┌──────────────────────────────────────────────────────────┐
│ YOLO11n Inference (PersonDetector)          75% - 85%    │
├──────────────────────────────────────────────────────────┤
│ Frame Decoding (OpenCV VideoCapture)         5% - 10%    │
├──────────────────────────────────────────────────────────┤
│ Output Encoding (OpenCV VideoWriter)         3% -  8%    │
├──────────────────────────────────────────────────────────┤
│ Frame HUD Rendering (Visualizer)             3% -  6%    │
├──────────────────────────────────────────────────────────┤
│ ByteTrack Association (PersonTracker)        1% -  3%    │
├──────────────────────────────────────────────────────────┤
│ Entry/Exit Geometry & FSM (EntryExitCounter) < 0.5%      │
└──────────────────────────────────────────────────────────┘
```

### 10.2 Hardware Performance Expectations
- **Standard Modern CPU (Intel i5/i7, AMD Ryzen 5/7):** ~10 – 18 FPS
- **NVIDIA GPU (RTX 3060+ via CUDA):** ~45 – 90 FPS
- **Embedded / Low-Power CPU:** ~4 – 8 FPS

---

## 11. Known Limitations & Edge Cases

| Scenario / Edge Case | System Impact | Mitigation / System Behavior |
| :--- | :--- | :--- |
| **Severe Occlusion behind furniture** | Detection box disappears temporarily while student is walking. | ByteTrack maintains track state for `max_age=45` frames; Kalman filter predicts trajectory until re-acquisition. |
| **Close Group Entry (2+ students touching)** | YOLO detector may group two overlapping students into a single bounding box. | Count will register as 1. Acknowledged MVP limitation (no multi-camera or semantic segmentation). |
| **Student Loitering on Counting Line** | Foot-point fluctuates around the zero boundary. | Finite state machine locks track state upon first transition; no redundant events are triggered. |
| **Student Enters, Leaves, and Re-Enters** | Student exits camera view and returns later. | ByteTrack drops old track after `max_age` frames and assigns a new `track_id` on return, correctly registering a new entry. |
| **False Positives (Bags, Chairs)** | Static objects briefly misdetected as a person. | Tracker requires `min_hits=2` consecutive frames before confirming a track; isolated false positives are discarded. |
| **Elevated Perspective Foreshortening** | Students far from the camera appear significantly smaller. | Optional `scene_profile: "classroom"` executes a two-pass far-ROI detection for high-recall in the upper frame area. |

---

## 12. Strict Non-Goals

The following features are **strictly out of scope** for product0.3 MVP:

- ❌ **Hardware / ESP32 Integration:** No GPIO, microcontrollers, relays, or light/fan automation.
- ❌ **Biometrics & Student Identification:** No face recognition, embeddings, or student ID mapping.
- ❌ **Databases & Storage:** No SQL/NoSQL databases, persistent student profiles, or logs.
- ❌ **Dashboards & Web UI:** No Flask, FastAPI, WebSockets, or browser frontend interfaces.
- ❌ **Cross-Camera Re-ID:** No multi-camera fusion or cross-camera person re-identification.
- ❌ **Model Training:** No custom neural network training or fine-tuning.
- ❌ **Live RTSP Camera Streams:** Processing is exclusively validated on prerecorded video files.
- ❌ **LLM Integrations:** No generative AI or natural language querying.

---

## 13. Future Extensions (Post-MVP)

1. **Hardware Bridge (`HardwareController`):** Connect `OccupancyState.classroom_state` to an ESP32 microcontroller via MQTT / HTTP / Serial to automate AC, lighting, and ventilation.
2. **Empty State Confirmation Delay:** Require occupancy to remain 0 for a configurable confirmation window (e.g. 60 seconds) before signaling power-down.
3. **Interactive Graphical Line Editor:** A lightweight UI tool to click and set `line_start` and `line_end` visually over the camera stream.
4. **Live RTSP Stream Consumer:** Add an `RTSPVideoSource` subclass inheriting from `VideoSource` for direct IP camera streaming.

---

## 14. Implementation & Audit Status

### 14.1 Component Audit Matrix

| Component | Target Location | Implementation Status | Test Status | Audit Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **Config Loader** | `src/config/loader.py` | `IMPLEMENTED` | `TESTED` | Fully validated with 4 unit tests. |
| **Video Source** | `src/video/file_video_source.py` | `IMPLEMENTED` | `TESTED` | Fully verified with 26 unit tests. |
| **Person Detector** | `src/detection/person_detector.py` | `IMPLEMENTED` | `TESTED` | Fully verified with 54 unit & profile tests. |
| **Person Tracker** | `src/tracking/person_tracker.py` | `IMPLEMENTED` | `TESTED` | Fully verified with 27 unit tests. |
| **Entry/Exit Counter** | `src/counting/entry_exit_counter.py` | `IMPLEMENTED` | `TESTED` | Fully verified with 8 unit tests (including dual-dot validation). |
| **Occupancy Manager**| `src/occupancy/occupancy_manager.py` | `IMPLEMENTED` | `TESTED` | Fully verified with 7 unit tests (including left/right state). |
| **Visualizer** | `src/visualization/visualizer.py` | `IMPLEMENTED` | `TESTED` | Fully verified with 3 unit tests (including dual-dot rendering). |
| **Pipeline** | `src/pipeline/pipeline.py` | `IMPLEMENTED` | `TESTED` | Fully verified with end-to-end integration test. |
| **CLI Runner** | `main.py` | `IMPLEMENTED` | `TESTED` | CLI arguments verified via `--help`. |
| **ESP32 Controller**| `src/hardware/...` | `NOT IMPLEMENTED` | `N/A` | Strict non-goal for MVP. |
| **Web Dashboard** | `dashboard/...` | `NOT IMPLEMENTED` | `N/A` | Strict non-goal for MVP. |

---

## 15. Audit History Log

| Date | Milestone / Action | Auditor | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **2026-08-25** | Initial Planning & Architecture Design | Lead CV Systems Engineer | `APPROVED` | Implementation plan and task list generated. |
| **2026-08-25** | Phase 1 Foundation & Directory Tree | Lead CV Systems Engineer | `COMPLETED` | Created directories, `requirements.txt`, `config.yaml`, `ConfigLoader`. |
| **2026-08-25** | Phase 2, 3, 4 Module Migration | Lead CV Systems Engineer | `COMPLETED` | Ported `video`, `detection`, and `tracking` modules from Product-0.2. |
| **2026-08-25** | Phase 5, 6, 7 New Modules | Lead CV Systems Engineer | `COMPLETED` | Implemented `EntryExitCounter`, `OccupancyManager`, `Visualizer`. |
| **2026-08-25** | Phase 8 Pipeline & Entry Point | Lead CV Systems Engineer | `COMPLETED` | Implemented `Pipeline`, `main.py`, and integration test. |
| **2026-08-25** | Automated Test Verification | Lead CV Systems Engineer | `VERIFIED` | 125/125 tests passing in `pytest`. |
| **2026-08-25** | Documentation & Audit Finalization | Lead CV Systems Engineer | `COMPLETED` | Finalized `SetupAndRun.md` and `ReadMeAndAudit.md`. |
| **2026-08-26** | Perpendicular Divider & Side Occupancy | Lead CV Systems Engineer | `COMPLETED` | Added perpendicular divider line, left/right occupancy tracking, and badges (129/129 tests passing). |
| **2026-08-26** | Dual-Dot (Foot + Head) Validation System | Lead CV Systems Engineer | `COMPLETED` | Added dual reference dots, head-point orientation validation, 1cm divider shift (130/130 tests passing). |
| **2026-08-26** | Separated Counting & Zone State Machines | Lead CV Systems Engineer | `COMPLETED` | Separated counting line occupancy from perpendicular divider; inside-only zone occupancy with exact conservation. |



