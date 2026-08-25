"""
Person Detector — Scene-aware YOLO-based person detection engine.

Wraps an Ultralytics YOLO model behind a clean API so that downstream
modules (tracking, dashboard) never import or reference YOLO directly.

Architecture (general)::

    Frame.image  →  validate  →  YOLO inference  →  filter + clamp
                                                  →  DetectionResult

Architecture (classroom)::

    Frame.image  →  validate
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
     GLOBAL inference        FAR-ROI crop
     (high-res 1280)         (second pass)
           │                       │
           │              map ROI→original
           │                       │
           └───────────┬───────────┘
                       ▼
              DETECTION FUSION
              (IoU de-dup, keep best)
                       │
                       ▼
              DetectionResult

The model is loaded **once** at construction time.  The detector is
synchronous and single-threaded.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from src.detection.exceptions import (
    InferenceError,
    InvalidFrameError,
    ModelLoadError,
)
from src.detection.models import (
    DetectionConfig,
    DetectionResult,
    PersonDetection,
)
from src.video.models import Frame

logger = logging.getLogger(__name__)


class PersonDetector:
    """YOLO-backed person detector with optional classroom scene awareness.

    Parameters
    ----------
    config : DetectionConfig | None
        Detection configuration.  Defaults to ``DetectionConfig()``
        which uses ``yolo11n.pt`` on CPU at 640px.

        For classroom use, pass ``DetectionConfig.classroom_default()``
        or a custom config with ``scene_profile="classroom"``.

    Example
    -------
    ::

        # General use
        detector = PersonDetector()

        # Classroom-optimised
        detector = PersonDetector(DetectionConfig.classroom_default())
        detector.warmup()

        result = detector.detect(frame)
        for det in result.detections:
            print(det)
    """

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self._config = config or DetectionConfig()
        self._model = None
        self._is_loaded = False
        self._load_model()

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the YOLO model.  Called once during __init__."""
        t0 = time.monotonic()
        try:
            from ultralytics import YOLO

            self._model = YOLO(self._config.model_path)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load YOLO model: {exc}",
                model_path=self._config.model_path,
            ) from exc

        self._is_loaded = True
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "Model loaded: %s on device=%s (%.0f ms)",
            self._config.model_path,
            self._config.device,
            elapsed_ms,
        )
        logger.info(
            "Detection config: conf=%.2f, iou=%.2f, input=%d, scene=%s",
            self._config.confidence_threshold,
            self._config.iou_threshold,
            self._config.input_size,
            self._config.scene_profile,
        )

    def warmup(self, iterations: int = 1) -> float:
        """Run warm-up inference(s) to prime the model.

        Parameters
        ----------
        iterations : int
            Number of warm-up passes.

        Returns
        -------
        float
            Total warm-up time in milliseconds.
        """
        if not self._is_loaded or self._model is None:
            raise ModelLoadError(
                "Cannot warm up: model is not loaded.",
                model_path=self._config.model_path,
            )
        dummy = np.zeros(
            (self._config.input_size, self._config.input_size, 3),
            dtype=np.uint8,
        )
        t0 = time.monotonic()
        for _ in range(iterations):
            self._model.predict(
                source=dummy,
                conf=self._config.confidence_threshold,
                iou=self._config.iou_threshold,
                imgsz=self._config.input_size,
                device=self._config.device,
                classes=[self._config.person_class_id],
                verbose=False,
            )
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info("Warm-up complete: %d iteration(s) in %.0f ms", iterations, elapsed_ms)
        return elapsed_ms

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def config(self) -> DetectionConfig:
        return self._config

    # ------------------------------------------------------------------
    # Detection — public entry point
    # ------------------------------------------------------------------

    def detect(self, frame: Frame) -> DetectionResult:
        """Run person detection on a single frame.

        For ``scene_profile="classroom"``, runs a two-pass detection:
        1. Global high-resolution inference on the full frame.
        2. Far/back-ROI crop inference to recover small distant students.
        3. Fusion of both passes with IoU-based duplicate suppression.

        Parameters
        ----------
        frame : Frame
            Decoded video frame from Module 1/2.

        Returns
        -------
        DetectionResult
            Zero or more person detections sorted by confidence
            descending, plus timing information.

        Raises
        ------
        InvalidFrameError
            If the frame or its image is invalid.
        InferenceError
            If the model fails during inference.
        """
        total_t0 = time.monotonic()

        # --- validate input ----------------------------------------------
        self._validate_frame(frame)

        if self._config.is_classroom:
            detections, infer_ms = self._detect_classroom(frame)
        else:
            detections, infer_ms = self._detect_general(frame)

        total_ms = (time.monotonic() - total_t0) * 1000

        return DetectionResult(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            detections=tuple(detections),
            inference_time_ms=infer_ms,
            total_time_ms=total_ms,
        )

    # ------------------------------------------------------------------
    # General (single-pass) detection
    # ------------------------------------------------------------------

    def _detect_general(
        self, frame: Frame
    ) -> tuple[list[PersonDetection], float]:
        """Single-pass detection for general scenes."""
        infer_t0 = time.monotonic()
        try:
            results = self._model.predict(
                source=frame.image,
                conf=self._config.confidence_threshold,
                iou=self._config.iou_threshold,
                imgsz=self._config.input_size,
                device=self._config.device,
                classes=[self._config.person_class_id],
                verbose=False,
            )
        except Exception as exc:
            raise InferenceError(
                f"YOLO inference failed: {exc}",
                frame_id=frame.frame_id,
            ) from exc
        infer_ms = (time.monotonic() - infer_t0) * 1000

        detections = self._parse_results(results, frame)
        return detections, infer_ms

    # ------------------------------------------------------------------
    # Classroom (dual-pass) detection
    # ------------------------------------------------------------------

    def _detect_classroom(
        self, frame: Frame
    ) -> tuple[list[PersonDetection], float]:
        """Two-pass detection for classroom scenes.

        Pass 1: Global high-resolution inference.
        Pass 2: Far/back-ROI crop inference at enlarged scale.
        Fusion: IoU-based duplicate suppression.
        """
        frame_h, frame_w = frame.image.shape[:2]
        total_infer = 0.0

        # ---- Pass 1: Global detection -----------------------------------
        t0 = time.monotonic()
        try:
            global_results = self._model.predict(
                source=frame.image,
                conf=self._config.confidence_threshold,
                iou=self._config.iou_threshold,
                imgsz=self._config.input_size,
                device=self._config.device,
                classes=[self._config.person_class_id],
                verbose=False,
            )
        except Exception as exc:
            raise InferenceError(
                f"Global inference failed: {exc}",
                frame_id=frame.frame_id,
            ) from exc
        total_infer += (time.monotonic() - t0) * 1000

        global_dets = self._parse_results(global_results, frame)

        # ---- Pass 2: Far/back ROI detection (if configured) -------------
        roi_dets: list[PersonDetection] = []

        roi_norm = self._config.classroom_far_roi
        if roi_norm is not None:
            rx1 = int(roi_norm[0] * frame_w)
            ry1 = int(roi_norm[1] * frame_h)
            rx2 = int(roi_norm[2] * frame_w)
            ry2 = int(roi_norm[3] * frame_h)

            # Clamp to valid frame region
            rx1 = max(0, min(rx1, frame_w))
            ry1 = max(0, min(ry1, frame_h))
            rx2 = max(0, min(rx2, frame_w))
            ry2 = max(0, min(ry2, frame_h))

            crop_w = rx2 - rx1
            crop_h = ry2 - ry1

            if crop_w > 10 and crop_h > 10:
                roi_crop = frame.image[ry1:ry2, rx1:rx2]

                t0 = time.monotonic()
                try:
                    roi_results = self._model.predict(
                        source=roi_crop,
                        conf=self._config.classroom_roi_confidence,
                        iou=self._config.iou_threshold,
                        imgsz=self._config.classroom_roi_input_size,
                        device=self._config.device,
                        classes=[self._config.person_class_id],
                        verbose=False,
                    )
                except Exception as exc:
                    raise InferenceError(
                        f"ROI inference failed: {exc}",
                        frame_id=frame.frame_id,
                    ) from exc
                total_infer += (time.monotonic() - t0) * 1000

                # Map ROI coordinates back to original frame
                roi_dets = self._parse_roi_results(
                    roi_results, frame, rx1, ry1, rx2, ry2
                )

        # ---- Fusion -----------------------------------------------------
        fused = self._fuse_detections(global_dets, roi_dets)

        return fused, total_infer

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------

    def _parse_results(
        self, results, frame: Frame
    ) -> list[PersonDetection]:
        """Convert YOLO results to :class:`PersonDetection` list.

        Ultralytics returns coordinates in original-frame space by default,
        so no manual coordinate rescaling is required.

        Detections are sorted by confidence descending for deterministic
        ordering.
        """
        detections: list[PersonDetection] = []

        if not results or len(results) == 0:
            return detections

        result = results[0]  # single-image inference → one result
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        # Extract tensors → numpy.
        xyxy = boxes.xyxy.cpu().numpy()      # (N, 4)  x1 y1 x2 y2
        confs = boxes.conf.cpu().numpy()      # (N,)
        cls_ids = boxes.cls.cpu().numpy()     # (N,)

        frame_h, frame_w = frame.image.shape[:2]

        for i in range(len(xyxy)):
            cls_id = int(cls_ids[i])
            if cls_id != self._config.person_class_id:
                continue  # extra safety filter

            x1, y1, x2, y2 = xyxy[i]

            # Clamp to frame boundaries.
            x1 = float(max(0.0, min(x1, frame_w)))
            y1 = float(max(0.0, min(y1, frame_h)))
            x2 = float(max(0.0, min(x2, frame_w)))
            y2 = float(max(0.0, min(y2, frame_h)))

            # Skip degenerate boxes.
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                PersonDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(confs[i]),
                    class_id=cls_id,
                )
            )

        # Deterministic ordering: confidence descending.
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _parse_roi_results(
        self,
        results,
        frame: Frame,
        roi_x1: int,
        roi_y1: int,
        roi_x2: int,
        roi_y2: int,
    ) -> list[PersonDetection]:
        """Parse detections from an ROI crop and map back to original frame.

        YOLO returns coordinates relative to the crop.  We add the ROI
        offset to translate back to original-frame space, then clamp.
        """
        detections: list[PersonDetection] = []

        if not results or len(results) == 0:
            return detections

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()

        frame_h, frame_w = frame.image.shape[:2]

        for i in range(len(xyxy)):
            cls_id = int(cls_ids[i])
            if cls_id != self._config.person_class_id:
                continue

            # ROI-local coordinates → original-frame coordinates
            x1 = float(xyxy[i][0]) + roi_x1
            y1 = float(xyxy[i][1]) + roi_y1
            x2 = float(xyxy[i][2]) + roi_x1
            y2 = float(xyxy[i][3]) + roi_y1

            # Clamp to frame boundaries
            x1 = max(0.0, min(x1, frame_w))
            y1 = max(0.0, min(y1, frame_h))
            x2 = max(0.0, min(x2, frame_w))
            y2 = max(0.0, min(y2, frame_h))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                PersonDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(confs[i]),
                    class_id=cls_id,
                )
            )

        return detections

    # ------------------------------------------------------------------
    # Detection fusion
    # ------------------------------------------------------------------

    def _fuse_detections(
        self,
        global_dets: list[PersonDetection],
        roi_dets: list[PersonDetection],
    ) -> list[PersonDetection]:
        """Merge global and ROI detections, suppressing duplicates.

        For each ROI detection, check IoU against every global detection.
        If IoU > fusion_threshold, the ROI detection is a duplicate of an
        already-found global detection → keep the one with higher confidence.
        Otherwise, the ROI detection is a new recovery → add it.

        Returns a single list sorted by confidence descending.
        """
        if not roi_dets:
            return global_dets

        threshold = self._config.fusion_iou_threshold
        fused = list(global_dets)  # start with all global detections

        for roi_det in roi_dets:
            is_duplicate = False
            best_match_idx = -1
            best_iou = 0.0

            for j, g_det in enumerate(fused):
                iou = self._compute_iou(roi_det.bbox, g_det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = j

            if best_iou > threshold and best_match_idx >= 0:
                # Duplicate — keep higher confidence version
                if roi_det.confidence > fused[best_match_idx].confidence:
                    fused[best_match_idx] = roi_det
                is_duplicate = True

            if not is_duplicate:
                # New recovery from ROI pass
                fused.append(roi_det)

        # Re-sort by confidence descending
        fused.sort(key=lambda d: d.confidence, reverse=True)
        return fused

    @staticmethod
    def _compute_iou(
        box_a: tuple[float, float, float, float],
        box_b: tuple[float, float, float, float],
    ) -> float:
        """Compute intersection-over-union between two (x1,y1,x2,y2) boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter == 0.0:
            return 0.0

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_frame(frame: Frame) -> None:
        """Raise :class:`InvalidFrameError` if the frame is unusable."""
        if frame is None:
            raise InvalidFrameError("Frame is None.")
        if not isinstance(frame.image, np.ndarray):
            raise InvalidFrameError(
                f"Frame image must be a numpy array, got {type(frame.image).__name__}."
            )
        if frame.image.size == 0:
            raise InvalidFrameError("Frame image is empty (size=0).")
        if frame.image.ndim < 2:
            raise InvalidFrameError(
                f"Frame image must be 2D or 3D, got ndim={frame.image.ndim}."
            )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PersonDetector("
            f"model={self._config.model_path!r}, "
            f"device={self._config.device!r}, "
            f"scene={self._config.scene_profile!r}, "
            f"loaded={self._is_loaded})"
        )

