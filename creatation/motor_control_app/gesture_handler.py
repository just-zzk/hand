"""
手势识别模块 - Gesture Recognition Handler
封装 MediaPipe + KNN 手势识别管线，供上位机 CameraThread 调用

Usage:
    handler = GestureHandler()
    handler.load_model("data/models/knn_model.pkl")

    # In camera loop:
    gesture_name, landmarks, confidence = handler.detect(frame)
    if gesture_name:
        frame = handler.draw(frame, landmarks, gesture_name, confidence)
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
from typing import Optional, List, Tuple

# 确保能找到 src
_MODEL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_MODEL_ROOT))

from src.detection import HandDetector
from src.features import FeatureExtractor
from src.classifier import KNNClassifier


# ─── 可视化配色常量 ──────────────────────────────────────
# Finger-specific colors (BGR for OpenCV drawing on RGB frame)
FINGER_COLORS_BGR = {
    "thumb":  (255, 140, 0),    # orange
    "index":  (70, 230, 70),    # green
    "middle": (255, 80, 50),    # blue
    "ring":   (180, 50, 255),   # purple
    "pinky":  (50, 160, 255),   # gold
    "wrist":  (50, 50, 255),    # red
}

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# Per-finger landmark ranges (MediaPipe topology)
FINGER_RANGES = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

# Skeleton connections (MediaPipe hand topology)
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (0, 9), (9, 10), (10, 11), (11, 12),       # middle
    (0, 13), (13, 14), (14, 15), (15, 16),     # ring
    (0, 17), (17, 18), (18, 19), (19, 20),     # pinky
    (5, 9), (9, 13), (13, 17),                  # palm arches
]

# Which finger a connection belongs to (for coloring)
CONNECTION_FINGER = {
    (0, 1): "thumb", (1, 2): "thumb", (2, 3): "thumb", (3, 4): "thumb",
    (0, 5): "index", (5, 6): "index", (6, 7): "index", (7, 8): "index",
    (0, 9): "middle", (9, 10): "middle", (10, 11): "middle", (11, 12): "middle",
    (0, 13): "ring", (13, 14): "ring", (14, 15): "ring", (15, 16): "ring",
    (0, 17): "pinky", (17, 18): "pinky", (18, 19): "pinky", (19, 20): "pinky",
    (5, 9): "palm", (9, 13): "palm", (13, 17): "palm",
}

PALM_COLOR = (180, 180, 180)  # light gray (BGR)


class GestureHandler:
    """手势识别处理器"""

    def __init__(self):
        self.detector: Optional[HandDetector] = None
        self.extractor: Optional[FeatureExtractor] = None
        self.classifier: Optional[KNNClassifier] = None
        self._model_loaded = False
        self._enabled = False

    # ─── 模型加载 ──────────────────────────────────────

    def load_model(self, model_path: str = None):
        """加载 MediaPipe 模型 + KNN 分类器"""
        if model_path is None:
            model_path = str(_MODEL_ROOT / "data" / "models" / "knn_model.pkl")

        if not os.path.exists(model_path):
            print(f"[Gesture] Model not found: {model_path}")
            return False

        try:
            self.detector = HandDetector(
                max_num_hands=1,
                min_detection_confidence=0.7,
            )
            self.extractor = FeatureExtractor()
            self.classifier = KNNClassifier(k=3)
            self.classifier.load(model_path)
            self._model_loaded = True
            print(f"[Gesture] Model loaded: {model_path}")
            return True
        except Exception as e:
            print(f"[Gesture] Model load failed: {e}")
            self._model_loaded = False
            return False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        if value and not self._model_loaded:
            self.load_model()
        self._enabled = value and self._model_loaded

    # ─── 检测 ────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> Tuple[Optional[str], Optional[List], float]:
        """
        检测手势

        Args:
            frame: BGR image (numpy array)

        Returns:
            (gesture_name, landmarks, confidence) — gesture_name 为 None 表示未检测到
        """
        if not self._enabled or not self._model_loaded:
            return None, None, 0.0

        try:
            detected, landmarks = self.detector.detect(frame)
            if not detected or landmarks is None:
                return None, None, 0.0

            features = self.extractor.extract(landmarks, "all")
            label, confidence = self.classifier.predict_with_confidence(features)

            # 拇指角度兜底
            if label == 4:  # "5"
                thumb_mcp = self.extractor.joint_angle(
                    landmarks[4], landmarks[2], landmarks[1])
                if thumb_mcp < 140.0:
                    label = 3  # force "4"

            gesture_name = self.classifier.get_label_name(label)
            return gesture_name, landmarks, confidence

        except Exception as e:
            print(f"[Gesture] Detection error: {e}")
            return None, None, 0.0

    # ─── 绘制（增强版） ────────────────────────────────

    def _get_finger_for_landmark(self, idx: int) -> Optional[str]:
        """Return which finger a landmark index belongs to."""
        if idx == 0:
            return "wrist"
        for name, indices in FINGER_RANGES.items():
            if idx in indices:
                return name
        return None

    def _draw_glow_circle(self, frame, cx, cy, radius, color, glow_radius=0,
                          glow_alpha=0.3):
        """Draw a circle with optional outer glow."""
        if glow_radius > 0:
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), glow_radius, color, -1)
            cv2.addWeighted(overlay, glow_alpha, frame, 1 - glow_alpha, 0, frame)
        cv2.circle(frame, (cx, cy), radius, color, -1)

    def draw(self, frame: np.ndarray, landmarks: List,
             gesture_name: str = None, confidence: float = 0.0) -> np.ndarray:
        """
        增强版绘制：关键节点 + 骨架连线 + 手势名称 + 置信度 + 手部边界框

        Args:
            frame: RGB image (numpy array)
            landmarks: 21 keypoints [(x, y, z), ...] normalized 0~1
            gesture_name: recognized gesture label
            confidence: recognition confidence (0~1)

        Returns:
            annotated frame (RGB)
        """
        h, w = frame.shape[:2]

        # Convert normalized coords to pixel
        pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks]

        # ── 1. Draw hand bounding box (subtle) ──
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        margin = 20
        x1, y1 = max(0, min(xs) - margin), max(0, min(ys) - margin)
        x2, y2 = min(w, max(xs) + margin), min(h, max(ys) + margin)

        # Rounded rectangle approximation (draw corners)
        corner_len = 15
        corner_color = (120, 120, 140)
        cv2.line(frame, (x1, y1 + corner_len), (x1, y1 + 5), corner_color, 1)
        cv2.line(frame, (x1, y1 + 5), (x1 + 5, y1), corner_color, 1)
        cv2.line(frame, (x2, y1 + corner_len), (x2, y1 + 5), corner_color, 1)
        cv2.line(frame, (x2, y1 + 5), (x2 - 5, y1), corner_color, 1)
        cv2.line(frame, (x1, y2 - corner_len), (x1, y2 - 5), corner_color, 1)
        cv2.line(frame, (x1, y2 - 5), (x1 + 5, y2), corner_color, 1)
        cv2.line(frame, (x2, y2 - corner_len), (x2, y2 - 5), corner_color, 1)
        cv2.line(frame, (x2, y2 - 5), (x2 - 5, y2), corner_color, 1)

        # ── 2. Draw skeleton connections ──
        for (a, b) in SKELETON_CONNECTIONS:
            finger = CONNECTION_FINGER.get((a, b), "palm")
            color = FINGER_COLORS_BGR.get(finger, PALM_COLOR)

            # Palm connections thinner/lighter
            if finger == "palm":
                cv2.line(frame, pts[a], pts[b], PALM_COLOR, 1, cv2.LINE_AA)
            else:
                cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)

        # ── 3. Draw keypoints with glow ──
        for i, (cx, cy) in enumerate(pts):
            finger = self._get_finger_for_landmark(i)
            color = FINGER_COLORS_BGR.get(finger, (200, 200, 200))

            # Wrist: bigger
            if i == 0:
                self._draw_glow_circle(frame, cx, cy, 6, color,
                                       glow_radius=14, glow_alpha=0.15)
                cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1, cv2.LINE_AA)

            # Fingertips: highlighted
            elif i in (4, 8, 12, 16, 20):
                self._draw_glow_circle(frame, cx, cy, 5, color,
                                       glow_radius=12, glow_alpha=0.20)
                cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1, cv2.LINE_AA)

            # Articulations
            else:
                cv2.circle(frame, (cx, cy), 3, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), 1, cv2.LINE_AA)

        # ── 4. Gesture info panel (bottom-left, semi-transparent) ──
        if gesture_name:
            self._draw_gesture_panel(frame, gesture_name, confidence, w, h)

        return frame

    def _draw_gesture_panel(self, frame, gesture_name, confidence, w, h):
        """绘制半透明的识别结果信息面板"""
        # Panel dimensions
        panel_w = 320
        panel_h = 110
        panel_x = 10
        panel_y = h - panel_h - 10

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Panel border with accent
        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      (80, 200, 120), 1)

        # Left accent bar
        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + 4, panel_y + panel_h),
                      (80, 200, 120), -1)

        # Gesture name (large)
        cv2.putText(frame, gesture_name,
                    (panel_x + 18, panel_y + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 128), 3, cv2.LINE_AA)

        # Confidence bar background
        bar_x = panel_x + 18
        bar_y = panel_y + 62
        bar_w = panel_w - 36
        bar_h = 8
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h),
                      (60, 60, 80), -1)

        # Confidence bar fill (color based on confidence level)
        fill_w = int(bar_w * confidence)
        if confidence >= 0.8:
            bar_color = (80, 220, 120)  # green
        elif confidence >= 0.5:
            bar_color = (0, 200, 255)   # gold
        else:
            bar_color = (80, 80, 255)   # red
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h),
                      bar_color, -1)

        # Confidence text
        cv2.putText(frame, f"Confidence: {confidence:.1%}",
                    (panel_x + 18, panel_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 210), 1, cv2.LINE_AA)

        # Gesture label on the right side of the panel
        label_map = {
            "1": "Index Up", "2": "Two Fingers", "3": "Three Fingers",
            "4": "Four Fingers", "5": "Five Fingers",
            "OK": "OK Sign", "Good": "Thumbs Up", "Fist": "Fist",
        }
        english_label = label_map.get(gesture_name, "")
        if english_label:
            cv2.putText(frame, english_label,
                        (panel_x + 18, panel_y + 102),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 170), 1, cv2.LINE_AA)

    # ─── 资源释放 ────────────────────────────────────

    def close(self):
        if self.detector:
            self.detector.close()
