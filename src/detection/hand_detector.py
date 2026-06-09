"""
手部关键点检测模块 - Hand Landmark Detection Module
基于 MediaPipe Hands 提取21个手部关键点
"""

import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from typing import List, Optional, Tuple


class HandDetector:
    """MediaPipe 手部关键点检测器 (使用 mediapipe.tasks API)"""

    # MediaPipe Hands 21个关键点索引与名称
    LANDMARK_NAMES = {
        0: "WRIST",
        1: "THUMB_CMC", 2: "THUMB_MCP", 3: "THUMB_IP", 4: "THUMB_TIP",
        5: "INDEX_FINGER_MCP", 6: "INDEX_FINGER_PIP", 7: "INDEX_FINGER_DIP", 8: "INDEX_FINGER_TIP",
        9: "MIDDLE_FINGER_MCP", 10: "MIDDLE_FINGER_PIP", 11: "MIDDLE_FINGER_DIP", 12: "MIDDLE_FINGER_TIP",
        13: "RING_FINGER_MCP", 14: "RING_FINGER_PIP", 15: "RING_FINGER_DIP", 16: "RING_FINGER_TIP",
        17: "PINKY_MCP", 18: "PINKY_PIP", 19: "PINKY_DIP", 20: "PINKY_TIP",
    }

    # 默认模型路径（搜索多个可能位置）
    _MODEL_CANDIDATES = [
        os.path.join("assets", "mediapipe", "hand_landmarker.task"),
        os.path.join("assets", "mediapipe", "hand_landmark_lite.tflite"),
        os.path.join(os.path.dirname(__file__), "..", "..", "assets", "mediapipe", "hand_landmarker.task"),
        os.path.join(os.path.dirname(__file__), "..", "..", "assets", "mediapipe", "hand_landmark_lite.tflite"),
        "hand_landmarker.task",
        "hand_landmark_lite.tflite",
    ]

    @staticmethod
    def _find_model() -> str:
        for candidate in HandDetector._MODEL_CANDIDATES:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
        raise FileNotFoundError(
            "未找到 MediaPipe Hands 模型文件 (hand_landmark_lite.tflite)。"
            "请将模型文件放到项目根目录。"
        )

    def __init__(self,
                 static_image_mode: bool = False,
                 max_num_hands: int = 1,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        """
        初始化手部检测器

        Args:
            static_image_mode:      静态图片模式（图片为True，视频流为False）
            max_num_hands:          最大检测手数
            min_detection_confidence: 检测置信度阈值
            min_tracking_confidence:  跟踪置信度阈值
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        model_path = self._find_model()
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, image) -> Tuple[bool, Optional[List[Tuple[float, float, float]]]]:
        """
        检测手部关键点

        Args:
            image: BGR格式图像 (numpy array)

        Returns:
            (detected, landmarks): 
                detected - 是否检测到手部
                landmarks - 21个关键点列表 [(x, y, z), ...] 或 None
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            return False, None

        # 取第一只手的关键点
        hand_landmarks = result.hand_landmarks[0]
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]

        return True, landmarks

    def get_landmark(self, landmarks: List, index: int) -> Tuple[float, float, float]:
        """获取指定索引的关键点坐标"""
        return landmarks[index]

    def draw_landmarks(self, image, landmarks: List):
        """在图像上绘制关键点和连线"""
        h, w, _ = image.shape
        for i, (x, y, z) in enumerate(landmarks):
            cx, cy = int(x * w), int(y * h)
            cv2.circle(image, (cx, cy), 4, (0, 255, 0), -1)
            cv2.putText(image, str(i), (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    def close(self):
        """释放资源"""
        self.detector.close()
