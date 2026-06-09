"""
特征提取模块 - Feature Extraction Module
对 MediaPipe 输出的21个关键点进行归一化、距离特征和角度特征提取
"""

import math
import numpy as np
from typing import List, Tuple


class FeatureExtractor:
    """
    手势特征提取器

    提取四类特征:
    1. 指尖间距离特征 (10维 + 5维到腕)
    2. 关节弯曲角度特征 (10维)
    3. 手指开合状态 (4维, 距离比法, 兼容2D/3D)
    4. 拇指专用连续特征 (3维)
    """

    # 手指关节定义: (指尖, 远端指间关节, 近端指间关节, 掌指关节)
    FINGER_JOINTS = {
        "thumb":   [4, 3, 2, 1],    # 拇指: TIP, IP, MCP, CMC
        "index":   [8, 7, 6, 5],    # 食指
        "middle":  [12, 11, 10, 9], # 中指
        "ring":    [16, 15, 14, 13],# 无名指
        "pinky":   [20, 19, 18, 17],# 小指
    }

    # 指尖索引
    FINGERTIPS = [4, 8, 12, 16, 20]

    def __init__(self):
        pass

    # ---------- 归一化 ----------

    def normalize(self, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        以手腕为原点进行归一化，消除手部位置影响

        Args:
            landmarks: 21个关键点 [(x, y, z), ...]

        Returns:
            归一化后的 (21, 3) 数组
        """
        wrist = np.array(landmarks[0])
        normalized = np.array(landmarks) - wrist
        return normalized

    def normalize_scale(self, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        以手腕为原点，并以手腕到中指MCP距离进行尺度归一化

        Args:
            landmarks: 21个关键点

        Returns:
            归一化后的 (21, 3) 数组
        """
        wrist = np.array(landmarks[0])
        middle_mcp = np.array(landmarks[9])
        scale = np.linalg.norm(middle_mcp - wrist)
        if scale < 1e-6:
            scale = 1.0
        normalized = (np.array(landmarks) - wrist) / scale
        return normalized

    # ---------- 距离特征 ----------

    def fingertip_distances(self, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        计算五指指尖两两之间的距离

        Args:
            landmarks: 21个关键点

        Returns:
            10个距离值 (C(5,2) = 10)
        """
        tips = [landmarks[i] for i in self.FINGERTIPS]
        distances = []
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                d = math.dist(tips[i], tips[j])
                distances.append(d)
        return np.array(distances)

    def fingertip_to_wrist_distances(self, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        计算各指尖到手腕的距离

        Returns:
            5个距离值
        """
        wrist = landmarks[0]
        distances = []
        for i in self.FINGERTIPS:
            d = math.dist(landmarks[i], wrist)
            distances.append(d)
        return np.array(distances)

    # ---------- 角度特征 ----------

    def joint_angle(self, a: Tuple[float, float, float],
                     b: Tuple[float, float, float],
                     c: Tuple[float, float, float]) -> float:
        """
        计算以b为顶点的三点夹角 (a-b-c)

        Returns:
            角度（度）
        """
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return math.degrees(math.acos(cos_angle))

    def finger_angles(self, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        计算各手指的关节弯曲角度

        每根手指计算2个角度:
        - PIP角度 (指尖-DIP关节-PIP关节)
        - MCP角度 (DIP关节-PIP关节-MCP关节)

        Returns:
            10个角度值 (5根手指 x 2个关节)
        """
        angles = []
        for finger_name, joints in self.FINGER_JOINTS.items():
            # PIP角度: joints[0] - joints[1] - joints[2]
            ang_pip = self.joint_angle(
                landmarks[joints[0]],  # 指尖
                landmarks[joints[1]],  # DIP
                landmarks[joints[2]],  # PIP
            )
            # MCP角度: joints[1] - joints[2] - joints[3]
            ang_mcp = self.joint_angle(
                landmarks[joints[1]],  # DIP
                landmarks[joints[2]],  # PIP
                landmarks[joints[3]],  # MCP
            )
            angles.extend([ang_pip, ang_mcp])
        return np.array(angles)

    # ---------- 手指状态特征 (距离比法, 兼容2D/3D) ----------

    # MCP关节索引 (每个手指的掌指关节)
    FINGER_MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}

    # 距离比阈值: tip_to_wrist / mcp_to_wrist >= 此值 → open
    OPEN_RATIO_THRESHOLD = 0.85

    def _finger_open_ratio(self, tip_idx: int, mcp_idx: int,
                           landmarks: List[Tuple[float, float, float]]) -> float:
        """
        计算指尖-腕距 / MCP-腕距 的比值

        伸直时 tip 比 mcp 离腕更远 → 比值 > 1.0
        弯曲时 tip 靠近手腕 → 比值 < 0.7
        """
        wrist = np.array(landmarks[0])
        tip = np.array(landmarks[tip_idx])
        mcp = np.array(landmarks[mcp_idx])
        tip_dist = np.linalg.norm(tip - wrist)
        mcp_dist = np.linalg.norm(mcp - wrist)
        if mcp_dist < 1e-8:
            return 0.0
        return tip_dist / mcp_dist

    def finger_states(self, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        判断食指/中指/无名指/小指的张开/闭合状态（拇指由 thumb_features 独立处理）

        用 tip_to_wrist / MCP_to_wrist 距离比 (>0.85=open)
        兼容 2D(HaGRID训练) 和 3D(实时MediaPipe)

        Returns:
            [index, middle, ring, pinky]  (0.0=closed, 1.0=open) — 4维
        """
        states = []
        for finger_name in ["index", "middle", "ring", "pinky"]:
            joints = self.FINGER_JOINTS[finger_name]
            mcp_idx = self.FINGER_MCP[finger_name]
            ratio = self._finger_open_ratio(joints[0], mcp_idx, landmarks)
            states.append(1.0 if ratio >= self.OPEN_RATIO_THRESHOLD else 0.0)
        return np.array(states, dtype=np.float32)

    # ---------- 拇指专用特征 (距离比，兼容2D) ----------

    def thumb_features(self, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        拇指伸展连续特征（2D/3D通用，距离比法）

        Returns:
            [thumb_tip_to_index_mcp_ratio, thumb_tip_to_pinky_mcp_ratio, thumb_open_ratio]
            全部经过尺度归一化
        """
        wrist = np.array(landmarks[0])
        scale = np.linalg.norm(np.array(landmarks[9]) - wrist) + 1e-8

        thumb_tip = np.array(landmarks[4])
        index_mcp = np.array(landmarks[5])
        pinky_mcp = np.array(landmarks[17])

        dist_to_index = np.linalg.norm(thumb_tip - index_mcp) / scale
        dist_to_pinky = np.linalg.norm(thumb_tip - pinky_mcp) / scale

        # 拇指 open ratio (用相同距离比逻辑)
        thumb_ratio = self._finger_open_ratio(4, 2, landmarks)

        return np.array([dist_to_index, dist_to_pinky, thumb_ratio], dtype=np.float32)

    # ---------- 综合特征 ----------

    def extract(self, landmarks: List[Tuple[float, float, float]],
                feature_set: str = "all") -> np.ndarray:
        """
        提取综合特征向量

        Args:
            landmarks: 21个关键点
            feature_set: 特征集类型
                - "distance": 仅距离特征 (15维)
                - "angle":    仅角度特征 (10维)
                - "all":      全部特征 (32维: 距离15 + 角度10 + 手指状态4 + 拇指3)

        Returns:
            特征向量
        """
        normalized = self.normalize_scale(landmarks)
        features = []

        if feature_set in ("distance", "all"):
            tip_dist = self.fingertip_distances(landmarks)          # 10维
            wrist_dist = self.fingertip_to_wrist_distances(landmarks) # 5维
            features.extend(tip_dist.tolist())
            features.extend(wrist_dist.tolist())

        if feature_set in ("angle", "all"):
            angles = self.finger_angles(landmarks)       # 10维 (PIP+MCP × 5)
            finger_st = self.finger_states(landmarks)     # 5维 (open/closed × 5)
            thumb_f = self.thumb_features(landmarks)      # 3维 (thumb extension)
            features.extend(angles.tolist())
            features.extend(finger_st.tolist())
            features.extend(thumb_f.tolist())

        return np.array(features, dtype=np.float32)
