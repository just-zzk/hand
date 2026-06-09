"""
KNN分类器模块 - KNN Classifier Module
基于特征向量进行手势分类
"""

import pickle
import numpy as np
from typing import List, Tuple, Optional
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class KNNClassifier:
    """
    KNN 手势分类器

    支持:
    - 训练: fit(X, y)
    - 预测: predict(X)
    - 模型持久化: save / load
    """

    # Gesture label mapping (English for OpenCV display compatibility)
    GESTURE_LABELS = {
        0: "1",
        1: "2",
        2: "3",
        3: "4",
        4: "5",
        5: "OK",
        6: "Good",
        7: "Fist",
    }

    def __init__(self, k: int = 5):
        """
        初始化KNN分类器

        Args:
            k: 近邻数量
        """
        self.k = k
        self.scaler = StandardScaler()
        self.classifier = KNeighborsClassifier(
            n_neighbors=k,
            weights='distance',
            metric='euclidean',
        )
        self._trained = False

    @property
    def trained(self) -> bool:
        return self._trained

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练分类器

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签数组 (n_samples,)
        """
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)
        self._trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测手势类别

        Args:
            X: 特征向量 (n_samples, n_features) 或 (n_features,)

        Returns:
            预测的标签
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回各类别的预测概率"""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)

    def predict_with_confidence(self, X: np.ndarray) -> Tuple[int, float]:
        """
        预测并返回置信度

        Returns:
            (label, confidence)
        """
        probs = self.predict_proba(X)[0]
        label = self.classifier.classes_[np.argmax(probs)]
        confidence = np.max(probs)
        return int(label), float(confidence)

    def get_label_name(self, label: int) -> str:
        """Get gesture display name"""
        return self.GESTURE_LABELS.get(label, f"Unknown({label})")

    def save(self, path: str):
        """保存模型到文件"""
        data = {
            'k': self.k,
            'scaler': self.scaler,
            'classifier': self.classifier,
            'trained': self._trained,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """从文件加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.k = data['k']
        self.scaler = data['scaler']
        self.classifier = data['classifier']
        self._trained = data['trained']
