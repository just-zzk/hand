"""
导出数据集手部骨架图 - Export Hand Skeleton Images from HaGRID Landmarks
从 HaGRID 标注的 21 个关键点用 matplotlib 绘制手部骨架
每类约 63 张，总计 ~500 张，保存到 picture/
"""

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# MediaPipe Hands 骨架连线 (21点之间的边)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),       # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
    (0, 13), (13, 14), (14, 15), (15, 16),# 无名指
    (0, 17), (17, 18), (18, 19), (19, 20),# 小指
    (5, 9), (9, 13), (13, 17),            # 手掌横向连线
]

JOINT_COLORS = [
    "#FF0000",  # 0  wrist  - 红
    "#FF9800", "#FF9800", "#FF9800", "#FF9800",  # 1-4  拇指 - 橙
    "#4CAF50", "#4CAF50", "#4CAF50", "#4CAF50",  # 5-8  食指 - 绿
    "#2196F3", "#2196F3", "#2196F3", "#2196F3",  # 9-12 中指 - 蓝
    "#9C27B0", "#9C27B0", "#9C27B0", "#9C27B0",  # 13-16 无名指 - 紫
    "#FF5722", "#FF5722", "#FF5722", "#FF5722",  # 17-20 小指 - 深橙
]

GESTURE_MAP = {
    "one": "1", "two_up": "2", "three": "3", "four": "4",
    "palm": "5", "ok": "OK", "like": "Good", "fist": "Fist",
}

TARGETS = ["one", "two_up", "three", "four", "palm", "ok", "like", "fist"]
PER_CLASS = 63  # 每类导出数量 → 8×63=504 ≈ 500


def parse_args():
    parser = argparse.ArgumentParser(description="导出HaGRID手势骨架图")
    parser.add_argument("--annotations", type=str,
                        default="data/hagrid_annotations/annotations",
                        help="标注目录")
    parser.add_argument("--output", type=str, default="picture",
                        help="输出目录")
    parser.add_argument("--per-class", type=int, default=PER_CLASS,
                        help="每类导出张数")
    return parser.parse_args()


def draw_hand_skeleton(landmarks_2d, gesture_name, img_id, output_dir):
    """
    用 matplotlib 绘制手部骨架

    Args:
        landmarks_2d: 21个 (x, y) 归一化坐标
        gesture_name: 手势显示名 (如 "1", "OK")
        img_id: 图片标识
        output_dir: 输出目录
    """
    # 坐标转换: MediaPipe (0,0)在左上 → matplotlib (0,0)在左下
    points = np.array([(p[0], 1.0 - p[1]) for p in landmarks_2d])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")

    # 画连线
    for start, end in HAND_CONNECTIONS:
        ax.plot([points[start][0], points[end][0]],
                [points[start][1], points[end][1]],
                color="#BDBDBD", linewidth=2.5, alpha=0.7, zorder=1)

    # 画关键点
    for i, (x, y) in enumerate(points):
        ax.scatter(x, y, s=80, color=JOINT_COLORS[i], edgecolors="white",
                   linewidth=1.5, zorder=2)

    ax.set_title(f"Gesture: {gesture_name}", fontsize=16, fontweight="bold",
                 color="#333333", pad=10)

    filepath = output_dir / gesture_name / f"{img_id}.png"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    ann_dir = Path(args.annotations)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for hagrid_label in TARGETS:
        gesture_name = GESTURE_MAP[hagrid_label]
        json_path = ann_dir / "train" / f"{hagrid_label}.json"

        if not json_path.exists():
            print(f"  [SKIP] {json_path} not found")
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        print(f"  {hagrid_label:10s} -> {gesture_name:6s}", end=" ", flush=True)

        count = 0
        for img_id, ann in data.items():
            labels = ann.get("labels", [])
            hand_landmarks = ann.get("hand_landmarks", [])

            for i, label in enumerate(labels):
                if label == hagrid_label and i < len(hand_landmarks):
                    lm = hand_landmarks[i]
                    if len(lm) == 21:
                        draw_hand_skeleton(lm, gesture_name, img_id[:8], output_dir)
                        count += 1
                        total += 1
                    break

            if count >= args.per_class:
                break

        print(f"({count} images)")

    print(f"\nTotal: {total} skeleton images saved to {output_dir}/")


if __name__ == "__main__":
    main()
