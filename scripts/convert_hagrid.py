"""
HaGRID 标注转换脚本 - Convert HaGRID Annotations to Feature Vectors
从 HaGRID 标注 JSON (含 MediaPipe landmarks) 提取特征，生成训练数据
"""

import json
import sys
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import FeatureExtractor


# 手势映射: HaGRID 标签 → 我们的标签(索引)
GESTURE_MAP = {
    "one":     0,   # 数字1
    "two_up":  1,   # 数字2
    "three":   2,   # 数字3
    "four":    3,   # 数字4
    "palm":    4,   # 数字5
    "ok":      5,   # OK手势
    "like":    6,   # 点赞手势
    "fist":    7,   # 握拳手势
}

# 手势名称
GESTURE_NAMES = {
    0: "1", 1: "2", 2: "3", 3: "4",
    4: "5", 5: "OK", 6: "Good", 7: "Fist",
}


def parse_args():
    parser = argparse.ArgumentParser(description="HaGRID标注转特征向量")
    parser.add_argument("--annotations", type=str, default="data/hagrid_annotations/annotations",
                        help="标注目录 (含 train/ val/ test/ 子目录)")
    parser.add_argument("--output", type=str, default="data/processed",
                        help="输出目录")
    parser.add_argument("--feature", type=str, default="all",
                        choices=["distance", "angle", "all"],
                        help="特征集类型")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test", "all"],
                        help="使用哪个数据集划分")
    parser.add_argument("--max-per-class", type=int, default=0,
                        help="每类最多样本数 (0=全部)")
    return parser.parse_args()


def load_landmarks_from_json(json_path: str, target_label: str, max_samples: int = 0):
    """
    从HaGRID标注JSON中提取指定手势的landmarks

    Args:
        json_path: JSON标注文件路径
        target_label: 目标手势标签 (如 "one", "fist")
        max_samples: 最多提取样本数 (0=全部)

    Returns:
        landmarks_list: [(21, 3) ndarray, ...] 的列表
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    landmarks_list = []
    for img_id, ann in data.items():
        labels = ann.get("labels", [])
        hand_landmarks = ann.get("hand_landmarks", [])

        if not labels or not hand_landmarks:
            continue

        # 找到目标手势对应的手
        for i, label in enumerate(labels):
            if label == target_label and i < len(hand_landmarks):
                lm_2d = hand_landmarks[i]
                if len(lm_2d) != 21:
                    continue
                # 转为 (21, 3) 格式: (x, y, z=0)
                lm_3d = np.array([[p[0], p[1], 0.0] for p in lm_2d], dtype=np.float32)
                landmarks_list.append(lm_3d)
                break  # 只取第一个匹配的手

        if max_samples > 0 and len(landmarks_list) >= max_samples:
            break

    return landmarks_list


def main():
    args = parse_args()
    ann_dir = Path(args.annotations)
    extractor = FeatureExtractor()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定要处理的划分
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    # 如果 split 是 "all"，合并所有划分
    if args.split == "all":
        splits = ["train", "val", "test"]

    X_all = []
    y_all = []
    stats = {}

    for split_name in splits:
        split_dir = ann_dir / split_name
        if not split_dir.exists():
            print(f"[WARN] 划分目录不存在: {split_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"处理划分: {split_name}")
        print(f"{'='*60}")

        for hagrid_label, our_label in GESTURE_MAP.items():
            json_path = split_dir / f"{hagrid_label}.json"
            if not json_path.exists():
                print(f"  [SKIP] {json_path} 不存在")
                continue

            print(f"  提取 '{hagrid_label}' → '{GESTURE_NAMES[our_label]}' (label={our_label})...", end=" ")
            landmarks_list = load_landmarks_from_json(
                str(json_path), hagrid_label, max_samples=args.max_per_class
            )

            n_features = 0
            for lm in landmarks_list:
                try:
                    features = extractor.extract(lm.tolist(), args.feature)
                    X_all.append(features)
                    y_all.append(our_label)
                    n_features += 1
                except Exception as e:
                    continue  # 跳过提取失败的样本

            stats[f"{split_name}/{hagrid_label}"] = n_features
            print(f"{n_features} 个样本")

    if len(X_all) == 0:
        print("[ERROR] 未提取到任何特征")
        return

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)

    print(f"\n{'='*60}")
    print(f"转换完成")
    print(f"{'='*60}")
    print(f"总样本: {X.shape[0]}")
    print(f"特征维度: {X.shape[1]}")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 保存
    feature_name = args.feature
    x_path = output_dir / f"X_{feature_name}.npy"
    y_path = output_dir / f"y_{feature_name}.npy"
    np.save(x_path, X)
    np.save(y_path, y)
    print(f"\n特征已保存: {x_path}")
    print(f"标签已保存: {y_path}")

    # 保存标签映射
    label_map = {v: k for k, v in GESTURE_MAP.items()}
    import pickle
    map_path = output_dir / "label_map.pkl"
    with open(map_path, 'wb') as f:
        pickle.dump(GESTURE_NAMES, f)
    print(f"标签映射已保存: {map_path}")


if __name__ == "__main__":
    main()
