"""
基于 HaGRID 标注的训练脚本 - Train with HaGRID Annotations
直接从 HaGRID landmarks 提取特征，训练 KNN 分类器
自动生成可视化图表到 graph/ 目录
"""

import sys
import time
import json
import pickle
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无头模式，不弹窗
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import FeatureExtractor
from src.classifier import KNNClassifier


# Gesture mapping: HaGRID label → class index
GESTURE_MAP = {
    "one": 0, "two_up": 1, "three": 2, "four": 3,
    "palm": 4, "ok": 5, "like": 6, "fist": 7,
}

GESTURE_NAMES = {
    0: "1", 1: "2", 2: "3", 3: "4",
    4: "5", 5: "OK", 6: "Good", 7: "Fist",
}


def parse_args():
    parser = argparse.ArgumentParser(description="从HaGRID landmarks训练KNN")
    parser.add_argument("--annotations", type=str,
                        default="data/hagrid_annotations/annotations",
                        help="标注目录")
    parser.add_argument("--output", type=str, default="data/models/knn_model.pkl",
                        help="模型输出路径")
    parser.add_argument("--k", type=int, default=5, help="KNN的K值")
    parser.add_argument("--feature", type=str, default="all",
                        choices=["distance", "angle", "all"],
                        help="特征集类型")
    parser.add_argument("--max-per-class", type=int, default=3000,
                        help="每类最多训练样本数")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="测试集比例")
    parser.add_argument("--graph-dir", type=str, default="graph",
                        help="可视化图表输出目录")
    return parser.parse_args()


# ---------- 版本管理 ----------

def get_version(project_root: Path) -> int:
    """读取当前版本号，不存在则从1开始"""
    ver_file = project_root / "data" / "models" / "version.txt"
    if ver_file.exists():
        return int(ver_file.read_text().strip())
    return 1


def bump_version(project_root: Path) -> int:
    """版本号+1并持久化"""
    v = get_version(project_root) + 1
    ver_file = project_root / "data" / "models" / "version.txt"
    ver_file.parent.mkdir(parents=True, exist_ok=True)
    ver_file.write_text(str(v))
    return v


# ---------- 可视化函数 ----------

def plot_confusion_matrix(cm, label_names, version, graph_dir, model_k):
    """混淆矩阵热力图"""
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues")

    # 标注数字
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=11)

    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, fontsize=12)
    ax.set_yticklabels(label_names, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)
    ax.set_title(f"Confusion Matrix  |  K={model_k}  |  v{version}", fontsize=14, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.82)
    plt.tight_layout()
    path = Path(graph_dir) / f"confusion_matrix_v{version}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [GRAPH] {path}")


def plot_k_accuracy(k_values, accuracies, best_k, version, graph_dir):
    """K值-准确率曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, accuracies, "o-", color="#2196F3", linewidth=2, markersize=8, zorder=3)
    ax.axvline(x=best_k, color="red", linestyle="--", alpha=0.7, label=f"Best K={best_k}")
    ax.set_xlabel("K (Number of Neighbors)", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title(f"K-Value vs Accuracy  |  v{version}", fontsize=14, fontweight="bold")
    ax.set_xticks(k_values)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))

    # 标注数值
    for k, a in zip(k_values, accuracies):
        ax.annotate(f"{a:.3f}", (k, a), textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=9)

    plt.tight_layout()
    path = Path(graph_dir) / f"k_accuracy_v{version}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [GRAPH] {path}")


def plot_class_metrics(report_dict, label_names, version, graph_dir):
    """各类别 Precision / Recall / F1 柱状图"""
    precision = [report_dict[name]["precision"] for name in label_names]
    recall = [report_dict[name]["recall"] for name in label_names]
    f1 = [report_dict[name]["f1-score"] for name in label_names]

    x = np.arange(len(label_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width, precision, width, label="Precision", color="#4CAF50", edgecolor="white")
    bars2 = ax.bar(x, recall, width, label="Recall", color="#2196F3", edgecolor="white")
    bars3 = ax.bar(x + width, f1, width, label="F1-Score", color="#FF9800", edgecolor="white")

    ax.set_xlabel("Gesture Class", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title(f"Per-Class Metrics  |  v{version}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, fontsize=12)
    ax.set_ylim(0.90, 1.005)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

    plt.tight_layout()
    path = Path(graph_dir) / f"class_metrics_v{version}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [GRAPH] {path}")


def plot_cv_scores(cv_scores, version, graph_dir):
    """5折交叉验证分数"""
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot([cv_scores], patch_artist=True, widths=0.4,
                     boxprops=dict(facecolor="#BBDEFB", edgecolor="#1976D2", linewidth=2),
                     medianprops=dict(color="red", linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))

    # 叠加散点
    x_jitter = np.random.normal(1, 0.04, len(cv_scores))
    ax.scatter(x_jitter, cv_scores, color="#1565C0", alpha=0.7, s=60, zorder=3)
    ax.set_xticklabels(["5-Fold CV"], fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title(f"Cross-Validation Scores  |  v{version}\n"
                 f"Mean={cv_scores.mean():.4f}  Std={cv_scores.std():.4f}",
                 fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = Path(graph_dir) / f"cv_scores_v{version}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [GRAPH] {path}")


# ---------- 特征提取 ----------

def extract_features_from_json(json_path, target_label, extractor, feature_set, max_samples=0):
    """从JSON标注提取特征"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    features_list = []
    for img_id, ann in data.items():
        labels = ann.get("labels", [])
        hand_landmarks = ann.get("hand_landmarks", [])

        if not labels or not hand_landmarks:
            continue

        for i, label in enumerate(labels):
            if label == target_label and i < len(hand_landmarks):
                lm_2d = hand_landmarks[i]
                if len(lm_2d) != 21:
                    continue
                lm_3d = [(p[0], p[1], 0.0) for p in lm_2d]
                try:
                    feat = extractor.extract(lm_3d, feature_set)
                    features_list.append(feat)
                except Exception:
                    continue
                break

        if max_samples > 0 and len(features_list) >= max_samples:
            break

    return features_list


def main():
    args = parse_args()
    ann_dir = Path(args.annotations)
    extractor = FeatureExtractor()

    # 版本管理
    project_root = Path(__file__).parent.parent
    version = bump_version(project_root)

    print("=" * 60)
    print(f"HaGRID → KNN Gesture Classifier Training  |  v{version}")
    print("=" * 60)

    # 1. 从训练集提取特征
    print("\n[1/4] Extracting features from training set...")
    X_list, y_list = [], []
    stats = {}

    for hagrid_label, our_label in GESTURE_MAP.items():
        json_path = ann_dir / "train" / f"{hagrid_label}.json"
        if not json_path.exists():
            print(f"  [WARN] {json_path} not found, skip")
            continue

        print(f"  {hagrid_label:10s} -> {GESTURE_NAMES[our_label]}", end=" ", flush=True)
        features = extract_features_from_json(
            json_path, hagrid_label, extractor, args.feature,
            max_samples=args.max_per_class
        )

        for f in features:
            X_list.append(f)
            y_list.append(our_label)
        stats[hagrid_label] = len(features)
        print(f"({len(features)} samples)")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"\n  Total: {X.shape[0]} samples, {X.shape[1]} features")

    # 2. 划分训练/测试集
    print(f"\n[2/4] Train/test split (test_size={args.test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    # 3. 训练模型
    print(f"\n[3/4] Training KNN (K={args.k}, feature={args.feature})...")
    start = time.time()
    classifier = KNNClassifier(k=args.k)
    classifier.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"  Training time: {train_time:.3f}s")

    # 4. 评估
    print(f"\n[4/4] Model Evaluation")
    print("=" * 60)

    y_pred = classifier.predict(X_test)
    if len(y_pred.shape) == 0:
        y_pred = np.array([y_pred])

    label_names = [GESTURE_NAMES[i] for i in range(len(GESTURE_MAP))]
    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    header = "        " + " ".join(f"{name:>6s}" for name in label_names)
    print(header)
    for i, name in enumerate(label_names):
        row = " ".join(f"{cm[i][j]:6d}" for j in range(len(GESTURE_MAP)))
        print(f"{name:>6s}  {row}")

    # K值实验
    print("\nK-Value Comparison:")
    k_values = [1, 3, 5, 7, 9, 11, 15]
    accuracies = []
    best_k, best_acc = args.k, 0
    for k_test in k_values:
        clf = KNNClassifier(k=k_test)
        clf.fit(X_train, y_train)
        acc = clf.classifier.score(clf.scaler.transform(X_test), y_test)
        accuracies.append(acc)
        marker = " <--" if k_test == args.k else ""
        print(f"  K={k_test:<3}  Accuracy={acc:.4f}{marker}")
        if acc > best_acc:
            best_acc = acc
            best_k = k_test
    print(f"\nBest K: {best_k} (Accuracy={best_acc:.4f})")

    # 交叉验证
    print("\nCross-Validation:")
    cv_scores = None
    try:
        cv_scores = cross_val_score(
            classifier.classifier, classifier.scaler.transform(X), y, cv=5
        )
        print(f"  5-Fold CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Individual folds: {[f'{s:.4f}' for s in cv_scores]}")
    except Exception as e:
        print(f"  Skipped: {e}")

    # ---------- Generate Visualizations ----------
    print("\n" + "=" * 60)
    print("Generating Graphs")
    print("=" * 60)

    graph_dir = Path(args.graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)

    # 1. 混淆矩阵热力图
    plot_confusion_matrix(cm, label_names, version, graph_dir, args.k)

    # 2. K值-准确率曲线
    plot_k_accuracy(k_values, accuracies, best_k, version, graph_dir)

    # 3. 各类别 Precision/Recall/F1
    report_dict = classification_report(y_test, y_pred, target_names=label_names,
                                         zero_division=0, output_dict=True)
    plot_class_metrics(report_dict, label_names, version, graph_dir)

    # 4. 交叉验证图
    if cv_scores is not None:
        plot_cv_scores(cv_scores, version, graph_dir)

    # ---------- 保存模型 ----------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(str(output_path))
    print(f"\nModel saved: {output_path}")

    label_path = output_path.parent / "label_map.pkl"
    with open(label_path, 'wb') as f:
        pickle.dump(GESTURE_NAMES, f)
    print(f"Label map saved: {label_path}")
    print(f"Version: {version}")


if __name__ == "__main__":
    main()
