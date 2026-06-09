"""
树莓派一键部署脚本 - Raspberry Pi Deployment Script
将项目最小运行集打包，复制到树莓派即可运行
Usage: python scripts/deploy_pi.py --output pi_deploy/
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


# ---- 需要复制到树莓派的文件清单 ----
DEPLOY_FILES = [
    # 源代码
    "src/__init__.py",
    "src/main.py",
    "src/capture/__init__.py",
    "src/capture/camera.py",
    "src/detection/__init__.py",
    "src/detection/hand_detector.py",
    "src/features/__init__.py",
    "src/features/extractor.py",
    "src/classifier/__init__.py",
    "src/classifier/knn_classifier.py",
    "src/control/__init__.py",
    "src/control/gpio_control.py",
    # 模型文件
    "data/models/knn_model.pkl",
    "data/models/label_map.pkl",
    # MediaPipe 模型
    "assets/mediapipe/hand_landmarker.task",
    # 配置文件
    "config/config.yaml",
    # 依赖清单
    "requirements.txt",
]

# ---- 树莓派专用 requirements ----
PI_REQUIREMENTS = """# Raspberry Pi Gesture Recognition - Dependencies
# Install: pip install -r requirements-pi.txt

opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
scikit-learn>=1.3.0
PyYAML>=6.0
# matplotlib>=3.7.0        # Pi上不需要（训练用）
RPi.GPIO>=0.7.1             # 树莓派 GPIO
"""

PI_README = """Gesture Recognition Control System - Raspberry Pi Deployment
================================================================

Quick Start:
  1. Install dependencies:
     pip install -r requirements-pi.txt
     (takes ~10-15 minutes on Pi 4)

  2. Test camera:
     python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

  3. Run (recognition only, no GPIO):
     python src/main.py --no-control

  4. Run (with GPIO control):
     python src/main.py

  5. Press 'q' to quit.

GPIO Wiring (BCM numbering):
  LED    → GPIO 17 → 220Ω resistor → LED → GND
  Fan    → GPIO 18 → transistor/MOSFET → Fan → 5V
  Buzzer → GPIO 27 → transistor → Buzzer → 3.3V

Performance Tips:
  - Pi 4 (4GB+): ~8-10 FPS with skip=2
  - Pi 3: ~4-6 FPS with skip=3
  - Lower camera resolution for better FPS:
    Edit config/config.yaml, set width: 320, height: 240
  - Use --skip 3 or --skip 4 flag for smoother experience
  - Close other GUI apps to free CPU
"""


def parse_args():
    parser = argparse.ArgumentParser(description="打包项目用于树莓派部署")
    parser.add_argument("--output", type=str, default="pi_deploy",
                        help="输出目录")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).parent.parent
    output = Path(args.output)

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    print(f"Packing project for Raspberry Pi → {output}/")
    print("=" * 50)

    # 1. 复制文件
    for rel_path in DEPLOY_FILES:
        src = root / rel_path
        dst = output / rel_path
        if not src.exists():
            print(f"  [SKIP] not found: {rel_path}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        size_kb = src.stat().st_size / 1024
        print(f"  COPY  {rel_path} ({size_kb:.0f} KB)")

    # 2. 写入树莓派专用文件
    (output / "requirements-pi.txt").write_text(PI_REQUIREMENTS, encoding="utf-8")
    (output / "README.md").write_text(PI_README, encoding="utf-8")
    print(f"  ADD   requirements-pi.txt")
    print(f"  ADD   README.md")

    # 3. 统计
    total_size = 0
    for f in output.rglob("*"):
        if f.is_file():
            total_size += f.stat().st_size

    print(f"\n{'='*50}")
    print(f"Package ready: {output}/")
    print(f"Total size: {total_size/1024:.0f} KB ({total_size/1024/1024:.1f} MB)")
    print(f"\nCopy to Raspberry Pi:")
    print(f"  scp -r {output}/* pi@raspberrypi:~/gesture-system/")
    print(f"\nOr use USB drive / network share.")


if __name__ == "__main__":
    main()
