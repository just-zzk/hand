"""
Gesture Recognition Control System - Main Entry

Pipeline:
    Camera → Hand Detection → Feature Extraction → Gesture Classification → GPIO → Display
"""

import sys
import time
import argparse
from collections import Counter, deque
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capture import Camera
from src.detection import HandDetector
from src.features import FeatureExtractor
from src.classifier import KNNClassifier
from src.control import GPIOController


def parse_args():
    parser = argparse.ArgumentParser(description="Gesture Recognition Control System")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--model", type=str, default="data/models/knn_model.pkl",
                        help="Path to trained KNN model")
    parser.add_argument("--k", type=int, default=3, help="KNN k value (if model not loaded)")
    parser.add_argument("--confirm", type=int, default=5,
                        help="Consecutive frame confirmations (anti-false-trigger)")
    parser.add_argument("--no-control", action="store_true",
                        help="Disable GPIO control (recognition-only mode)")
    parser.add_argument("--feature", type=str, default="all",
                        choices=["distance", "angle", "all"],
                        help="Feature set type")
    parser.add_argument("--skip", type=int, default=2,
                        help="Process every N frames (1 = every frame, 2 = half, etc.)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Init modules
    print("=" * 50)
    print("Gesture Recognition Control System v0.1.1")
    print("=" * 50)

    # 1. Classifier
    classifier = KNNClassifier(k=args.k)
    model_path = Path(args.model)
    if model_path.exists():
        print(f"[INFO] Loading model: {model_path}")
        classifier.load(str(model_path))
    else:
        print(f"[WARN] Model not found: {model_path}")
        print("[WARN] Run train_hagrid.py first; landmark detection demo only")

    # 2. Detector
    detector = HandDetector(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
    )

    # 3. Feature extractor
    extractor = FeatureExtractor()

    # 4. GPIO controller
    controller = None if args.no_control else GPIOController(
        confirm_frames=args.confirm
    )

    # 5. Camera
    camera = Camera(camera_id=args.camera)

    # Smoothing: sliding window of recent predictions (majority vote)
    VOTE_WINDOW = 5
    vote_history = deque(maxlen=VOTE_WINDOW)

    try:
        if not camera.open():
            print("[ERROR] Cannot open camera")
            return

        print(f"[INFO] System ready (skip={args.skip}, vote_window={VOTE_WINDOW}). Press 'q' to quit.")
        frame_count = 0
        detect_count = 0
        start_time = time.time()
        last_result = None   # cached detection result for skipped frames
        TARGET_FPS = 30.0    # cap display refresh rate
        frame_budget = 1.0 / TARGET_FPS

        while True:
            t0 = time.perf_counter()
            success, frame = camera.read()
            if not success:
                print("[WARN] Frame read failed")
                continue

            # Mirror flip (selfie view)
            frame = cv2.flip(frame, 1)
            frame_count += 1

            # Frame skipping: only run heavy detection every N frames
            if frame_count % args.skip == 1:
                detect_count += 1
                detected, landmarks = detector.detect(frame)
                last_result = (detected, landmarks)
            elif last_result is not None:
                detected, landmarks = last_result
            else:
                detected, landmarks = False, None

            gesture_text = "No hand detected"

            if detected and landmarks is not None:
                # Draw landmarks (always, even on skipped frames)
                detector.draw_landmarks(frame, landmarks)

                if classifier.trained:
                    try:
                        # Only re-extract & classify on detection frames
                        if frame_count % args.skip == 1:
                            features = extractor.extract(landmarks, args.feature)
                            label, confidence = classifier.predict_with_confidence(features)
                            vote_history.append(label)

                        # Majority vote over recent predictions
                        if len(vote_history) > 0:
                            voted_label = Counter(vote_history).most_common(1)[0][0]

                            # Thumb-angle guard: KNN relies on thumb features for 4 vs 5,
                            # but 3D real-time landmarks shift the boundary. If model says
                            # "5" but thumb is clearly folded, force "4".
                            if voted_label == 4:  # 4 = index of "5"
                                # Thumb MCP angle: folded < 140°, extended > 160°
                                thumb_mcp = extractor.joint_angle(
                                    landmarks[4], landmarks[2], landmarks[1])
                                if thumb_mcp < 140.0:
                                    voted_label = 3  # force "4"

                            gesture_name = classifier.get_label_name(voted_label)
                            gesture_text = gesture_name

                            # GPIO control (uses majority-voted label)
                            if controller is not None:
                                controller.execute(gesture_name)
                    except Exception as e:
                        gesture_text = f"Err: {e}"
                else:
                    gesture_text = "Model not loaded"

            # FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # --- Display ---
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, gesture_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            if classifier.trained and len(vote_history) >= VOTE_WINDOW // 2:
                cv2.putText(frame, f"[LOCKED]", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            cv2.imshow("Gesture Control", frame)

            # FPS cap: sleep excess time to keep display smooth
            elapsed_frame = time.perf_counter() - t0
            if elapsed_frame < frame_budget:
                time.sleep(frame_budget - elapsed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        camera.release()
        detector.close()
        if controller is not None:
            controller.cleanup()
        cv2.destroyAllWindows()
        print(f"[INFO] Stopped. {frame_count} frames, {detect_count} detections")


if __name__ == "__main__":
    import cv2
    main()
