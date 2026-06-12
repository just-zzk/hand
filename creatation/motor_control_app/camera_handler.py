"""
摄像头处理模块
支持：
  · 普通摄像头（本地/RTSP）
  · 红外摄像头（灰度 + 可选伪彩色渲染）
  · YOLOv5 ONNX 物品识别（GPU加速）
"""
import cv2
import threading
import time
import numpy as np
import os
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

from gesture_handler import GestureHandler


def list_available_cameras(max_devices=6):
    """
    Auto-detect available cameras with priority:
      HDMI (high-res) > USB serial cam > built-in laptop cam

    Returns list of (device_id, display_name)
    """
    available = []
    print(f"[Camera Detection] Scanning cameras 0~{max_devices-1}...")

    for i in range(max_devices):
        try:
            cap = cv2.VideoCapture(i)
            if not cap.isOpened():
                cap.release()
                continue
            # Read one frame to confirm it's really working
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                continue

            h, w = frame.shape[:2]
            cap.release()

            # Classify camera type by resolution
            if w >= 1920 or h >= 1080:
                label = f"相机 {i} (HDMI {w}x{h})"
                priority = 0  # HDMI first
            elif w >= 1280 or h >= 720:
                label = f"相机 {i} (USB {w}x{h})"
                priority = 1  # USB second
            else:
                label = f"相机 {i} (内置 {w}x{h})"
                priority = 2  # built-in last

            available.append((priority, i, label))
            print(f"  [Camera] 设备 {i}: {w}x{h} → {label}")

        except Exception as e:
            print(f"  [Camera] 设备 {i}: error - {e}")
            continue

    # Sort by priority (HDMI > USB > built-in), then by device id
    available.sort(key=lambda x: (x[0], x[1]))

    if not available:
        # Fallback: add device 0 as built-in even if detection failed
        available.append((2, 0, "相机 0 (内置 默认)"))
        print(f"  [Camera] 未检测到摄像头，使用默认设备 0")

    # Return (device_id, display_name) in priority order
    result = [(dev_id, label) for _, dev_id, label in available]
    print(f"[Camera Detection] 检测完成: {len(result)} 个设备")
    return result

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    print("[YOLO] onnxruntime 未安装，将使用 CPU 推理")

# 获取当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── 伪彩色方案 ───────────────────────────────────────────
COLORMAP_OPTIONS = {
    "灰度":       None,                   # 不映射，原始灰度
    "JET":        cv2.COLORMAP_JET,
    "HOT":        cv2.COLORMAP_HOT,
    "RAINBOW":    cv2.COLORMAP_RAINBOW,
    "INFERNO":    cv2.COLORMAP_INFERNO,
    "PLASMA":     cv2.COLORMAP_PLASMA,
    "MAGMA":      cv2.COLORMAP_MAGMA,
    "BONE":       cv2.COLORMAP_BONE,
    "OCEAN":      cv2.COLORMAP_OCEAN,
}

# ─── YOLOv5 配置 ─────────────────────────────────────────
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov5s.onnx")  # ONNX模型路径（绝对路径）
YOLO_INPUT_SIZE = 640             # 输入尺寸
YOLO_CONF_THRESH = 0.5           # 置信度阈值
YOLO_NMS_THRESH = 0.45           # NMS阈值

# COCO 类别标签
YOLO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 颜色映射（用于绘制检测框）
COLORS = np.random.randint(0, 255, size=(len(YOLO_CLASSES), 3), dtype=np.uint8)


class YOLOv5Detector:
    """YOLOv5 ONNX 推理器（支持 GPU 加速）"""
    def __init__(self, model_path=YOLO_MODEL_PATH):
        self.model_path = model_path
        self.session = None
        self.load_model()

    def load_model(self):
        """加载ONNX模型"""
        if not os.path.exists(self.model_path):
            print(f"[YOLO] 错误: 模型文件不存在: {self.model_path}")
            self.session = None
            return
        
        try:
            if ONNXRUNTIME_AVAILABLE:
                # 使用 onnxruntime-gpu（会自动调用 CUDA）
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.session = ort.InferenceSession(
                    self.model_path,
                    providers=providers
                )
                print(f"[YOLO] 模型加载成功 (GPU加速): {self.model_path}")
            else:
                # 回退到 OpenCV DNN
                self.session = cv2.dnn.readNetFromONNX(self.model_path)
                print(f"[YOLO] 模型加载成功 (CPU): {self.model_path}")
        except Exception as e:
            print(f"[YOLO] 模型加载失败: {e}")
            self.session = None

    def detect(self, image):
        """执行推理，返回检测结果"""
        if self.session is None:
            return []
        
        try:
            blob, scale_w, scale_h = self.preprocess(image)
            
            if ONNXRUNTIME_AVAILABLE:
                # onnxruntime 推理
                outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
            else:
                # OpenCV DNN 推理
                self.session.setInput(blob)
                outputs = self.session.forward()
            
            detections = self.postprocess(outputs, image.shape, scale_w, scale_h)
            return detections
        except Exception as e:
            import traceback
            print(f"[YOLO] 推理失败: {e}")
            traceback.print_exc()
            return []

    def preprocess(self, image):
        """图像预处理：直接缩放到 640x640，不保持宽高比"""
        h, w = image.shape[:2]
        
        # 计算缩放比例（用于后处理坐标转换）
        scale_w = w / YOLO_INPUT_SIZE
        scale_h = h / YOLO_INPUT_SIZE
        
        # 直接缩放到 640x640（YOLOv5 要求精确尺寸）
        image_resized = cv2.resize(image, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
        
        # 创建blob（YOLOv5格式）
        blob = cv2.dnn.blobFromImage(
            image_resized, 
            scalefactor=1/255.0, 
            size=(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), 
            mean=(0, 0, 0), 
            swapRB=True, 
            crop=False
        )
        
        return blob, scale_w, scale_h

    def postprocess(self, output, original_shape, scale_w, scale_h):
        """后处理：NMS、坐标转换"""
        orig_h, orig_w = original_shape[:2]
        boxes = []
        confidences = []
        class_ids = []

        # onnxruntime 返回 list，取第一个元素
        if isinstance(output, list):
            output = output[0]

        # 输出形状: [1, 25195, 85] -> [25195, 85]
        output = output.squeeze(0)

        # YOLOv5 输出格式: [num_detections, 85]
        # 85 = x_center, y_center, width, height, confidence, class0, class1, ..., class79
        for detection in output:
            conf = detection[4]
            if conf < YOLO_CONF_THRESH:
                continue

            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]
            confidence = conf * class_confidence

            if confidence >= YOLO_CONF_THRESH:
                x_center = detection[0] * scale_w
                y_center = detection[1] * scale_h
                box_w = detection[2] * scale_w
                box_h = detection[3] * scale_h

                x1 = int(max(0, x_center - box_w / 2))
                y1 = int(max(0, y_center - box_h / 2))
                x2 = int(min(orig_w, x_center + box_w / 2))
                y2 = int(min(orig_h, y_center + box_h / 2))

                boxes.append([x1, y1, x2-x1, y2-y1])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, YOLO_CONF_THRESH, YOLO_NMS_THRESH)

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                results.append({
                    "box": [x, y, x+w, y+h],
                    "confidence": confidences[i],
                    "class_id": class_ids[i],
                    "class_name": YOLO_CLASSES[class_ids[i]]
                })

        return results

    def draw_detections(self, image, detections):
        """在图像上绘制检测结果"""
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            class_name = det["class_name"]
            confidence = det["confidence"]
            class_id = det["class_id"]
            
            # 获取颜色
            color = [int(c) for c in COLORS[class_id]]
            
            # 绘制矩形框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y1 = max(y1, label_size[1] + 10)
            
            cv2.rectangle(image, (x1, label_y1 - label_size[1] - 10), 
                          (x1 + label_size[0], label_y1), color, -1)
            cv2.putText(image, label, (x1, label_y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image


class CameraThread(QThread):
    """
    摄像头捕获线程，通过信号将帧传递给 UI。

    参数
    ----
    camera_url : int | str
        本地设备 ID（0,1,...）或 RTSP/HTTP 地址
    fps : int
        目标帧率
    ir_mode : bool
        True = 红外模式（强制灰度处理）
    colormap : str
        伪彩色方案名称，见 COLORMAP_OPTIONS；仅 ir_mode=True 时生效
    enable_detection : bool
        是否启用YOLOv5物品识别
    """
    frame_signal     = pyqtSignal(QImage)
    error_signal     = pyqtSignal(str)
    connected_signal = pyqtSignal(bool)
    detection_signal = pyqtSignal(list)  # YOLO detection results
    gesture_signal   = pyqtSignal(str, float)   # gesture name, confidence

    def __init__(self, camera_url=0, fps=30, ir_mode=False, colormap="JET"):
        super().__init__()
        self.camera_url        = camera_url
        self.fps               = fps
        self.ir_mode           = ir_mode
        self.colormap          = colormap     # thread-safe: can be changed at runtime
        self.enable_detection  = False        # YOLO object detection
        self.enable_gesture    = False        # hand gesture recognition
        self.running           = False
        self._lock             = threading.Lock()
        self.cap               = None
        self.detector            = YOLOv5Detector()   # YOLO
        self.gesture             = GestureHandler()   # hand gesture
        self._gesture_frame_count = 0                  # frame counter for logging

    def set_camera_url(self, url):
        self.camera_url = url

    def toggle_detection(self, enable):
        """Toggle YOLO object detection"""
        self.enable_detection = enable
        if enable and self.detector.session is None:
            self.detector.load_model()

    def toggle_gesture(self, enable):
        """Toggle hand gesture recognition"""
        self.enable_gesture = enable
        if enable:
            self.gesture.enabled = True

    # ──────────────────────────────────────────────
    # 帧处理：普通 / 红外
    # ──────────────────────────────────────────────
    def _process_frame(self, frame):
        """返回适合显示的 RGB numpy array"""
        if not self.ir_mode:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── 红外模式 ──────────────────────────────
        # 先转灰度（兼容彩色输入和原生灰度输入）
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame  # 已是单通道

        # 对比度增强（CLAHE），让细节更清晰
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        cmap_id = COLORMAP_OPTIONS.get(self.colormap)
        if cmap_id is None:
            # 灰度模式：转为 3 通道灰度
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            colored = cv2.applyColorMap(gray, cmap_id)
            rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        return rgb

    # ──────────────────────────────────────────────
    # 线程主循环
    # ──────────────────────────────────────────────
    def run(self):
        self.running = True
        
        # 调试日志：显示尝试打开的摄像头URL
        print(f"[Camera] 尝试打开摄像头: {self.camera_url}")
        
        # 尝试不同的后端打开摄像头
        backends = [
            (cv2.CAP_MSMF, "MSMF"),
            (cv2.CAP_DSHOW, "DSHOW"),
            (cv2.CAP_ANY, "ANY"),
        ]
        
        self.cap = None
        for backend, backend_name in backends:
            print(f"[Camera] 尝试后端: {backend_name}")
            self.cap = cv2.VideoCapture(self.camera_url, backend)
            if self.cap.isOpened():
                print(f"[Camera] 后端 {backend_name} 打开成功")
                # 尝试读取一帧确认设备可用
                ret, _ = self.cap.read()
                if ret:
                    print(f"[Camera] 成功读取第一帧")
                    # Set resolution to 640x480 for performance
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    break
                else:
                    print(f"[Camera] 后端 {backend_name} 打开但无法读取帧")
                    self.cap.release()
                    self.cap = None
            else:
                print(f"[Camera] 后端 {backend_name} 打开失败")
        
        if self.cap is None or not self.cap.isOpened():
            self.error_signal.emit(f"无法打开摄像头: {self.camera_url}")
            self.connected_signal.emit(False)
            print(f"[Camera] 所有后端尝试失败")
            return

        # 记录当前使用的后端索引
        current_backend_idx = 0
        for i, (backend, backend_name) in enumerate(backends):
            if self.cap.getBackendName() == backend_name:
                current_backend_idx = i
                break
        
        self.connected_signal.emit(True)
        interval = 1.0 / self.fps

        while self.running:
            t0 = time.time()
            ret, frame = self.cap.read()

            if not ret:
                self.error_signal.emit("摄像头读取失败，尝试重连...")
                self.cap.release()
                time.sleep(2)
                
                # 尝试切换到下一个后端
                current_backend_idx = (current_backend_idx + 1) % len(backends)
                backend, backend_name = backends[current_backend_idx]
                print(f"[Camera] 切换到后端: {backend_name}")
                
                self.cap = cv2.VideoCapture(self.camera_url, backend)
                if not self.cap.isOpened():
                    # 继续尝试下一个后端
                    continue
                print(f"[Camera] 后端 {backend_name} 重连成功")
                continue

            # YOLO object detection (on BGR)
            gesture_name = None
            landmarks = None
            if self.enable_detection and self.detector.session is not None:
                detections = self.detector.detect(frame)
                self.detection_signal.emit(detections)
                frame = self.detector.draw_detections(frame, detections)

            # Hand gesture recognition (on BGR)
            gesture_name = None
            landmarks = None
            gesture_confidence = 0.0
            if self.enable_gesture and self.gesture.enabled:
                gesture_name, landmarks, gesture_confidence = self.gesture.detect(frame)

            # Process frame (BGR → RGB conversion, IR colormap, etc.)
            rgb = self._process_frame(frame)
            h, w, ch = rgb.shape

            # Draw gesture overlay on RGB frame (after conversion for correct colors)
            if gesture_name and landmarks:
                self._gesture_frame_count += 1
                rgb = self.gesture.draw(rgb, landmarks, gesture_name, gesture_confidence)
                # Emit gesture signal every 5 frames with confidence
                if self._gesture_frame_count % 5 == 0:
                    self.gesture_signal.emit(gesture_name, gesture_confidence)
            else:
                # Reset gesture frame counter when no hand detected
                self._gesture_frame_count = 0

            # Add "Gesture ON" indicator when gesture mode is active
            if self.enable_gesture:
                status = "Gesture: ON" if self.gesture.enabled else "Gesture: LOADING..."
                cv2.putText(rgb, status, (w - 195, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 100), 1, cv2.LINE_AA)

            # Create QImage
            q_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            self.frame_signal.emit(q_image)

            elapsed = time.time() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if self.cap:
            self.cap.release()
        self.connected_signal.emit(False)

    def stop(self):
        self.running = False
        self.wait(3000)
        self.gesture.close()
