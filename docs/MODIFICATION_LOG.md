# 上位机手势识别集成 & 界面美化 — 修改记录

## 项目信息

| 项目 | 说明 |
|------|------|
| 名称 | Hand Gesture Recognition Control System |
| 上位机版本 | v2.0 |
| 修改日期 | 2026-06-11 |
| 涉及文件 | `gesture_handler.py`, `camera_handler.py`, `main_window.py` |
| 修改类型 | 模型接入、关键节点可视化、界面美化、布局修复 |

---

## 1. 手势识别模型接入上位机

### 1.1 架构概览

```
摄像头帧 (BGR)
    │
    ▼
┌─────────────────────────────────────────────┐
│  GestureHandler.detect(frame)               │
│  ├── HandDetector.detect()  → 21个关键点     │
│  ├── FeatureExtractor.extract() → 32维特征   │
│  └── KNNClassifier.predict() → 手势名+置信度 │
│                                              │
│  Returns: (gesture_name, landmarks, conf)    │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  GestureHandler.draw(frame, lm, name, conf) │
│  ├── 骨架连线 (按手指分色)                    │
│  ├── 关键节点 (指尖+手腕发光效果)              │
│  ├── 手部边界框                              │
│  └── 识别结果面板 (半透明, 含置信度进度条)      │
└─────────────────────────────────────────────┘
    │
    ▼
  Qt 信号 (每5帧) → main_window.py → 日志 + 串口指令
```

### 1.2 数据流

- **CameraThread** (`camera_handler.py`) 在子线程中捕获帧
- 每帧调用 `GestureHandler.detect()` 返回 `(名称, 关键点, 置信度)`
- 检测到手势后调用 `GestureHandler.draw()` 绘制可视化叠加层
- 每 **5 帧**通过 Qt 信号 `gesture_signal(str, float)` 发送结果到主线程
- `MainWindow._on_gesture_result()` 接收后写入日志 + 发送串口指令
- 手势变化时触发电机/外设控制，不变时仅记录日志

---

## 2. 文件修改详情

### 2.1 `gesture_handler.py` — 重写

**文件路径**: `creatation/motor_control_app/gesture_handler.py`

#### 2.1.1 `detect()` 返回值增强

| 修改前 | 修改后 |
|--------|--------|
| `return gesture_name, landmarks` | `return gesture_name, landmarks, confidence` |

置信度来自 KNN 的 `predict_proba()`，范围 0.0~1.0。

#### 2.1.2 `draw()` 可视化重写

新增手指分色常量：

```python
FINGER_COLORS_BGR = {
    "thumb":  (255, 140, 0),   # 橙色
    "index":  (70, 230, 70),   # 绿色
    "middle": (255, 80, 50),   # 蓝色
    "ring":   (180, 50, 255),  # 紫色
    "pinky":  (50, 160, 255),  # 金色
    "wrist":  (50, 50, 255),   # 红色
}
```

新增绘制元素：

| 元素 | 实现方式 | 参数 |
|------|---------|------|
| 手部边界框 | 四角装饰线 | 灰色, 1px, 距手15px外边距 |
| 骨架连线 | `cv2.line()` 抗锯齿 | 按手指分色, 掌弓灰色, 2px |
| 关键点-手指尖 | `_draw_glow_circle()` | 5px 内圈 + 12px 发光外圈(alpha=0.2) |
| 关键点-手腕 | `_draw_glow_circle()` | 6px 内圈 + 14px 发光外圈(alpha=0.15) |
| 关键点-关节 | `cv2.circle()` 双环 | 3px 彩色 + 4px 白色描边 |
| 识别结果面板 | `_draw_gesture_panel()` | 左下角, 半透明暗色背景, 绿色强调条 |
| 置信度进度条 | 矩形填充 | ≥80%绿 / ≥50%金 / <50%红 |
| 英文标签 | `cv2.putText()` | 灰色小字 (Index Up, OK Sign...) |

#### 2.1.3 手势名称映射

```python
label_map = {
    "1": "Index Up", "2": "Two Fingers", "3": "Three Fingers",
    "4": "Four Fingers", "5": "Five Fingers",
    "OK": "OK Sign", "Good": "Thumbs Up", "Fist": "Fist",
}
```

---

### 2.2 `camera_handler.py` — 适配

**文件路径**: `creatation/motor_control_app/camera_handler.py`

#### 2.2.1 信号签名变更

```python
# 修改前
gesture_signal = pyqtSignal(str)

# 修改后
gesture_signal = pyqtSignal(str, float)  # (gesture_name, confidence)
```

#### 2.2.2 手势检测调用适配

```python
# 修改前
gesture_name, landmarks = self.gesture.detect(frame)

# 修改后
gesture_name, landmarks, gesture_confidence = self.gesture.detect(frame)
```

#### 2.2.3 绘制调用适配

```python
# 修改前
rgb = self.gesture.draw(rgb, landmarks, gesture_name)

# 修改后
rgb = self.gesture.draw(rgb, landmarks, gesture_name, gesture_confidence)
```

#### 2.2.4 每5帧日志

```python
if gesture_name and landmarks:
    self._gesture_frame_count += 1
    # 只在每第5帧发射信号
    if self._gesture_frame_count % 5 == 0:
        self.gesture_signal.emit(gesture_name, gesture_confidence)
else:
    # 无手时重置计数器
    self._gesture_frame_count = 0
```

#### 2.2.5 手势状态指示器

```python
if self.enable_gesture:
    status = "Gesture: ON" if self.gesture.enabled else "Gesture: LOADING..."
    cv2.putText(rgb, status, (w - 195, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 100), 1, cv2.LINE_AA)
```

---

### 2.3 `main_window.py` — 界面美化 + 布局修复

**文件路径**: `creatation/motor_control_app/main_window.py`

#### 2.3.1 调色板升级

| 颜色常量 | 修改前 | 修改后 | 用途 |
|---------|--------|--------|------|
| `COLOR_BG` | `#1e1e2e` | `#0f0f1a` | 主背景 (更深) |
| `COLOR_SURFACE` | `#2a2a3e` | `#1a1a2e` | 面板背景 |
| `COLOR_PANEL` | `#313149` | `#222240` | 控件背景 |
| `COLOR_ACCENT` | `#7c6af7` | `#8b7cf7` | 主强调色 (偏紫) |
| `COLOR_ACCENT2` | `#5ad3c1` | `#45d9c1` | 辅助色 (青色) |
| `COLOR_SUCCESS` | `#6bcb77` | `#4ade80` | 成功/就绪 |
| `COLOR_WARNING` | `#ffb347` | `#f0a040` | 警告 |

#### 2.3.2 全局样式增强

| 元素 | 修改前 | 修改后 |
|------|--------|--------|
| QGroupBox 背景 | 纯色 `#2a2a3e` | 渐变 `#1f1f38 → #1a1a2e` |
| QGroupBox 圆角 | 8px | 10px |
| 按钮圆角 | 6px | 8px |
| 按钮背景 | 纯色 | 垂直渐变 |
| 按钮 padding | 7px 14px | 8px 16px |
| 滑块手柄 | 16px 纯色圆 | 18px 渐变圆 + 2px 边框 |
| 滑块轨道 | 6px | 8px |
| 滑块已过区域 | 纯色 | 水平渐变 (青色→紫色) |
| QComboBox | 无 hover | hover 时边框变色 |
| 滚动条 | 8px | 10px, hover 变色 |
| 复选框 | 16px | 18px, hover 边框变色 |

#### 2.3.3 按钮样式

所有按钮增加 `qlineargradient` 渐变背景：

```css
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #222240, stop:1 #1e1e38);
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #8b7cf7, stop:1 #6a5cd4);
}
```

#### 2.3.4 面板标题

| 面板 | 修改前 | 修改后 |
|------|--------|--------|
| 摄像头 | "摄像头监控" | "📹 摄像头监控" |
| 串口 | "串口连接" | "🔌 串口连接" |
| 电机 | "电机控制" | "⚙️ 电机控制" |
| 速度 | "速度控制" | "🏃 速度控制" |
| 日志 | "操作日志" | "📋 操作日志" |

#### 2.3.5 手势状态显示

新增 `gesture_status_lbl` 在摄像头面板状态栏，实时显示：

```
✋ 1 (95%)           ← 高置信度=绿色
✋ OK (65%)          ← 中置信度=金色
✋ Fist (40%)        ← 低置信度=灰色
```

#### 2.3.6 手势映射图例

勾选"手势识别"后，摄像头面板底部显示：

```
✋ Gestures: 1→AB▶ | 2→AB◀ | 3→CD▶ | 4→CD◀ | 5→AB■ | OK→CD■ | 👍→Brake | ✊→All■
```

关闭手势识别或断开摄像头时自动隐藏。

#### 2.3.7 日志格式增强

```
修改前:  [手势 #5] 1 *
修改后:  [Gesture #5] 1 (conf: 95.2%) [CHANGED]
```

- 置信度百分比精确到 0.1%
- 手势变化时标记 `[CHANGED]`
- 手势→串口指令也记录到日志

#### 2.3.8 QSplitter 布局修复

**问题**: 日志面板 `stretch=1` 导致电机控制面板被压缩，文字不可见。

**修复**: 用 `QSplitter(Qt.Vertical)` 替换 stretch 布局：

```python
splitter = QSplitter(Qt.Vertical)
splitter.setChildrenCollapsible(False)     # 禁止完全折叠
splitter.addWidget(control_stack)          # 控制区 (电机+速度)
splitter.addWidget(log_panel)              # 日志区
splitter.setSizes([420, 180])              # 初始比例 7:3
```

| 参数 | 修改前 | 修改后 |
|------|--------|--------|
| 控制区初始高度 | ~340px | **420px** |
| 日志区初始高度 | ~300px | **180px** |
| 控制区最小高度 | 无 | **320px** |
| 窗口默认尺寸 | 1100×780 | **1200×920** |
| 窗口最小尺寸 | 900×640 | **1020×760** |
| 拖拽条样式 | 默认 | 灰色 3px, hover 变紫色 |

#### 2.3.9 电机控制面板紧凑化

| 改动 | 效果 |
|------|------|
| 按钮 `setMinimumHeight(36)` | 防止按钮被压缩到不可用 |
| "重置位置"和"自动调零"并排一行 | 节省垂直空间 |
| 面板 `spacing` 从 8→4px | 减少空白 |
| 面板 `margins` 统一为 `(10,14,10,8)` | 紧凑统一 |

#### 2.3.10 窗口标题

```
修改前: "电机控制上位机  v1.1"
修改后: "🎮 电机控制上位机  v2.0"
```

---

## 3. 手势识别可视化效果对比

### 修改前
- 关键点: 简单绿点 (cv2.circle, 4px)
- 骨架: 灰色直线
- 识别结果: `cv2.putText` 简单绿色文字
- 无置信度显示
- 无手部边界框
- 无英文标签

### 修改后
- 关键点: **5指分色** + 指尖/手腕**发光效果** + 关节**双环描边**
- 骨架: 按手指分色的**抗锯齿连线**, 掌弓为浅灰色
- 识别结果: **半透明面板** (75%不透明度暗色背景 + 绿色强调条)
- 置信度: **彩色进度条** (≥80%绿 / ≥50%金 / <50%红)
- 手部边界框: 四角装饰线风格
- 英文标签: 面板右下角灰色小字

---

## 4. 5帧日志机制

```
帧计数逻辑:
  有手势 → counter += 1 → counter % 5 == 0 → 发射信号 → 日志输出
  无手势 → counter = 0 → 无信号 → 无日志

日志格式:
  [HH:MM:SS] [Gesture #N] 手势名 (conf: XX.X%) [CHANGED]
```

---

## 5. 文件修改清单

| 文件 | 行数变化 | 修改类型 |
|------|---------|---------|
| `creatation/motor_control_app/gesture_handler.py` | 122 → 320 行 | 重写 (可视化增强 + 置信度) |
| `creatation/motor_control_app/camera_handler.py` | 489 → 492 行 | 适配 (信号签名 + 检测调用) |
| `creatation/motor_control_app/main_window.py` | ~1700 行 | UI美化 + 布局修复 + 手势状态 |

### 关键修改点索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `gesture_handler.py` | 95-101 | `detect()` 返回置信度 |
| `gesture_handler.py` | 109-113 | 手指→findger映射函数 |
| `gesture_handler.py` | 115-121 | 发光圆圈绘制函数 |
| `gesture_handler.py` | 123-185 | 增强版 `draw()` 方法 |
| `gesture_handler.py` | 247-275 | 手势面板绘制函数 |
| `camera_handler.py` | 307 | `gesture_signal` 信号签名 |
| `camera_handler.py` | 450-471 | 手势检测+绘制+5帧发射 |
| `main_window.py` | 43-54 | 新调色板 |
| `main_window.py` | 57-135 | 按钮样式升级 |
| `main_window.py` | 233-370 | 全局样式表升级 |
| `main_window.py` | 448-473 | QSplitter 布局 |
| `main_window.py` | 499-505 | 手势状态标签 |
| `main_window.py` | 555-562 | 手势映射图例 |
| `main_window.py` | 605-646 | `_on_gesture_result()` 重构 |
| `main_window.py` | 697-763 | 电机面板紧凑化 |

---

## 6. 依赖项 (未变更)

所有修改仅使用项目已有的依赖：

```
opencv-python, mediapipe, scikit-learn, numpy, PyQt5, pyserial
```

无需安装新的 Python 包。

---

## 7. 已知限制

1. **单手检测**: MediaPipe 配置 `max_num_hands=1`，多只手时只取第一只
2. **无手势时不绘图**: 未检测到手时不显示任何覆盖层
3. **PyQt5 信号线程安全**: `gesture_signal` 跨线程发射，Qt 自动排队处理
4. **MediaPipe 清理**: `HandLandmarker.__del__` 在 Windows 上偶发异常（已知 MediaPipe bug），不影响功能

---

## 8. 后续计划 (规划中，未实施)

- [ ] 树莓派移植 (camera_handler 后端适配、YOLO 移除、分辨率优化)
- [ ] STM32 串口下位机 (扩展协议: 3字节外设指令 + 传感器读取)
- [ ] 命令行 headless 模式 (无 GUI 直接运行)
- [ ] CSI 灰度摄像头原生支持
