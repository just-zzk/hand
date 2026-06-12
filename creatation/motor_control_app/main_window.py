"""
电机控制上位机 - 主界面
PyQt5 实现，包含：
  · 网络/本地摄像头实时预览
  · YOLOv5物品识别（可选）
  · 串口连接管理
  · 电机控制（正转/反转/停止/刹车）
  · 速度调节
  · 操作日志
  · 自动对焦
  
适配单片机单字节协议：
  - 心跳包: 0x00（刷新看门狗）
  - AB电机: 0x01~0x04 (正转/反转/停止/刹车)
  - CD电机: 0x05~0x08 (正转/反转/停止/刹车)
  - 速度: 0x10~0x73 对应 1~100
"""
import time
import sys
import numpy as np
import json
import os

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QPushButton,
    QComboBox, QSlider, QTextEdit, QLineEdit,
    QSplitter, QStatusBar, QFrame, QSizePolicy,
    QCheckBox,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QFont, QColor, QPalette, QIcon

from camera_handler import CameraThread, COLORMAP_OPTIONS, list_available_cameras
from serial_communicator import (
    SerialCommunicator, list_available_ports,
    CMD_AB_FORWARD, CMD_AB_REVERSE, CMD_AB_STOP, CMD_AB_BRAKE,
    CMD_CD_FORWARD, CMD_CD_REVERSE, CMD_CD_STOP, CMD_CD_BRAKE,
)
from auto_focus import AutoFocusThread, AF_FOCUS_THRESHOLD
from zero_calibration_dialog import ZeroCalibrationDialog

# ─── 调色板常量 (Premium Dark Theme) ───────────────────
COLOR_BG        = "#0f0f1a"
COLOR_SURFACE   = "#1a1a2e"
COLOR_PANEL     = "#222240"
COLOR_ACCENT    = "#8b7cf7"
COLOR_ACCENT2   = "#45d9c1"
COLOR_DANGER    = "#f25f5c"
COLOR_WARNING   = "#f0a040"
COLOR_SUCCESS   = "#4ade80"
COLOR_TEXT      = "#e8e8f4"
COLOR_TEXT_DIM  = "#8888aa"
COLOR_BORDER    = "#333355"
COLOR_GLOW      = "rgba(139, 124, 247, 0.15)"

BTN_BASE = f"""
    QPushButton {{
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 13px;
        font-weight: 600;
        color: {COLOR_TEXT};
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 {COLOR_PANEL}, stop:1 #1e1e38);
    }}
    QPushButton:hover {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 {COLOR_ACCENT}, stop:1 #6a5cd4);
        border-color: {COLOR_ACCENT};
    }}
    QPushButton:pressed {{
        background: #5a4cd4;
    }}
    QPushButton:disabled {{
        color: {COLOR_TEXT_DIM};
        background: {COLOR_SURFACE};
        border-color: {COLOR_BORDER};
    }}
"""

BTN_DANGER = f"""
    QPushButton {{
        border: 1px solid {COLOR_DANGER};
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 13px;
        font-weight: 600;
        color: {COLOR_DANGER};
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 {COLOR_PANEL}, stop:1 #1e1e38);
    }}
    QPushButton:hover {{
        background: {COLOR_DANGER};
        color: #fff;
    }}
    QPushButton:pressed {{
        background: #c04040;
    }}
"""

BTN_SUCCESS = f"""
    QPushButton {{
        border: 1px solid {COLOR_SUCCESS};
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 13px;
        font-weight: 600;
        color: {COLOR_SUCCESS};
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 {COLOR_PANEL}, stop:1 #1e1e38);
    }}
    QPushButton:hover {{
        background: {COLOR_SUCCESS};
        color: #000;
    }}
    QPushButton:pressed {{
        background: #3ab868;
    }}
"""

BTN_WARNING = f"""
    QPushButton {{
        border: 1px solid #f0a040;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 13px;
        font-weight: 600;
        color: #f0a040;
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 {COLOR_PANEL}, stop:1 #1e1e38);
    }}
    QPushButton:hover {{
        background: #f0a040;
        color: #000;
    }}
    QPushButton:pressed {{
        background: #d09030;
    }}
"""

BTN_TOGGLE_ON = f"""
    QPushButton {{
        border: 1px solid {COLOR_SUCCESS};
        border-radius: 6px;
        padding: 7px 14px;
        font-size: 13px;
        font-weight: 600;
        color: #fff;
        background: {COLOR_SUCCESS};
    }}
    QPushButton:hover {{
        background: #5ab865;
    }}
"""

BTN_TOGGLE_OFF = f"""
    QPushButton {{
        border: 1px solid {COLOR_BORDER};
        border-radius: 6px;
        padding: 7px 14px;
        font-size: 13px;
        font-weight: 600;
        color: {COLOR_TEXT_DIM};
        background: {COLOR_PANEL};
    }}
    QPushButton:hover {{
        border-color: {COLOR_TEXT_DIM};
    }}
"""


def _label(text, bold=False, color=COLOR_TEXT, size=13):
    lbl = QLabel(text)
    font = QFont("Segoe UI", size)
    font.setBold(bold)
    lbl.setFont(font)
    lbl.setStyleSheet(f"color: {color}; background: transparent;")
    return lbl


def _hline():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"color: {COLOR_BORDER};")
    return line


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.serial = None
        self.camera_thread = None
        self.auto_focus_thread = None
        
        # 配置文件路径
        self._config_path = os.path.join(os.path.dirname(__file__), 'motor_config.json')
        
        # 先加载配置
        self._load_config()
        
        self._setup_window()
        self._build_ui()
        self._apply_global_style()
        self._refresh_ports()

    def _load_config(self):
        """加载电机配置"""
        self._config = {
            'channel_0_boundary': {'left': 0, 'right': 32},
            'channel_1_boundary': {'left': 0, 'right': 32}
        }
        
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, 'r') as f:
                    saved_config = json.load(f)
                    self._config.update(saved_config)
            except Exception as e:
                pass  # 配置文件可能损坏，使用默认值
    
    def _save_config(self):
        """保存电机配置"""
        try:
            with open(self._config_path, 'w') as f:
                json.dump(self._config, f, indent=2)
            self._log("配置保存成功", "info")
        except Exception as e:
            self._log(f"配置保存失败: {str(e)}", "warning")

    # ══════════════════════════════════════════════
    # 窗口基础配置
    # ══════════════════════════════════════════════
    def _setup_window(self):
        self.setWindowTitle("🎮 电机控制上位机  v2.0")
        self.resize(1200, 920)
        self.setMinimumSize(1020, 760)

    def _apply_global_style(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLOR_BG};
            }}
            QWidget {{
                background-color: {COLOR_BG};
                color: {COLOR_TEXT};
                font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
            }}
            QGroupBox {{
                border: 1px solid {COLOR_BORDER};
                border-radius: 10px;
                margin-top: 16px;
                padding-top: 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1f1f38, stop:1 {COLOR_SURFACE});
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 8px;
                color: {COLOR_ACCENT};
                font-weight: bold;
                font-size: 13px;
                letter-spacing: 0.5px;
            }}
            QComboBox {{
                background: {COLOR_PANEL};
                border: 1px solid {COLOR_BORDER};
                border-radius: 6px;
                padding: 5px 10px;
                color: {COLOR_TEXT};
                min-width: 100px;
            }}
            QComboBox:hover {{
                border-color: {COLOR_ACCENT};
            }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox QAbstractItemView {{
                background: {COLOR_PANEL};
                selection-background-color: {COLOR_ACCENT};
                color: {COLOR_TEXT};
                border: 1px solid {COLOR_BORDER};
                border-radius: 4px;
                outline: none;
            }}
            QSlider::groove:horizontal {{
                height: 8px;
                background: {COLOR_BORDER};
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                width: 18px; height: 18px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {COLOR_ACCENT}, stop:1 #6a5cd4);
                border-radius: 9px;
                margin: -5px 0;
                border: 2px solid {COLOR_ACCENT2};
            }}
            QSlider::handle:horizontal:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #9b8cf7, stop:1 #7a6ce4);
            }}
            QSlider::sub-page:horizontal {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLOR_ACCENT2}, stop:1 {COLOR_ACCENT});
                border-radius: 4px;
            }}
            QTextEdit {{
                background: {COLOR_PANEL};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
                color: {COLOR_TEXT};
                font-family: "Cascadia Code", Consolas, monospace;
                font-size: 12px;
                padding: 4px;
            }}
            QLineEdit {{
                background: {COLOR_PANEL};
                border: 1px solid {COLOR_BORDER};
                border-radius: 6px;
                padding: 5px 10px;
                color: {COLOR_TEXT};
            }}
            QLineEdit:focus {{
                border-color: {COLOR_ACCENT};
                border-width: 1.5px;
            }}
            QStatusBar {{
                background: {COLOR_SURFACE};
                color: {COLOR_TEXT_DIM};
                border-top: 1px solid {COLOR_BORDER};
                font-size: 12px;
            }}
            QScrollBar:vertical {{
                background: {COLOR_SURFACE};
                width: 10px;
                border-radius: 5px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical {{
                background: {COLOR_BORDER};
                border-radius: 5px;
                min-height: 24px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {COLOR_ACCENT};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QCheckBox {{
                spacing: 8px;
                color: {COLOR_TEXT};
                font-size: 13px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {COLOR_BORDER};
                border-radius: 5px;
                background: {COLOR_PANEL};
            }}
            QCheckBox::indicator:hover {{
                border-color: {COLOR_ACCENT};
            }}
            QCheckBox::indicator:checked {{
                background: {COLOR_ACCENT};
                border-color: {COLOR_ACCENT};
                image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMiIgaGVpZ2h0PSIxMiI+PHBhdGggZmlsbD0icmdiYSgyNTUsMjU1LDI1NSwwLjk0KSIgZD0iTTExLjQgMy40bC04IDhjLS40LjQtLjQgMSAwIDEuNGw1LjIgNS4yYy40LjQuMSAxLjEtLjMgMS40bC0xMC40IDQuNmMtLjQuMy0xIC4yLTEuNC0uMWwtLjgtLjhjLS40LS40LS40LTEgMC0xLjRsNi44LTguNmMuNC0uNCAxLS40IDEuNCAwbDguMiA4LjJjLjQuNC40IDEgMCAxLjR6Ii8+PC9zdmc+);
            }}
        """)

    # ══════════════════════════════════════════════
    # UI 构建
    # ══════════════════════════════════════════════
    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # 左侧：视频预览
        root.addWidget(self._build_video_panel(), stretch=3)

        # 右侧：控制面板（垂直）
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        
        # 串口面板始终显示
        right_layout.addWidget(self._build_serial_panel())
        
        # 控制模式切换（手动/自动/调零）
        mode_switch = QWidget()
        mode_layout = QHBoxLayout(mode_switch)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(3)
        
        self.manual_mode_btn = QPushButton("手动控制")
        self.manual_mode_btn.setStyleSheet(BTN_SUCCESS)
        self.manual_mode_btn.setCheckable(True)
        self.manual_mode_btn.setChecked(True)
        self.manual_mode_btn.clicked.connect(lambda: self._switch_control_mode(0))
        mode_layout.addWidget(self.manual_mode_btn)
        
        self.auto_mode_btn = QPushButton("自动对焦")
        self.auto_mode_btn.setStyleSheet(BTN_BASE)
        self.auto_mode_btn.setCheckable(True)
        self.auto_mode_btn.clicked.connect(lambda: self._switch_control_mode(1))
        mode_layout.addWidget(self.auto_mode_btn)
        
        self.zero_mode_btn = QPushButton("调零校准")
        self.zero_mode_btn.setStyleSheet(BTN_BASE)
        self.zero_mode_btn.setCheckable(True)
        self.zero_mode_btn.clicked.connect(lambda: self._switch_control_mode(2))
        mode_layout.addWidget(self.zero_mode_btn)
        
        right_layout.addWidget(mode_switch)
        
        # 分页面板
        self.control_stack = QWidget()
        self.control_stack.setMinimumHeight(320)
        self.control_stack_layout = QVBoxLayout(self.control_stack)
        self.control_stack_layout.setContentsMargins(0, 0, 0, 0)
        self.control_stack_layout.setSpacing(8)
        
        # 手动控制面板（电机控制 + 速度控制）
        self.manual_panel = QWidget()
        self.manual_panel_layout = QVBoxLayout(self.manual_panel)
        self.manual_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.manual_panel_layout.setSpacing(6)
        self.manual_panel_layout.addWidget(self._build_motor_panel())
        self.manual_panel_layout.addWidget(self._build_speed_panel())
        
        # 自动对焦面板
        self.auto_panel = QWidget()
        self.auto_panel_layout = QVBoxLayout(self.auto_panel)
        self.auto_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.auto_panel_layout.setSpacing(10)
        self.auto_panel_layout.addWidget(self._build_auto_focus_panel())
        
        # 调零校准面板
        self.zero_panel = QWidget()
        self.zero_panel_layout = QVBoxLayout(self.zero_panel)
        self.zero_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.zero_panel_layout.setSpacing(10)
        self.zero_panel_layout.addWidget(self._build_zero_calibration_panel())
        
        # 默认显示手动面板
        self.control_stack_layout.addWidget(self.manual_panel)
        self.auto_panel.hide()
        self.zero_panel.hide()

        # ── 用 QSplitter 分割控制区和日志区（可拖拽调整） ──
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.control_stack)
        splitter.addWidget(self._build_log_panel())
        # 初始分配：控制区 420px ← 主空间，日志区 180px ← 辅助
        splitter.setSizes([420, 180])
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background: {COLOR_BORDER};
                height: 3px;
            }}
            QSplitter::handle:hover {{
                background: {COLOR_ACCENT};
            }}
        """)

        right_layout.addWidget(splitter, stretch=1)

        root.addWidget(right, stretch=2)

        self.setCentralWidget(central)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._set_status("就绪", COLOR_TEXT_DIM)
        
        # 初始化摄像头设备列表（需要在log_text创建后调用）
        self._refresh_camera_list()

    # ──────────────────────────────────────────────
    # 视频面板（含物品识别）
    # ──────────────────────────────────────────────
    def _build_video_panel(self):
        group = QGroupBox("📹 摄像头监控")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 16, 10, 10)
        layout.setSpacing(6)

        # ── 设备选择行 ──────────────────────────────
        device_row = QHBoxLayout()
        device_row.addWidget(_label("设备:", color=COLOR_TEXT_DIM))
        
        # 设备下拉框
        self.cam_device_combo = QComboBox()
        self.cam_device_combo.setPlaceholderText("选择摄像头设备")
        self.cam_device_combo.currentIndexChanged.connect(self._on_cam_device_selected)
        self.cam_device_combo.setFixedWidth(200)
        device_row.addWidget(self.cam_device_combo)
        
        # 刷新按钮
        self.refresh_cam_btn = QPushButton("刷新")
        self.refresh_cam_btn.setStyleSheet(BTN_BASE)
        self.refresh_cam_btn.setFixedWidth(60)
        self.refresh_cam_btn.clicked.connect(self._refresh_camera_list)
        device_row.addWidget(self.refresh_cam_btn)
        
        # 打开/关闭按钮
        self.cam_btn = QPushButton("打开摄像头")
        self.cam_btn.setStyleSheet(BTN_SUCCESS)
        self.cam_btn.setFixedWidth(110)
        self.cam_btn.clicked.connect(self._toggle_camera)
        device_row.addWidget(self.cam_btn)
        
        device_row.addStretch()
        layout.addLayout(device_row)

        # ── 摄像头选项行 ────────────────────────────
        opt_row = QHBoxLayout()

        # 摄像头类型
        opt_row.addWidget(_label("类型:", color=COLOR_TEXT_DIM))
        self.cam_type_combo = QComboBox()
        self.cam_type_combo.addItems(["普通摄像头", "红外摄像头"])
        self.cam_type_combo.setFixedWidth(120)
        self.cam_type_combo.currentIndexChanged.connect(self._on_cam_type_changed)
        opt_row.addWidget(self.cam_type_combo)

        # 红外伪彩色选项
        opt_row.addSpacing(12)
        self.colormap_lbl = _label("伪彩色:", color=COLOR_TEXT_DIM)
        opt_row.addWidget(self.colormap_lbl)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(list(COLORMAP_OPTIONS.keys()))
        self.colormap_combo.setCurrentText("JET")
        self.colormap_combo.setFixedWidth(110)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        opt_row.addWidget(self.colormap_combo)

        # 物品识别开关
        opt_row.addSpacing(12)
        self.detection_checkbox = QCheckBox("物品识别")
        self.detection_checkbox.setEnabled(False)
        self.detection_checkbox.stateChanged.connect(self._on_detection_toggled)
        opt_row.addWidget(self.detection_checkbox)

        # 手势识别开关
        self.gesture_checkbox = QCheckBox("手势识别")
        self.gesture_checkbox.setEnabled(False)
        self.gesture_checkbox.stateChanged.connect(self._on_gesture_toggled)
        opt_row.addWidget(self.gesture_checkbox)

        # 红外标识角标
        opt_row.addStretch()
        self.ir_badge = _label("● IR", bold=True, color="#ff9900", size=12)
        self.ir_badge.setVisible(False)
        opt_row.addWidget(self.ir_badge)

        layout.addLayout(opt_row)

        # 默认禁用红外选项（普通模式）
        self._set_ir_controls_visible(False)

        # ── 视频显示 ────────────────────────────────
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(480, 360)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #0a0a18, stop:1 #0f0f20);
            border: 2px solid {COLOR_BORDER};
            border-radius: 10px;
            color: {COLOR_TEXT_DIM};
            font-size: 14px;
        """)
        self.video_label.setText("📷 摄像头未连接\n\n请选择设备后点击「打开摄像头」")
        layout.addWidget(self.video_label)

        # ── 状态指示 ────────────────────────────────
        status_row = QHBoxLayout()
        self.cam_status_lbl = _label("● 未连接", color=COLOR_TEXT_DIM, size=12)
        status_row.addWidget(self.cam_status_lbl)

        # Gesture status indicator
        self.gesture_status_lbl = _label("", color=COLOR_TEXT_DIM, size=11)
        status_row.addWidget(self.gesture_status_lbl)

        # 检测状态指示
        status_row.addStretch()
        self.detection_status_lbl = _label("识别: 关", color=COLOR_TEXT_DIM, size=12)
        status_row.addWidget(self.detection_status_lbl)

        layout.addLayout(status_row)

        # ── 手势映射图例（折叠式） ────────────────────
        self.gesture_legend = QLabel(
            '<span style="color:#8888aa;font-size:10px;">'
            '✋ <b>Gestures:</b> 1→AB▶ | 2→AB◀ | 3→CD▶ | 4→CD◀ | '
            '5→AB■ | OK→CD■ | 👍→Brake | ✊→All■'
            '</span>'
        )
        self.gesture_legend.setVisible(False)
        self.gesture_legend.setStyleSheet("background: transparent; padding: 2px 0;")
        layout.addWidget(self.gesture_legend)

        return group

    def _set_ir_controls_visible(self, visible: bool):
        """显示/隐藏红外专属控件"""
        self.colormap_lbl.setVisible(visible)
        self.colormap_combo.setVisible(visible)
        self.ir_badge.setVisible(visible)

    def _on_cam_type_changed(self, index: int):
        is_ir = (index == 1)
        self._set_ir_controls_visible(is_ir)
        # 若摄像头正在运行，热切换模式
        if self.camera_thread and self.camera_thread.running:
            self.camera_thread.ir_mode   = is_ir
            self.camera_thread.colormap  = self.colormap_combo.currentText()
            mode_text = "红外" if is_ir else "普通"
            self._log(f"切换为{mode_text}模式", "info")

    def _on_colormap_changed(self, name: str):
        # 运行中实时切换伪彩色方案
        if self.camera_thread and self.camera_thread.running:
            self.camera_thread.colormap = name
            self._log(f"伪彩色方案切换为 {name}", "info")

    def _on_detection_toggled(self, state: int):
        """切换物品识别状态"""
        if self.camera_thread and self.camera_thread.running:
            enable = (state == Qt.Checked)
            self.camera_thread.toggle_detection(enable)
            
            if enable:
                self.detection_status_lbl.setText("识别: 开")
                self.detection_status_lbl.setStyleSheet(f"color: {COLOR_SUCCESS}; background: transparent;")
                self._log("物品识别已启用", "success")
            else:
                self.detection_status_lbl.setText("识别: 关")
                self.detection_status_lbl.setStyleSheet(f"color: {COLOR_TEXT_DIM}; background: transparent;")
                self._log("物品识别已禁用", "info")

    # ─── Gesture → Serial Command Mapping ─────────────────

    GESTURE_ACTIONS = {
        "1":    ("CMD_AB_FORWARD", "AB正转"),
        "2":    ("CMD_AB_REVERSE", "AB反转"),
        "3":    ("CMD_CD_FORWARD", "CD正转"),
        "4":    ("CMD_CD_REVERSE", "CD反转"),
        "5":    ("CMD_AB_STOP",    "AB停止"),
        "OK":   ("CMD_CD_STOP",    "CD停止"),
        "Good": ("CMD_AB_BRAKE",   "AB刹车"),
        "Fist": ("CMD_CD_BRAKE",   "CD刹车"),
    }

    _last_gesture = None
    _gesture_log_count = 0   # counter for per-frame logging

    def _on_gesture_toggled(self, state: int):
        """Toggle gesture recognition on/off"""
        if self.camera_thread and self.camera_thread.running:
            enable = (state == Qt.Checked)
            self.camera_thread.toggle_gesture(enable)
            if enable:
                self.gesture_status_lbl.setText("✋ 手势识别: ON")
                self.gesture_status_lbl.setStyleSheet(f"color: {COLOR_SUCCESS}; background: transparent; font-size: 11px;")
                self.gesture_legend.setVisible(True)
                self._log("手势识别已启用 [打开摄像头 + 勾选手势识别]", "success")
                self._log("手势映射: 1=AB正转 2=AB反转 3=CD正转 4=CD反转 5=AB停 OK=CD停 Good=刹车 Fist=全关", "info")
            else:
                self.gesture_status_lbl.setText("")
                self.gesture_legend.setVisible(False)
                self._log("手势识别已禁用", "info")

    def _on_gesture_result(self, gesture_name: str, confidence: float = 0.0):
        """Handle gesture detection (called every 5 frames) → log + serial command"""
        self._gesture_log_count += 1

        # Update gesture status indicator in camera panel
        conf_str = f" ({confidence:.0%})" if confidence > 0 else ""
        self.gesture_status_lbl.setText(f"✋ {gesture_name}{conf_str}")
        if confidence >= 0.8:
            self.gesture_status_lbl.setStyleSheet(f"color: {COLOR_SUCCESS}; background: transparent; font-size: 11px;")
        elif confidence >= 0.5:
            self.gesture_status_lbl.setStyleSheet(f"color: {COLOR_WARNING}; background: transparent; font-size: 11px;")
        else:
            self.gesture_status_lbl.setStyleSheet(f"color: {COLOR_TEXT_DIM}; background: transparent; font-size: 11px;")

        # Log gesture to console every call (every ~5 frames)
        changed = (gesture_name != self._last_gesture)
        marker = " [CHANGED]" if changed else ""
        conf_str_log = f" (conf: {confidence:.1%})"
        self._log(f"[Gesture #{self._gesture_log_count}] {gesture_name}{conf_str_log}{marker}", "info")

        # Only send serial command when gesture changes
        if not changed:
            self._last_gesture = gesture_name
            return
        self._last_gesture = gesture_name

        action = self.GESTURE_ACTIONS.get(gesture_name)
        if action is None:
            return

        cmd_name, desc = action

        if not self._check_serial():
            self._log(f"  → {desc} (串口未连接，仅识别)", "warning")
            return

        from serial_communicator import (
            CMD_AB_FORWARD, CMD_AB_REVERSE, CMD_AB_STOP, CMD_AB_BRAKE,
            CMD_CD_FORWARD, CMD_CD_REVERSE, CMD_CD_STOP, CMD_CD_BRAKE,
        )
        cmd_map = {
            "CMD_AB_FORWARD": CMD_AB_FORWARD, "CMD_AB_REVERSE": CMD_AB_REVERSE,
            "CMD_AB_STOP": CMD_AB_STOP, "CMD_AB_BRAKE": CMD_AB_BRAKE,
            "CMD_CD_FORWARD": CMD_CD_FORWARD, "CMD_CD_REVERSE": CMD_CD_REVERSE,
            "CMD_CD_STOP": CMD_CD_STOP, "CMD_CD_BRAKE": CMD_CD_BRAKE,
        }
        cmd_byte = cmd_map.get(cmd_name)
        if cmd_byte is not None:
            self.serial.send_command(cmd_byte, f"手势: {gesture_name} → {desc}")
            self._log(f"  → {desc} 已发送", "success")

    # ──────────────────────────────────────────────
    # 串口面板
    # ──────────────────────────────────────────────
    def _build_serial_panel(self):
        """构建串口连接面板"""
        group = QGroupBox("🔌 串口连接")
        grid = QGridLayout(group)
        grid.setContentsMargins(10, 16, 10, 10)
        grid.setSpacing(6)

        # 端口选择
        grid.addWidget(_label("端口:", color=COLOR_TEXT_DIM), 0, 0)
        self.port_combo = QComboBox()
        # 【核心修改点】：设置 editable 为 False（或直接删掉 setEditable），禁止键盘输入修改
        self.port_combo.setEditable(False)
        self.port_combo.setPlaceholderText("请点击刷新获取端口")
        grid.addWidget(self.port_combo, 0, 1)

        # 刷新端口按钮
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setStyleSheet(BTN_BASE)
        self.refresh_btn.setFixedWidth(70)
        self.refresh_btn.clicked.connect(self._refresh_ports)
        grid.addWidget(self.refresh_btn, 0, 2)

        # 波特率选择
        grid.addWidget(_label("波特率:", color=COLOR_TEXT_DIM), 1, 0)
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["9600", "19200", "38400", "57600", "115200", "230400"])
        self.baud_combo.setCurrentText("115200")
        grid.addWidget(self.baud_combo, 1, 1)

        # 连接开关按钮
        self.connect_btn = QPushButton("连接串口")
        self.connect_btn.setStyleSheet(BTN_SUCCESS)
        self.connect_btn.clicked.connect(self._toggle_serial)
        grid.addWidget(self.connect_btn, 1, 2)

        # 状态指示灯
        self.serial_status_lbl = _label("● 未连接", color=COLOR_DANGER, size=12)
        grid.addWidget(self.serial_status_lbl, 2, 0, 1, 3)

        return group

    # ──────────────────────────────────────────────
    # 电机控制面板
    # ──────────────────────────────────────────────
    def _build_motor_panel(self):
        group = QGroupBox("⚙️ 电机控制")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 14, 10, 8)
        layout.setSpacing(4)

        ch_row = QHBoxLayout()
        ch_row.setSpacing(6)
        ch_row.addWidget(_label("通道:", color=COLOR_TEXT_DIM, size=12))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["通道 0 (AB相)", "通道 1 (CD相)"])
        self.channel_combo.currentIndexChanged.connect(self._update_position_display)
        ch_row.addWidget(self.channel_combo, stretch=1)
        layout.addLayout(ch_row)

        # 位置显示
        pos_row = QHBoxLayout()
        pos_row.setSpacing(4)
        pos_row.addWidget(_label("位置:", color=COLOR_TEXT_DIM, size=12))
        self.position_lbl = _label("16", bold=True, color=COLOR_ACCENT2, size=14)
        self.position_lbl.setMinimumWidth(30)
        pos_row.addWidget(self.position_lbl)
        pos_row.addWidget(_label("/", color=COLOR_TEXT_DIM, size=12))
        self.boundary_lbl = _label("0-32", color=COLOR_TEXT_DIM, size=11)
        pos_row.addWidget(self.boundary_lbl)
        pos_row.addStretch()
        layout.addLayout(pos_row)

        layout.addWidget(_hline())

        btn_grid = QGridLayout()
        btn_grid.setSpacing(6)

        self.btn_forward = QPushButton("▶  正转")
        self.btn_forward.setStyleSheet(BTN_BASE)
        self.btn_forward.setMinimumHeight(36)
        self.btn_forward.clicked.connect(lambda: self._motor_cmd(0x01))

        self.btn_reverse = QPushButton("◀  反转")
        self.btn_reverse.setStyleSheet(BTN_BASE)
        self.btn_reverse.setMinimumHeight(36)
        self.btn_reverse.clicked.connect(lambda: self._motor_cmd(0x02))

        self.btn_stop = QPushButton("■  停止")
        self.btn_stop.setStyleSheet(BTN_DANGER)
        self.btn_stop.setMinimumHeight(36)
        self.btn_stop.clicked.connect(lambda: self._motor_cmd(0x03))

        self.btn_brake = QPushButton("⊠  刹车")
        self.btn_brake.setStyleSheet(BTN_DANGER)
        self.btn_brake.setMinimumHeight(36)
        self.btn_brake.clicked.connect(lambda: self._motor_cmd(0x04))

        btn_grid.addWidget(self.btn_forward, 0, 0)
        btn_grid.addWidget(self.btn_reverse, 0, 1)
        btn_grid.addWidget(self.btn_stop,    1, 0)
        btn_grid.addWidget(self.btn_brake,   1, 1)
        layout.addLayout(btn_grid)

        layout.addWidget(_hline())

        # 重置位置 + 自动调零 并排
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(6)
        self.reset_pos_btn = QPushButton("重置位置")
        self.reset_pos_btn.setStyleSheet(BTN_BASE)
        self.reset_pos_btn.clicked.connect(self._reset_position)
        bottom_row.addWidget(self.reset_pos_btn)

        self.auto_zero_btn = QPushButton("自动调零")
        self.auto_zero_btn.setStyleSheet(BTN_BASE)
        self.auto_zero_btn.clicked.connect(self._zero_calibration)
        bottom_row.addWidget(self.auto_zero_btn)
        layout.addLayout(bottom_row)

        return group

    # ──────────────────────────────────────────────
    # 速度面板
    # ──────────────────────────────────────────────
    def _build_speed_panel(self):
        group = QGroupBox("🏃 速度控制")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 14, 10, 8)
        layout.setSpacing(4)

        slider_row = QHBoxLayout()
        slider_row.setSpacing(8)
        slider_row.addWidget(_label("速度:", color=COLOR_TEXT_DIM, size=12))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(
            lambda v: self.speed_val_lbl.setText(str(v))
        )
        slider_row.addWidget(self.speed_slider, stretch=1)
        self.speed_val_lbl = _label("50", bold=True, color=COLOR_ACCENT2, size=14)
        self.speed_val_lbl.setMinimumWidth(30)
        slider_row.addWidget(self.speed_val_lbl)
        layout.addLayout(slider_row)

        self.set_speed_btn = QPushButton("设置速度")
        self.set_speed_btn.setStyleSheet(BTN_BASE)
        self.set_speed_btn.clicked.connect(self._set_speed)
        layout.addWidget(self.set_speed_btn)

        return group

    # ──────────────────────────────────────────────
    # 自动对焦面板
    # ──────────────────────────────────────────────
    def _build_auto_focus_panel(self):
        group = QGroupBox("自动对焦")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 16, 10, 10)
        layout.setSpacing(6)

        # 清晰度指示
        sharpness_row = QHBoxLayout()
        sharpness_row.addWidget(_label("清晰度:", color=COLOR_TEXT_DIM))
        self.sharpness_val_lbl = _label("0.00", bold=True, color=COLOR_ACCENT2, size=14)
        self.sharpness_val_lbl.setMinimumWidth(50)
        sharpness_row.addWidget(self.sharpness_val_lbl)
        
        self.sharpness_bar = QSlider(Qt.Horizontal)
        self.sharpness_bar.setRange(0, 200)
        self.sharpness_bar.setValue(0)
        self.sharpness_bar.setDisabled(True)
        sharpness_row.addWidget(self.sharpness_bar)
        layout.addLayout(sharpness_row)

        # 状态指示
        status_row = QHBoxLayout()
        self.af_status_lbl = _label("状态: 就绪", color=COLOR_TEXT_DIM, size=12)
        status_row.addWidget(self.af_status_lbl)
        self.af_position_lbl = _label("位置: 0", color=COLOR_TEXT_DIM, size=12)
        status_row.addWidget(self.af_position_lbl)
        layout.addLayout(status_row)

        # 控制按钮
        btn_row = QHBoxLayout()
        
        self.af_start_btn = QPushButton("启动对焦")
        self.af_start_btn.setStyleSheet(BTN_SUCCESS)
        self.af_start_btn.setFixedWidth(80)
        self.af_start_btn.clicked.connect(self._toggle_auto_focus)
        btn_row.addWidget(self.af_start_btn)

        self.af_stop_btn = QPushButton("停止对焦")
        self.af_stop_btn.setStyleSheet(BTN_DANGER)
        self.af_stop_btn.setFixedWidth(80)
        self.af_stop_btn.setEnabled(False)
        self.af_stop_btn.clicked.connect(self._stop_auto_focus)
        btn_row.addWidget(self.af_stop_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # 自动检测开关
        self.auto_detect_checkbox = QCheckBox("自动检测模糊并对焦")
        self.auto_detect_checkbox.setEnabled(False)
        self.auto_detect_checkbox.stateChanged.connect(self._on_auto_detect_toggled)
        layout.addWidget(self.auto_detect_checkbox)

        return group

    # ──────────────────────────────────────────────
    # 调零校准面板
    # ──────────────────────────────────────────────
    def _build_zero_calibration_panel(self):
        group = QGroupBox("调零校准")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        
        # 操作说明
        desc = QLabel("""
            <p style="color: #aaaaaa; font-size: 12px; line-height: 1.5;">
            <b style="color: #ffffff;">操作步骤：</b><br>
            1. 选择通道后，点击<b style="color: #ff6b6b;">反转</b>按钮移动电机到左边界<br>
            2. 点击"设置左边界"记录左边界位置<br>
            3. 点击<b style="color: #6ac975;">正转</b>按钮移动电机到右边界<br>
            4. 点击"设置右边界"记录右边界位置<br>
            5. 点击"确定校准"保存边界并自动移动到中间位置
            </p>
        """)
        layout.addWidget(desc)
        
        # 通道选择
        ch_row = QHBoxLayout()
        ch_row.addWidget(_label("通道:", color=COLOR_TEXT_DIM))
        self.zero_channel_combo = QComboBox()
        self.zero_channel_combo.addItems(["通道 0 (AB相)", "通道 1 (CD相)"])
        self.zero_channel_combo.currentIndexChanged.connect(self._zero_on_channel_changed)
        ch_row.addWidget(self.zero_channel_combo, stretch=1)
        layout.addLayout(ch_row)
        
        # 当前位置显示
        pos_row = QHBoxLayout()
        pos_row.addWidget(_label("当前位置:", color=COLOR_TEXT_DIM))
        self.zero_pos_lbl = _label("0", bold=True, color=COLOR_ACCENT, size=18)
        pos_row.addWidget(self.zero_pos_lbl)
        pos_row.addStretch()
        layout.addLayout(pos_row)
        
        layout.addWidget(_hline())
        
        # 边界设置
        bounds_group = QGroupBox("边界设置")
        bounds_layout = QVBoxLayout(bounds_group)
        
        # 左边界
        left_row = QHBoxLayout()
        left_row.addWidget(_label("左边界:", color=COLOR_TEXT_DIM))
        self.zero_left_edit = QLineEdit("0")
        self.zero_left_edit.setStyleSheet(f"background: {COLOR_PANEL}; color: {COLOR_TEXT}; border: 1px solid {COLOR_BORDER}; border-radius: 4px; padding: 4px; width: 60px;")
        self.zero_left_edit.setReadOnly(True)
        left_row.addWidget(self.zero_left_edit)
        self.zero_set_left_btn = QPushButton("设置左边界")
        self.zero_set_left_btn.setStyleSheet(BTN_BASE)
        self.zero_set_left_btn.clicked.connect(self._zero_set_left_bound)
        left_row.addWidget(self.zero_set_left_btn)
        left_row.addStretch()
        bounds_layout.addLayout(left_row)
        
        # 右边界
        right_row = QHBoxLayout()
        right_row.addWidget(_label("右边界:", color=COLOR_TEXT_DIM))
        self.zero_right_edit = QLineEdit("32")
        self.zero_right_edit.setStyleSheet(f"background: {COLOR_PANEL}; color: {COLOR_TEXT}; border: 1px solid {COLOR_BORDER}; border-radius: 4px; padding: 4px; width: 60px;")
        self.zero_right_edit.setReadOnly(True)
        right_row.addWidget(self.zero_right_edit)
        self.zero_set_right_btn = QPushButton("设置右边界")
        self.zero_set_right_btn.setStyleSheet(BTN_BASE)
        self.zero_set_right_btn.clicked.connect(self._zero_set_right_bound)
        right_row.addWidget(self.zero_set_right_btn)
        right_row.addStretch()
        bounds_layout.addLayout(right_row)
        
        # 总行程
        total_row = QHBoxLayout()
        total_row.addWidget(_label("总行程:", color=COLOR_TEXT_DIM))
        self.zero_total_lbl = _label("32 步", color=COLOR_SUCCESS)
        total_row.addWidget(self.zero_total_lbl)
        total_row.addStretch()
        bounds_layout.addLayout(total_row)
        
        # 初始化边界值（从配置加载）
        channel = self.zero_channel_combo.currentIndex()
        config_key = f'channel_{channel}_boundary'
        if config_key in self._config:
            bounds = self._config[config_key]
            self._zero_left_bound = bounds['left']
            self._zero_right_bound = bounds['right']
            self.zero_left_edit.setText(str(self._zero_left_bound))
            self.zero_right_edit.setText(str(self._zero_right_bound))
            self.zero_total_lbl.setText(f"{self._zero_right_bound - self._zero_left_bound} 步")
        else:
            self._zero_left_bound = 0
            self._zero_right_bound = 32
        
        self._zero_calibrating = False  # 防止重复点击确定校准
        
        layout.addWidget(bounds_group)
        
        layout.addWidget(_hline())
        
        # 电机控制
        control_group = QGroupBox("电机控制")
        control_layout = QHBoxLayout(control_group)
        control_layout.setSpacing(10)
        
        self.zero_forward_btn = QPushButton("▶ 正转")
        self.zero_forward_btn.setStyleSheet(BTN_BASE)
        self.zero_forward_btn.clicked.connect(lambda: self._zero_motor_cmd(0x01))
        control_layout.addWidget(self.zero_forward_btn)
        
        self.zero_reverse_btn = QPushButton("◀ 反转")
        self.zero_reverse_btn.setStyleSheet(BTN_BASE)
        self.zero_reverse_btn.clicked.connect(lambda: self._zero_motor_cmd(0x02))
        control_layout.addWidget(self.zero_reverse_btn)
        
        self.zero_stop_btn = QPushButton("■ 停止")
        self.zero_stop_btn.setStyleSheet(BTN_DANGER)
        self.zero_stop_btn.clicked.connect(lambda: self._zero_motor_cmd(0x03))
        control_layout.addWidget(self.zero_stop_btn)
        
        layout.addWidget(control_group)
        
        # 底部按钮
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        
        self.zero_ok_btn = QPushButton("确定校准")
        self.zero_ok_btn.setStyleSheet(BTN_SUCCESS)
        self.zero_ok_btn.clicked.connect(self._zero_apply_calibration)
        btn_row.addWidget(self.zero_ok_btn)
        
        self.zero_reset_btn = QPushButton("重置")
        self.zero_reset_btn.setStyleSheet(BTN_BASE)
        self.zero_reset_btn.clicked.connect(self._zero_reset_bounds)
        btn_row.addWidget(self.zero_reset_btn)
        
        layout.addLayout(btn_row)
        
        # 初始化边界值
        self._zero_left_bound = 0
        self._zero_right_bound = 32
        
        return group
    
    def _zero_motor_cmd(self, action):
        """调零模式下的电机控制"""
        if not self._check_serial():
            return
        
        channel = self.zero_channel_combo.currentIndex()
        speed = self.speed_slider.value()
        self.serial.motor_control(channel, action, speed=speed)
    
    def _zero_on_channel_changed(self):
        """通道切换处理"""
        # 停止所有电机
        if self.serial and self.serial.is_open:
            self.serial.motor_control(0, 0x03)  # 停止AB相
            self.serial.motor_control(1, 0x03)  # 停止CD相
        
        # 更新当前位置显示
        channel = self.zero_channel_combo.currentIndex()
        if self.serial and self.serial.is_open:
            pos = self.serial.get_position(channel)
            self.zero_pos_lbl.setText(str(pos))
        
        # 重置边界显示为默认值
        self._zero_left_bound = 0
        self._zero_right_bound = 32
        self.zero_left_edit.setText("0")
        self.zero_right_edit.setText("32")
        self._zero_update_total()
    
    def _zero_set_left_bound(self):
        """设置左边界"""
        if not self._check_serial():
            return
        
        channel = self.zero_channel_combo.currentIndex()
        self._zero_left_bound = self.serial.get_position(channel)
        self.zero_left_edit.setText(str(self._zero_left_bound))
        self._zero_update_total()
    
    def _zero_set_right_bound(self):
        """设置右边界"""
        if not self._check_serial():
            return
        
        channel = self.zero_channel_combo.currentIndex()
        self._zero_right_bound = self.serial.get_position(channel)
        self.zero_right_edit.setText(str(self._zero_right_bound))
        self._zero_update_total()
    
    def _zero_update_total(self):
        """更新总行程显示"""
        total = abs(self._zero_right_bound - self._zero_left_bound)
        self.zero_total_lbl.setText(f"{total} 步")
    
    def _zero_reset_bounds(self):
        """重置边界（包括电机位置）"""
        self._zero_left_bound = 0
        self._zero_right_bound = 32
        self.zero_left_edit.setText("0")
        self.zero_right_edit.setText("32")
        self.zero_total_lbl.setText("32 步")
        
        # 重置电机位置到0
        if self.serial and self.serial.is_open:
            channel = self.zero_channel_combo.currentIndex()
            self.serial.reset_position(channel, 0)
            self._log(f"通道{channel} 位置已重置为0", "info")
    
    def _zero_apply_calibration(self):
        """应用校准"""
        if self._zero_calibrating:
            self._log("正在校准中，请等待完成...", "warning")
            return
        
        if not self._check_serial():
            return
        
        if self._zero_left_bound >= self._zero_right_bound:
            self._log("左边界必须小于右边界！", "warning")
            return
        
        # 设置校准标志
        self._zero_calibrating = True
        
        try:
            # 停止电机
            self._zero_motor_cmd(0x03)
            
            # 计算新边界
            channel = self.zero_channel_combo.currentIndex()
            total_steps = self._zero_right_bound - self._zero_left_bound
            
            if total_steps <= 0:
                self._log("边界设置无效！", "warning")
                return
            
            # 更新边界配置
            self.serial.update_boundaries(channel, 0, total_steps)
            self.serial.reset_position(channel, 0)
            
            # 保存配置
            if channel == 0:
                self._config['channel_0_boundary'] = {'left': 0, 'right': total_steps}
            else:
                self._config['channel_1_boundary'] = {'left': 0, 'right': total_steps}
            self._save_config()
            
            # 移动到中间位置
            mid_pos = total_steps // 2
            speed = self.speed_slider.value()
            self.serial.move_to_position(channel, mid_pos, speed=speed)
            
            self._log(f"校准完成！新边界: 0-{total_steps}, 中间位置: {mid_pos}", "success")
            
            # 更新手动控制界面的位置显示
            self._update_position_display()
        finally:
            # 重置校准标志
            self._zero_calibrating = False

    # ──────────────────────────────────────────────
    # 日志面板
    # ──────────────────────────────────────────────
    def _build_log_panel(self):
        group = QGroupBox("📋 操作日志")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 16, 10, 10)
        layout.setSpacing(6)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(80)
        layout.addWidget(self.log_text)

        clear_btn = QPushButton("清空日志")
        clear_btn.setStyleSheet(BTN_BASE)
        clear_btn.setFixedHeight(28)
        clear_btn.clicked.connect(self.log_text.clear)
        layout.addWidget(clear_btn, alignment=Qt.AlignRight)

        return group

    # ══════════════════════════════════════════════
    # 摄像头逻辑
    # ══════════════════════════════════════════════
    def _toggle_camera(self):
        """Toggle camera on/off. Auto-selects best available camera if none chosen."""
        if self.camera_thread and self.camera_thread.running:
            self._stop_camera()
        else:
            # Auto-detect & refresh list if empty
            if self.cam_device_combo.count() == 0:
                self._refresh_camera_list()

            # Auto-select first (highest priority) camera if none selected
            device_id = self.cam_device_combo.currentData()
            if device_id is None and self.cam_device_combo.count() > 0:
                self.cam_device_combo.setCurrentIndex(0)
                device_id = self.cam_device_combo.currentData()
                self._log(f"自动选择: {self.cam_device_combo.currentText()}", "info")

            if device_id is None:
                self._log("未检测到摄像头设备", "warning")
                return
            self._start_camera()

    def _refresh_camera_list(self):
        """Auto-detect available cameras (HDMI > USB > built-in priority)"""
        self.cam_device_combo.clear()
        try:
            devices = list_available_cameras(max_devices=6)

            for device_id, name in devices:
                self.cam_device_combo.addItem(name, device_id)

            self._log(f"检测到 {len(devices)} 个摄像头设备", "info")
        except Exception as e:
            self._log(f"摄像头检测失败: {e}", "error")
    def _switch_control_mode(self, mode):
        """切换控制模式（0=手动, 1=自动, 2=调零）"""
        if mode == 0:
            # 切换到手动模式
            self.manual_mode_btn.setChecked(True)
            self.auto_mode_btn.setChecked(False)
            self.zero_mode_btn.setChecked(False)
            self.manual_mode_btn.setStyleSheet(BTN_SUCCESS)
            self.auto_mode_btn.setStyleSheet(BTN_BASE)
            self.zero_mode_btn.setStyleSheet(BTN_BASE)
            
            # 停止自动对焦
            if self.auto_focus_thread and self.auto_focus_thread.isRunning():
                self._stop_auto_focus()
            
            # 退出调零模式
            if self.serial:
                self.serial.set_zeroing_mode(0, False)
                self.serial.set_zeroing_mode(1, False)
            
            # 显示手动面板，隐藏其他面板
            self.auto_panel.hide()
            self.zero_panel.hide()
            self.manual_panel.show()
            self._log("切换到手动控制模式", "info")
            
        elif mode == 1:
            # 切换到自动模式
            self.manual_mode_btn.setChecked(False)
            self.auto_mode_btn.setChecked(True)
            self.zero_mode_btn.setChecked(False)
            self.manual_mode_btn.setStyleSheet(BTN_BASE)
            self.auto_mode_btn.setStyleSheet(BTN_SUCCESS)
            self.zero_mode_btn.setStyleSheet(BTN_BASE)
            
            # 退出调零模式
            if self.serial:
                self.serial.set_zeroing_mode(0, False)
                self.serial.set_zeroing_mode(1, False)
            
            # 显示自动面板，隐藏其他面板
            self.manual_panel.hide()
            self.zero_panel.hide()
            self.auto_panel.show()
            self._log("切换到自动对焦模式", "info")
            
        elif mode == 2:
            # 切换到调零模式
            self.manual_mode_btn.setChecked(False)
            self.auto_mode_btn.setChecked(False)
            self.zero_mode_btn.setChecked(True)
            self.manual_mode_btn.setStyleSheet(BTN_BASE)
            self.auto_mode_btn.setStyleSheet(BTN_BASE)
            self.zero_mode_btn.setStyleSheet(BTN_SUCCESS)
            
            # 停止自动对焦
            if self.auto_focus_thread and self.auto_focus_thread.isRunning():
                self._stop_auto_focus()
            
            # 进入调零模式（禁用边界限制）
            if self.serial:
                self.serial.set_zeroing_mode(0, True)
                self.serial.set_zeroing_mode(1, True)
            
            # 显示调零面板，隐藏其他面板
            self.manual_panel.hide()
            self.auto_panel.hide()
            self.zero_panel.show()
            self._log("切换到调零校准模式（边界限制已禁用）", "info")

    def _on_cam_device_selected(self, index):
        """选择摄像头设备"""
        if index >= 0:
            device_id = self.cam_device_combo.currentData()
            if device_id is not None:
                self._log(f"已选择摄像头设备 {device_id}", "info")

    def _start_camera(self):
        # 从设备下拉框获取选中的设备ID
        device_id = self.cam_device_combo.currentData()

        # 【优化点 3】安全规避：若拿到 None 或非有效值，降级采用默认通道 0
        if device_id is None or not isinstance(device_id, int):
            url = 0
            url_text = "0 (默认)"
        else:
            url = device_id
            url_text = str(device_id)

        is_ir = (self.cam_type_combo.currentIndex() == 1)
        colormap = self.colormap_combo.currentText()

        self.camera_thread = CameraThread(url, ir_mode=is_ir, colormap=colormap)
        self.camera_thread.frame_signal.connect(self._update_video)
        self.camera_thread.error_signal.connect(self._on_cam_error)
        self.camera_thread.connected_signal.connect(self._on_cam_connected)
        self.camera_thread.detection_signal.connect(self._on_detection_result)
        self.camera_thread.gesture_signal.connect(self._on_gesture_result)
        self.camera_thread.start()
        self.cam_btn.setText("关闭摄像头")
        self.cam_btn.setStyleSheet(BTN_DANGER)
        self.detection_checkbox.setEnabled(True)
        self.gesture_checkbox.setEnabled(True)
        self.auto_detect_checkbox.setEnabled(True)
        mode_text = "红外" if is_ir else "普通"
        self._log(f"正在连接{mode_text}摄像头: {url_text}")

    def _stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        self.cam_btn.setText("打开摄像头")
        self.cam_btn.setStyleSheet(BTN_SUCCESS)
        self.detection_checkbox.setChecked(False)
        self.detection_checkbox.setEnabled(False)
        self.gesture_checkbox.setChecked(False)
        self.gesture_checkbox.setEnabled(False)
        self.auto_detect_checkbox.setChecked(False)
        self.auto_detect_checkbox.setEnabled(False)
        self.cam_status_lbl.setText("● 未连接")
        self.cam_status_lbl.setStyleSheet(f"color: {COLOR_TEXT_DIM}; background: transparent;")
        self.detection_status_lbl.setText("识别: 关")
        self.detection_status_lbl.setStyleSheet(f"color: {COLOR_TEXT_DIM}; background: transparent;")
        self.gesture_status_lbl.setText("")
        self.gesture_legend.setVisible(False)
        self._last_gesture = None
        self._gesture_log_count = 0
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText("📷 摄像头未连接\n\n请选择设备后点击「打开摄像头」")
        self._log("摄像头已关闭")

    def _update_video(self, q_image):
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        if self.auto_focus_thread and self.auto_focus_thread.isRunning():
            img = q_image.copy()
            img = img.convertToFormat(4)
            width = img.width()
            height = img.height()
            ptr = img.bits()
            ptr.setsize(height * width * 4)
            arr = np.array(ptr).reshape(height, width, 4)[:, :, :3]
            self.auto_focus_thread.set_frame(arr)

    def _on_cam_error(self, msg):
        self._log(f"[摄像头] {msg}", "warning")

    def _on_cam_connected(self, ok):
        if ok:
            self.cam_status_lbl.setText("● 已连接")
            self.cam_status_lbl.setStyleSheet(f"color: {COLOR_SUCCESS}; background: transparent;")
            self._log("摄像头连接成功", "success")
        else:
            self.cam_status_lbl.setText("● 断开")
            self.cam_status_lbl.setStyleSheet(f"color: {COLOR_DANGER}; background: transparent;")
            self.gesture_status_lbl.setText("")
            self.gesture_legend.setVisible(False)
            self._log("摄像头已断开", "warning")
            self.cam_btn.setText("打开摄像头")
            self.cam_btn.setStyleSheet(BTN_SUCCESS)
            self.detection_checkbox.setChecked(False)
            self.detection_checkbox.setEnabled(False)
            self.gesture_checkbox.setChecked(False)
            self.gesture_checkbox.setEnabled(False)

    def _on_detection_result(self, detections):
        """处理物品识别结果（可选：显示识别信息）"""
        if detections:
            detected_items = [f"{det['class_name']} ({det['confidence']:.2f})" for det in detections]
            self._log(f"检测到: {', '.join(detected_items)}", "info")

    # ══════════════════════════════════════════════
    # 串口逻辑
    # ══════════════════════════════════════════════
    def _refresh_ports(self):
        ports = list_available_ports()
        self.port_combo.clear()
        if ports:
            self.port_combo.addItems(ports)
        else:
            self.port_combo.addItem("（无可用串口）")
        self._log(f"刷新串口列表：{ports or '无'}")

    def _toggle_serial(self):
        if self.serial and self.serial.is_open:
            self.serial.close_port()
            self.connect_btn.setText("连接串口")
            self.connect_btn.setStyleSheet(BTN_SUCCESS)
            self.serial_status_lbl.setText("● 未连接")
            self.serial_status_lbl.setStyleSheet(f"color: {COLOR_DANGER}; background: transparent;")
            self._set_status("串口已断开", COLOR_WARNING)
        else:
            port = self.port_combo.currentText().strip().upper()
            if not port:
                self._log("请输入或选择串口端口号", "warning")
                return
            baud = int(self.baud_combo.currentText())
            self.serial = SerialCommunicator(port, baud)
            self.serial.log_signal.connect(self._log)
            self.serial.status_signal.connect(self._on_status_response)
            self.serial.position_signal.connect(self._on_position_update)
            self.serial.boundary_warning_signal.connect(self._on_boundary_warning)
            if self.serial.open_port():
                self.connect_btn.setText("断开串口")
                self.connect_btn.setStyleSheet(BTN_DANGER)
                self.serial_status_lbl.setText(f"● 已连接 {port}")
                self.serial_status_lbl.setStyleSheet(f"color: {COLOR_SUCCESS}; background: transparent;")
                self._set_status(f"串口已连接: {port} @ {baud}", COLOR_SUCCESS)
                
                # 加载保存的边界配置到串口通信器
                for channel in (0, 1):
                    key = f'channel_{channel}_boundary'
                    if key in self._config:
                        bounds = self._config[key]
                        self.serial.update_boundaries(channel, bounds['left'], bounds['right'])
                        self._log(f"通道{channel} 已加载配置边界: {bounds['left']}-{bounds['right']}", "info")
                
                # 更新位置显示
                self._update_position_display()
            else:
                self._set_status("串口连接失败", COLOR_DANGER)

    # ══════════════════════════════════════════════
    # 电机 / 速度控制
    # ══════════════════════════════════════════════
    def _check_serial(self):
        if not self.serial or not self.serial.is_open:
            self._log("请先连接串口", "warning")
            self._set_status("串口未连接", COLOR_WARNING)
            return False
        return True

    def _motor_cmd(self, action):
        """
        发送电机控制指令
        action: 0x01=正转, 0x02=反转, 0x03=停止, 0x04=刹车
        """
        if not self._check_serial():
            return
        
        channel = self.channel_combo.currentIndex()
        speed   = self.speed_slider.value()
        
        action_names = {
            0x01: "正转", 0x02: "反转",
            0x03: "停止", 0x04: "刹车",
        }
        
        # 使用新的motor_control接口
        result = self.serial.motor_control(channel, action, speed)
        
        if result and result["success"]:
            self._log(f"电机{action_names[action]} (通道{channel}, 速度{speed}) → {result['status_text']}", "success")
        else:
            status_text = result["status_text"] if result else "无响应"
            self._log(f"电机{action_names[action]}指令失败 → {status_text}", "error")

    def _set_speed(self):
        """单独设置速度（不改变运动状态）"""
        if not self._check_serial():
            return
        
        speed = self.speed_slider.value()
        result = self.serial.send_command(speed + 0x0F, f"设置速度 {speed}")
        
        if result and result["success"]:
            self._log(f"设置速度 {speed} → {result['status_text']}", "success")
        else:
            status_text = result["status_text"] if result else "无响应"
            self._log(f"速度设置失败 → {status_text}", "error")

    def _open_calibration_dialog(self):
        """打开调零校准对话框"""
        if not self._check_serial():
            return
        
        channel = self.channel_combo.currentIndex()
        
        # 进入调零模式（禁用边界限制）
        self.serial.set_zeroing_mode(channel, True)
        
        # 创建并显示校准对话框
        dialog = ZeroCalibrationDialog(self.serial, channel, self)
        dialog.exec_()
        
        # 退出调零模式（恢复边界限制）
        self.serial.set_zeroing_mode(channel, False)
        
        # 刷新位置显示
        self._update_position_display()

    def _zero_calibration(self):
        """执行电机自动调零：移动到中间位置"""
        if not self._check_serial():
            return
        
        channel = self.channel_combo.currentIndex()
        bounds = self.serial.get_boundaries(channel)
        mid_pos = (bounds['left'] + bounds['right']) // 2
        
        self._log(f"开始电机{channel}自动调零...", "info")
        
        # 禁用按钮防止重复点击
        self.auto_zero_btn.setEnabled(False)
        self.auto_zero_btn.setText("调零中...")
        
        # 在主线程中获取速度值
        speed = self.speed_slider.value()
        current_pos = self.serial.get_position(channel)
        
        # 计算预计完成时间并设置恢复定时器（保障措施）
        estimated_time = abs(mid_pos - current_pos) * 0.1 + 3.0
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(int(estimated_time * 1000 + 1000), self._reset_auto_zero_btn)
        
        import threading
        def do_zero():
            try:
                # 确定移动方向
                if mid_pos > current_pos:
                    action = 0x01  # 正转
                    target_reached = lambda cur: cur >= mid_pos
                else:
                    action = 0x02  # 反转
                    target_reached = lambda cur: cur <= mid_pos
                
                # 发送移动指令（带速度）
                self.serial.motor_control(channel, action, speed=speed)
                
                # 等待到达目标位置
                import time
                wait_time = abs(mid_pos - current_pos) * 0.1
                end_time = time.time() + wait_time + 2.0
                
                while time.time() < end_time:
                    current = self.serial.get_position(channel)
                    if target_reached(current):
                        break
                    time.sleep(0.02)
                
                # 刹车
                self.serial.motor_control(channel, 0x04)
                time.sleep(0.1)
                
                self._log(f"电机{channel}自动调零完成，已移动到位置 {mid_pos}", "success")
            except Exception as e:
                self._log(f"电机{channel}自动调零失败: {e}", "error")
            finally:
                # 使用 QTimer.singleShot 在主线程中更新GUI
                QTimer.singleShot(0, self._update_position_display)
        
        # 在子线程中执行调零
        threading.Thread(target=do_zero, daemon=True).start()
    
    def _reset_auto_zero_btn(self):
        """重置自动调零按钮状态"""
        self.auto_zero_btn.setEnabled(True)
        self.auto_zero_btn.setText("自动调零")

    def _reset_position(self):
        """重置当前通道的位置为0"""
        if not self._check_serial():
            return
        
        channel = self.channel_combo.currentIndex()
        self.serial.reset_position(channel, 0)
        self._log(f"通道{channel} 位置已重置为0", "success")

    def _on_position_update(self, data):
        """处理位置更新信号"""
        channel = data['channel']
        position = data['position']
        left_bound = data['left_bound']
        right_bound = data['right_bound']
        
        # 更新当前选中通道的显示
        if channel == self.channel_combo.currentIndex():
            self.position_lbl.setText(str(position))
            self.boundary_lbl.setText(f"{left_bound}-{right_bound}")

    def _on_boundary_warning(self, data):
        """处理边界警告信号"""
        channel = data['channel']
        side = data['side']
        position = data['position']
        self._log(f"[边界警告] 电机{channel}到达{side}边界位置 {position}", "warning")
        self._set_status(f"边界警告: 电机{channel}已到达{side}边界", COLOR_WARNING)

    def _update_position_display(self):
        """更新位置显示"""
        if self.serial:
            channel = self.channel_combo.currentIndex()
            position = self.serial.get_position(channel)
            bounds = self.serial.get_boundaries(channel)
            self.position_lbl.setText(str(position))
            self.boundary_lbl.setText(f"{bounds['left']}-{bounds['right']}")

    def _on_status_response(self, data: dict):
        """处理串口响应信号"""
        if data.get("success"):
            self.last_status_info = f"指令0x{data['command']:02X} → {data['status_text']}"
        else:
            self.last_status_info = f"指令失败: {data.get('status_text', '未知')}"

    # ══════════════════════════════════════════════
    # 日志 & 状态栏
    # ══════════════════════════════════════════════
    def _log(self, message, level="info"):
        colors = {
            "info":    COLOR_TEXT,
            "success": COLOR_SUCCESS,
            "warning": COLOR_WARNING,
            "error":   COLOR_DANGER,
        }
        color = colors.get(level, COLOR_TEXT)
        ts = time.strftime("%H:%M:%S")
        self.log_text.append(
            f'<span style="color:{COLOR_TEXT_DIM}">[{ts}]</span> '
            f'<span style="color:{color}">{message}</span>'
        )
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def _set_status(self, msg, color=COLOR_TEXT_DIM):
        self.status_bar.showMessage(msg)
        self.status_bar.setStyleSheet(
            f"QStatusBar {{ color: {color}; background: {COLOR_SURFACE}; "
            f"border-top: 1px solid {COLOR_BORDER}; }}"
        )

    # ══════════════════════════════════════════════
    # 关闭事件
    # ══════════════════════════════════════════════
    # ══════════════════════════════════════════════
    # 自动对焦逻辑
    # ══════════════════════════════════════════════
    def _toggle_auto_focus(self):
        """启动/停止自动对焦"""
        if self.auto_focus_thread and self.auto_focus_thread.isRunning():
            self._stop_auto_focus()
        else:
            self._start_auto_focus()

    def _start_auto_focus(self):
        """启动自动对焦"""
        if not self._check_serial():
            return
        if not self.camera_thread or not self.camera_thread.running:
            self._log("请先打开摄像头", "warning")
            return

        # 获取当前界面设置的速度
        speed = self.speed_slider.value()
        self.auto_focus_thread = AutoFocusThread(self.serial, speed=speed)
        self.auto_focus_thread.status_signal.connect(self._on_af_status)
        self.auto_focus_thread.focus_signal.connect(self._on_af_focus)
        self.auto_focus_thread.position_signal.connect(self._on_af_position)
        self.auto_focus_thread.completed_signal.connect(self._on_af_completed)
        self.auto_focus_thread.start()

        self.af_start_btn.setEnabled(False)
        self.af_stop_btn.setEnabled(True)
        self.af_status_lbl.setText("状态: 搜索中")
        self.af_status_lbl.setStyleSheet(f"color: {COLOR_WARNING}; background: transparent;")
        self._log("自动对焦已启动", "success")

    def _stop_auto_focus(self):
        """停止自动对焦"""
        if self.auto_focus_thread:
            self.auto_focus_thread.stop()
            # 使用 QTimer.singleShot 延迟清理，避免阻塞
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(100, self._cleanup_auto_focus_thread)

        self.af_start_btn.setEnabled(True)
        self.af_stop_btn.setEnabled(False)
        self.af_status_lbl.setText("状态: 已停止")
        self.af_status_lbl.setStyleSheet(f"color: {COLOR_TEXT_DIM}; background: transparent;")
        self._log("自动对焦已停止", "info")

    def _cleanup_auto_focus_thread(self):
        """清理自动对焦线程"""
        if self.auto_focus_thread:
            if self.auto_focus_thread.isRunning():
                self.auto_focus_thread.wait(2000)  # 最多等待2秒
            self.auto_focus_thread = None

    def _on_af_status(self, status: str):
        """处理自动对焦状态更新"""
        self.af_status_lbl.setText(f"状态: {status}")

    def _on_af_focus(self, focus_value: float):
        """处理清晰度值更新"""
        self.sharpness_val_lbl.setText(f"{focus_value:.2f}")
        self.sharpness_bar.setValue(min(int(focus_value), 200))
        
        # 更新颜色指示
        if focus_value >= AF_FOCUS_THRESHOLD:
            self.sharpness_val_lbl.setStyleSheet(f"color: {COLOR_SUCCESS}; background: transparent;")
        else:
            self.sharpness_val_lbl.setStyleSheet(f"color: {COLOR_WARNING}; background: transparent;")

    def _on_af_position(self, position: int):
        """处理位置更新"""
        self.af_position_lbl.setText(f"位置: {position}")

    def _on_af_completed(self, success: bool, position: int, sharpness: float):
        """处理自动对焦完成"""
        if success:
            self.af_status_lbl.setText("状态: 完成")
            self.af_status_lbl.setStyleSheet(f"color: {COLOR_SUCCESS}; background: transparent;")
            self._log(f"自动对焦完成，位置: {position}, 清晰度: {sharpness:.2f}", "success")
        else:
            self.af_status_lbl.setText("状态: 已停止")
            self.af_status_lbl.setStyleSheet(f"color: {COLOR_TEXT_DIM}; background: transparent;")
        
        self.af_start_btn.setEnabled(True)
        self.af_stop_btn.setEnabled(False)

    def _on_auto_detect_toggled(self, state: int):
        """切换自动检测模糊功能"""
        enable = (state == Qt.Checked)
        if enable:
            self._log("已启用自动检测模糊", "info")
        else:
            self._log("已禁用自动检测模糊", "info")

    # ══════════════════════════════════════════════
    # 关闭事件
    # ══════════════════════════════════════════════
    def closeEvent(self, event):
        self._stop_camera()
        self._stop_auto_focus()
        if self.serial and self.serial.is_open:
            self.serial.close_port()
        event.accept()
