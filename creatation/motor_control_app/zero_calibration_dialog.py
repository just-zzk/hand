"""调零校准对话框"""
import sys
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QMessageBox, QWidget
)
from PyQt5.QtCore import Qt

COLOR_BG = "#1a1a2e"
COLOR_PANEL = "#252540"
COLOR_TEXT = "#ffffff"
COLOR_TEXT_DIM = "#cccccc"
COLOR_ACCENT = "#8a7aff"
COLOR_SUCCESS = "#6ac975"
COLOR_DANGER = "#ff6b6b"
COLOR_BORDER = "#666688"

BTN_BASE = f"""
    QPushButton {{
        border: 1px solid {COLOR_BORDER};
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 14px;
        font-weight: 600;
        color: {COLOR_TEXT};
        background: {COLOR_PANEL};
        min-width: 80px;
    }}
    QPushButton:hover {{
        background: {COLOR_ACCENT};
        border-color: {COLOR_ACCENT};
    }}
"""

BTN_SUCCESS = f"""
    QPushButton {{
        border: 1px solid {COLOR_SUCCESS};
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 14px;
        font-weight: 600;
        color: {COLOR_SUCCESS};
        background: {COLOR_PANEL};
        min-width: 80px;
    }}
    QPushButton:hover {{
        background: {COLOR_SUCCESS};
        color: #fff;
    }}
"""

BTN_DANGER = f"""
    QPushButton {{
        border: 1px solid {COLOR_DANGER};
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 14px;
        font-weight: 600;
        color: {COLOR_DANGER};
        background: {COLOR_PANEL};
        min-width: 80px;
    }}
    QPushButton:hover {{
        background: {COLOR_DANGER};
        color: #fff;
    }}
"""


class ZeroCalibrationDialog(QDialog):
    """调零校准对话框"""
    
    def __init__(self, communicator, channel, parent=None):
        super().__init__(parent)
        self.communicator = communicator
        self.channel = channel
        self.left_bound = 0
        self.right_bound = 32
        
        self.setWindowTitle(f"通道 {channel} 调零校准")
        self.setFixedSize(400, 450)
        self.setStyleSheet(f"background: {COLOR_BG};")
        
        self._build_ui()
        
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(25, 25, 25, 25)
        
        # 标题
        title = QLabel("电机调零校准")
        title.setStyleSheet(f"color: {COLOR_TEXT}; font-size: 18px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 操作说明
        desc = QLabel("""
            <p style="color: #cccccc; font-size: 13px; line-height: 1.6;">
            <b style="color: #ffffff;">操作步骤：</b><br>
            1. 点击正转/反转按钮移动电机到左边界<br>
            2. 点击"设置左边界"记录左边界位置<br>
            3. 点击正转按钮移动电机到右边界<br>
            4. 点击"设置右边界"记录右边界位置<br>
            5. 点击"确定"保存边界并自动移动到中间位置
            </p>
        """)
        desc.setAlignment(Qt.AlignLeft)
        layout.addWidget(desc)
        
        # 当前位置显示
        pos_group = QGroupBox("当前位置")
        pos_group.setStyleSheet(f"""
            QGroupBox {{
                color: {COLOR_TEXT}; 
                border: 1px solid {COLOR_BORDER}; 
                border-radius: 8px;
                padding: 15px;
                font-weight: bold;
                font-size: 14px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                background: {COLOR_BG};
            }}
        """)
        pos_layout = QVBoxLayout(pos_group)
        
        self.pos_display = QLabel("0")
        self.pos_display.setStyleSheet(f"color: {COLOR_ACCENT}; font-size: 36px; font-weight: bold;")
        self.pos_display.setAlignment(Qt.AlignCenter)
        pos_layout.addWidget(self.pos_display)
        layout.addWidget(pos_group)
        
        # 边界显示
        bounds_group = QGroupBox("边界设置")
        bounds_group.setStyleSheet(f"""
            QGroupBox {{
                color: {COLOR_TEXT}; 
                border: 1px solid {COLOR_BORDER}; 
                border-radius: 8px;
                padding: 15px;
                font-weight: bold;
                font-size: 14px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                background: {COLOR_BG};
            }}
        """)
        bounds_layout = QVBoxLayout(bounds_group)
        
        # 左边界
        left_row = QHBoxLayout()
        left_label = QLabel("左边界:")
        left_label.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 13px;")
        left_row.addWidget(left_label)
        self.left_bound_edit = QLineEdit(str(self.left_bound))
        self.left_bound_edit.setStyleSheet(f"background: {COLOR_PANEL}; color: {COLOR_TEXT}; border: 1px solid {COLOR_BORDER}; border-radius: 4px; padding: 6px; font-size: 14px; width: 60px;")
        self.left_bound_edit.setReadOnly(True)
        left_row.addWidget(self.left_bound_edit)
        self.set_left_btn = QPushButton("设置左边界")
        self.set_left_btn.setStyleSheet(BTN_BASE)
        self.set_left_btn.clicked.connect(self._set_left_bound)
        left_row.addWidget(self.set_left_btn)
        left_row.addStretch()
        bounds_layout.addLayout(left_row)
        
        # 右边界
        right_row = QHBoxLayout()
        right_label = QLabel("右边界:")
        right_label.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 13px;")
        right_row.addWidget(right_label)
        self.right_bound_edit = QLineEdit(str(self.right_bound))
        self.right_bound_edit.setStyleSheet(f"background: {COLOR_PANEL}; color: {COLOR_TEXT}; border: 1px solid {COLOR_BORDER}; border-radius: 4px; padding: 6px; font-size: 14px; width: 60px;")
        self.right_bound_edit.setReadOnly(True)
        right_row.addWidget(self.right_bound_edit)
        self.set_right_btn = QPushButton("设置右边界")
        self.set_right_btn.setStyleSheet(BTN_BASE)
        self.set_right_btn.clicked.connect(self._set_right_bound)
        right_row.addWidget(self.set_right_btn)
        right_row.addStretch()
        bounds_layout.addLayout(right_row)
        
        # 总行程
        total_row = QHBoxLayout()
        total_label = QLabel("总行程:")
        total_label.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 13px;")
        total_row.addWidget(total_label)
        self.total_display = QLabel("32 步")
        self.total_display.setStyleSheet(f"color: {COLOR_SUCCESS}; font-weight: bold; font-size: 14px;")
        total_row.addWidget(self.total_display)
        total_row.addStretch()
        bounds_layout.addLayout(total_row)
        
        layout.addWidget(bounds_group)
        
        # 电机控制按钮
        control_group = QGroupBox("电机控制")
        control_group.setStyleSheet(f"""
            QGroupBox {{
                color: {COLOR_TEXT}; 
                border: 1px solid {COLOR_BORDER}; 
                border-radius: 8px;
                padding: 15px;
                font-weight: bold;
                font-size: 14px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                background: {COLOR_BG};
            }}
        """)
        control_layout = QVBoxLayout(control_group)
        
        btn_grid = QHBoxLayout()
        btn_grid.setSpacing(10)
        
        self.forward_btn = QPushButton("▶ 正转")
        self.forward_btn.setStyleSheet(BTN_BASE)
        self.forward_btn.clicked.connect(self._motor_forward)
        
        self.reverse_btn = QPushButton("◀ 反转")
        self.reverse_btn.setStyleSheet(BTN_BASE)
        self.reverse_btn.clicked.connect(self._motor_reverse)
        
        self.stop_btn = QPushButton("■ 停止")
        self.stop_btn.setStyleSheet(BTN_DANGER)
        self.stop_btn.clicked.connect(self._motor_stop)
        
        btn_grid.addWidget(self.forward_btn)
        btn_grid.addWidget(self.reverse_btn)
        btn_grid.addWidget(self.stop_btn)
        control_layout.addLayout(btn_grid)
        
        layout.addWidget(control_group)
        
        # 底部按钮
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)
        
        self.ok_btn = QPushButton("确定")
        self.ok_btn.setStyleSheet(BTN_SUCCESS)
        self.ok_btn.clicked.connect(self._apply_calibration)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setStyleSheet(BTN_BASE)
        self.cancel_btn.clicked.connect(self.reject)
        
        self.reset_btn = QPushButton("重置")
        self.reset_btn.setStyleSheet(BTN_BASE)
        self.reset_btn.clicked.connect(self._reset_bounds)
        
        bottom_row.addWidget(self.reset_btn)
        bottom_row.addWidget(self.cancel_btn)
        bottom_row.addWidget(self.ok_btn)
        layout.addLayout(bottom_row)
        
        # 连接位置更新信号
        if self.communicator:
            self.communicator.position_signal.connect(self._on_position_update)
            
    def _on_position_update(self, data):
        """处理位置更新"""
        if data['channel'] == self.channel:
            self.pos_display.setText(str(data['position']))
            
    def _motor_forward(self):
        """正转电机"""
        if self.communicator and self.communicator.is_open:
            self.communicator.motor_control(self.channel, 0x01, speed=30)
            
    def _motor_reverse(self):
        """反转电机"""
        if self.communicator and self.communicator.is_open:
            self.communicator.motor_control(self.channel, 0x02, speed=30)
            
    def _motor_stop(self):
        """停止电机"""
        if self.communicator and self.communicator.is_open:
            self.communicator.motor_control(self.channel, 0x03)
            
    def _set_left_bound(self):
        """设置左边界"""
        if self.communicator:
            self.left_bound = self.communicator.get_position(self.channel)
            self.left_bound_edit.setText(str(self.left_bound))
            self._update_total()
            
    def _set_right_bound(self):
        """设置右边界"""
        if self.communicator:
            self.right_bound = self.communicator.get_position(self.channel)
            self.right_bound_edit.setText(str(self.right_bound))
            self._update_total()
            
    def _update_total(self):
        """更新总行程显示"""
        total = abs(self.right_bound - self.left_bound)
        self.total_display.setText(f"{total} 步")
        
    def _reset_bounds(self):
        """重置边界"""
        self.left_bound = 0
        self.right_bound = 32
        self.left_bound_edit.setText("0")
        self.right_bound_edit.setText("32")
        self.total_display.setText("32 步")
        
    def _apply_calibration(self):
        """应用校准"""
        if self.left_bound >= self.right_bound:
            QMessageBox.warning(self, "警告", "左边界必须小于右边界！")
            return
            
        # 停止电机
        self._motor_stop()
        
        # 计算新边界
        total_steps = self.right_bound - self.left_bound
        if total_steps <= 0:
            QMessageBox.warning(self, "警告", "边界设置无效！")
            return
            
        # 更新边界配置
        if self.communicator:
            self.communicator.update_boundaries(self.channel, 0, total_steps)
            # 重置位置为0
            self.communicator.reset_position(self.channel, 0)
            # 移动到中间位置
            mid_pos = total_steps // 2
            self.communicator.move_to_position(self.channel, mid_pos, speed=30)
            
            QMessageBox.information(self, "成功", f"校准完成！\n新边界: 0-{total_steps}\n中间位置: {mid_pos}")
        
        self.accept()
        
    def closeEvent(self, event):
        """关闭时停止电机"""
        self._motor_stop()
        super().closeEvent(event)


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    dialog = ZeroCalibrationDialog(None, 0)
    dialog.exec_()