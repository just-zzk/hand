"""
自动对焦模块
实现基于优化粗精双阶段扫描算法的自动对焦功能
"""
import cv2
import numpy as np
import time
from PyQt5.QtCore import QObject, pyqtSignal, QThread

# ==========================================
# 自动对焦模块全局常量配置
# ==========================================
AF_MIN_STEP = 2
AF_INITIAL_STEP = 20
AF_MAX_POSITION = 200
AF_MIN_POSITION = 0
AF_FOCUS_THRESHOLD = 50
AF_STABILITY_THRESHOLD = 3
AF_MOTOR_DELAY = 0.15
AF_EVAL_SKIP_FRAMES = 3


class SharpnessEvaluator:
    """清晰度评价器，使用拉普拉斯方差"""

    @staticmethod
    def laplacian_variance(image):
        """评价图像清晰度（值越大越清晰）"""
        if image is None:
            return 0
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 适当缩小分辨率加快计算，同时过滤高频噪声
        small = cv2.resize(gray, (240, 180))
        return cv2.Laplacian(small, cv2.CV_64F).var()


class AutoFocusController(QObject):
    status_signal = pyqtSignal(str)
    focus_signal = pyqtSignal(float)
    position_signal = pyqtSignal(int)
    completed_signal = pyqtSignal(bool, int, float)

    def __init__(self, communicator, speed=50):
        super().__init__()
        self.communicator = communicator
        self.running = False
        self._current_frame = None
        self._speed = speed  # 保存速度参数

    def set_frame(self, frame):
        self._current_frame = frame

    def _get_current_position(self):
        """从通信器获取实际位置"""
        if self.communicator:
            try:
                return self.communicator.get_position(0)
            except Exception:
                pass
        return 0

    def start(self):
        self.running = True
        self._search_loop()

    def stop(self):
        self.running = False
        if self.communicator:
            try:
                self.communicator.motor_control(0, 0x04)  # 强制刹车停止
            except Exception:
                pass

    def _search_loop(self):
        """
        改良版两阶段自动对焦算法：全局粗扫 + 局部精扫
        完全基于绝对位置控制，通过加大延迟彻底解决画面滞后导致的位置偏差问题
        """
        self.status_signal.emit("开始自动对焦...")

        # 1. 获取物理边界配置
        bounds = self.communicator.get_boundaries(0)
        min_pos = bounds['left']
        max_pos = bounds['right']
        self.status_signal.emit(f"物理边界范围: {min_pos} - {max_pos}")

        # 用于存储所有扫描过位置的清晰度，避免重复扫描
        scores = {}
        global_best_score = -1
        global_best_pos = min_pos

        # ==========================================
        # 阶段 1: 全局粗扫 (大步长快速定位峰值区间)
        # ==========================================
        COARSE_STEP = 15
        self.status_signal.emit(f"【阶段1】开始全局粗扫，步长: {COARSE_STEP}...")

        # 生成粗扫绝对位置序列
        coarse_positions = list(range(min_pos, max_pos + 1, COARSE_STEP))
        if coarse_positions[-1] != max_pos:
            coarse_positions.append(max_pos)

        for pos in coarse_positions:
            if not self.running:
                self._stop_and_exit(global_best_pos, global_best_score)
                return

            # 驱动电机移动并非阻塞等待到达
            if self._move_and_wait(pos):
                # 【核心修改】加大延迟：等待 0.35 秒
                # 0.05秒让电机机械抖动和惯性彻底静止 + 0.30秒确保相机硬件缓冲区积压的旧帧完全刷新
                time.sleep(0.35)

                # 评估当前帧清晰度
                score = self._evaluate_and_log(pos)
                scores[pos] = score

                if score > global_best_score:
                    global_best_score = score
                    global_best_pos = pos
                self.status_signal.emit(f"位置 {pos:3d} -> 清晰度: {score:.2f}")

        # ==========================================
        # 阶段 2: 局部精扫 (在粗扫最佳位置附近细化)
        # ==========================================
        if not self.running:
            self._stop_and_exit(global_best_pos, global_best_score)
            return

        FINE_STEP = 2  # 精细微调步长
        FINE_RANGE = 12  # 在粗扫最佳位置的左右各扩展12步范围内扫描

        fine_min = max(min_pos, global_best_pos - FINE_RANGE)
        fine_max = min(max_pos, global_best_pos + FINE_RANGE)

        self.status_signal.emit(f"【阶段2】开始精细微调，目标范围: {fine_min} ~ {fine_max}...")

        fine_positions = list(range(fine_min, fine_max + 1, FINE_STEP))

        for pos in fine_positions:
            if not self.running:
                self._stop_and_exit(global_best_pos, global_best_score)
                return

            # 如果粗扫已经评估过这个绝对位置，直接跳过
            if pos in scores:
                continue

            if self._move_and_wait(pos):
                # 同步加大精扫的曝光与刷新延迟
                time.sleep(0.35)

                score = self._evaluate_and_log(pos)
                scores[pos] = score

                if score > global_best_score:
                    global_best_score = score
                    global_best_pos = pos
                self.status_signal.emit(f"位置 {pos:3d} (精扫) -> 清晰度: {score:.2f}")

        # ==========================================
        # 阶段 3: 打印扫描日志并最终回位
        # ==========================================
        self.status_signal.emit("对焦扫描全记录:")
        for pos in sorted(scores.keys()):
            marker = " ← 【最大清晰度峰值】" if pos == global_best_pos else ""
            self.status_signal.emit(f"  位置 {pos:3d}: 清晰度 {scores[pos]:.2f}{marker}")

        self.status_signal.emit(f"正在驱动镜头回到最终最佳位置: {global_best_pos}")
        self._move_and_wait(global_best_pos)
        time.sleep(0.2)  # 最终回位的稳定

        if self.running:
            self.status_signal.emit(f"自动对焦成功！最佳位置: {global_best_pos}")
            self.completed_signal.emit(True, global_best_pos, global_best_score)

    def _move_and_wait(self, target_pos, timeout=4.0):
        """
        安全的绝对位置移动与非阻塞等待
        """
        if not self.communicator:
            return False

        try:
            # 统一通过底层move_to_position发送绝对位置控制指令
            self.communicator.move_to_position(0, target_pos, speed=self._speed)
        except Exception as e:
            self.status_signal.emit(f"发送移动指令异常: {e}")
            return False

        # 非阻塞循环：等待实际位置到达目标位置
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self.running:
                return False

            curr_pos = self._get_current_position()
            # 绝对位置完全相等则判定到达
            if abs(curr_pos - target_pos) <= 0:
                time.sleep(0.05)  # 释放CPU前给电机机械结构50ms的微小驻留消抖时间
                return True

            time.sleep(0.02)  # 防止死循环过度占用CPU

        self.status_signal.emit(f"提示：移动到位置 {target_pos} 超过预设响应时间，强行继续")
        return True

    def _evaluate_and_log(self, pos):
        """计算图像清晰度并发送实时信号"""
        score = SharpnessEvaluator.laplacian_variance(self._current_frame)
        self.focus_signal.emit(score)
        self.position_signal.emit(pos)
        return score

    def _stop_and_exit(self, best_pos, best_score):
        """异常或中途停止时的退出处理"""
        if self.communicator:
            try:
                self.communicator.motor_control(0, 0x04)
            except Exception:
                pass
        self.status_signal.emit("自动对焦已被用户终止")
        self.completed_signal.emit(False, best_pos, best_score)


class AutoFocusThread(QThread):
    status_signal = pyqtSignal(str)
    focus_signal = pyqtSignal(float)
    position_signal = pyqtSignal(int)
    completed_signal = pyqtSignal(bool, int, float)

    def __init__(self, communicator, speed=50):
        super().__init__()
        self.controller = AutoFocusController(communicator, speed=speed)
        self.controller.status_signal.connect(self.status_signal.emit)
        self.controller.focus_signal.connect(self.focus_signal.emit)
        self.controller.position_signal.connect(self.position_signal.emit)
        self.controller.completed_signal.connect(self.completed_signal.emit)

    def set_frame(self, frame):
        self.controller.set_frame(frame)

    def run(self):
        self.controller.start()

    def stop(self):
        self.controller.stop()