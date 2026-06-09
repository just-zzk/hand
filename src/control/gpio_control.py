"""
GPIO外设控制模块 - GPIO Peripheral Control Module
根据手势识别结果控制 LED、风扇、蜂鸣器等设备
"""

import time
from typing import Dict, Optional

# 嵌入式平台GPIO库（开发阶段可使用模拟模式）
try:
    import RPi.GPIO as GPIO
    _GPIO_AVAILABLE = True
except ImportError:
    _GPIO_AVAILABLE = False
    # 模拟 GPIO 用于非嵌入式开发环境
    class _MockGPIO:
        BCM = "BCM"
        BOARD = "BOARD"
        OUT = "OUT"
        IN = "IN"
        HIGH = True
        LOW = False
        _pins = {}

        @staticmethod
        def setmode(mode): pass

        @staticmethod
        def setup(pin, mode, initial=None):
            _MockGPIO._pins[pin] = initial if initial is not None else False

        @staticmethod
        def output(pin, state):
            _MockGPIO._pins[pin] = state

        @staticmethod
        def input(pin):
            return _MockGPIO._pins.get(pin, False)

        @staticmethod
        def cleanup():
            _MockGPIO._pins.clear()

    GPIO = _MockGPIO


class GPIOController:
    """
    GPIO设备控制器

    管理手势与外设的映射关系，带连续帧确认防误触发
    """

    # GPIO引脚定义 (BCM编号)
    PIN_LED = 17       # LED
    PIN_FAN = 18       # 风扇
    PIN_BUZZER = 27    # 蜂鸣器

    # Gesture → action mapping (keys match GESTURE_LABELS in knn_classifier.py)
    GESTURE_ACTIONS = {
        "1":    ("led", "on"),
        "2":    ("led", "off"),
        "3":    ("fan", "on"),
        "4":    ("fan", "off"),
        "5":    ("buzzer", "beep"),
        "OK":   ("buzzer", "beep"),
        "Good": ("welcome", None),
        "Fist": ("all_off", None),
    }

    def __init__(self, confirm_frames: int = 5):
        """
        初始化GPIO控制器

        Args:
            confirm_frames: 连续识别确认帧数，防止误触发
        """
        self.confirm_frames = confirm_frames
        self._last_gesture: Optional[str] = None
        self._confirm_count: int = 0
        self._state: Dict[str, bool] = {"led": False, "fan": False}
        self._init_gpio()

    def _init_gpio(self):
        """初始化GPIO引脚"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.PIN_LED, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.PIN_FAN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.PIN_BUZZER, GPIO.OUT, initial=GPIO.LOW)

    def execute(self, gesture_name: str) -> Optional[str]:
        """
        执行手势对应的控制动作（含连续帧确认）

        Args:
            gesture_name: 识别到的手势名称

        Returns:
            执行的动作描述，或 None（确认中/无动作）
        """
        # 连续帧确认逻辑
        if gesture_name == self._last_gesture:
            self._confirm_count += 1
        else:
            self._last_gesture = gesture_name
            self._confirm_count = 1
            return None  # 新手势，开始计数

        if self._confirm_count < self.confirm_frames:
            return None  # 未达到确认帧数

        # 达到确认帧数，执行动作
        action = self.GESTURE_ACTIONS.get(gesture_name)
        if action is None:
            return None

        device, cmd = action
        desc = self._do_action(device, cmd)
        return desc

    def _do_action(self, device: str, cmd: str) -> str:
        """Execute device-specific control action"""
        if device == "led":
            if cmd == "on":
                GPIO.output(self.PIN_LED, GPIO.HIGH)
                self._state["led"] = True
                return "LED ON"
            elif cmd == "off":
                GPIO.output(self.PIN_LED, GPIO.LOW)
                self._state["led"] = False
                return "LED OFF"

        elif device == "fan":
            if cmd == "on":
                GPIO.output(self.PIN_FAN, GPIO.HIGH)
                self._state["fan"] = True
                return "Fan ON"
            elif cmd == "off":
                GPIO.output(self.PIN_FAN, GPIO.LOW)
                self._state["fan"] = False
                return "Fan OFF"

        elif device == "buzzer":
            if cmd == "beep":
                GPIO.output(self.PIN_BUZZER, GPIO.HIGH)
                time.sleep(0.3)
                GPIO.output(self.PIN_BUZZER, GPIO.LOW)
                return "Buzzer beep"

        elif device == "welcome":
            # Welcome mode: LED flash + buzzer chirp
            for _ in range(3):
                GPIO.output(self.PIN_LED, GPIO.HIGH)
                GPIO.output(self.PIN_BUZZER, GPIO.HIGH)
                time.sleep(0.15)
                GPIO.output(self.PIN_LED, GPIO.LOW)
                GPIO.output(self.PIN_BUZZER, GPIO.LOW)
                time.sleep(0.15)
            return "Welcome mode"

        elif device == "all_off":
            GPIO.output(self.PIN_LED, GPIO.LOW)
            GPIO.output(self.PIN_FAN, GPIO.LOW)
            GPIO.output(self.PIN_BUZZER, GPIO.LOW)
            self._state["led"] = False
            self._state["fan"] = False
            return "All OFF"

        return f"Unknown: {device}/{cmd}"

    def cleanup(self):
        """清理GPIO资源"""
        GPIO.cleanup()
