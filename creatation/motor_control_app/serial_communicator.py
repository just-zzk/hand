"""
串口通信模块
实现与STM32开发板的串口通信，支持单字节指令协议

协议说明（根据单片机HAL_UART_RxCpltCallback实现）：
- 单字节指令，无需起始/结束字节
- 心跳包：0x00 → 响应0x55（刷新看门狗计时）
- 速度设置：0x10~0x73 (16~115) → 对应速度1~100
- AB电机控制：0x01~0x04 (正转/反转/停止/刹车)
- CD电机控制：0x05~0x08 (正转/反转/停止/刹车)
- 响应：0x55表示成功，0xEE表示失败

发送策略（简化版）：
- 发送任何指令后，都以心跳维持（100ms间隔）
- 只有改变指令时才发送新指令
- 心跳包0x00即可刷新看门狗，无需频繁发送电机指令

边界保护：
- 实时记录电机位置
- 超出边界时自动刹车抱死并发出警告
"""
import serial
import serial.tools.list_ports
import threading
import os
import time
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

# ───────── 指令类型 ─────────
CMD_HEARTBEAT   = 0x00  # 心跳包（刷新看门狗）

# AB电机控制 (通道0)
CMD_AB_FORWARD = 0x01  # AB电机正转
CMD_AB_REVERSE = 0x02  # AB电机反转
CMD_AB_STOP    = 0x03  # AB电机自由停止
CMD_AB_BRAKE   = 0x04  # AB电机刹车抱死

# CD电机控制 (通道1)
CMD_CD_FORWARD = 0x05  # CD电机正转
CMD_CD_REVERSE = 0x06  # CD电机反转
CMD_CD_STOP    = 0x07  # CD电机自由停止
CMD_CD_BRAKE   = 0x08  # CD电机刹车抱死

# 速度设置范围 (上位机发送值)
SPEED_CMD_MIN = 0x10   # 对应速度1
SPEED_CMD_MAX = 0x73   # 对应速度100

# 响应码
ACK_OK  = 0x55
ACK_ERR = 0xEE

# 心跳间隔 (毫秒)
HEARTBEAT_INTERVAL_MS = 100  # 心跳包发送间隔（防止看门狗超时）

# 响应状态映射
STATUS_MAP = {
    ACK_OK:  "执行成功",
    ACK_ERR: "指令错误",
}


def list_available_ports():
    """
    枚举系统可用串口。
    策略（按优先级）：
    1. pyserial comports()  —— 读注册表，速度快
    2. 暴力扫描 COM1~COM256 —— 兜底，覆盖 com0com 等不写注册表的驱动
    两次结果取并集，去重后按端口号排序返回。
    """
    found = set()

    # ── 方法①：注册表枚举 ────────────────────────────────
    for p in serial.tools.list_ports.comports():
        found.add(p.device.upper())

    # ── 方法②：暴力扫描（只在①结果较少时执行，避免拖慢启动） ──
    if len(found) < 2:
        for i in range(1, 257):
            name = f"COM{i}"
            try:
                s = serial.Serial(name, timeout=0)
                s.close()
                found.add(name)
            except serial.SerialException:
                pass
            except Exception:
                pass

    def _port_key(name):
        # 按数字排序：COM3 < COM10
        digits = "".join(c for c in name if c.isdigit())
        return int(digits) if digits else 0

    return sorted(found, key=_port_key)


def speed_to_cmd(speed):
    """
    将速度值(1~100)转换为对应的指令字节(0x10~0x73)
    """
    speed = max(1, min(100, speed))
    return speed + 0x0F  # 0x10~0x73 对应速度 1~100


def cmd_to_speed(cmd):
    """
    将指令字节转换回速度值
    """
    if cmd >= SPEED_CMD_MIN and cmd <= SPEED_CMD_MAX:
        return cmd - 0x0F
    return 0


class SerialCommunicator(QObject):
    """
    串口通信类（单字节协议版本）
    信号：
        log_signal(str)           ── 日志消息
        status_signal(dict)       ── 解析后的响应状态
        heartbeat_signal(bool)    ── 心跳响应状态
        position_signal(dict)     ── 位置更新信号
        boundary_warning_signal(dict) ── 边界警告信号
    """
    log_signal                 = pyqtSignal(str)
    status_signal              = pyqtSignal(dict)
    heartbeat_signal           = pyqtSignal(bool)
    position_signal            = pyqtSignal(dict)
    boundary_warning_signal    = pyqtSignal(dict)

    def __init__(self, port="COM3", baudrate=115200, timeout=0.5):
        super().__init__()
        self.port     = port
        self.baudrate = baudrate
        self.timeout  = timeout
        self.ser      = None
        self._lock    = threading.Lock()
        self._current_speed = 50  # 当前速度缓存
        
        # 心跳定时器（100ms间隔）
        self._heartbeat_timer = QTimer()
        self._heartbeat_timer.setInterval(HEARTBEAT_INTERVAL_MS)
        self._heartbeat_timer.timeout.connect(self._on_heartbeat_timer)

        # ─── 边界配置 ────────────────────────────────────────────
        self._boundaries = {
            0: {'left': 0, 'right': 32},  # AB电机（聚焦）边界
            1: {'left': 0, 'right': 32},  # CD电机（变焦）边界
        }
        
        # ─── 位置跟踪 ────────────────────────────────────────────
        # 根据边界范围计算中间位置
        self._positions = {
            0: (self._boundaries[0]['left'] + self._boundaries[0]['right']) // 2,  # AB电机当前位置（中间位置）
            1: (self._boundaries[1]['left'] + self._boundaries[1]['right']) // 2,  # CD电机当前位置（中间位置）
        }
        
        # ─── 运动状态跟踪 ────────────────────────────────────────
        # 0=停止, 1=正转, -1=反转
        self._motion_state = {
            0: 0,  # AB电机运动状态
            1: 0,  # CD电机运动状态
        }
        
        # ─── 上次位置更新时间 ────────────────────────────────────
        self._last_position_update = {
            0: time.time(),
            1: time.time(),
        }
        
        # ─── 调零模式 ────────────────────────────────────────────
        # 调零模式下边界限制不生效，用于重新校准边界
        self._zeroing_mode = {
            0: False,  # AB电机调零模式
            1: False,  # CD电机调零模式
        }

        # ─── 加载校准配置 ────────────────────────────────────────
        self._load_calibration()

    def _load_calibration(self):
        """加载电机校准配置"""
        config_path = os.path.join(os.path.dirname(__file__), "motor_calibration.txt")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 解析配置文件
                lines = content.split('\n')
                current_section = None
                
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if line.startswith('[') and line.endswith(']'):
                        current_section = line[1:-1]
                    elif '=' in line and current_section:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = int(value.strip())
                        
                        if 'AB' in current_section:
                            if key == 'left_bound':
                                self._boundaries[0]['left'] = value
                            elif key == 'right_bound':
                                self._boundaries[0]['right'] = value
                        elif 'CD' in current_section:
                            if key == 'left_bound':
                                self._boundaries[1]['left'] = value
                            elif key == 'right_bound':
                                self._boundaries[1]['right'] = value
                
                self.log_signal.emit(f"已加载校准配置: AB边界={self._boundaries[0]['left']}-{self._boundaries[0]['right']}, CD边界={self._boundaries[1]['left']}-{self._boundaries[1]['right']}")
            except Exception as e:
                self.log_signal.emit(f"加载校准配置失败: {e}")
        else:
            self.log_signal.emit(f"未找到校准配置文件 {config_path}，使用默认边界")

    def get_boundaries(self, channel):
        """获取指定通道的边界"""
        return self._boundaries.get(channel, {'left': 0, 'right': 32})

    def get_position(self, channel):
        """获取指定通道的当前位置"""
        return self._positions.get(channel, 0)

    def _update_position(self, channel, direction, steps=1):
        """更新位置（根据调零模式决定是否限制边界）
        
        Parameters
        ----------
        channel : int
            通道号
        direction : int
            方向: 1=正转, -1=反转
        steps : int
            步数（默认1步）
            
        Returns
        -------
        bool
            是否允许继续移动（非调零模式下到达边界返回False）
        """
        old_pos = self._positions.get(channel, 16)
        new_pos = old_pos + direction * steps
        
        # 获取边界
        bounds = self._boundaries.get(channel, {'left': 0, 'right': 32})
        left_bound = bounds['left']
        right_bound = bounds['right']
        
        # 检查是否在调零模式
        is_zeroing = self._zeroing_mode.get(channel, False)
        
        if is_zeroing:
            # 调零模式：不限制边界，但记录位置（允许超出0-32）
            self._positions[channel] = new_pos
            self.position_signal.emit({
                'channel': channel,
                'position': new_pos,
                'left_bound': left_bound,
                'right_bound': right_bound,
            })
            return True
        else:
            # 正常模式：限制在边界范围内
            if new_pos < left_bound:
                new_pos = left_bound
            elif new_pos > right_bound:
                new_pos = right_bound
            
            # 更新位置
            self._positions[channel] = new_pos
            
            # 发送位置更新信号
            self.position_signal.emit({
                'channel': channel,
                'position': new_pos,
                'left_bound': left_bound,
                'right_bound': right_bound,
            })
            
            # 检查是否到达边界
            if new_pos <= left_bound or new_pos >= right_bound:
                side = 'left' if new_pos <= left_bound else 'right'
                self._trigger_boundary_warning(channel, side, new_pos)
                return False
            
            return True

    def set_zeroing_mode(self, channel, enabled):
        """设置调零模式
        
        Parameters
        ----------
        channel : int
            通道号
        enabled : bool
            是否启用调零模式
        """
        self._zeroing_mode[channel] = enabled
        if enabled:
            self.log_signal.emit(f"通道{channel} 已进入调零模式（边界限制已禁用）")
        else:
            self.log_signal.emit(f"通道{channel} 已退出调零模式（边界限制已启用）")

    def get_zeroing_mode(self, channel):
        """获取调零模式状态"""
        return self._zeroing_mode.get(channel, False)

    def reset_position(self, channel, position=16):
        """重置位置（用于调零后设置新的原点）"""
        self._positions[channel] = position
        bounds = self._boundaries.get(channel, {'left': 0, 'right': 32})
        self.position_signal.emit({
            'channel': channel,
            'position': position,
            'left_bound': bounds['left'],
            'right_bound': bounds['right'],
        })
        self.log_signal.emit(f"通道{channel} 位置已重置为 {position}")

    def update_boundaries(self, channel, left_bound, right_bound):
        """更新边界配置，并将位置重置为新边界的中间位置"""
        self._boundaries[channel] = {'left': left_bound, 'right': right_bound}
        
        # 计算新的中间位置
        mid_pos = (left_bound + right_bound) // 2
        self._positions[channel] = mid_pos
        
        self.log_signal.emit(f"通道{channel} 边界已更新: {left_bound}-{right_bound}")
        self.log_signal.emit(f"通道{channel} 位置已重置为中间位置: {mid_pos}")

    def _trigger_boundary_warning(self, channel, side, position):
        """触发边界警告"""
        self.log_signal.emit(f"[警告] 电机{channel}到达{side}边界位置 {position}")
        self.boundary_warning_signal.emit({
            'channel': channel,
            'side': side,
            'position': position,
            'timestamp': time.time(),
        })

    # ──────────────────────────────────────────────
    # 连接管理
    # ──────────────────────────────────────────────
    def open_port(self, port=None):
        if port:
            self.port = port
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
            )
            if self.ser.is_open:
                self.log_signal.emit(f"串口 {self.port} 打开成功（波特率 {self.baudrate}）")
                # 启动心跳保活
                self._start_heartbeat()
                return True
            return False
        except Exception as e:
            self.log_signal.emit(f"打开串口失败: {e}")
            return False

    def close_port(self):
        self._stop_heartbeat()
        
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.log_signal.emit(f"串口 {self.port} 已关闭")

    @property
    def is_open(self):
        return self.ser is not None and self.ser.is_open

    # ──────────────────────────────────────────────
    # 心跳定时器控制
    # ──────────────────────────────────────────────
    def _start_heartbeat(self):
        """启动心跳保活（100ms间隔）"""
        if not self._heartbeat_timer.isActive():
            self._heartbeat_timer.start()
            self.log_signal.emit(f"心跳保活已启动（间隔{HEARTBEAT_INTERVAL_MS}ms）")

    def _stop_heartbeat(self):
        """停止心跳保活"""
        if self._heartbeat_timer.isActive():
            self._heartbeat_timer.stop()
            self.log_signal.emit("心跳保活已停止")

    def _on_heartbeat_timer(self):
        """心跳定时器回调，每100ms发送一次心跳包"""
        if not self.is_open:
            return
        
        # 更新持续运动中的位置
        self._update_moving_position()
        
        result = self._send_command_internal(CMD_HEARTBEAT, "心跳包")
        
        if result:
            self.heartbeat_signal.emit(result["success"])
        else:
            self.heartbeat_signal.emit(False)

    def _update_moving_position(self):
        """更新持续运动中的位置（基于速度估算）"""
        now = time.time()
        
        for channel in (0, 1):
            motion = self._motion_state.get(channel, 0)
            if motion == 0:
                continue  # 电机未运动
            
            last_time = self._last_position_update.get(channel, now)
            elapsed = now - last_time
            
            # 每约100ms更新一次位置（对应电机运动）
            if elapsed >= 0.1:
                # 更新位置
                if not self._update_position(channel, motion):
                    # 到达边界，停止运动
                    self._motion_state[channel] = 0
                    # 发送刹车指令
                    brake_cmd = CMD_AB_BRAKE if channel == 0 else CMD_CD_BRAKE
                    self._send_command_internal(brake_cmd, "边界刹车")
                
                self._last_position_update[channel] = now

    # ──────────────────────────────────────────────
    # 内部指令发送
    # ──────────────────────────────────────────────
    def _send_command_internal(self, cmd_byte, desc="指令"):
        """
        内部指令发送方法，不触发日志（由定时器调用时避免重复日志）
        """
        if not self.is_open:
            return None

        frame = bytes([cmd_byte])

        with self._lock:
            try:
                self.ser.write(frame)
                response = self.ser.read(1)
            except Exception as e:
                self.log_signal.emit(f"发送异常: {e}")
                return None

        if len(response) != 1:
            return None

        resp_byte = response[0]
        status_text = STATUS_MAP.get(resp_byte, f"未知(0x{resp_byte:02X})")
        
        result = {
            "command":     cmd_byte,
            "response":    resp_byte,
            "status":      resp_byte,
            "status_text": status_text,
            "success":     (resp_byte == ACK_OK),
        }
        return result

    # ──────────────────────────────────────────────
    # 指令发送 & 响应接收（带日志）
    # ──────────────────────────────────────────────
    def send_command(self, cmd_byte, desc="指令"):
        """
        发送单字节指令并等待响应。

        Parameters
        ----------
        cmd_byte : int
            要发送的指令字节 (0x00~0xFF)
        desc : str
            指令描述，用于日志显示

        Returns
        -------
        dict | None
            解析后的响应字典，失败返回 None
        """
        if not self.is_open:
            self.log_signal.emit("串口未打开，无法发送指令")
            return None

        frame = bytes([cmd_byte])
        self.log_signal.emit(f"发送: 0x{cmd_byte:02X} ({desc})")

        with self._lock:
            try:
                self.ser.reset_input_buffer()
                self.ser.write(frame)
                response = self.ser.read(1)
            except Exception as e:
                self.log_signal.emit(f"发送/接收异常: {e}")
                return None

        if len(response) != 1:
            self.log_signal.emit(f"响应长度错误: 期望1字节，实际{len(response)}字节")
            return None

        resp_byte = response[0]
        status_text = STATUS_MAP.get(resp_byte, f"未知(0x{resp_byte:02X})")
        
        self.log_signal.emit(f"响应: 0x{resp_byte:02X} → {status_text}")
        
        result = {
            "command":     cmd_byte,
            "response":    resp_byte,
            "status":      resp_byte,
            "status_text": status_text,
            "success":     (resp_byte == ACK_OK),
        }
        self.status_signal.emit(result)
        return result

    def send_heartbeat(self):
        """
        手动发送心跳包
        
        Returns
        -------
        dict | None
        """
        return self.send_command(CMD_HEARTBEAT, "心跳包")

    # ──────────────────────────────────────────────
    # 电机控制指令（高层接口）
    # ──────────────────────────────────────────────
    def motor_control(self, channel, action, speed=None):
        """
        电机控制：发送指令后持续心跳维持，不频繁发送指令
        
        Parameters
        ----------
        channel : int
            通道号: 0=AB相, 1=CD相
        action : int
            动作类型: 0x01=正转, 0x02=反转, 0x03=停止, 0x04=刹车
        speed : int | None
            速度值(1~100)，None表示使用当前缓存的速度
            
        Returns
        -------
        dict | None
            指令的响应结果
        """
        action_map = {
            0x01: (CMD_AB_FORWARD, CMD_CD_FORWARD),  # 正转
            0x02: (CMD_AB_REVERSE, CMD_CD_REVERSE),  # 反转
            0x03: (CMD_AB_STOP, CMD_CD_STOP),        # 停止
            0x04: (CMD_AB_BRAKE, CMD_CD_BRAKE),      # 刹车
        }
        
        cmd_desc = {
            0x01: "正转", 0x02: "反转", 0x03: "停止", 0x04: "刹车"
        }
        
        if action not in action_map:
            self.log_signal.emit(f"未知动作: 0x{action:02X}")
            return None
        
        cmd = action_map[action][channel]
        desc = f"通道{channel} {cmd_desc[action]}"
        
        # 更新运动状态
        if action == 0x01:
            self._motion_state[channel] = 1  # 正转
        elif action == 0x02:
            self._motion_state[channel] = -1  # 反转
        else:
            self._motion_state[channel] = 0  # 停止
        
        # 边界保护：正转/反转时检查边界
        if action in (0x01, 0x02):
            direction = 1 if action == 0x01 else -1
            
            # 检查边界
            if not self._update_position(channel, direction):
                # 超出边界，发送刹车指令
                self.log_signal.emit(f"[边界保护] 电机{channel} {cmd_desc[action]}超出边界，已自动刹车")
                brake_cmd = CMD_AB_BRAKE if channel == 0 else CMD_CD_BRAKE
                self._motion_state[channel] = 0  # 更新运动状态为停止
                result = self.send_command(brake_cmd, f"通道{channel} 边界刹车")
                self._start_heartbeat()
                return result
        
        # 如果指定了新速度，先发送速度设置指令
        if speed is not None and speed != self._current_speed:
            speed_cmd = speed_to_cmd(speed)
            result = self.send_command(speed_cmd, f"设置速度 {speed}")
            if not result or not result["success"]:
                return result
            self._current_speed = speed
        
        # 发送电机指令
        result = self.send_command(cmd, desc)
        
        # 确保心跳正在运行（维持当前状态）
        self._start_heartbeat()
        
        return result

    def move_to_position(self, channel, target_position, speed=None):
        """
        移动电机到指定位置（带边界保护）
        
        Parameters
        ----------
        channel : int
            通道号: 0=AB相, 1=CD相
        target_position : int
            目标位置
        speed : int | None
            速度值(1~100)，None表示使用当前缓存的速度
            
        Returns
        -------
        bool
            是否成功到达目标位置
        """
        bounds = self._boundaries.get(channel, {'left': 0, 'right': 32})
        left_bound = bounds['left']
        right_bound = bounds['right']
        
        target_position = max(left_bound, min(right_bound, target_position))
        
        current_pos = self._positions.get(channel, 0)
        diff = target_position - current_pos
        
        if diff == 0:
            self.log_signal.emit(f"电机{channel}已在目标位置 {target_position}")
            return True
        
        direction = 1 if diff > 0 else -1
        
        self.log_signal.emit(f"电机{channel} 移动到位置 {target_position}（方向: {'正转' if direction > 0 else '反转'}）")
        
        if speed is not None:
            speed_cmd = speed_to_cmd(speed)
            self.send_command(speed_cmd, f"设置速度 {speed}")
            self._current_speed = speed
        
        action = 0x01 if direction > 0 else 0x02
        self.motor_control(channel, action)
        
        move_start_time = time.time()
        last_log_time = move_start_time
        
        from PyQt5.QtCore import QEventLoop, QTimer
        
        loop = QEventLoop()
        timer = QTimer()
        timer.setInterval(50)
        timer.start()
        
        reached_target = False
        timeout = False
        stopped = False
        
        def check_position():
            nonlocal reached_target, timeout, stopped
            
            current_pos = self._positions.get(channel, 0)
            elapsed = time.time() - move_start_time
            
            # 检查是否被外部停止
            if stopped:
                timer.stop()
                self.motor_control(channel, 0x04)
                self.log_signal.emit(f"电机{channel} 移动已被停止于位置 {current_pos}")
                loop.quit()
                return
            
            if direction > 0 and current_pos >= target_position:
                timer.stop()
                self.motor_control(channel, 0x04)
                self.log_signal.emit(f"电机{channel} 已到达位置 {current_pos}")
                reached_target = True
                loop.quit()
                return
            
            if direction < 0 and current_pos <= target_position:
                timer.stop()
                self.motor_control(channel, 0x04)
                self.log_signal.emit(f"电机{channel} 已到达位置 {current_pos}")
                reached_target = True
                loop.quit()
                return
            
            if elapsed > 10.0:
                timer.stop()
                self.motor_control(channel, 0x04)
                self.log_signal.emit(f"电机{channel} 移动超时，已停止于位置 {current_pos}")
                timeout = True
                loop.quit()
                return
            
            if time.time() - last_log_time > 0.5:
                self.log_signal.emit(f"电机{channel} 移动中... 当前位置: {current_pos}, 目标: {target_position}")
        
        timer.timeout.connect(check_position)
        loop.exec_()
        
        return reached_target

    def zero_calibration(self, channel):
        """
        调零：移动到左边界(位置0)，然后移动到中间位置
        
        Parameters
        ----------
        channel : int
            通道号: 0=AB相, 1=CD相
            
        Returns
        -------
        bool
            是否成功
        """
        bounds = self._boundaries.get(channel, {'left': 0, 'right': 32})
        left_bound = bounds['left']
        right_bound = bounds['right']
        middle_pos = (left_bound + right_bound) // 2
        
        self.log_signal.emit(f"电机{channel} 开始调零: 左边界={left_bound}, 右边界={right_bound}, 中间位置={middle_pos}")
        
        # 先移动到左边界
        if not self.move_to_position(channel, left_bound):
            self.log_signal.emit(f"电机{channel} 调零失败")
            return False
        
        # 再移动到中间位置
        if not self.move_to_position(channel, middle_pos):
            self.log_signal.emit(f"电机{channel} 移动到中间位置失败")
            return False
        
        self.log_signal.emit(f"电机{channel} 调零完成，当前位置: {self._positions.get(channel)}")
        return True

    def stop_motor(self, channel):
        """
        停止指定通道的电机
        """
        cmd = CMD_AB_STOP if channel == 0 else CMD_CD_STOP
        result = self.send_command(cmd, f"通道{channel} 停止")
        
        # 确保心跳正在运行
        self._start_heartbeat()
        
        return result
