"""
虚拟串口模拟器 - 单字节协议版本
用途：配合虚拟串口工具（VSPD / com0com）调试上位机，
      无需真实 STM32 硬件即可验证指令是否正确。

协议说明（与单片机HAL_UART_RxCpltCallback一致）：
- 单字节指令，无需起始/结束字节
- 速度设置：0x10~0x73 (16~115) → 对应速度1~100
- AB电机控制：0x01~0x04 (正转/反转/停止/刹车)
- CD电机控制：0x05~0x08 (正转/反转/停止/刹车)
- 响应：0x55表示成功，0xEE表示失败

使用步骤
--------
1. 安装虚拟串口软件，创建一对互联串口，例如 COM10 ↔ COM11
2. 本脚本监听 COM11（模拟 STM32 端）
3. 上位机连接 COM10 发送指令
4. 本脚本收到指令后打印解析结果，并回复合法响应

运行方式
--------
    python virtual_serial_simulator.py            # 默认 COM11, 115200
    python virtual_serial_simulator.py COM11 9600
"""

import sys
import time
import serial

# ─── 指令定义（与单片机保持一致） ──────────────────────────────────────
CMD_NAMES = {
    0x01: "AB_FORWARD",
    0x02: "AB_REVERSE",
    0x03: "AB_STOP",
    0x04: "AB_BRAKE",
    0x05: "CD_FORWARD",
    0x06: "CD_REVERSE",
    0x07: "CD_STOP",
    0x08: "CD_BRAKE",
}

# 响应码
ACK_OK  = 0x55
ACK_ERR = 0xEE

# 速度指令范围
SPEED_MIN = 0x10   # 对应速度1
SPEED_MAX = 0x73   # 对应速度100

# ─── 模拟电机状态 ─────────────────────────────────────────────────────
motor_state = {
    0: {"running": False, "direction": 1, "speed": 1, "braking": False},  # AB
    1: {"running": False, "direction": 1, "speed": 1, "braking": False},  # CD
}


def run(port="COM11", baudrate=115200):
    print(f"{'='*60}")
    print(f"  虚拟串口模拟器 (单字节协议)")
    print(f"  监听: {port}  波特率: {baudrate}")
    print(f"  上位机请连接另一端（如 COM10）")
    print(f"{'='*60}")
    print(f"\n指令协议:")
    print(f"  AB电机: 0x01~0x04 (正转/反转/停止/刹车)")
    print(f"  CD电机: 0x05~0x08 (正转/反转/停止/刹车)")
    print(f"  速度:   0x10~0x73 (16~115) → 速度1~100")
    print(f"  响应:   0x55=成功, 0xEE=失败")
    print(f"\n等待上位机发送指令...\n")

    try:
        ser = serial.Serial(port, baudrate, timeout=2)
    except serial.SerialException as e:
        print(f"[错误] 无法打开 {port}: {e}")
        print("请先用 VSPD 或 com0com 创建虚拟串口对，再运行本脚本。")
        return

    print(f"[OK] {port} 已打开\n")

    try:
        while True:
            raw = ser.read(1)
            if not raw:
                continue

            cmd = raw[0]
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] ← 收到指令: 0x{cmd:02X}")

            # 判断指令类型
            if cmd >= SPEED_MIN and cmd <= SPEED_MAX:
                # 速度设置指令
                speed = cmd - 0x0F
                print(f"       → 速度设置: {speed}")
                # 更新两个通道的速度
                motor_state[0]["speed"] = speed
                motor_state[1]["speed"] = speed
                ser.write(bytes([ACK_OK]))
                print(f"       → 回复: 0x{ACK_OK:02X} (成功)\n")
                continue

            # 电机控制指令
            if cmd in CMD_NAMES:
                cmd_name = CMD_NAMES[cmd]
                if cmd <= 0x04:
                    channel = 0
                    ch_name = "AB"
                else:
                    channel = 1
                    ch_name = "CD"

                state = motor_state[channel]

                if cmd in (0x01, 0x05):  # 正转
                    state["running"] = True
                    state["direction"] = 1
                    state["braking"] = False
                    print(f"       → {ch_name}电机正转 (速度: {state['speed']})")

                elif cmd in (0x02, 0x06):  # 反转
                    state["running"] = True
                    state["direction"] = 0
                    state["braking"] = False
                    print(f"       → {ch_name}电机反转 (速度: {state['speed']})")

                elif cmd in (0x03, 0x07):  # 自由停止
                    state["running"] = False
                    state["braking"] = False
                    print(f"       → {ch_name}电机自由停止")

                elif cmd in (0x04, 0x08):  # 刹车抱死
                    state["running"] = False
                    state["braking"] = True
                    print(f"       → {ch_name}电机刹车抱死")

                ser.write(bytes([ACK_OK]))
                print(f"       → 回复: 0x{ACK_OK:02X} (成功)\n")
                continue

            # 未知指令
            print(f"       → 未知指令")
            ser.write(bytes([ACK_ERR]))
            print(f"       → 回复: 0x{ACK_ERR:02X} (失败)\n")

    except KeyboardInterrupt:
        print("\n[停止] 用户中断")
    finally:
        ser.close()
        print(f"[OK] {port} 已关闭")


if __name__ == "__main__":
    _port = sys.argv[1] if len(sys.argv) > 1 else "COM11"
    _baud = int(sys.argv[2]) if len(sys.argv) > 2 else 115200
    run(_port, _baud)
