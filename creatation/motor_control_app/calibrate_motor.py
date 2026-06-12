"""
电机边界校准脚本
用于手动测试和确定聚焦/变焦电机的左右边界
"""
import serial
import time
import os

# 串口配置
PORT = 'COM5'  # 请根据实际情况修改
BAUD = 115200
TIMEOUT = 1

# 指令定义
CMD_HEARTBEAT = 0x00
CMD_AB_FORWARD = 0x01  # AB电机正转

CMD_AB_REVERSE = 0x02  # AB电机反转
CMD_AB_STOP = 0x03     # AB电机停止
CMD_AB_BRAKE = 0x04    # AB电机刹车
CMD_CD_FORWARD = 0x05  # CD电机正转
CMD_CD_REVERSE = 0x06  # CD电机反转
CMD_CD_STOP = 0x07     # CD电机停止
CMD_CD_BRAKE = 0x08    # CD电机刹车

def connect_serial():
    """连接串口"""
    try:
        ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
        time.sleep(2)  # 等待连接稳定
        print(f"[连接] 成功连接到 {PORT}")
        return ser
    except Exception as e:
        print(f"[错误] 无法连接串口: {e}")
        return None

def send_command(ser, cmd, description=""):
    """发送单字节指令并接收响应"""
    try:
        ser.write(bytes([cmd]))
        time.sleep(0.05)  # 等待单片机处理

        # 读取响应
        if ser.in_waiting > 0:
            response = ser.read(1)
            if response == b'\x55':
                print(f"[发送] {description} -> 0x{cmd:02X} | [响应] 0x55 OK")
                return True
            else:
                print(f"[发送] {description} -> 0x{cmd:02X} | [响应] 0x{response[0]:02X} ERROR")
                return False
        else:
            print(f"[发送] {description} -> 0x{cmd:02X} | [响应] 无响应")
            return False
    except Exception as e:
        print(f"[错误] 发送指令失败: {e}")
        return False

def calibrate_motor(ser, motor_name, forward_cmd, reverse_cmd, stop_cmd):
    """
    校准单个电机的边界

    参数:
        motor_name: 电机名称 (用于显示)
        forward_cmd: 正转指令
        reverse_cmd: 反转指令
        stop_cmd: 停止指令
    """
    import threading

    print(f"\n{'='*60}")
    print(f"开始校准 {motor_name} 电机")
    print(f"{'='*60}")
    print("操作说明:")
    print("  W - 连续正转 (按住不放持续移动)")
    print("  S - 连续反转 (按住不放持续移动)")
    print("  Q - 停止电机")
    print("  X - 完成校准并保存边界")
    print(f"{'='*60}\n")

    position = 0
    left_bound = 0
    right_bound = 0
    is_moving = False
    move_direction = 0  # 0=停止, 1=正转, -1=反转
    stop_heartbeat = threading.Event()
    move_thread = None

    def heartbeat_thread():
        """后台心跳线程，保持电机控制器活跃"""
        while not stop_heartbeat.is_set():
            try:
                ser.write(bytes([CMD_HEARTBEAT]))
                time.sleep(0.1)
            except:
                break

    def continuous_move(direction, cmd):
        """持续发送移动指令直到停止"""
        nonlocal position, is_moving, move_direction, left_bound, right_bound

        consecutive_errors = 0
        max_consecutive_errors = 3  # 连续3次错误认为到达边界

        while is_moving and move_direction == direction:
            response = send_command(ser, cmd, f"{motor_name}{'正转' if direction > 0 else '反转'}")
            if response:
                consecutive_errors = 0
                position += direction
                print(f"  [位置] {position}")
            else:
                consecutive_errors += 1
                print(f"  [警告] 电机无响应 ({consecutive_errors}/{max_consecutive_errors})")

                # 连续多次无响应，认为到达边界
                if consecutive_errors >= max_consecutive_errors:
                    print(f"  [检测] 电机似乎已到达边界位置: {position}")
                    is_moving = False
                    move_direction = 0

                    # 确定边界
                    if direction > 0:
                        right_bound = position
                        print(f"  [记录] 右边界设置为: {right_bound}")
                    else:
                        left_bound = position
                        print(f"  [记录] 左边界设置为: {left_bound}")
                    break

            time.sleep(0.15)

    hb_thread = threading.Thread(target=heartbeat_thread, daemon=True)
    hb_thread.start()
    print("[心跳] 自动心跳已启动 (100ms间隔)")

    input("按回车开始校准，确保镜头/电机处于自由状态...")

    print("\n请使用 W(正转)/S(反转) 键让电机移动到边界")
    print("提示: 按住 W 或 S 不放，电机将持续移动\n")

    def stop_motor():
        """停止电机移动"""
        nonlocal is_moving, move_direction
        if is_moving:
            is_moving = False
            move_direction = 0
            time.sleep(0.2)
            send_command(ser, stop_cmd, f"{motor_name}停止")
            print(f"  [位置] {position} (已停止)")

    while True:
        key = input("请输入指令 (W/S/Q/X): ").strip().upper()

        if key == 'W':
            # 停止当前的任何移动
            stop_motor()
            time.sleep(0.3)

            # 开始新的正转
            is_moving = True
            move_direction = 1
            move_thread = threading.Thread(target=continuous_move, args=(1, forward_cmd), daemon=True)
            move_thread.start()
            print(f"  [开始] 正转中... 按 Q 停止")

        elif key == 'S':
            stop_motor()
            time.sleep(0.3)

            # 开始新的反转
            is_moving = True
            move_direction = -1
            move_thread = threading.Thread(target=continuous_move, args=(-1, reverse_cmd), daemon=True)
            move_thread.start()
            print(f"  [开始] 反转中... 按 Q 停止")

        elif key == 'Q':
            stop_motor()

        elif key == 'X':
            stop_motor()

            # 确定边界
            if position > 0:
                right_bound = position
                left_bound = 0
            elif position < 0:
                left_bound = position
                right_bound = 0
            else:
                left_bound = 0
                right_bound = 0

            print(f"\n{'='*60}")
            print(f"{motor_name} 电机校准完成!")
            print(f"  左边界: {left_bound}")
            print(f"  右边界: {right_bound}")
            print(f"  总行程: {right_bound - left_bound} 步")
            print(f"{'='*60}")

            stop_heartbeat.set()
            time.sleep(0.2)

            # 移动到中间位置
            middle = (left_bound + right_bound) // 2
            move_steps = middle - position

            print(f"\n移动到中间位置 (步数: {abs(move_steps)}, 方向: {'正向' if move_steps > 0 else '反向'})...")

            # 先确保电机停止
            send_command(ser, stop_cmd, f"{motor_name}停止")
            time.sleep(0.3)

            # 逐步移动到中间位置
            for i in range(abs(move_steps)):
                if move_steps > 0:
                    send_command(ser, forward_cmd, f"{motor_name}正转")
                    position += 1
                else:
                    send_command(ser, reverse_cmd, f"{motor_name}反转")
                    position -= 1
                print(f"  [移动] {i+1}/{abs(move_steps)} -> 当前位置: {position}")
                time.sleep(0.2)  # 增加延迟确保电机完成一步

            # 连续发送多次停止指令确保电机停止
            print("  [停止] 发送停止指令...")
            for _ in range(3):
                send_command(ser, stop_cmd, f"{motor_name}停止")
                time.sleep(0.1)

            print(f"已移动到中间位置，最终位置: {position}")

            return left_bound, right_bound
        else:
            print("无效指令，请重新输入!")

def main():
    """主函数"""
    print("\n" + "="*60)
    print("        电机边界校准工具")
    print("="*60)
    print(f"串口配置: {PORT} @ {BAUD} baud")
    print("="*60)

    # 连接串口
    ser = connect_serial()
    if ser is None:
        input("\n按回车退出...")
        return

    try:
        # 发送心跳包测试连接
        print("\n测试串口通信...")
        if not send_command(ser, CMD_HEARTBEAT, "心跳包"):
            print("[警告] 心跳包无响应，可能连接有问题")

        # 校准AB电机 (聚焦)
        ab_left, ab_right = calibrate_motor(ser, "AB(聚焦)", CMD_AB_FORWARD, CMD_AB_REVERSE, CMD_AB_STOP)

        # 校准CD电机 (变焦)
        cd_left, cd_right = calibrate_motor(ser, "CD(变焦)", CMD_CD_FORWARD, CMD_CD_REVERSE, CMD_CD_STOP)

        # 保存校准结果
        print("\n" + "="*60)
        print("校准结果汇总:")
        print("="*60)
        print(f"AB电机(聚焦): 左边界={ab_left}, 右边界={ab_right}")
        print(f"CD电机(变焦): 左边界={cd_left}, 右边界={cd_right}")
        print("="*60)

        # 保存到文件
        config_path = os.path.join(os.path.dirname(__file__), "motor_calibration.txt")
        with open(config_path, 'w') as f:
            f.write(f"# 电机边界校准结果\n")
            f.write(f"# 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\n")
            f.write(f"[AB电机 - 聚焦]\n")
            f.write(f"left_bound={ab_left}\n")
            f.write(f"right_bound={ab_right}\n")
            f.write(f"\n")
            f.write(f"[CD电机 - 变焦]\n")
            f.write(f"left_bound={cd_left}\n")
            f.write(f"right_bound={cd_right}\n")

        print(f"\n校准结果已保存到: {config_path}")

    except KeyboardInterrupt:
        print("\n\n[中断] 用户取消操作")
    except Exception as e:
        print(f"\n[错误] {e}")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("\n[关闭] 串口已关闭")

    input("\n按回车退出...")

if __name__ == "__main__":
    main()
