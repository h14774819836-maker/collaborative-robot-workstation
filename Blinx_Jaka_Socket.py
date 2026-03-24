import configparser
import json
import math
import threading
import time
from socket import *

# endregion
class Blinx_Jaka_Socket():
    def __init__(self, public_class):
        self.max_attempts = 2 # 设置最大尝试次数
        self.attempt_count = 0  # 初始化尝试次数计数器
        self.public_class = public_class
        # 坐标系
        self.COORD_BASE = 0  # 基座标/世界坐标系
        self.COORD_JOINT = 1  # 关节坐标系
        self.COORD_TOOL = 2  # 工具坐标系
        self.speeds_joint = [10, 90, 180]  # 机器人关节速度，小中大°/s
        self.speeds_linear = [100, 1000, 1800]  # 机器人直线速度，小中大mm/s
        self.jaka_connect_sucess = False
        while self.attempt_count < self.max_attempts:
            try:
                self.socket_jaka = socket(AF_INET, SOCK_STREAM)
                self.socket_jaka.setsockopt(SOL_SOCKET, SO_KEEPALIVE, 1)  # 在客户端开启心跳维护
                self.socket_jaka.connect((str(self.public_class.jaka_ip), int(self.public_class.jaka_port)))
                self.BUF_SIZE = 10000
                # 开线程，防止堵塞
                self.jaka_t = threading.Thread(target=self.blinx_jaka_msg)
                self.jaka_t.start()
                self.jaka_connect_sucess = True
                self.blinx_get_robot_state()#获取机器人状态
                time.sleep(0.5)
                if self.public_class.jaka_power==0:
                    self.blinx_power_on()
                    #上电需要一个过程时间
                    while not self.public_class.jaka_enable:
                        self.blinx_get_robot_state()  # 获取机器人状态
                        time.sleep(0.5)
                        if self.public_class.jaka_power == 1 and self.public_class.jaka_enable==False:
                            print("上电完成，等待使能")
                            self.blinx_enable_robot()
                            self.blinx_get_robot_state()  # 获取机器人状态
                            time.sleep(0.5)
                            if self.public_class.jaka_enable:
                                print("使能完成")
                                break
                        else:
                            print("等待上电")
                            # self.blinx_power_on()
                elif self.public_class.jaka_power==1:
                    if not self.public_class.jaka_enable:
                        self.blinx_enable_robot()
                        print("使能完成")
                break
            except Exception as e:
                self.socket_jaka.close()
                self.attempt_count += 1  # 增加尝试次数
                if self.attempt_count >= self.max_attempts:
                    self.jaka_connect_sucess = False
                    print("机器人连接失败超过最大尝试次数，停止尝试")
                    break

    # 度数转弧度
    def blinx_d2r(self, degree):
        return degree / 180.0 * math.pi

    # 弧度转度数
    def blinx_r2d(self, radian):
        return radian / math.pi * 180.0

    # 通讯关闭方法
    def blinx_jaka_close(self):
        self.socket_jaka.close()

    # 消息接收
    def blinx_jaka_msg(self):
        while True:
            time.sleep(0.01)
            try:
                # 消息接受
                data = self.socket_jaka.recv(self.BUF_SIZE)
                if len(data) > 0:
                    jason_data = json.loads(data.decode())
                    """判断是否是获取机器人数据"""
                    if "get_joint_pos" in str(data):
                        self.public_class.joint_pos = jason_data['joint_pos']
                    elif "get_tcp_pos" in str(data):
                        self.public_class.tcp_pos = jason_data['tcp_pos']
                    elif "din_status" in str(data):
                        self.public_class.din_status = jason_data['din_status']
                    elif "get_robot_state" in str(data):
                        self.public_class.jaka_enable = jason_data['enable']
                        self.public_class.jaka_power = jason_data['power']
                    elif "get_analog_output" in str(data):
                        self.public_class.ao_value = jason_data['value']
                    elif "get_analog_input" in str(data):
                        self.public_class.ai_value = jason_data['value']
            except Exception as e:
                print("blinx_jaka_msg: ", e)
                continue

    # 发送消息
    def blinx_jaka_send(self, msg):
        time.sleep(0.1)
        self.socket_jaka.send(msg.encode('utf-8'))

    # 获取角度数据
    def blinx_get_joint_pos(self):
        data = '{"cmdName":"get_joint_pos"}'
        """发送消息"""
        self.blinx_jaka_send(data)

    # 获取位姿数据
    def blinx_get_tcp_pos(self):
        data = '{"cmdName":"get_tcp_pos"}'
        """发送消息"""
        self.blinx_jaka_send(data)

    # 上电
    def blinx_power_on(self):
        data = '{"cmdName":"power_on"}'
        """发送消息"""
        self.blinx_jaka_send(data)

    # 下电
    def blinx_power_off(self):
        data = '{"cmdName":"power_off"}'
        """发送消息"""
        self.blinx_jaka_send(data)

    # 机器人上使能
    def blinx_enable_robot(self):
        data = '{"cmdName":"enable_robot"}'
        """发送消息"""
        self.blinx_jaka_send(data)

    # 机器人下使能
    def blinx_disable_robot(self):
        data = '{"cmdName":"disable_robot"}'
        """发送消息"""
        self.blinx_jaka_send(data)
    #获取机器人状态
    def blinx_get_robot_state(self):
        data='{"cmdName":"get_robot_state"}'
        """发送消息"""
        self.blinx_jaka_send(data)

    # 关闭机器人和控制器
    def blinx_shutdown(self):
        data = '{"cmdName":"shutdown"}'
        """发送消息"""
        self.blinx_jaka_send(data)

    # 设置工具坐标系
    def blinx_set_tool_offsets(self, tooloffset, id, name):
        data = '{"cmdName": "set_tool_offsets",'+' "tooloffset": '+str(tooloffset)+', "id":'+str(id)+', "name": '+str(name)+'}'
        """发送消息"""
        self.blinx_jaka_send(data)

    # 选择工具坐标系
    def blinx_set_tool_id(self, toolid):
        data =' {"cmdName":"set_tool_id","tool_id":'+str(toolid)+'}'
        """发送消息"""
        self.blinx_jaka_send(data)
    """
    机器人角度控制
    jog_mode 代表运动模式可填的值有 3 个:
        Jog_stop(Jog 停止): 0
        Continue(连续运动): 1
        Increment(步进运动): 2
        ABS 绝对运动: 3
    coord_map 代表坐标系的选择可填的值有 3 个:
        在世界坐标系下运动: 0
        在关节空间运动: 1
        在工具坐标系下运动: 2
    jnum 在关节空间下代表轴号 1 轴到六轴的轴号分别对应数字 0 到 5
    jnum 在笛卡尔空间代表 x，y，z，rx，ry，rz 分别对应数字 0 到 5
    jogvel 代表速度
    poscmd 代表步进值，单关节运动为 deg,空间单轴运动为 mm
    """

    def blinx_jog(self, jog_mode, coord_map, jnum, jogvel, poscmd):
        data = ''
        if jog_mode == 0:
            data = '{"cmdName":"jog","jog_mode":' + str(jog_mode) + ', "coord_map":' \
                   + str(coord_map) + ', "jnum":' + str(jnum) + '}'
        elif jog_mode == 1:
            data = '{"cmdName":"jog","jog_mode":' + str(jog_mode) + ', "coord_map":' \
                   + str(coord_map) + ', "jnum":' + str(jnum) + ', "jogvel":' + str(jogvel) + '}'
        elif jog_mode == 2 or jog_mode == 3:
            data = '{"cmdName":"jog","jog_mode":' + str(jog_mode) + ', "coord_map":' \
                   + str(coord_map) + ', "jnum":' + str(jnum) + ', "jogvel":' \
                   + str(jogvel) + ', "poscmd":' + str(poscmd) + '}'
        if len(data) > 0:
            self.blinx_jaka_send(data)
            return True
        else:
            return False

    """
    角度运动
    relFlag: 可选值为 0 或者 1, 0 代表绝对运动，1 代表相对运动
    jointPosition：[j1,j2,j3,j4,j5,j6] 填入的是每一个关节的角度值
        单位是度，不是弧度。值的注意的是运动的正负由 jointPosition 值的正负来确定。
    speed: speed_val 代表关节的速度，单位是 (°/S)，用户可以自行填入适合的参数
    accel：accel_val 代表关节的加速度，单位是 (°/S²)，用户可以自行填入适合的参数，加速度的值建议不
        要超过 720。
    joint_move 是阻塞的运动指令，必须一条运动指令执行完才会执行下一条运动指令。如果需要立即执行下
        一条指令，建议先使用 stop_program 停止当前的运动，再发送下一条运动指令。
    """

    def blinx_joint_move(self, relFlag, jointPosition, speed, accel):
        if (relFlag and jointPosition and speed and accel) is not None:
            data = '{"cmdName":"joint_move","relFlag":' + str(relFlag) + ',"jointPosition":' \
                   + str(jointPosition) + ',"speed":' + str(speed) + ',"accel":' + str(accel) + '}'
            self.blinx_jaka_send(data)
            return True
        else:
            return False

    """
    末端运动到指定位置
    endPosition: [x,y,z,a,b,c] 指定 TCP 末端 xyzabc 的值
    speed: speed_val 代表关节的速度，单位是 (°/S)，用户可以自行填入适合的参数。如果 speed_val
        设置为 20,则代表关节速度为 20 °/S。
    accel：accel_val 代表关节的加速度，单位是 (°/S²)，用户可以自行填入适合的参数，加速度的值建议不
        要超过 720。
    """

    def blinx_end_move(self, endPosition, speed, accel):
        if (endPosition and speed and accel) is not None:
            data = '{"cmdName":"end_move","endPosition":' + str(endPosition) + ', "speed":' \
                   + str(speed) + ', "accel":' + str(accel) + '}'
            self.blinx_jaka_send(data)
            return True
        else:
            return False

    """
    直线运动
    cartPosition: [x,y,z,rx,ry,rz] 指定笛卡尔空间 x,y,z,rx,ry,rz 的值
    speed: speed_val 代表线速度，单位是 (mm/s)，用户可以自行填入适合的参数。如果 speed_val
        设置为 20,则代表线速度为 20 mm/s。
    accel: accel_val 代表直线运动的加速度，单位是 (mm/S²)，用户可以自行填入适合的参数，加速度的值
        建议不要超过 8000.
    relFlag: flag_val 可选值为 0 或者 1。 0 代表绝对运动，1 代表相对运动。
    """

    def blinx_moveL(self, cartPosition, speed, accel, relFlag):
        if len(str(cartPosition)) > 0 and len(str(speed)) > 0 and len(str(accel)) > 0 and len(str(relFlag)) > 0:
            data = '{"cmdName":"moveL","relFlag":' + str(relFlag) + ',"cartPosition":' \
                   + str(cartPosition) + ',"speed":' + str(speed) + ',"accel":' + str(accel) + ',"tol":0.5}'
            self.blinx_jaka_send(data)
            return True
        else:
            return False

    """
    圆弧运动
    pos_mid: [x,y,z,rx,ry,rz] 指定笛卡尔空间 x,y,z,rx,ry,rz 的值
    pos_end: [x,y,z,rx,ry,rz] 指定笛卡尔空间 x,y,z,rx,ry,rz 的值
    speed: speed_val 代表关节的速度，单位是 (°/S)，用户可以自行填入适合的参数。如果 speed_val
        设置为 20,则代表关节速度为 20 °/S。
    accel：accel_val 代表关节的加速度，单位是 (°/S²)，用户可以自行填入适合的参数，加速度的值建议不
        要超过 720。
    tol：tol 代表最大允许误差，若不需要则取 0。
    """

    def blinx_movc(self, pos_mid, pos_end, speed, accel, tol):
        if len(str(pos_mid)) > 0 and len(str(pos_end)) > 0 and len(str(speed)) > 0 and len(str(accel)) > 0 and len(
                str(tol)) > 0:
            data = '{"cmdName":"movc","relFlag":move_mode,"pos_mid":' + str(pos_mid) + ',"pos_end":' \
                   + str(pos_end) + ',"speed":' + str(speed) + ',"accel":' + str(accel) + ',"tol":' + str(tol) + '}'
            self.blinx_jaka_send(data)
            return True
        else:
            return False

    """提出连接"""

    def blinx_quit(self):
        data = '{"cmdName":"quit"}'
        self.blinx_jaka_send(data)

    """
    设置数字输出变量(DO) 的值
    type 是 DO 的类型： 0 是控制器 DO，1 代表工具端 DO，2 代表扩展 DO
    index 是 DO 的编号，例如控制的 DO 编号为 1~31，如果想要控制第 31 个 DO，index 的值设为 31.
    value 是 DO 的值，可选值为 0 或者 1
    """

    def blinx_set_digital_output(self, type, index, value):
        data = '{"cmdName":"set_digital_output","type":' + str(type) + ',"index":' \
               + str(index) + ',"value":' + str(value) + '}'
        self.blinx_jaka_send(data)


    """
    获得数字输入状态
    """

    def blinx_get_digital_input_status(self):
        data = '{"cmdName":"get_digital_input_status"}'
        self.blinx_jaka_send(data)

    """
       获得指定数字输入状态
    """
    def blinx_get_digital_input(self,type, index):
        data = '{"cmdName":"get_digital_input","type":' + str(type) + ',"index":' + str(index) + '}'
        self.blinx_jaka_send(data)
    """"
    获取模拟输出变量的值（AO）的值
    说明：
    type 是AO的类型：0--控制器AO；1--工具IO；2--扩展AO；3--保留；4--modbusIO；5--Profinet IO；6--Ethernet/IP IO 。
    index 是AO的编号，例如控制的AO编号为0~7，如果想要控制第7个AO，index的值设为7。
    value 是AO的值，可输入一个浮点数来满足编程要求，例如4.32。
    """
    def blinx_get_analog_output(self,type, index):
        data='{"cmdName":"get_analog_output","type":' + str(type) + ',"index":' + str(index) +'}'
        self.blinx_jaka_send(data)

    """"
        获取模拟输入变量的值（AI）的值（tcp协议目前没有这个指令）
    """
    def blinx_get_analog_input(self,type, index):
        data='{"cmdName":"get_analog_input","type":' + str(type) + ',"index":' + str(index) +'}'
        self.blinx_jaka_send(data)
    """"
    设置模拟输出变量的值（AO）的值
    说明：
    type 是AO的类型：0--控制器AO；1--工具IO；2--扩展AO；3--保留；4--modbusIO；5--Profinet IO；6--Ethernet/IP IO 。
    index 是AO的编号，例如控制的AO编号为0~7，如果想要控制第7个AO，index的值设为7。
    value 是AO的值，可输入一个浮点数来满足编程要求。
    """
    def blinx_set_analog_output(self,type, index, value):
        data = '{"cmdName":"set_analog_output","type":' + str(type) + ',"index":' \
               + str(index) + ',"value":' + str(value) + '}'
        self.blinx_jaka_send(data)
    """
    机器人运动学正解
    jointPosition:[j1,j2,j3,j4,j5,j6]是需要发送的 6 个关节角度用来求运动学正解，
    """

    def blinx_kine_forward(self, jointPosition):
        if len(str(jointPosition)) > 0:
            data = '{"cmdName":"kine_forward","jointPosition":' + str(jointPosition) + '}'
            self.blinx_jaka_send(data)
            return True
        else:
            return False

    """
    机器人运动学逆解
    发送的 jointPosition 是机器人的参考关节角，推荐用户选取机器人当前的关节角作为
    参考关节角。单位是度。
    发送消息里的 cartPosition 是机器人的末端位姿。[x,y,z,a,b,c] xyz 的单位是毫米，abc 的单位是度。
    """

    def blinx_kine_inverse(self, jointPosition, cartPosition):
        if len(str(jointPosition)) > 0 and len(str(cartPosition)) > 0:
            data = '{"cmdName":"kine_inverse","jointPosition":' + str(jointPosition) + ',"cartPosition":' \
                   + str(cartPosition) + '}'
            self.blinx_jaka_send(data)
            return True
        else:
            return False

    """
    碰撞后从碰撞保护模式恢复
    """

    def blinx_clear_error(self):
        data = '{"cmdName":"clear_error"}'
        self.blinx_jaka_send(data)

    # 机器人停止程序
    def blinx_stop_run(self):
        data = '{"cmdName":"stop_program"}'
        self.blinx_jaka_send(data)
