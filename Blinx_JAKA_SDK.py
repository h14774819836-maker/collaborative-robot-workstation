import math
import time
import configparser
from JakaPyLib import jkrc
class Blinx_JAKA_SDK():
    def __init__(self,public_class):
        self.public_class = public_class
        # 坐标系
        self.COORD_BASE = 0  # 基座标/世界坐标系
        self.COORD_JOINT = 1  # 关节坐标系
        self.COORD_TOOL = 2  # 工具坐标系
        # 运动模式
        self.ABS = 0  # 绝对运动
        self.INCR = 1  # 相对运动
        self.CONT = 2  # 连续运动

        self.IO_CABINET = 0  # 控制柜面板IO
        self.IO_TOOL = 1 #工具IO
        self.IO_EXTEND = 2 #扩展IO
        self.jaka_open_sucess=self.open_device(self.public_class.jaka_ip)
        if self.jaka_open_sucess:
            self.set_status_data_update_time_interval(4)
        self.speeds_joint = [self.d2r(10), self.d2r(90), self.d2r(180)]  # 机器人关节速度，小中大rad/s
        self.speeds_linear = [100, 1000, 1800]  # 机器人直线速度，小中大mm/s

    # 打开机器人（连接-登录-上电-使能）
    def open_device(self,ip):
        try:
            self.rc = jkrc.RC(ip)
            ret = self.rc.login()  # 登录
            if ret[0] == 0:
                print("login success")
            else:
                print("login fail")
                return False
            ret = self.rc.power_on()  # 上电
            if ret[0] == 0:
                print("power_on success")
            else:
                print("power_on fail")
                return False
            ret = self.rc.enable_robot()  # 使能
            if ret[0] == 0:
                print("enable_robot success")
            else:
                print("enable_robot fail")
                return False
            return True
        except  Exception as e:
            print("打开机器人失败：", e)
            return False
    # 关闭机器人（下使能-下电-登出）
    def close_device(self):
        try:
            ret = self.rc.disable_robot()
            if ret[0] == 0:
                print("disable_robot success")
            else:
                print("disable_robot fail")
                return False
            time.sleep(1)  # 下使能需要一定时间
            ret = self.rc.power_off()
            if ret[0] == 0:
                print("power_off success")
            else:
                print("power_off fail")
                return False
            ret = self.rc.logout()
            if ret[0] == 0:
                print("logout success")
            else:
                print("logout fail")
                return False
            return True
        except  Exception as e:
            print("关闭机器人失败：",e)
            return False
    # 机器人上电
    def robot_power_on(self):
        ret = self.rc.power_on()
        if ret[0] == 0:
            print("power_on success")
        else:
            print("power_on fail")
            return False
        return True
    # 机器人下电
    def robot_power_off(self):
        ret = self.rc.power_off()
        if ret[0] == 0:
            print("power_off success")
        else:
            print("power_off fail")
            return False
        return True
    # 机器人使能
    def robot_enable(self):
        ret = self.rc.enable_robot()
        if ret[0] == 0:
            print("enable_robot success")
        else:
            print("enable_robot fail")
            return False
        return True
    # 机器人下使能
    def robot_disable(self):
        ret = self.rc.disable_robot()
        if ret[0] == 0:
            print("disable_robot success")
        else:
            print("disable_robot fail")
            return False
        return True    #控制器关机
    def controller_shut_down(self):
        ret = self.rc.shut_down()
        print("ret:",ret)
        if ret[0] == 0:
            print("control_shut_down success")
        else:
            print("control_shut_down fail")
            return False
        return True
    # 获取机器人状态数据监测数据,支持多线程安全
    def get_robot_status(self):
        ret = self.rc.get_robot_status()
        if ret[0] == 0:
            print("the robot status is :", ret[1][3],ret[1][4],ret[1][23])
            return ret[1]
        else:
            print("some things happend,the errcode is: ", ret[0])
            return ret[0]
    #度数转弧度
    def d2r(self,degree):
        return degree / 180.0 * math.pi
    #弧度转度数
    def r2d(self,radian):
        return radian / math.pi * 180.0
    # 控制机器人手动模式下运动
    # aj_num：axis_joint_based 标识值，在关节空间下代表轴号，1 轴到六轴的轴号分别
    # 对应数字 0 到 5，笛卡尔空间下依次为 x，y，z，rx，ry，rz 分别对应数字0 到 5
    # move_mode：0 代表绝对运动，1 代表相对运动，2 代表连续运动
    # coord_type：机器人运动坐标系，工具坐标系，基坐标系（当前的世界/用户坐标系）或关节空间
    # jog_vel：指令速度，旋转轴或关节运动单位为 rad/s，移动轴单位为 mm/s，速度的正负决定运动方向的正负。
    # pos_cmd：指令位置，旋转轴或关节运动单位为 rad，移动轴单位为 mm，当 move_mdoe是绝对运动时参数可忽略
    def jog(self, aj_num, move_mode, coord_type, jog_vel, pos_cmd):
        ret = self.rc.jog(aj_num, move_mode, coord_type, jog_vel, pos_cmd)
        if ret[0] == 0:
            print("jog success")
        else:
            print("jog fail")
        return ret[0]

    # 控制机器人手动模式下运动停止，用于停止 jog
    # joint_num: num代表要停止运动的关节轴号，轴号0到5，分别代表关节1到关节6。值的注意的是 - 1,代表停止所有轴的运动。
    def jog_stop(self, joint_num):
        ret = self.rc.jog_stop(joint_num)
        if ret[0] == 0:
            print("jog_stop success")
        else:
            print("jog_stop fail")
        return ret[0]

    # 机器人关节运动到目标点位
    # joint_pos: 机器人关节运动目标位置。弧度
    # move_mode: 0代表绝对运动，1代表相对运动
    # is_block：设置接口是否为阻塞接口，TRUE为阻塞接口,FALSE为非阻塞接口，阻塞表示机器人运动完成才会有返回值，非阻塞表示接口调用完成立刻就有返回值。
    # speed: 机器人关节运动速度，单位：rad / s,最大180°/s
    def joint_move(self, joint_pos, move_mode, is_block, speed):
        joint_pos_rad = [self.d2r(joint_pos[0]), self.d2r(joint_pos[1]), self.d2r(joint_pos[2]),
                         self.d2r(joint_pos[3]), self.d2r(joint_pos[4]), self.d2r(joint_pos[5])]
        ret = self.rc.joint_move(joint_pos_rad, move_mode, is_block, speed)
        if ret[0] == 0:
            print("joint_move success")
        else:
            print("joint_move fail")
        return ret[0]

    # 机器人扩展关节运动。增加关节角加速度和关节运动终点误差。
    # joint_pos: 机器人关节运动各目标关节角度。弧度
    # move_mode: 指定运动模式, 0为绝对运动，1为相对运动，2代表连续运动。
    # is_block: 是否为阻塞接口，True为阻塞接口，False为非阻塞接口。
    # speed: 机器人关节运动速度，单位：rad / s ，最大180°/s
    # acc：机器人关节运动角加速度。
    # tol：机器人运动终点误差。
    def joint_move_extend(self, joint_pos, move_mode, is_block, speed, acc, tol):
        joint_pos_rad = [self.d2r(joint_pos[0]), self.d2r(joint_pos[1]), self.d2r(joint_pos[2]),
                         self.d2r(joint_pos[3]), self.d2r(joint_pos[4]), self.d2r(joint_pos[5])]
        ret = self.rc.joint_move_extend(joint_pos_rad, move_mode, is_block, speed, acc, tol)
        if ret[0] == 0:
            print("joint_move_extend success")
        else:
            print("joint_move_extend fail")
        return ret[0]

    # 机器人末端直线运动
    # end_pos: 机器人末端运动目标位置
    # move_mode: 0代表绝对运动，1代表相对运动
    # is_block: 设置接口是否为阻塞接口，TRUE为阻塞接口FALSE为非阻塞接口，阻塞表示机器人运动完成才会有返回值，非阻塞表示接口调用完成立刻就有返回值。
    # speed: 机器人直线运动速度，单位：mm / s,最大1500mm/s
    def linear_move(self, end_pos, move_mode, is_block, speed):
        end_pos_rad = [end_pos[0], end_pos[1], end_pos[2], self.d2r(end_pos[3]), self.d2r(end_pos[4]),
                       self.d2r(end_pos[5])]
        ret = self.rc.linear_move(end_pos_rad, move_mode, is_block, speed)
        if ret[0] == 0:
            print("linear_move success")
        else:
            print("linear_move fail")
        return ret[0]

    # 机器人扩展末端直线运动
    # end_pos: 机器人末端运动目标位置。
    # move_mode: 指定运动模式, 0为绝对运动，1为相对运动
    # is_block: 是否为阻塞接口，True为阻塞接口，False为非阻塞接口。
    # speed: 机器人笛卡尔空间运动速度，单位：mm / s
    # acc：机器人笛卡尔空间加速度，单位：mm / s ^ 2。
    # tol：机器人运动终点误差。
    def linear_move_extend(self, end_pos, move_mode, is_block, speed, acc, tol):
        end_pos_rad=[end_pos[0],end_pos[1],end_pos[2],self.d2r(end_pos[3]),self.d2r(end_pos[4]),self.d2r(end_pos[5])]
        ret = self.rc.linear_move_extend(end_pos_rad, move_mode, is_block, speed, acc, tol)
        if ret[0] == 0:
            print("linear_move_extend success")
        else:
            print("linear_move_extend fail")
        return ret[0]

    # 机器人末端圆弧运动
    # end_pos: 机器人末端运动目标位置。
    # mid_pos: 机器人末端运动中间点。
    # move_mode: 指定运动模式, 0为绝对运动，1为相对运动，2代表连续运动。
    # is_block: 是否为阻塞接口，True为阻塞接口，False为非阻塞接口。
    # speed: 机器人直线运动速度，单位：mm / s
    # acc：机器人直线运动角加速度，单位：mm / s ^ 2
    # tol：机器人运动终点误差。
    def circular_move(self, end_pos, mid_pos, move_mode, is_block, speed, acc, tol):
        end_pos_rad = [end_pos[0], end_pos[1], end_pos[2], self.d2r(end_pos[3]), self.d2r(end_pos[4]),
                       self.d2r(end_pos[5])]
        mid_pos_rad = [mid_pos[0], mid_pos[1], mid_pos[2], self.d2r(mid_pos[3]), self.d2r(mid_pos[4]),
                       self.d2r(mid_pos[5])]
        ret = self.rc.circular_move(end_pos_rad, mid_pos_rad, move_mode, is_block, speed, acc, tol)
        if ret[0] == 0:
            print("circular_move success")
        else:
            print("circular_move fail")
        return ret[0]

    # circle_cnt: 圆弧运动圈数
    def circular_move_extend(self, end_pos, mid_pos, move_mode, is_block, speed, acc, tol, cricle_cnt):
        end_pos_rad = [end_pos[0], end_pos[1], end_pos[2], self.d2r(end_pos[3]), self.d2r(end_pos[4]),
                       self.d2r(end_pos[5])]
        mid_pos_rad = [mid_pos[0], mid_pos[1], mid_pos[2], self.d2r(mid_pos[3]), self.d2r(mid_pos[4]),
                       self.d2r(mid_pos[5])]
        ret = self.rc.circular_move_extend(end_pos_rad, mid_pos_rad, move_mode, is_block, speed, acc, tol, cricle_cnt)
        if ret[0] == 0:
            print("circular_move_extend success")
        else:
            print("circular_move_extend fail")
        return ret[0]

    # 终止当前机械臂运动
    def motion_abort(self):
        ret = self.rc.motion_abort()
        if ret[0] == 0:
            print("motion_abort success")
        else:
            print("motion_abort fail")
        return ret[0]

    # 设置机器人状态数据自动更新时间间隔
    def set_status_data_update_time_interval(self, millisecond):
        ret = self.rc.set_status_data_update_time_interval(millisecond)
        if ret[0] == 0:
            print("set_status_data_update_time_interval success")
        else:
            print("set_status_data_update_time_interval fail")
        return ret[0]

    # 获取当前机器人的六个关节角度值
    def get_joint_position(self):
        ret = self.rc.get_joint_position()
        if ret[0] == 0:
            # print("the joint position is :", ret[1])
            return ret[1]
        else:
            print("some things happend,the errcode is: ", ret[0])
            return ret[0]

    # 获取当前设置下工具末端的位姿
    def get_tcp_position(self):
        ret = self.rc.get_tcp_position()
        if ret[0] == 0:
            # print("the tcp position is :", ret[1])
            return ret[1]
        else:
            print("some things happend,the errcode is: ", ret[0])
            return ret[0]

    # 设置用户坐标系信息
    # id: 用户坐标系ID，可选ID为1到10, 0代表机器人基坐标系
    # user_frame: 用户坐标系参数[x, y, z, rx, ry, rz]
    # name: 用户坐标系别名
    def set_user_frame_data(self, id, user_frame, name):
        ret = self.rc.set_user_frame_data(id, user_frame, name)
        if ret[0] == 0:
            print("set_user_frame_data success")
        else:
            print("set_user_frame_data fail")
        return ret[0]

    # 获取用户坐标系信息
    # 返回值成功：(0, id, tcp), id : 用户坐标系 ID，可选 ID 为 1 到 10, 0 代表机器人基坐标系tcp: 用户坐标系参数[x,y,z,rx,ry,rz]
    def get_user_frame_data(self, id):
        ret = self.rc.get_user_frame_data(id)
        if ret[0] == 0:
            print("the user frame data is :", ret[1])
            return ret[1]
        else:
            print("some things happend,the errcode is: ", ret[0])
            return ret[0]

    # 查询当前使用的用户坐标系 ID
    # 返回值：成功：(0, id)，id 值范围为 0 到 10, 0 代表机器人基坐标系
    def get_user_frame_id(self):
        ret = self.rc.get_user_frame_id()
        if ret[0] == 0:
            print("the user frame id is :", ret[1])
            return ret[1]
        else:
            print("some things happend,the errcode is: ", ret[0])
            return ret[0]

    # 设置当前使用的用户坐标系 ID
    def set_user_frame_id(self, id):
        ret = self.rc.set_user_frame_id(id)
        if ret[0] == 0:
            print("set_user_frame_id success")
        else:
            print("set_user_frame_id fail")
        return ret[0]

    # 查询机器人当前使用的工具 ID
    # 返回值：成功：(0, id), id 值范围为 0 到 10，0 代表末端法兰盘, 已被控制器使用。
    def get_tool_id(self):
        ret = self.rc.get_tool_id()
        if ret[0] == 0:
            print("the tool id is :", ret[1])
            return ret[1]
        else:
            print("some things happend,the errcode is: ", ret[0])
            return ret[0]

    # 设置数字输出变量(DO)的值
    # IO_CABINET = 0  # 控制柜面板 IO
    # IO_TOOL = 1  # 工具 IO
    # IO_EXTEND = 2  # 扩展 IO
    # ex：set_digital_output(IO_CABINET, 2, 1)  # 设置 DO3 的引脚输出值为 1
    def set_digital_output(self, iotype, index, value):
        ret = self.rc.set_digital_output(iotype, index, value)
        if ret[0] == 0:
            print("set_digital_output success")
        else:
            print("set_digital_output fail")
        return ret[0]

    # 设置模拟输出变量的值(AO)的值
    # IO_CABINET = 0  # 控制柜面板 IO
    # IO_TOOL = 1  # 工具 IO
    # IO_EXTEND = 2  # 扩展 IO
    # ex:set_analog_output(iotype = IO_CABINET,index = 3,value = 1.55)#设置 AO4 的值为 1.55
    def set_analog_output(self, iotype, index, value):
        ret = self.rc.set_analog_output(iotype, index, value)
        if ret[0] == 0:
            print("set_analog_output success")
        else:
            print("set_analog_output fail")
        return ret[0]

    # 查询数字输入(DI)状态
    # 返回值成功：(0, value)，value: DI 状态查询结果
    # iotype: DI类型
    # index: DI索引
    def get_digital_input(self, iotype, index):
        ret = self.rc.get_digital_input(iotype, index)
        if ret[0] == 0:
            # print("the digital input is :", ret[1])
            return ret[1]
        else:
            print("get_digital_input fail", ret[0])
            return ret[0]

    # 查询数字输出(DO)状态
    def get_digital_output(self, iotype, index):
        ret = self.rc.get_digital_output(iotype, index)
        if ret[0] == 0:
            print("the digital output is :", ret[1])
            return ret[1]
        else:
            print("get_digital_output fail", ret[0])
            return ret[0]

    # 机器人负载设置
    # mass: 负载质量，单位: kg
    # centroid: 负载质心坐标[x, y, z], 单位: mm
    def set_payload(self, mass, centroid):
        ret = self.rc.set_payload(mass, centroid)
        if ret[0] == 0:
            print("set_payload success")
        else:
            print("set_payload fail")
        return ret[0]

    # 设置 tioV3 电压参数
    # vout_enable电压使能，0:关，1开
    # vout_vol电压大小0:24v 1:12v
    def set_tio_vout_param(self, vout_enable, vout_vol):
        ret = self.rc.set_tio_vout_param(vout_enable, vout_vol)
        if ret[0] == 0:
            print("set_tio_vout_param success")
        else:
            print("set_tio_vout_param fail")
        return ret[0]

    # 获取 tioV3 电压参数
    # vout_enable电压使能，0:关，1开
    # vout_vol电压大小0:24v 1:12v
    # 返回值 成功：(0,(vout_enable ,vout_vol))
    def get_tio_vout_param(self, vout_enable, vout_vol):
        ret = self.rc.get_tio_vout_param(vout_enable, vout_vol)
        if ret[0] == 0:
            print("the tio_vout_param is :", ret[1])
            return ret[1]
        else:
            print("get_tio_vout_param fail", ret[0])
            return ret[0]

    # 获取机械臂状态
    # 返回值 成功：(0,(estoped, power_on, servo_enabled))
    # estoped: 急停0: 关， 1: 开
    # power_on: 上电0: 关，1: 开
    # servo_enabled: 伺服使能0: 关，1: 开
    def get_robot_state(self):
        ret = self.rc.get_robot_state()
        if ret[0] == 0:
            print("the robot state is :", ret[1])
            return ret[1]
        else:
            print("get_robot_state fail:", ret[0])
            return ret[0]
    #机器人逆解 逆解失败返回-4
    def kine_inverse(self,ref_pos, cartesian_pose):
        ret = self.rc.kine_inverse(ref_pos, cartesian_pose)
        if ret[0] == 0:
            print("kine_inverse success")
            return ret
        else:
            print("kine_inverse fail")
            return ret
    # 碰撞之后从碰撞保护模式恢复
    def collision_recover(self):
        ret = self.rc.collision_recover()
        if ret[0] == 0:
            print("collision_recover success")
        else:
            print("collision_recover fail")
        return ret[0]

    # 错误状态清除
    def clear_error(self):
        ret = self.rc.clear_error()
        if ret[0] == 0:
            print("clear_error success")
        else:
            print("clear_error fail")
        return ret[0]

    # 设置网络异常时机器人自动终止运动类型
    # millisecond: 时间参数，单位：ms。
    # mnt: 网络异常时机器人需要进行的动作类型, 0代表机器人保持原来的运动，1代表暂停运动，2代表终止运动。
    def set_network_exception_handle(self, millisecond, mnt):
        ret = self.rc.set_network_exception_handle(millisecond, mnt)
        if ret[0] == 0:
            print("set_network_exception_handle success")
        else:
            print("set_network_exception_handle fail")
        return ret[0]


# if __name__ == '__main__':
#     robot = Blinx_JAKA_SDK()
#     robot.get_robot_state()
#     robot.get_robot_status()
    # robot.jog_stop(-1)
    # robot.jog(2,robot.ABS,robot.COORD_JOINT,100,0)
    # for i in range(5):
    #     robot.joint_move([0, 0, 0, 0, 0, 0], robot.ABS, True, math.pi)
    #     robot.joint_move([-70, 0, 0, 0, 0, 0], robot.ABS, True, math.pi)
    #     robot.joint_move([-70, 0, 90, 0, 0, 0], robot.ABS, True, math.pi)
    #     robot.joint_move([-70, 0, 90, 0, 90, 0], robot.ABS, True, math.pi)
    # for i in range(5):
    #     robot.joint_move_extend([0, 0, 0, 0, 0, 0], robot.ABS, False, math.pi, 5, 0.1)
    #     robot.joint_move_extend([0, 0, 90, 0, 90, 0], robot.ABS, False, math.pi, 5, 0.1)
    #     robot.linear_move([0, 0, -30, 0, 0, 0], robot.INCR, False, 1500)
    #     robot.linear_move([100, 100, 30, 0, 0, 0], robot.INCR, False, 1500)
    #     robot.linear_move([0, 0, -30, 0, 0, 0], robot.INCR, False, 1500)
    #     robot.linear_move([100, 100, 30, 0, 0, 0], robot.INCR, False, 1500)
