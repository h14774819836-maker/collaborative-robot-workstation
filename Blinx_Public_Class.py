import configparser
class Blinx_Public_Class():
    def __init__(self):
        # region 配置文件读取
        config = configparser.ConfigParser()
        # -read读取ini文件
        config.read('Config/config.ini', encoding="utf-8")
        self.cam_sn = config.get('CamSN', 'cam_sn')
        self.jaka_ip = config.get('DeviceIP', 'jaka_ip')
        self.jaka_port = config.get('DeviceIP', 'jaka_port')
        self.jaka_port_10000 = config.get('DeviceIP', 'jaka_port_10000')
        self.minArea = config.get('Image_Process', 'minArea')
        self.maxArea = config.get('Image_Process','maxArea')
        self.M = config.get('RelationalMatrix', 'm')    # 二次定位
        self.M1 = config.get('RelationalMatrix', 'm1')   # 钢筋捆扎
        # endregion
        # jaka机器人角度，坐标，输入数字IO数据，模拟IO
        self.joint_pos = ""
        self.tcp_pos = ""
        self.din_status = ""
        self.ao_value = ""
        self.ai_value = ""
        # jaka机器人状态
        self.jaka_power = ""
        self.jaka_enable = ""
        # Mech2D相机图像
        self.mech_2d_image = None
        # Mech3D相机图像
        self.mech_depth_map = None
        # Mech点云图
        self.mech_point_cloud = None
        # 梅卡相机链接标识
        self.mech_connected = False
        # 相机是否连续采集显示
        self.is_continue_show = False

        # 机械臂角度判断与坐标判断使用参数
        self.new_data = None

        # 吸盘切换
        self.sucker_process = "0-0"
        self.sucker_type = 0  # 0取  1放
        self.sucker_state = False

        # 捆扎机切换
        self.bundle_process = "0-0"
        self.bundle_type = 0  # 0取   1放
        self.bundle_state = False

        # 瓷砖流程参数
        self.ceramic_process_state = False   # 瓷砖流程状态
        self.ceramic_process_node = "0-0"   # 瓷砖流程节点
        self.ceramic_process_num = 0   # 瓷砖流程抓取次数
        self.ceramic_process_list = []   # 瓷砖识别结果

        # 墙砖流程参数
        self.brick_process_state = False   # 墙砖流程状态
        self.brick_process_node = "0-0"  # 墙砖流程节点
        self.brick_process_num = 0   # 墙砖抓取次数
        self.brick_process_list = []  # 瓷砖识别结果

        # 钢筋捆扎流程参数
        self.rebar_process_state = False  # 钢筋捆扎流程状态
        self.rebar_process_node = "0-0"  # 钢筋捆扎流程节点
        self.rebar_process_num = 0   # 钢筋捆扎次数
        self.rebar_data_list = None   # 钢筋捆扎识别结果

        # 机械臂点位数据
        self.initial_angle = eval(config.get('Positioning', 'initial_angle'))   # 初始角度
        self.sucker_actuator_loc = eval(config.get('Positioning', 'sucker_actuator_loc'))   # 吸盘位置
        self.bundle_actuator_loc = eval(config.get('Positioning', 'bundle_actuator_loc'))   # 困扎机位置
        self.identify_loc1 = eval(config.get('Positioning', 'identify_loc1'))   # 钢筋捆扎拍照位置
        self.identify_loc2 = eval(config.get('Positioning', 'identify_loc2'))   # 砖块拍照点位
        self.ceramic_place_loc = eval(config.get('Positioning', 'ceramic_place_loc'))   # 瓷砖放置零点
        self.ceramic_excessive_loc = eval(config.get('Positioning', 'ceramic_excessive_loc'))   # 瓷砖放置过度点
        self.brick_place_loc = eval(config.get('Positioning', 'brick_place_loc'))   # 红砖放置零点
        self.brick_excessive_loc = eval(config.get('Positioning', 'brick_excessive_loc'))   # 红砖放置过度点
        self.secondary_positioning_loc = eval(config.get('Positioning', 'secondary_positioning_loc'))   # 二次定位点
        self.secondary_photography_loc = eval(config.get('Positioning', 'secondary_photography_loc'))   # 二次拍照点



