import configparser
from ast import literal_eval


def _parse_literal(config, section, option, fallback):
    raw_value = config.get(section, option, fallback=None)
    if raw_value is None:
        return fallback
    try:
        return literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return fallback


class Blinx_Public_Class:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read("Config/config.ini", encoding="utf-8")

        self.cam_sn = config.get("CamSN", "cam_sn", fallback="")
        self.jaka_ip = config.get("DeviceIP", "jaka_ip", fallback="")
        self.jaka_port = config.get("DeviceIP", "jaka_port", fallback="")
        self.jaka_port_10000 = config.get("DeviceIP", "jaka_port_10000", fallback="")
        self.minArea = config.get("Image_Process", "minArea", fallback="80000")
        self.maxArea = config.get("Image_Process", "maxArea", fallback="150000")
        self.M = config.get("RelationalMatrix", "m", fallback="[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]")
        self.M1 = config.get("RelationalMatrix", "m1", fallback="[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]")

        self.depth_min_valid_mm = config.getfloat("Image_Process", "depth_min_valid_mm", fallback=10.0)
        self.depth_trim_percent = config.getfloat("Image_Process", "depth_trim_percent", fallback=5.0)
        self.depth_min_valid_pixels = config.getint("Image_Process", "depth_min_valid_pixels", fallback=200)
        self.depth_min_valid_ratio = config.getfloat("Image_Process", "depth_min_valid_ratio", fallback=0.2)
        self.depth_erode_kernel = config.getint("Image_Process", "depth_erode_kernel", fallback=3)
        self.depth_erode_iterations = config.getint("Image_Process", "depth_erode_iterations", fallback=1)
        self.pca_min_axis_ratio = config.getfloat("Image_Process", "pca_min_axis_ratio", fallback=1.10)
        self.vision_debug = config.getint("Image_Process", "vision_debug", fallback=1)

        self.joint_pos = ""
        self.tcp_pos = ""
        self.din_status = ""
        self.ao_value = ""
        self.ai_value = ""

        self.jaka_power = ""
        self.jaka_enable = ""

        self.mech_2d_image = None
        self.mech_depth_map = None
        self.mech_point_cloud = None
        self.mech_connected = False
        self.is_continue_show = False
        self.new_data = None

        self.sucker_process = "0-0"
        self.sucker_type = 0
        self.sucker_state = False

        self.bundle_process = "0-0"
        self.bundle_type = 0
        self.bundle_state = False

        self.ceramic_process_state = False
        self.ceramic_process_node = "0-0"
        self.ceramic_process_num = 0
        self.ceramic_process_list = []
        self.ceramic_process_result = None

        self.brick_process_state = False
        self.brick_process_node = "0-0"
        self.brick_process_num = 0
        self.brick_process_list = []
        self.brick_process_result = None

        self.rebar_process_state = False
        self.rebar_process_node = "0-0"
        self.rebar_process_num = 0
        self.rebar_data_list = None

        self.last_brick_rotation_delta = None
        self.last_ceramic_rotation_delta = None

        zero_pose = [0, 0, 0, 0, 0, 0]
        self.initial_angle = _parse_literal(config, "Positioning", "initial_angle", zero_pose)
        self.sucker_actuator_loc = _parse_literal(config, "Positioning", "sucker_actuator_loc", zero_pose)
        self.bundle_actuator_loc = _parse_literal(config, "Positioning", "bundle_actuator_loc", zero_pose)
        self.identify_loc1 = _parse_literal(config, "Positioning", "identify_loc1", zero_pose)
        self.identify_loc2 = _parse_literal(config, "Positioning", "identify_loc2", zero_pose)
        self.ceramic_place_loc = _parse_literal(config, "Positioning", "ceramic_place_loc", zero_pose)
        self.ceramic_excessive_loc = _parse_literal(config, "Positioning", "ceramic_excessive_loc", zero_pose)
        self.brick_place_loc = _parse_literal(config, "Positioning", "brick_place_loc", zero_pose)
        self.brick_excessive_loc = _parse_literal(config, "Positioning", "brick_excessive_loc", zero_pose)
        self.secondary_positioning_loc = _parse_literal(
            config,
            "Positioning",
            "secondary_positioning_loc",
            zero_pose,
        )
        self.secondary_photography_loc = _parse_literal(
            config,
            "Positioning",
            "secondary_photography_loc",
            zero_pose,
        )
