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
        self.primary_pick_depth_roi = _parse_literal(config, "Image_Process", "primary_pick_depth_roi", [0, 0, 0, 0])
        self.secondary_alignment_depth_roi = _parse_literal(
            config,
            "Image_Process",
            "secondary_alignment_depth_roi",
            list(self.primary_pick_depth_roi),
        )
        self.depth_geom_board_estimation_mode = config.get(
            "Image_Process",
            "depth_geom_board_estimation_mode",
            fallback="global_hist",
        )
        self.depth_geom_hist_bin_mm = config.getfloat("Image_Process", "depth_geom_hist_bin_mm", fallback=2.0)
        self.depth_geom_hist_peak_refine_window_mm = config.getfloat(
            "Image_Process",
            "depth_geom_hist_peak_refine_window_mm",
            fallback=3.0,
        )
        self.depth_geom_expected_long_mm = config.getfloat("Image_Process", "depth_geom_expected_long_mm", fallback=200.0)
        self.depth_geom_expected_short_mm = config.getfloat("Image_Process", "depth_geom_expected_short_mm", fallback=100.0)
        self.depth_geom_expected_height_mm = config.getfloat("Image_Process", "depth_geom_expected_height_mm", fallback=70.0)
        self.depth_geom_long_tol_mm = config.getfloat("Image_Process", "depth_geom_long_tol_mm", fallback=40.0)
        self.depth_geom_short_tol_mm = config.getfloat("Image_Process", "depth_geom_short_tol_mm", fallback=30.0)
        self.depth_geom_height_tol_mm = config.getfloat("Image_Process", "depth_geom_height_tol_mm", fallback=20.0)
        self.depth_geom_min_brick_height_mm = config.getfloat(
            "Image_Process",
            "depth_geom_min_brick_height_mm",
            fallback=25.0,
        )
        self.depth_geom_max_brick_height_mm = config.getfloat(
            "Image_Process",
            "depth_geom_max_brick_height_mm",
            fallback=130.0,
        )
        self.depth_geom_min_region_area_px = config.getint("Image_Process", "depth_geom_min_region_area_px", fallback=2000)
        self.depth_geom_max_region_area_px = config.getint("Image_Process", "depth_geom_max_region_area_px", fallback=400000)
        self.secondary_depth_geom_min_region_area_px = config.getint(
            "Image_Process",
            "secondary_depth_geom_min_region_area_px",
            fallback=self.depth_geom_min_region_area_px,
        )
        self.secondary_depth_geom_max_region_area_px = config.getint(
            "Image_Process",
            "secondary_depth_geom_max_region_area_px",
            fallback=self.depth_geom_max_region_area_px,
        )
        self.depth_geom_median_kernel = config.getint("Image_Process", "depth_geom_median_kernel", fallback=3)
        self.depth_geom_open_kernel = config.getint("Image_Process", "depth_geom_open_kernel", fallback=3)
        self.depth_geom_close_kernel = config.getint("Image_Process", "depth_geom_close_kernel", fallback=5)
        self.depth_geom_border_margin_px = config.getint("Image_Process", "depth_geom_border_margin_px", fallback=12)
        self.depth_geom_planarity_max_std_mm = config.getfloat(
            "Image_Process",
            "depth_geom_planarity_max_std_mm",
            fallback=3.0,
        )
        self.depth_geom_min_rectangularity = config.getfloat(
            "Image_Process",
            "depth_geom_min_rectangularity",
            fallback=0.65,
        )
        self.depth_geom_min_completeness = config.getfloat(
            "Image_Process",
            "depth_geom_min_completeness",
            fallback=0.60,
        )
        self.primary_pick_rgb_depth_center_thresh_px = config.getfloat(
            "Image_Process",
            "primary_pick_rgb_depth_center_thresh_px",
            fallback=80.0,
        )
        self.primary_pick_rgb_depth_angle_thresh_deg = config.getfloat(
            "Image_Process",
            "primary_pick_rgb_depth_angle_thresh_deg",
            fallback=20.0,
        )
        self.primary_pick_rgb_depth_iou_thresh = config.getfloat(
            "Image_Process",
            "primary_pick_rgb_depth_iou_thresh",
            fallback=0.20,
        )
        self.primary_pick_rgb_depth_mm_thresh = config.getfloat(
            "Image_Process",
            "primary_pick_rgb_depth_mm_thresh",
            fallback=10.0,
        )
        self.primary_pick_rgb_low_score_thresh = config.getfloat(
            "Image_Process",
            "primary_pick_rgb_low_score_thresh",
            fallback=0.90,
        )
        self.primary_pick_rgb_low_valid_ratio_thresh = config.getfloat(
            "Image_Process",
            "primary_pick_rgb_low_valid_ratio_thresh",
            fallback=0.70,
        )
        self.primary_pick_depth_fallback_geom_thresh = config.getfloat(
            "Image_Process",
            "primary_pick_depth_fallback_geom_thresh",
            fallback=0.88,
        )

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
