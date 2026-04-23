# region 库文件
import inspect
import os
import re
import signal
import sys
import time
import numpy as np
from brick_process_recording import BrickProcessRecorder
from ImageProcessing.BLX_Image_Camera import Blinx_Image_Camera
from ImageProcessing.Conversion_3D import Blinx_3D_Conversion

import cv2
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import QRegExp, Qt, QEvent, QPropertyAnimation
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QRegExpValidator, QPixmap, QPainter, QColor
from PyQt5.QtWidgets import *
from Blinx_Jaka_Socket import *
from Blinx_Jaka_Rev_Socket import *
from Blinx_Object_Recongnition import *
from Blinx_Public_Class import *
from Blinx_Mech3D_Viewer_SDK import *
from ImageProcessing.image_recognition import *
from MainWindows import Ui_MainWindow


# endregion
# endregion
class Blinx_XXXY_Robot_Vision(QMainWindow, Ui_MainWindow, ):
    # region 变量初始化
    def __init__(self):  # 创建构造函数
        super().__init__()  # 调用父类函数，继承
        self.setupUi(self)  # 调用UI
        self.graphics_view_init()  # graphics_view初始化
        # 绑定空格键,按空格键保存图像
        self._display_source_pixmaps = {}
        self._init_recognition_display_widgets()
        self.shortcut = QShortcut(Qt.Key_Space, self)
        self.shortcut.activated.connect(self.on_space_pressed)
        # 对象初始化
        self.public_class = Blinx_Public_Class()
        # yolo图像识别是初始
        self.yolo_iamge = Blinx_image_rec(self.public_class)
        # 标定转换3D
        self.image_camera = Blinx_Image_Camera()   # 图像坐标转相机坐标
        self.conversion_3d = Blinx_3D_Conversion()   # 标定转化
        self.brick_process_recorder = None


        # 创建 QTimer 实例
        self.timer_ini = QTimer(self)  # 连接初始化定时器
        # self.time_judge_client_state = QTimer(self)  # 通讯状态连接定时器
        # 设置定时器触发的时间间隔
        self.timer_ini.setInterval(5000)
        # self.time_judge_client_state.setInterval(500)
        # 连接定时器的 timeout() 信号到槽函数
        self.timer_ini.timeout.connect(self.ini_class)
        # self.time_judge_client_state.timeout.connect(self.judge_client_state)
        # 设置定时器为单次触发
        self.timer_ini.setSingleShot(True)  # 初始化一次
        # 启动定时器
        self.timer_ini.start()
        # 机器人状态获取线程
        self.timer_jaka_state = QTimer()
        self.timer_jaka_state.setInterval(100)
        self.timer_jaka_state.timeout.connect(self.get_jaka_state)
        # Mech3D相机数据采集显示
        self.timer_mech = QTimer()
        self.timer_mech.setInterval(30)
        self.timer_mech.timeout.connect(self.show_camera_mech)

        # 流程程序定时器监控
        self.timer_process = QTimer()
        self.timer_process.setInterval(200)
        self.timer_process.timeout.connect(self.blinx_timer_process)
        self.timer_process.start()
        # 标定是否ok
        self.biaoding_ok = False
        self.biaoding_image = None
        self.btn_systermRun_click()  # 程序启动在系统运行界面


        # 设置窗口大小
        # 隐藏所有的Tab widget页面
        self.tabBar = self.tabWidget.findChild(QTabBar)
        self.tabBar.hide()
        self.calibration_parm_ini()  # 标定参数初始化
        self.pixel_point = []
        # 槽函数
        self.slot_init()  # 设置槽函数
        self.joint_or_xyz = False
        self.lineEdit_speed.setValidator(QRegExpValidator(QRegExp("[0-9]+$")))
        self.lineEdit_P_x_1.setValidator(QRegExpValidator(QRegExp("[0-9]+$")))
        self.lineEdit_P_y_1.setValidator(QRegExpValidator(QRegExp("[0-9]+$")))
        self.lineEdit_P_x_2.setValidator(QRegExpValidator(QRegExp("[0-9]+$")))
        self.lineEdit_P_y_2.setValidator(QRegExpValidator(QRegExp("[0-9]+$")))
        self.lineEdit_P_x_3.setValidator(QRegExpValidator(QRegExp("[0-9]+$")))
        self.lineEdit_P_y_3.setValidator(QRegExpValidator(QRegExp("[0-9]+$")))
        self.lineEdit_P_x_4.setValidator(QRegExpValidator(QRegExp("[0-9]+$")))
        self.lineEdit_P_y_4.setValidator(QRegExpValidator(QRegExp("[0-9]+$")))
        self.lineEdit_R_x_1.setValidator(QRegExpValidator(QRegExp(r"^[-+]?[0-9]*\.?[0-9]+$")))  # 允许输入正负数和最多一位小数
        self.lineEdit_R_y_1.setValidator(QRegExpValidator(QRegExp(r"^[-+]?[0-9]*\.?[0-9]+$")))
        self.lineEdit_R_x_2.setValidator(QRegExpValidator(QRegExp(r"^[-+]?[0-9]*\.?[0-9]+$")))
        self.lineEdit_R_y_2.setValidator(QRegExpValidator(QRegExp(r"^[-+]?[0-9]*\.?[0-9]+$")))
        self.lineEdit_R_x_3.setValidator(QRegExpValidator(QRegExp(r"^[-+]?[0-9]*\.?[0-9]+$")))
        self.lineEdit_R_y_3.setValidator(QRegExpValidator(QRegExp(r"^[-+]?[0-9]*\.?[0-9]+$")))
        self.lineEdit_R_x_4.setValidator(QRegExpValidator(QRegExp(r"^[-+]?[0-9]*\.?[0-9]+$")))
        self.lineEdit_R_y_4.setValidator(QRegExpValidator(QRegExp(r"^[-+]?[0-9]*\.?[0-9]+$")))
        self.lineEdit_absolute_loaction.setValidator(QRegExpValidator(QRegExp(r"^[-+]?[0-9]*\.?[0-9]+$")))
        # 背景文字提示
        self.lineEdit_absolute_loaction.setPlaceholderText("输入位置")

    # endregion
    # region 槽函数
    def reset_rotation_history(self, process_name):
        if process_name == "brick":
            self.public_class.last_brick_rotation_delta = None
        elif process_name == "ceramic":
            self.public_class.last_ceramic_rotation_delta = None

    def select_shortest_rotation_delta(self, process_name, angle_deg):
        attr_name = f"last_{process_name}_rotation_delta"
        previous_angle = getattr(self.public_class, attr_name, None)
        return resolve_shortest_rotation_delta(previous_angle, angle_deg)

    def remember_rotation_delta(self, process_name, angle_deg):
        attr_name = f"last_{process_name}_rotation_delta"
        setattr(self.public_class, attr_name, float(angle_deg))

    def _normalize_pose_values(self, values):
        if not isinstance(values, (list, tuple, np.ndarray)) or len(values) < 6:
            return None
        try:
            return [float(values[index]) for index in range(6)]
        except (TypeError, ValueError):
            return None

    def _summarize_pick_result(self, pick_result):
        if not isinstance(pick_result, dict):
            return None

        summary = {}
        fields = [
            "source",
            "class_id",
            "score",
            "pixel_x",
            "pixel_y",
            "depth_mm",
            "angle_deg",
            "raw_pca_angle_deg",
            "valid_depth_count",
            "valid_depth_ratio",
            "angle_fallback",
            "axis_ratio",
            "is_valid",
            "geometry_score",
            "decision_status",
            "decision_reason",
            "decision_warning",
            "rgb_low_quality",
            "rgb_low_quality_reasons",
            "rgb_depth_match_found",
            "match_center_distance_px",
            "match_angle_delta_deg",
            "match_depth_delta_mm",
            "match_bbox_iou",
            "match_mask_iou",
            "matched_depth_geometry_score",
            "robot_x",
            "robot_y",
        ]
        for field in fields:
            if field not in pick_result:
                continue
            value = pick_result[field]
            if isinstance(value, np.generic):
                value = value.item()
            summary[field] = value
        return summary

    def _brick_record_snapshot(self):
        return {
            "brick_process_state": bool(self.public_class.brick_process_state),
            "brick_process_node": self.public_class.brick_process_node,
            "brick_process_num": int(self.public_class.brick_process_num),
            "tcp_pose": self._normalize_pose_values(self.public_class.tcp_pos),
            "joint_pos": self._normalize_pose_values(self.public_class.joint_pos),
            "new_data": self._normalize_pose_values(self.public_class.new_data),
            "brick_process_result": self._summarize_pick_result(self.public_class.brick_process_result),
            "brick_secondary_alignment_result": self._summarize_pick_result(
                getattr(self.public_class, "brick_secondary_alignment_result", None)
            ),
            "last_brick_rotation_delta": getattr(self.public_class, "last_brick_rotation_delta", None),
        }

    def _start_brick_process_recording_session(self):
        try:
            if self.brick_process_recorder is not None:
                self.brick_process_recorder.finalize_case(
                    status="interrupted_by_new_session",
                    process_node=self.public_class.brick_process_node,
                    public_snapshot=self._brick_record_snapshot(),
                    extra={"reason": "A new brick recording session replaced the previous one."},
                )
            session_name = time.strftime("brick_%Y%m%d_%H%M%S", time.localtime())
            self.brick_process_recorder = BrickProcessRecorder(session_name=session_name)
            session_message = f"墙砖记录目录: {self.brick_process_recorder.session_dir}"
            print(session_message)
            self.textEdit_log.append(session_message + "\n")
        except Exception as exc:
            self.brick_process_recorder = None
            print("墙砖记录器初始化失败:", exc)
            self.textEdit_log.append(f"墙砖记录器初始化失败: {exc}\n")

    def _brick_record_start_case(self, raw_rgb, depth_map, display_rgb, extra=None):
        if self.brick_process_recorder is None:
            return
        try:
            self.brick_process_recorder.start_case(
                process_num=self.public_class.brick_process_num,
                process_node=self.public_class.brick_process_node,
                raw_rgb=raw_rgb,
                depth_map=depth_map,
                display_rgb=display_rgb,
                public_snapshot=self._brick_record_snapshot(),
                extra=extra or {},
            )
        except Exception as exc:
            print("墙砖记录器 start_case 失败:", exc)

    def _brick_record_capture(self, capture_name, raw_rgb, depth_map, display_rgb, extra=None):
        if self.brick_process_recorder is None:
            return
        try:
            self.brick_process_recorder.record_capture(
                capture_name=capture_name,
                raw_rgb=raw_rgb,
                depth_map=depth_map,
                display_rgb=display_rgb,
                process_node=self.public_class.brick_process_node,
                public_snapshot=self._brick_record_snapshot(),
                extra=extra or {},
            )
        except Exception as exc:
            print("墙砖记录器 record_capture 失败:", exc)

    def _brick_record_event(self, event_name, command_target=None, extra=None):
        if self.brick_process_recorder is None:
            return
        try:
            self.brick_process_recorder.record_event(
                event_name=event_name,
                process_node=self.public_class.brick_process_node,
                public_snapshot=self._brick_record_snapshot(),
                command_target=command_target,
                extra=extra or {},
            )
        except Exception as exc:
            print("墙砖记录器 record_event 失败:", exc)

    def _brick_record_finalize_case(self, status, extra=None):
        if self.brick_process_recorder is None:
            return
        try:
            self.brick_process_recorder.finalize_case(
                status=status,
                process_node=self.public_class.brick_process_node,
                public_snapshot=self._brick_record_snapshot(),
                extra=extra or {},
            )
        except Exception as exc:
            print("墙砖记录器 finalize_case 失败:", exc)

    def _brick_move_linear_fast(self, data):
        self.jaka.blinx_moveL(data, self.public_class.brick_linear_speed_fast, 5000, 0)

    def _brick_move_linear_pick(self, data):
        self.jaka.blinx_moveL(data, self.public_class.brick_linear_speed_pick, 5000, 0)

    def _brick_joint_move_rotate(self, data):
        self.jaka.blinx_joint_move(
            0,
            data,
            self.public_class.brick_joint_speed_rotate,
            50,
        )

    def _brick_slider_move_to(self, position):
        self.jaka.blinx_set_analog_output(6, 26, self.public_class.brick_slider_speed)
        self.jaka.blinx_set_analog_output(6, 25, position)
        self.jaka.blinx_set_digital_output(6, 5, 1)
        self.jaka.blinx_set_digital_output(6, 4, 1)
        self.jaka.blinx_set_digital_output(6, 4, 0)
        self.jaka.blinx_set_digital_output(6, 5, 0)

    def _brick_restart_from_primary_pick(self, reason, extra=None):
        restart_extra = {"reason": reason}
        if isinstance(extra, dict):
            restart_extra.update(extra)
        self._brick_record_event(
            "secondary_alignment_restart_to_primary",
            extra=restart_extra,
        )
        self._brick_record_finalize_case(
            "secondary_alignment_failed_restart",
            extra=restart_extra,
        )
        self.public_class.brick_process_result = None
        self.public_class.brick_secondary_alignment_result = None
        self.public_class.new_data = None
        self.reset_rotation_history("brick")
        self.public_class.brick_process_node = "3-0"

    def _init_recognition_display_widgets(self):
        self._display_source_pixmaps = {
            "Image_Show_1": None,
            "Image_Show_2": None,
        }
        for label in (self.Image_Show_1, self.Image_Show_2):
            label.setAlignment(Qt.AlignCenter)
            label.setScaledContents(False)
        self._update_recognition_display_layout()

    def _update_recognition_display_layout(self):
        main_rect = self.Image_Show_1.geometry()
        if main_rect.width() <= 0 or main_rect.height() <= 0:
            return
        margin = max(12, int(min(main_rect.width(), main_rect.height()) * 0.02))
        preview_width = max(320, int(main_rect.width() * 0.38))
        preview_width = min(preview_width, int(main_rect.width() * 0.48))
        preview_height = max(200, int(main_rect.height() * 0.40))
        preview_height = min(preview_height, int(main_rect.height() * 0.48))
        self.Image_Show_2.setGeometry(
            main_rect.x() + margin,
            main_rect.y() + margin,
            preview_width,
            preview_height,
        )
        self.Image_Show_2.raise_()
        self._refresh_recognition_display_widgets()

    def _render_display_pixmap(self, label, pixmap):
        if pixmap is None or pixmap.isNull():
            label.clear()
            return
        target_size = label.contentsRect().size()
        if not target_size.isValid() or target_size.width() <= 0 or target_size.height() <= 0:
            label.setPixmap(pixmap)
            return
        scaled_pixmap = pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        label.setAlignment(Qt.AlignCenter)
        label.setScaledContents(False)
        label.setPixmap(scaled_pixmap)

    def _set_display_pixmap(self, label, pixmap):
        key = label.objectName()
        if pixmap is None or pixmap.isNull():
            self._display_source_pixmaps[key] = None
            label.clear()
            return
        self._display_source_pixmaps[key] = QPixmap(pixmap)
        self._render_display_pixmap(label, self._display_source_pixmaps[key])

    def _set_display_image(self, label, image, image_format):
        if image is None:
            self._set_display_pixmap(label, None)
            return
        image_array = np.ascontiguousarray(image)
        if image_array.ndim < 2:
            self._set_display_pixmap(label, None)
            return
        height, width = image_array.shape[:2]
        bytes_per_line = image_array.strides[0]
        show_image = QtGui.QImage(
            image_array.data,
            width,
            height,
            bytes_per_line,
            image_format,
        ).copy()
        self._set_display_pixmap(label, QtGui.QPixmap.fromImage(show_image))

    def _refresh_recognition_display_widgets(self):
        for label_name in ("Image_Show_1", "Image_Show_2"):
            pixmap = self._display_source_pixmaps.get(label_name)
            if pixmap is not None:
                self._render_display_pixmap(getattr(self, label_name), pixmap)

    def showEvent(self, event):
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self._update_recognition_display_layout)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_recognition_display_layout()

    def slot_init(self):
        # 主界面按钮
        self.btn_systermRun.clicked.connect(self.btn_systermRun_click)
        self.btn_camSet.clicked.connect(self.btn_camSet_click)
        self.btn_calibrate.clicked.connect(self.btn_calibrate_click)

        # 末端执行器
        self.btn_suction_get.clicked.connect(self.blinx_btn_suction_get)   # 吸盘取
        self.btn_suction_set.clicked.connect(self.blinx_btn_suction_set)   # 吸盘放
        self.btn_bundle_get.clicked.connect(self.blinx_btn_bundle_get)   # 捆扎机取
        self.btn_bundle_set.clicked.connect(self.blinx_btn_bundle_set)   # 捆扎机放

        # 流程启动按钮
        self.btn_initialization.clicked.connect(self.blinx_btn_initialization)   # 初始化按钮
        self.btn_porcelain.clicked.connect(self.blinx_btn_porcelain)   # 瓷砖贴片流程按钮
        self.btn_brick.clicked.connect(self.blinx_btn_brick)   # 墙砖堆砌流程按钮
        self.btn_rebar.clicked.connect(self.blinx_btn_rebar)   # 钢筋捆扎流程按钮

        # TabPage2页面相机参数设置
        self.bnOnceFrame.clicked.connect(self.get_one_image)
        self.bnContinueFrame.clicked.connect(self.get_continue_frame)
        self.bnSaveImage.clicked.connect(self.save_jpg)
        # TabPage3页面相机参数设置
        self.btn_trigger_biaoding.clicked.connect(self.get_image_mech)
        self.bnSaveImage_biaoding.clicked.connect(self.save_jpg_biaoding)
        # TabPage3标定参数设置
        self.graphics_view_biaoding.mouseDoubleClickEvent = self.getPixel
        self.btn_biaoding_star.clicked.connect(self.pixel_to_object)
        self.btn_biaoding_save.clicked.connect(self.matrix_saveto_ini)
        self.lab_R_X_1.mousePressEvent = self.lab_R_X_1_clicked
        self.lab_R_Y_1.mousePressEvent = self.lab_R_Y_1_clicked
        self.lab_R_X_2.mousePressEvent = self.lab_R_X_2_clicked
        self.lab_R_Y_2.mousePressEvent = self.lab_R_Y_2_clicked
        self.lab_R_X_3.mousePressEvent = self.lab_R_X_3_clicked
        self.lab_R_Y_3.mousePressEvent = self.lab_R_Y_3_clicked
        self.lab_R_X_4.mousePressEvent = self.lab_R_X_4_clicked
        self.lab_R_Y_4.mousePressEvent = self.lab_R_Y_4_clicked
        # TabPage3机器人控制器参数设置
        self.btn_controller_power_on.clicked.connect(self.btn_controller_power_on_click)
        self.btn_controller_power_off.clicked.connect(self.btn_controller_power_off_click)
        self.btn_robot_power_on.clicked.connect(self.btn_robot_power_on_click)
        self.btn_robot_power_off.clicked.connect(self.btn_robot_power_off_click)
        self.btn_enable_robot.clicked.connect(self.btn_enable_robot_click)
        self.btn_disable_robot.clicked.connect(self.btn_disable_robot_click)
        self.btn_home.clicked.connect(self.btn_home_click)
        self.btn_play.clicked.connect(self.btn_play_click)
        # TabPage3机器人位置控制
        self.btn_joint.clicked.connect(self.btn_joint_click)
        self.btn_xyz.clicked.connect(self.btn_xyz_click)
        # 关节1：加,减
        self.btn_j1_add.clicked.connect(self.btn_j1_add_click)
        self.btn_j1_subtract.clicked.connect(self.btn_j1_subtract_click)
        # 关节2：加,减
        self.btn_j2_add.clicked.connect(self.btn_j2_add_click)
        self.btn_j2_subtract.clicked.connect(self.btn_j2_subtract_click)
        # 关节3：加,减
        self.btn_j3_add.clicked.connect(self.btn_j3_add_click)
        self.btn_j3_subtract.clicked.connect(self.btn_j3_subtract_click)
        # 关节4：加,减
        self.btn_j4_add.clicked.connect(self.btn_j4_add_click)
        self.btn_j4_subtract.clicked.connect(self.btn_j4_subtract_click)
        # 关节5：加,减
        self.btn_j5_add.clicked.connect(self.btn_j5_add_click)
        self.btn_j5_subtract.clicked.connect(self.btn_j5_subtract_click)
        # 关节6：加,减
        self.btn_j6_add.clicked.connect(self.btn_j6_add_click)
        self.btn_j6_subtract.clicked.connect(self.btn_j6_subtract_click)
        # 速度按钮
        self.btn_speed_min.clicked.connect(self.btn_speed_min_click)
        self.btn_speed_max.clicked.connect(self.btn_speed_max_click)
        # 步长按钮
        self.btn_step_add.clicked.connect(self.btn_step_add_click)
        self.btn_step_subtract.clicked.connect(self.btn_step_subtract_click)
        # 扎丝机开/关
        self.btn_zsj_open.clicked.connect(self.btn_zsj_open_click)
        self.btn_zsj_close.clicked.connect(self.btn_zsj_close_click)
        # 吸盘打开/关闭
        self.btn_xipan_open.clicked.connect(self.btn_xipan_open_click)
        self.btn_xipan_close.clicked.connect(self.btn_xipan_close_click)
        # 快换打开/关闭
        self.btn_kh_open.clicked.connect(self.btn_kh_open_click)
        self.btn_kh_close.clicked.connect(self.btn_kh_close_click)
        # 滑轨向左/向右
        self.btn_servor_left.clicked.connect(self.btn_servor_left_click)
        self.btn_servor_right.clicked.connect(self.btn_servor_right_click)
        # 滑轨启动/停止/回原点
        self.btn_servor_run.clicked.connect(self.btn_servor_run_click)
        self.btn_servor_reset.clicked.connect(self.btn_servor_reset_click)
        # 滑轨绝对速度
        self.lineEdit_absolute_speed.textChanged.connect(self.absolute_speed_changed)
        # 绝对位置
        self.lineEdit_absolute_loaction.textChanged.connect(self.absolute_loaction_changed)

    def graphics_view_init(self):
        # 创建场景
        self.scene_2d = QtWidgets.QGraphicsScene()
        self.scene_depth = QtWidgets.QGraphicsScene()
        self.scene_biaoding = QtWidgets.QGraphicsScene()

        # 设置视图
        self.graphics_view_2d.setScene(self.scene_2d)
        self.graphics_view_depth.setScene(self.scene_depth)
        self.graphics_view_biaoding.setScene(self.scene_biaoding)

        # 启用抗锯齿
        self.graphics_view_2d.setRenderHint(QtGui.QPainter.Antialiasing)
        self.graphics_view_depth.setRenderHint(QtGui.QPainter.Antialiasing)
        self.graphics_view_biaoding.setRenderHint(QtGui.QPainter.Antialiasing)

        # 设置缩放控制
        self.graphics_view_2d.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.graphics_view_depth.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.graphics_view_biaoding.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

        # 设置缩放策略
        self.graphics_view_2d.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.graphics_view_depth.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.graphics_view_biaoding.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

        # 在初始化时添加这些设置
        self.graphics_view_2d.setInteractive(True)
        self.graphics_view_depth.setInteractive(True)
        self.graphics_view_biaoding.setInteractive(True)

        # 设置缩放限制
        self.graphics_view_2d.setMinimumSize(200, 200)
        self.graphics_view_depth.setMinimumSize(200, 200)
        self.graphics_view_biaoding.setMinimumSize(200, 200)

        # 设置滚动条策略
        self.graphics_view_2d.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.graphics_view_2d.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.graphics_view_depth.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.graphics_view_depth.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.graphics_view_biaoding.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.graphics_view_biaoding.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        self.graphics_view_biaoding.viewport().setCursor(QtCore.Qt.ArrowCursor)  # 设置为普通鼠标箭头

        # 禁用所有内置光标变更逻辑
        self.graphics_view_biaoding.viewport().setAttribute(Qt.WA_SetCursor, False)  # 关键设置1
        # 强制安装事件过滤器
        self.graphics_view_biaoding.viewport().installEventFilter(self)
        # 初始光标
        self._current_cursor = Qt.ArrowCursor

    # endregion
    # region连接始化
    def ini_class(self):
        self.Object_recongnition = Blinx_Object_Recongnition(self.public_class)
        self.jaka = Blinx_Jaka_Socket(self.public_class)
        if self.jaka.jaka_connect_sucess == True:
            self.label_jaka.setText("机器人连接成功")
            self.label_jaka.setStyleSheet("color:rgb(0,200,0);border:none;")
            self.textEdit_log.append("机器人连接成功！\n")
            self.timer_jaka_state.start()
            # self.jaka_rev=Blinx_Jaka_Rev_Socket(self.public_class)#机器人接受数据通道
        else:
            self.label_jaka.setText("机器人连接失败")
            self.label_jaka.setStyleSheet("color:rgb(255,0,0);border:none;")
            self.textEdit_log.append("Error：机器人连接失败\n")
        self.mechCam = Blinx_Mech3D_Viewer_SDK(self.public_class)
        self.mechCam.ConnectCamera()  # 连接相机
        if self.public_class.mech_connected:
            self.label_mech.setText("Mech3D相机连接成功")
            self.label_mech.setStyleSheet("color:rgb(0,200,0);border:none;")
            self.textEdit_log.append("Mech3D相机连接成功！\n")
            self.timer_mech.start()
            device = self.mechCam.camera_infos[self.mechCam.cam_index].model
            devices = []
            devices.append(device)
            self.ComboDevices.addItems(devices)

        else:
            self.label_mech.setText("Mech3D相机连接失败")
            self.label_mech.setStyleSheet("color:rgb(255,0,0);border:none;")
            self.textEdit_log.append("Error：Mech3D相机连接失败\n")

    # endregion

    # 图像识别并经过标定转换为机器人XY坐标---Mech3D相机00
    def image_robotic_coordinate_transformation_mechCam(self):
        if self.public_class.mech_connected:
            # 获取图像数据
            self.public_class.mech_2d_image, self.public_class.mech_depth_map, self.public_class.mech_point_cloud = self.mechCam.GrabImages()
            result_robot = []
            if self.public_class.mech_2d_image is not None:
                all_result_list, image_result = self.Object_recongnition.recognition(self.public_class.mech_img)
                print("all_result_list_return:", all_result_list)
                if len(str(all_result_list)) > 0:
                    for index, result in enumerate(all_result_list):
                        point_x, point_y = self.calibration(result[0][0], result[0][1])
                        result_robot.append((round(point_x, 3), round(point_y, 3), result[1], result[2], result[3]))
                        self.textEdit_log.append(
                            "Mech3D相机图像中心" + "_" + str(index) + ":" + str(result[0][0]) + "," + str(
                                result[0][1]) + "\n")
                        self.textEdit_log.append(
                            "机器人抓取坐标" + "_" + str(index) + ":" + str(round(point_x, 3)) + "," + str(
                                round(point_y, 3)) + "\n")
                print("result_robot:", result_robot)
                self.Image_Show_1.setPixmap(QPixmap())
                showImage = QtGui.QImage(image_result.data, image_result.shape[1], image_result.shape[0],
                                         QtGui.QImage.Format_RGB888)  # 设置显示RGB格式,opencv默认BGR
                self.Image_Show_1.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 显示图像
                self.Image_Show_1.setScaledContents(True)  # 图像自适应窗口大小
                self._set_display_image(self.Image_Show_1, image_result, QtGui.QImage.Format_RGB888)
                return result_robot

    # 图像识别并经过标定转换为机器人XY坐标
    def image_robotic_coordinate_transformation(self, image):
        if image is not None:
            center, image_result = self.Object_recongnition.recognition(image)
            if center[0] != 0 and center[1] != 0:
                point_x, point_y = self.calibration(center[0], center[1])
                self.textEdit_log.append("图像中心：" + str(center[0]) + "," + str(center[1]) + "\n")
                self.textEdit_log.append("机器人抓取坐标：" + str(round(point_x, 3)) + "," + str(round(point_y, 3)) + "\n")
                self.image_show_enable = False  # 显示使能关闭
                self.public_class.mech_img = None
                self.Image_Show_1.setPixmap(QPixmap())
                showImage = QtGui.QImage(image_result.data, image_result.shape[1], image_result.shape[0],
                                         QtGui.QImage.Format_BGR888)  # 设置显示BGR格式,opencv默认BGR
                self.Image_Show_1.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 显示图像
                self.Image_Show_1.setScaledContents(True)  # 图像自适应窗口大小
                self._set_display_image(self.Image_Show_1, image_result, QtGui.QImage.Format_BGR888)
                self.image_show_enable = True  # 显示使能打开
                return [round(point_x, 3), round(point_y, 3)]
            else:
                showImage = QtGui.QImage(image_result.data, image_result.shape[1], image_result.shape[0],
                                         QtGui.QImage.Format_BGR888)  # 设置显示BGR格式,opencv默认BGR
                self.Image_Show_1.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 显示图像
                self.Image_Show_1.setScaledContents(True)  # 图像自适应窗口大小
                self._set_display_image(self.Image_Show_1, image_result, QtGui.QImage.Format_BGR888)
                return [0, 0]

    # endregion
    # region Page2相机参数设置
    def show_camera_mech(self):
        try:
            if self.public_class.mech_connected and self.public_class.is_continue_show:
                # 获取图像数据
                self.public_class.mech_2d_image, self.public_class.mech_depth_map, self.public_class.mech_point_cloud = self.mechCam.GrabImages()

                # 处理并显示2D图像
                self.process_and_show_2d_image()

                # 处理并显示深度图像
                self.process_and_show_depth_image()
        except Exception as e:
            print("show_camera_mech:", e)

    def get_one_image(self):
        self.public_class.is_continue_show = False
        try:
            if self.public_class.mech_connected:
                # 获取图像数据
                self.public_class.mech_2d_image, self.public_class.mech_depth_map, self.public_class.mech_point_cloud = self.mechCam.GrabImages()

                # 处理并显示2D图像
                self.process_and_show_2d_image()

                # 处理并显示深度图像
                self.process_and_show_depth_image()

        except Exception as e:
            print("show_camera_mech:", e)

    def process_and_show_2d_image(self):
        # 清除旧场景
        self.scene_2d.clear()

        # 转换图像格式
        height, width = self.public_class.mech_2d_image.shape[:2]
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(self.public_class.mech_2d_image.data, width, height,
                             bytes_per_line, QtGui.QImage.Format_RGB888)

        # 创建QPixmap并添加到场景
        pixmap = QtGui.QPixmap.fromImage(q_img)
        pixmap_item = self.scene_2d.addPixmap(pixmap)
        pixmap_item.setTransformationMode(QtCore.Qt.SmoothTransformation)

        # 设置场景大小
        self.scene_2d.setSceneRect(0, 0, width, height)

        # 重置视图变换
        self.graphics_view_2d.resetTransform()

        # 新增：初始自适应视图大小
        self.fit_2d_image_to_view()

    def process_and_show_depth_image(self):
        # 清除旧场景
        self.scene_depth.clear()

        # 处理深度图像
        depth_8bit = cv2.normalize(self.public_class.mech_depth_map, None, 0, 255,
                                   cv2.NORM_MINMAX, cv2.CV_8UC1)
        depth_color = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
        depth_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

        # 转换图像格式
        height, width = depth_rgb.shape[:2]
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(depth_rgb.data, width, height,
                             bytes_per_line, QtGui.QImage.Format_RGB888)

        # 创建QPixmap并添加到场景
        pixmap = QtGui.QPixmap.fromImage(q_img)
        pixmap_item = self.scene_depth.addPixmap(pixmap)
        pixmap_item.setTransformationMode(QtCore.Qt.SmoothTransformation)

        # 设置场景大小
        self.scene_depth.setSceneRect(0, 0, width, height)

        # 重置视图变换
        self.graphics_view_depth.resetTransform()

        # 新增：初始自适应视图大小
        self.fit_depth_image_to_view()

    # 新增方法：自适应2D图像到视图
    def fit_2d_image_to_view(self):
        if not self.scene_2d.items():
            return

        # 获取视图和场景的尺寸
        view_rect = self.graphics_view_2d.viewport().rect()
        scene_rect = self.scene_2d.sceneRect()

        # 计算合适的缩放比例
        x_ratio = view_rect.width() / scene_rect.width()
        y_ratio = view_rect.height() / scene_rect.height()
        ratio = min(x_ratio, y_ratio)

        # 应用缩放
        self.graphics_view_2d.resetTransform()
        self.graphics_view_2d.scale(ratio, ratio)

        # 确保图像居中
        self.center_2d_image()

    # 新增方法：自适应深度图像到视图
    def fit_depth_image_to_view(self):
        if not self.scene_depth.items():
            return

        # 获取视图和场景的尺寸
        view_rect = self.graphics_view_depth.viewport().rect()
        scene_rect = self.scene_depth.sceneRect()

        # 计算合适的缩放比例
        x_ratio = view_rect.width() / scene_rect.width()
        y_ratio = view_rect.height() / scene_rect.height()
        ratio = min(x_ratio, y_ratio)

        # 应用缩放
        self.graphics_view_depth.resetTransform()
        self.graphics_view_depth.scale(ratio, ratio)

        # 确保图像居中
        self.center_depth_image()

    # 新增方法：居中2D图像
    def center_2d_image(self):
        if not self.scene_2d.items():
            return
        self.graphics_view_2d.setSceneRect(self.scene_2d.sceneRect())
        self.graphics_view_2d.centerOn(self.scene_2d.items()[0])

    # 新增方法：居中深度图像
    def center_depth_image(self):
        if not self.scene_depth.items():
            return
        self.graphics_view_depth.setSceneRect(self.scene_depth.sceneRect())
        self.graphics_view_depth.centerOn(self.scene_depth.items()[0])

    def wheelEvent(self, event):
        # 检查鼠标在哪个视图上
        if self.graphics_view_2d.underMouse():
            view = self.graphics_view_2d
        elif self.graphics_view_depth.underMouse():
            view = self.graphics_view_depth
        elif self.graphics_view_biaoding.underMouse():
            view = self.graphics_view_biaoding
        else:
            return
        # 计算缩放因子
        factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        # 应用缩放
        view.scale(factor, factor)

    def eventFilter(self, obj, event):
        # 拦截所有可能改变光标的事件
        if event.type() in [QEvent.MouseButtonPress,
                            QEvent.MouseButtonRelease,
                            QEvent.MouseMove,
                            QEvent.HoverEnter,
                            QEvent.HoverLeave]:
            obj.setCursor(self._current_cursor)

        # 额外拦截 QGraphicsView 内部事件
        if event.type() == QEvent.GraphicsSceneMouseMove:
            self.graphics_view_biaoding.viewport().setCursor(self._current_cursor)

        return super().eventFilter(obj, event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 拖拽开始时强制锁定
            self._current_cursor = Qt.ArrowCursor
            self.graphics_view_biaoding.viewport().setCursor(Qt.ArrowCursor)
            self.graphics_view_biaoding.setDragMode(QGraphicsView.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # 实时矫正（每秒60次检查）
        if self.graphics_view_biaoding.viewport().cursor().shape() != self._current_cursor:
            self.graphics_view_biaoding.viewport().setCursor(self._current_cursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # 拖拽结束后保持箭头
        self._current_cursor = Qt.ArrowCursor
        self.graphics_view_biaoding.viewport().setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            # 物理锁定滚动条
            self.graphics_view_2d.verticalScrollBar().setEnabled(False)
            self.graphics_view_2d.horizontalScrollBar().setEnabled(False)
            self.graphics_view_depth.verticalScrollBar().setEnabled(False)
            self.graphics_view_depth.horizontalScrollBar().setEnabled(False)
            self.graphics_view_biaoding.verticalScrollBar().setEnabled(False)
            self.graphics_view_biaoding.horizontalScrollBar().setEnabled(False)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            # 恢复滚动条但不显示
            self.graphics_view_2d.verticalScrollBar().setEnabled(True)
            self.graphics_view_2d.horizontalScrollBar().setEnabled(True)
            self.graphics_view_depth.verticalScrollBar().setEnabled(True)
            self.graphics_view_depth.horizontalScrollBar().setEnabled(True)
            self.graphics_view_biaoding.verticalScrollBar().setEnabled(True)
            self.graphics_view_biaoding.horizontalScrollBar().setEnabled(True)
        super().keyReleaseEvent(event)

    def get_continue_frame(self):
        self.public_class.is_continue_show = True

    # ch:存图 | en:save image
    def save_jpg(self):
        now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))  # 获取系统时间
        if self.public_class.mech_2d_image is None:
            strError = "Save jpg failed ret"
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            cv2.imwrite('pic/cam1/image_' + str(now_time) + '.jpg', self.public_class.mech_2d_image)  # 保存2D图片
            cv2.imwrite('pic/cam1/image_' + str(now_time) + '.tiff', self.public_class.mech_depth_map)  # 保存深度图片
            self.show_auto_close_message("图片已保存！")
            print("Save image_1 success")
    #按空格保存图像
    def on_space_pressed(self):
        if self.tabWidget.currentIndex() == 1:
            if self.public_class.mech_2d_image is not None:
                self.save_jpg()
    def show_auto_close_message(self,message):
        self.msg = QMessageBox(self)
        self.msg.setWindowTitle("提示")
        self.msg.setText(str(message))
        self.msg.show()
        # 定时关闭消息框
        QTimer.singleShot(100, self.msg.close)

    # endregion

    # region 流程定时器逻辑
    def blinx_timer_process(self):
        # 判断是否是吸盘取放功能
        if self.public_class.sucker_state:
            # 判断吸盘是取还是放
            if self.public_class.sucker_type == 0:   # 如果类别是取
                # 判断执行器放置区域
                if self.public_class.sucker_process == "0-0":
                    # 获取机械臂控制器的数字输入信号
                    self.jaka.blinx_get_digital_input_status()
                    time.sleep(0.5)
                    # 将吸盘捆扎机的放置区的状态提出
                    xp_state = self.robot_DI_1
                    kzj_state = self.robot_DI_2
                    # 判断吸盘与捆扎机是否都在
                    if xp_state == 1 and kzj_state == 1:
                        self.public_class.sucker_process = "1-0"
                    # 判断如果吸盘在捆扎机不在
                    elif xp_state == 1 and kzj_state != 1:
                        print("提示捆扎机在末端的提示语")
                    # 判断如果捆扎机在，吸盘不在
                    elif xp_state != 1 and kzj_state == 1:
                        print("提示吸盘已在末端上")
                    # 如果两个都不在
                    elif xp_state != 1 and kzj_state != 1:
                        print("报警，请将末端执行器归位")
                        self.public_class.sucker_state = False
                # 控制机械臂旋转180度
                elif self.public_class.sucker_process == "1-0":
                    data = [float(self.public_class.joint_pos[0]) - 180.00, float(self.public_class.joint_pos[1]),
                            float(self.public_class.joint_pos[2]), float(self.public_class.joint_pos[3]),
                            float(self.public_class.joint_pos[4]), float(self.public_class.joint_pos[5])]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.sucker_process = "1-2"
                # 角度是否到达，如果到达控制机械臂到达吸盘上方
                elif self.public_class.sucker_process == "1-2":
                    error_j1 = (float(self.public_class.new_data[0]) - float(
                        self.public_class.joint_pos[0]))
                    error_j2 = (float(self.public_class.new_data[1]) - float(
                        self.public_class.joint_pos[1]))
                    error_j3 = (float(self.public_class.new_data[2]) - float(
                        self.public_class.joint_pos[2]))
                    error_j4 = (float(self.public_class.new_data[3]) - float(
                        self.public_class.joint_pos[3]))
                    error_j5 = (float(self.public_class.new_data[4]) - float(
                        self.public_class.joint_pos[4]))
                    error_j6 = (float(self.public_class.new_data[5]) - float(
                        self.public_class.joint_pos[5]))
                    if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                            -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                        # 控制机械臂到达吸盘上方
                        data = [float(self.public_class.sucker_actuator_loc[0]),
                                float(self.public_class.sucker_actuator_loc[1]),
                                float(self.public_class.sucker_actuator_loc[2]) + 30.00,
                                float(self.public_class.sucker_actuator_loc[3]),
                                float(self.public_class.sucker_actuator_loc[4]),
                                float(self.public_class.sucker_actuator_loc[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.sucker_process = "1-3"
                # 判断坐标是否到达，如果到达先打开快换夹具，在控制机械臂到达吸盘位置
                elif self.public_class.sucker_process == "1-3":
                    error_x = (float(self.public_class.new_data[0]) - float(
                        self.public_class.tcp_pos[0]))
                    error_y = (float(self.public_class.new_data[1]) - float(
                        self.public_class.tcp_pos[1]))
                    error_z = (float(self.public_class.new_data[2]) - float(
                        self.public_class.tcp_pos[2]))
                    if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                        # 快换夹具开
                        self.jaka.blinx_set_digital_output(6, 5, 1)
                        self.jaka.blinx_set_digital_output(6, 8, 1)
                        time.sleep(0.5)
                        self.jaka.blinx_set_digital_output(6, 5, 0)
                        # 控制机械臂到达吸盘位置
                        data = [float(self.public_class.sucker_actuator_loc[0]),
                                float(self.public_class.sucker_actuator_loc[1]),
                                float(self.public_class.sucker_actuator_loc[2]),
                                float(self.public_class.sucker_actuator_loc[3]),
                                float(self.public_class.sucker_actuator_loc[4]),
                                float(self.public_class.sucker_actuator_loc[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.sucker_process = "1-4"
                # 判断坐标是否到达，如果到达控制快换夹具关，并控制机械臂上升
                elif self.public_class.sucker_process == "1-4":
                    error_x = (float(self.public_class.new_data[0]) - float(
                        self.public_class.tcp_pos[0]))
                    error_y = (float(self.public_class.new_data[1]) - float(
                        self.public_class.tcp_pos[1]))
                    error_z = (float(self.public_class.new_data[2]) - float(
                        self.public_class.tcp_pos[2]))
                    if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                        # 快换夹具开
                        self.jaka.blinx_set_digital_output(6, 5, 1)
                        self.jaka.blinx_set_digital_output(6, 8, 0)
                        time.sleep(0.5)
                        self.jaka.blinx_set_digital_output(6, 5, 0)
                        # 控制机械臂到达吸盘位置
                        data = [float(self.public_class.tcp_pos[0]),
                                float(self.public_class.tcp_pos[1]),
                                float(self.public_class.tcp_pos[2] + 10.00),
                                float(self.public_class.tcp_pos[3]),
                                float(self.public_class.tcp_pos[4]),
                                float(self.public_class.tcp_pos[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.sucker_process = "1-5"
                # 判断坐标是否到达，如果到达控制Y轴移动到
                elif self.public_class.sucker_process == "1-5":
                    error_x = (float(self.public_class.new_data[0]) - float(
                        self.public_class.tcp_pos[0]))
                    error_y = (float(self.public_class.new_data[1]) - float(
                        self.public_class.tcp_pos[1]))
                    error_z = (float(self.public_class.new_data[2]) - float(
                        self.public_class.tcp_pos[2]))
                    if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                        # 控制机械臂退出吸盘放置位置
                        data = [float(self.public_class.tcp_pos[0]),
                                float(self.public_class.tcp_pos[1] + 100.00),
                                float(self.public_class.tcp_pos[2]),
                                float(self.public_class.tcp_pos[3]),
                                float(self.public_class.tcp_pos[4]),
                                float(self.public_class.tcp_pos[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.sucker_process = "1-6"
                # 判断坐标是否到达，如果到达控制机械臂到达回收位置
                elif self.public_class.sucker_process == "1-6":
                    error_x = (float(self.public_class.new_data[0]) - float(
                        self.public_class.tcp_pos[0]))
                    error_y = (float(self.public_class.new_data[1]) - float(
                        self.public_class.tcp_pos[1]))
                    error_z = (float(self.public_class.new_data[2]) - float(
                        self.public_class.tcp_pos[2]))
                    if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                        data = [float(self.public_class.initial_angle[0]) - 180.00, float(self.public_class.initial_angle[1]),
                                float(self.public_class.initial_angle[2]), float(self.public_class.initial_angle[3]),
                                float(self.public_class.initial_angle[4]), float(self.public_class.initial_angle[5])]
                        self.jaka.blinx_joint_move(0, data, 50, 50)
                        self.public_class.new_data = data
                        self.public_class.sucker_process = "1-7"
                # 角度是否到达，如果到达控制机械臂到达吸盘上方
                elif self.public_class.sucker_process == "1-7":
                    error_j1 = (float(self.public_class.new_data[0]) - float(
                        self.public_class.joint_pos[0]))
                    error_j2 = (float(self.public_class.new_data[1]) - float(
                        self.public_class.joint_pos[1]))
                    error_j3 = (float(self.public_class.new_data[2]) - float(
                        self.public_class.joint_pos[2]))
                    error_j4 = (float(self.public_class.new_data[3]) - float(
                        self.public_class.joint_pos[3]))
                    error_j5 = (float(self.public_class.new_data[4]) - float(
                        self.public_class.joint_pos[4]))
                    error_j6 = (float(self.public_class.new_data[5]) - float(
                        self.public_class.joint_pos[5]))
                    if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                            -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                            data = [float(self.public_class.initial_angle[0]),
                                    float(self.public_class.initial_angle[1]),
                                    float(self.public_class.initial_angle[2]),
                                    float(self.public_class.initial_angle[3]),
                                    float(self.public_class.initial_angle[4]),
                                    float(self.public_class.initial_angle[5])]
                            self.jaka.blinx_joint_move(0, data, 50, 50)
                            self.public_class.new_data = data
                            self.public_class.sucker_process = "1-8"
                # 角度是否到达，如果到达控制机械臂到达吸盘上方
                elif self.public_class.sucker_process == "1-8":
                    error_j1 = (float(self.public_class.new_data[0]) - float(
                        self.public_class.joint_pos[0]))
                    error_j2 = (float(self.public_class.new_data[1]) - float(
                        self.public_class.joint_pos[1]))
                    error_j3 = (float(self.public_class.new_data[2]) - float(
                        self.public_class.joint_pos[2]))
                    error_j4 = (float(self.public_class.new_data[3]) - float(
                        self.public_class.joint_pos[3]))
                    error_j5 = (float(self.public_class.new_data[4]) - float(
                        self.public_class.joint_pos[4]))
                    error_j6 = (float(self.public_class.new_data[5]) - float(
                        self.public_class.joint_pos[5]))
                    if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (
                            -0.1 <= error_j3 <= 0.1) and (
                            -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (
                            -0.1 <= error_j6 <= 0.1):
                        data = [float(self.public_class.initial_angle[0]),
                                float(self.public_class.initial_angle[1]),
                                float(self.public_class.initial_angle[2]),
                                float(self.public_class.initial_angle[3]),
                                float(self.public_class.initial_angle[4]),
                                float(self.public_class.initial_angle[5])]
                        self.jaka.blinx_joint_move(0, data, 50, 50)
                        self.public_class.new_data = None
                        self.public_class.sucker_process = "0-0"
                        self.public_class.sucker_state = False
            else:
                # 判断执行器放置区域
                if self.public_class.sucker_process == "0-0":
                    # 获取机械臂控制器的数字输入信号
                    self.jaka.blinx_get_digital_input_status()
                    time.sleep(0.5)
                    # 将吸盘捆扎机的放置区的状态提出
                    xp_state = self.robot_DI_1
                    kzj_state = self.robot_DI_2
                    # 判断吸盘与捆扎机是否都在
                    if xp_state == 1 and kzj_state == 1:
                        print("吸盘已在放置区")
                    # 判断如果吸盘在捆扎机不在
                    elif xp_state == 1 and kzj_state != 1:
                        print("提示目前末端是捆扎机")
                    # 判断如果捆扎机在，吸盘不在
                    elif xp_state != 1 and kzj_state == 1:
                        self.public_class.sucker_process = "1-0"
                    # 如果两个都不在
                    elif xp_state != 1 and kzj_state != 1:
                        print("报警，请将末端执行器归位")
                        self.public_class.sucker_state = False
                # 控制机械臂旋转180度
                elif self.public_class.sucker_process == "1-0":
                    data = [float(self.public_class.joint_pos[0]) - 180.00, float(self.public_class.joint_pos[1]),
                            float(self.public_class.joint_pos[2]), float(self.public_class.joint_pos[3]),
                            float(self.public_class.joint_pos[4]), float(self.public_class.joint_pos[5])]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.sucker_process = "1-2"
                # 角度是否到达，如果到达控制机械臂待放置位置
                elif self.public_class.sucker_process == "1-2":
                    error_j1 = (float(self.public_class.new_data[0]) - float(
                        self.public_class.joint_pos[0]))
                    error_j2 = (float(self.public_class.new_data[1]) - float(
                        self.public_class.joint_pos[1]))
                    error_j3 = (float(self.public_class.new_data[2]) - float(
                        self.public_class.joint_pos[2]))
                    error_j4 = (float(self.public_class.new_data[3]) - float(
                        self.public_class.joint_pos[3]))
                    error_j5 = (float(self.public_class.new_data[4]) - float(
                        self.public_class.joint_pos[4]))
                    error_j6 = (float(self.public_class.new_data[5]) - float(
                        self.public_class.joint_pos[5]))
                    if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                            -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                        # 控制机械臂待放置位置
                        data = [float(self.public_class.sucker_actuator_loc[0]),
                                float(self.public_class.sucker_actuator_loc[1]) + 100.00,
                                float(self.public_class.sucker_actuator_loc[2]) + 10.00,
                                float(self.public_class.sucker_actuator_loc[3]),
                                float(self.public_class.sucker_actuator_loc[4]),
                                float(self.public_class.sucker_actuator_loc[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.sucker_process = "1-3"
                # 判断坐标是否到达，如果到达先控制机械臂到达吸盘放置区位置上一公分位置
                elif self.public_class.sucker_process == "1-3":
                    error_x = (float(self.public_class.new_data[0]) - float(
                        self.public_class.tcp_pos[0]))
                    error_y = (float(self.public_class.new_data[1]) - float(
                        self.public_class.tcp_pos[1]))
                    error_z = (float(self.public_class.new_data[2]) - float(
                        self.public_class.tcp_pos[2]))
                    if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                        # 控制机械臂到达吸盘位置
                        data = [float(self.public_class.tcp_pos[0]),
                                float(self.public_class.tcp_pos[1]) - 100.00,
                                float(self.public_class.tcp_pos[2]),
                                float(self.public_class.tcp_pos[3]),
                                float(self.public_class.tcp_pos[4]),
                                float(self.public_class.tcp_pos[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.sucker_process = "1-4"
                # 判断坐标是否到达，并控制机械臂下降，再释放快换夹具
                elif self.public_class.sucker_process == "1-4":
                    error_x = (float(self.public_class.new_data[0]) - float(
                        self.public_class.tcp_pos[0]))
                    error_y = (float(self.public_class.new_data[1]) - float(
                        self.public_class.tcp_pos[1]))
                    error_z = (float(self.public_class.new_data[2]) - float(
                        self.public_class.tcp_pos[2]))
                    if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                        # 控制机械臂到达吸盘位置
                        data = [float(self.public_class.tcp_pos[0]),
                                float(self.public_class.tcp_pos[1]),
                                float(self.public_class.tcp_pos[2]) - 10.00,
                                float(self.public_class.tcp_pos[3]),
                                float(self.public_class.tcp_pos[4]),
                                float(self.public_class.tcp_pos[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.sucker_process = "1-5"
                        time.sleep(1)
                        # 快换夹具释放
                        self.jaka.blinx_set_digital_output(6, 5, 1)
                        self.jaka.blinx_set_digital_output(6, 8, 1)
                        time.sleep(0.5)
                        self.jaka.blinx_set_digital_output(6, 5, 0)
                # 判断坐标是否到达，如果到达控制z轴上升
                elif self.public_class.sucker_process == "1-5":
                    error_x = (float(self.public_class.new_data[0]) - float(
                        self.public_class.tcp_pos[0]))
                    error_y = (float(self.public_class.new_data[1]) - float(
                        self.public_class.tcp_pos[1]))
                    error_z = (float(self.public_class.new_data[2]) - float(
                        self.public_class.tcp_pos[2]))
                    if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                        # 控制机械臂退出吸盘放置位置
                        data = [float(self.public_class.tcp_pos[0]),
                                float(self.public_class.tcp_pos[1]),
                                float(self.public_class.tcp_pos[2] + 30.00),
                                float(self.public_class.tcp_pos[3]),
                                float(self.public_class.tcp_pos[4]),
                                float(self.public_class.tcp_pos[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.sucker_process = "1-6"
                # 判断坐标是否到达，如果到达控制机械臂到达回收位置
                elif self.public_class.sucker_process == "1-6":
                    error_x = (float(self.public_class.new_data[0]) - float(
                        self.public_class.tcp_pos[0]))
                    error_y = (float(self.public_class.new_data[1]) - float(
                        self.public_class.tcp_pos[1]))
                    error_z = (float(self.public_class.new_data[2]) - float(
                        self.public_class.tcp_pos[2]))
                    if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                        data = [float(self.public_class.initial_angle[0]) - 180.00,
                                float(self.public_class.initial_angle[1]),
                                float(self.public_class.initial_angle[2]), float(self.public_class.initial_angle[3]),
                                float(self.public_class.initial_angle[4]), float(self.public_class.initial_angle[5])]
                        self.jaka.blinx_joint_move(0, data, 50, 50)
                        self.public_class.new_data = data
                        self.public_class.sucker_process = "1-7"
                # 角度是否到达，如果到达控制机械臂回到初始角度
                elif self.public_class.sucker_process == "1-7":
                    error_j1 = (float(self.public_class.new_data[0]) - float(
                        self.public_class.joint_pos[0]))
                    error_j2 = (float(self.public_class.new_data[1]) - float(
                        self.public_class.joint_pos[1]))
                    error_j3 = (float(self.public_class.new_data[2]) - float(
                        self.public_class.joint_pos[2]))
                    error_j4 = (float(self.public_class.new_data[3]) - float(
                        self.public_class.joint_pos[3]))
                    error_j5 = (float(self.public_class.new_data[4]) - float(
                        self.public_class.joint_pos[4]))
                    error_j6 = (float(self.public_class.new_data[5]) - float(
                        self.public_class.joint_pos[5]))
                    if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                            -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                        data = [float(self.public_class.initial_angle[0]),
                                float(self.public_class.initial_angle[1]),
                                float(self.public_class.initial_angle[2]),
                                float(self.public_class.initial_angle[3]),
                                float(self.public_class.initial_angle[4]),
                                float(self.public_class.initial_angle[5])]
                        self.jaka.blinx_joint_move(0, data, 50, 50)
                        self.public_class.new_data = data
                        self.public_class.sucker_process = "1-8"
                # 角度是否到达，如果到达控制机械臂到达吸盘上方
                elif self.public_class.sucker_process == "1-8":
                    error_j1 = (float(self.public_class.new_data[0]) - float(
                        self.public_class.joint_pos[0]))
                    error_j2 = (float(self.public_class.new_data[1]) - float(
                        self.public_class.joint_pos[1]))
                    error_j3 = (float(self.public_class.new_data[2]) - float(
                        self.public_class.joint_pos[2]))
                    error_j4 = (float(self.public_class.new_data[3]) - float(
                        self.public_class.joint_pos[3]))
                    error_j5 = (float(self.public_class.new_data[4]) - float(
                        self.public_class.joint_pos[4]))
                    error_j6 = (float(self.public_class.new_data[5]) - float(
                        self.public_class.joint_pos[5]))
                    if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (
                            -0.1 <= error_j3 <= 0.1) and (
                            -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (
                            -0.1 <= error_j6 <= 0.1):
                        data = [float(self.public_class.initial_angle[0]),
                                float(self.public_class.initial_angle[1]),
                                float(self.public_class.initial_angle[2]),
                                float(self.public_class.initial_angle[3]),
                                float(self.public_class.initial_angle[4]),
                                float(self.public_class.initial_angle[5])]
                        self.jaka.blinx_joint_move(0, data, 50, 50)
                        self.public_class.new_data = None
                        self.public_class.sucker_process = "0-0"
                        self.public_class.sucker_state = False
        # 判断是否是捆扎机取放功能
        elif self.public_class.bundle_state:
            # 判断捆扎机是取还是放
            if self.public_class.bundle_type == 0:   # 如果类别是取
                # 判断执行器放置区域
                if self.public_class.bundle_process == "0-0":
                    # 获取机械臂控制器的数字输入信号
                    self.jaka.blinx_get_digital_input_status()
                    time.sleep(0.5)
                    # 将吸盘捆扎机的放置区的状态提出
                    xp_state = self.robot_DI_1
                    kzj_state = self.robot_DI_2
                    # 判断吸盘与捆扎机是否都在
                    if xp_state == 1 and kzj_state == 1:
                        self.public_class.bundle_process = "1-0"
                    # 判断如果吸盘在捆扎机不在
                    elif xp_state == 1 and kzj_state != 1:
                        print("提示捆扎机在末端的提示语")
                        self.public_class.bundle_state = False
                    # 判断如果捆扎机在，吸盘不在
                    elif xp_state != 1 and kzj_state == 1:
                        print("提示吸盘在末端上")
                        self.public_class.bundle_state = False
                    # 如果两个都不在
                    elif xp_state != 1 and kzj_state != 1:
                        print("报警，请将末端执行器归位")
                        self.public_class.bundle_state = False
                # 控制机械臂旋转180度
                elif self.public_class.bundle_process == "1-0":
                    data = [float(self.public_class.joint_pos[0]) - 180.00, float(self.public_class.joint_pos[1]),
                            float(self.public_class.joint_pos[2]), float(self.public_class.joint_pos[3]),
                            float(self.public_class.joint_pos[4]), float(self.public_class.joint_pos[5])]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.bundle_process = "1-1"
                # 角度是否到达，如果到达控制J6旋转90都
                elif self.public_class.bundle_process == "1-1":
                    error_j1 = (float(self.public_class.new_data[0]) - float(
                        self.public_class.joint_pos[0]))
                    error_j2 = (float(self.public_class.new_data[1]) - float(
                        self.public_class.joint_pos[1]))
                    error_j3 = (float(self.public_class.new_data[2]) - float(
                        self.public_class.joint_pos[2]))
                    error_j4 = (float(self.public_class.new_data[3]) - float(
                        self.public_class.joint_pos[3]))
                    error_j5 = (float(self.public_class.new_data[4]) - float(
                        self.public_class.joint_pos[4]))
                    error_j6 = (float(self.public_class.new_data[5]) - float(
                        self.public_class.joint_pos[5]))
                    if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                            -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                        # 控制末端移动90度
                        data = [float(self.public_class.joint_pos[0]), float(self.public_class.joint_pos[1]),
                                float(self.public_class.joint_pos[2]), float(self.public_class.joint_pos[3]),
                                float(self.public_class.joint_pos[4]),
                                float(self.public_class.joint_pos[5]) - 90.00]
                        self.jaka.blinx_joint_move(0, data, 50, 50)
                        self.public_class.new_data = data
                        self.public_class.bundle_process = "1-2"
                # 角度是否到达，如果到达控制机械臂到达捆扎机上方
                elif self.public_class.bundle_process == "1-2":
                    error_j1 = (float(self.public_class.new_data[0]) - float(
                        self.public_class.joint_pos[0]))
                    error_j2 = (float(self.public_class.new_data[1]) - float(
                        self.public_class.joint_pos[1]))
                    error_j3 = (float(self.public_class.new_data[2]) - float(
                        self.public_class.joint_pos[2]))
                    error_j4 = (float(self.public_class.new_data[3]) - float(
                        self.public_class.joint_pos[3]))
                    error_j5 = (float(self.public_class.new_data[4]) - float(
                        self.public_class.joint_pos[4]))
                    error_j6 = (float(self.public_class.new_data[5]) - float(
                        self.public_class.joint_pos[5]))
                    if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                            -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                        # 控制机械臂到达捆扎机上方
                        data = [float(self.public_class.bundle_actuator_loc[0]),
                                float(self.public_class.bundle_actuator_loc[1]),
                                float(self.public_class.bundle_actuator_loc[2]) + 30.00,
                                float(self.public_class.bundle_actuator_loc[3]),
                                float(self.public_class.bundle_actuator_loc[4]),
                                float(self.public_class.bundle_actuator_loc[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.bundle_process = "1-3"
                # 判断坐标是否到达，如果到达先打开快换夹具，在控制机械臂到达捆扎机位置
                elif self.public_class.bundle_process == "1-3":
                    error_x = (float(self.public_class.new_data[0]) - float(
                        self.public_class.tcp_pos[0]))
                    error_y = (float(self.public_class.new_data[1]) - float(
                        self.public_class.tcp_pos[1]))
                    error_z = (float(self.public_class.new_data[2]) - float(
                        self.public_class.tcp_pos[2]))
                    if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                        # 快换夹具开
                        self.jaka.blinx_set_digital_output(6, 5, 1)
                        self.jaka.blinx_set_digital_output(6, 8, 1)
                        time.sleep(0.5)
                        self.jaka.blinx_set_digital_output(6, 5, 0)
                        # 控制机械臂到达捆扎机位置
                        data = [float(self.public_class.bundle_actuator_loc[0]),
                                float(self.public_class.bundle_actuator_loc[1]),
                                float(self.public_class.bundle_actuator_loc[2]),
                                float(self.public_class.bundle_actuator_loc[3]),
                                float(self.public_class.bundle_actuator_loc[4]),
                                float(self.public_class.bundle_actuator_loc[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.bundle_process = "1-4"
                # 判断坐标是否到达，如果到达控制快换夹具关，并控制机械臂上升
                elif self.public_class.bundle_process == "1-4":
                    error_x = (float(self.public_class.new_data[0]) - float(
                        self.public_class.tcp_pos[0]))
                    error_y = (float(self.public_class.new_data[1]) - float(
                        self.public_class.tcp_pos[1]))
                    error_z = (float(self.public_class.new_data[2]) - float(
                        self.public_class.tcp_pos[2]))
                    if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                        # 快换夹具开
                        self.jaka.blinx_set_digital_output(6, 5, 1)
                        self.jaka.blinx_set_digital_output(6, 8, 0)
                        time.sleep(0.5)
                        self.jaka.blinx_set_digital_output(6, 5, 0)
                        # 控制机械臂到达吸盘位置
                        data = [float(self.public_class.tcp_pos[0]),
                                float(self.public_class.tcp_pos[1]),
                                float(self.public_class.tcp_pos[2] + 10.00),
                                float(self.public_class.tcp_pos[3]),
                                float(self.public_class.tcp_pos[4]),
                                float(self.public_class.tcp_pos[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.bundle_process = "1-5"
                # 判断坐标是否到达，如果到达控制Y轴移动到
                elif self.public_class.bundle_process == "1-5":
                    error_x = (float(self.public_class.new_data[0]) - float(
                        self.public_class.tcp_pos[0]))
                    error_y = (float(self.public_class.new_data[1]) - float(
                        self.public_class.tcp_pos[1]))
                    error_z = (float(self.public_class.new_data[2]) - float(
                        self.public_class.tcp_pos[2]))
                    if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                        # 控制机械臂退出吸盘放置位置
                        data = [float(self.public_class.tcp_pos[0]),
                                float(self.public_class.tcp_pos[1] + 100.00),
                                float(self.public_class.tcp_pos[2]),
                                float(self.public_class.tcp_pos[3]),
                                float(self.public_class.tcp_pos[4]),
                                float(self.public_class.tcp_pos[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.bundle_process = "1-6"
                # 判断坐标是否到达，如果到达控制机械臂到达回收位置
                elif self.public_class.bundle_process == "1-6":
                    error_x = (float(self.public_class.new_data[0]) - float(
                        self.public_class.tcp_pos[0]))
                    error_y = (float(self.public_class.new_data[1]) - float(
                        self.public_class.tcp_pos[1]))
                    error_z = (float(self.public_class.new_data[2]) - float(
                        self.public_class.tcp_pos[2]))
                    if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                        data = [float(self.public_class.initial_angle[0]) - 180.00,
                                float(self.public_class.initial_angle[1]),
                                float(self.public_class.initial_angle[2]),
                                float(self.public_class.initial_angle[3]),
                                float(self.public_class.initial_angle[4]),
                                float(self.public_class.initial_angle[5])]
                        self.jaka.blinx_joint_move(0, data, 50, 50)
                        self.public_class.new_data = data
                        self.public_class.bundle_process = "1-7"
                # 角度是否到达，如果到达控制机械臂到达捆扎机过度位置
                elif self.public_class.bundle_process == "1-7":
                    error_j1 = (float(self.public_class.new_data[0]) - float(
                        self.public_class.joint_pos[0]))
                    error_j2 = (float(self.public_class.new_data[1]) - float(
                        self.public_class.joint_pos[1]))
                    error_j3 = (float(self.public_class.new_data[2]) - float(
                        self.public_class.joint_pos[2]))
                    error_j4 = (float(self.public_class.new_data[3]) - float(
                        self.public_class.joint_pos[3]))
                    error_j5 = (float(self.public_class.new_data[4]) - float(
                        self.public_class.joint_pos[4]))
                    error_j6 = (float(self.public_class.new_data[5]) - float(
                        self.public_class.joint_pos[5]))
                    if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                            -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                        # 控制机械臂捆扎机过度位置
                        data = [float(self.public_class.tcp_pos[0]),
                                float(self.public_class.tcp_pos[1] + 50.00),
                                float(self.public_class.tcp_pos[2]),
                                float(self.public_class.tcp_pos[3]),
                                float(self.public_class.tcp_pos[4]),
                                float(self.public_class.tcp_pos[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.bundle_process = "1-8"
                # 判断坐标是否到达，如果到达控制机械臂旋转180度
                elif self.public_class.bundle_process == "1-8":
                    error_x = (float(self.public_class.new_data[0]) - float(
                        self.public_class.tcp_pos[0]))
                    error_y = (float(self.public_class.new_data[1]) - float(
                        self.public_class.tcp_pos[1]))
                    error_z = (float(self.public_class.new_data[2]) - float(
                        self.public_class.tcp_pos[2]))
                    if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                        data = [float(self.public_class.joint_pos[0]) + 180.00,
                                float(self.public_class.joint_pos[1]),
                                float(self.public_class.joint_pos[2]),
                                float(self.public_class.joint_pos[3]),
                                float(self.public_class.joint_pos[4]),
                                float(self.public_class.joint_pos[5])]
                        self.jaka.blinx_joint_move(0, data, 50, 50)
                        self.public_class.new_data = data
                        self.public_class.bundle_process = "1-9"
                # 角度是否到达，如果到达控制机械臂到达初始位置
                elif self.public_class.bundle_process == "1-9":
                    error_j1 = (float(self.public_class.new_data[0]) - float(
                        self.public_class.joint_pos[0]))
                    error_j2 = (float(self.public_class.new_data[1]) - float(
                        self.public_class.joint_pos[1]))
                    error_j3 = (float(self.public_class.new_data[2]) - float(
                        self.public_class.joint_pos[2]))
                    error_j4 = (float(self.public_class.new_data[3]) - float(
                        self.public_class.joint_pos[3]))
                    error_j5 = (float(self.public_class.new_data[4]) - float(
                        self.public_class.joint_pos[4]))
                    error_j6 = (float(self.public_class.new_data[5]) - float(
                        self.public_class.joint_pos[5]))
                    if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (
                            -0.1 <= error_j3 <= 0.1) and (
                            -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (
                            -0.1 <= error_j6 <= 0.1):
                        data = [float(self.public_class.initial_angle[0]),
                                float(self.public_class.initial_angle[1]),
                                float(self.public_class.initial_angle[2]),
                                float(self.public_class.initial_angle[3]),
                                float(self.public_class.initial_angle[4]),
                                float(self.public_class.initial_angle[5])]
                        self.jaka.blinx_joint_move(0, data, 50, 50)
                        self.public_class.new_data = None
                        self.public_class.bundle_process = "0-0"
                        self.public_class.bundle_state = False
            else:
                # 判断执行器放置区域
                if self.public_class.bundle_process == "0-0":
                    # 获取机械臂控制器的数字输入信号
                    self.jaka.blinx_get_digital_input_status()
                    time.sleep(0.5)
                    # 将吸盘捆扎机的放置区的状态提出
                    xp_state = 1  # self.robot_DI_1
                    kzj_state = 0  # self.robot_DI_2
                    # 判断吸盘与捆扎机是否都在
                    if xp_state == 1 and kzj_state == 1:
                        print("捆扎机已在放置区")
                        self.public_class.sucker_state = False
                    # 判断如果吸盘在捆扎机不在
                    elif xp_state == 1 and kzj_state != 1:
                        self.public_class.bundle_process = "1-0"
                    # 判断如果捆扎机在，吸盘不在
                    elif xp_state != 1 and kzj_state == 1:
                        print("提示目前末端是吸盘")
                        self.public_class.sucker_state = False
                    # 如果两个都不在
                    elif xp_state != 1 and kzj_state != 1:
                        print("报警，请将末端执行器归位")
                        self.public_class.sucker_state = False
                # 控制机械臂到达过度点位
                elif self.public_class.bundle_process == "1-0":
                    # 控制机械臂到达吸盘位置
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]) - 50.00,
                            float(self.public_class.tcp_pos[2]),
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.bundle_process = "1-1"
                # 控制机械臂旋转180度
                elif self.public_class.bundle_process == "1-1":
                    error_x = (float(self.public_class.new_data[0]) - float(
                        self.public_class.tcp_pos[0]))
                    error_y = (float(self.public_class.new_data[1]) - float(
                        self.public_class.tcp_pos[1]))
                    error_z = (float(self.public_class.new_data[2]) - float(
                        self.public_class.tcp_pos[2]))
                    if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                        data = [float(self.public_class.joint_pos[0]) - 180.00, float(self.public_class.joint_pos[1]),
                                float(self.public_class.joint_pos[2]), float(self.public_class.joint_pos[3]),
                                float(self.public_class.joint_pos[4]), float(self.public_class.joint_pos[5])]
                        self.jaka.blinx_joint_move(0, data, 50, 50)
                        self.public_class.new_data = data
                        self.public_class.bundle_process = "1-2"
                # 角度是否到达，如果到达控制机械臂待放置位置
                elif self.public_class.bundle_process == "1-2":
                    error_j1 = (float(self.public_class.new_data[0]) - float(
                        self.public_class.joint_pos[0]))
                    error_j2 = (float(self.public_class.new_data[1]) - float(
                        self.public_class.joint_pos[1]))
                    error_j3 = (float(self.public_class.new_data[2]) - float(
                        self.public_class.joint_pos[2]))
                    error_j4 = (float(self.public_class.new_data[3]) - float(
                        self.public_class.joint_pos[3]))
                    error_j5 = (float(self.public_class.new_data[4]) - float(
                        self.public_class.joint_pos[4]))
                    error_j6 = (float(self.public_class.new_data[5]) - float(
                        self.public_class.joint_pos[5]))
                    if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                            -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                        # 控制机械臂待放置位置
                        data = [float(self.public_class.bundle_actuator_loc[0]),
                                float(self.public_class.bundle_actuator_loc[1]) + 100.00,
                                float(self.public_class.bundle_actuator_loc[2]) + 10.00,
                                float(self.public_class.bundle_actuator_loc[3]),
                                float(self.public_class.bundle_actuator_loc[4]),
                                float(self.public_class.bundle_actuator_loc[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.bundle_process = "1-3"
                # 判断坐标是否到达，如果到达先控制机械臂到达捆扎机放置区上方一公分位置
                elif self.public_class.bundle_process == "1-3":
                    error_x = (float(self.public_class.new_data[0]) - float(
                        self.public_class.tcp_pos[0]))
                    error_y = (float(self.public_class.new_data[1]) - float(
                        self.public_class.tcp_pos[1]))
                    error_z = (float(self.public_class.new_data[2]) - float(
                        self.public_class.tcp_pos[2]))
                    if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                        # 控制机械臂到达捆扎机位置
                        data = [float(self.public_class.tcp_pos[0]),
                                float(self.public_class.tcp_pos[1]) - 100.00,
                                float(self.public_class.tcp_pos[2]),
                                float(self.public_class.tcp_pos[3]),
                                float(self.public_class.tcp_pos[4]),
                                float(self.public_class.tcp_pos[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.bundle_process = "1-5"
                # 判断坐标是否到达，并控制机械臂下降，再释放快换夹具
                elif self.public_class.bundle_process == "1-5":
                    error_x = (float(self.public_class.new_data[0]) - float(
                        self.public_class.tcp_pos[0]))
                    error_y = (float(self.public_class.new_data[1]) - float(
                        self.public_class.tcp_pos[1]))
                    error_z = (float(self.public_class.new_data[2]) - float(
                        self.public_class.tcp_pos[2]))
                    if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                        # 控制机械臂到达捆扎机位置
                        data = [float(self.public_class.tcp_pos[0]),
                                float(self.public_class.tcp_pos[1]),
                                float(self.public_class.tcp_pos[2]) - 10.00,
                                float(self.public_class.tcp_pos[3]),
                                float(self.public_class.tcp_pos[4]),
                                float(self.public_class.tcp_pos[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.bundle_process = "1-6"
                        time.sleep(1)
                        # 快换夹具释放
                        self.jaka.blinx_set_digital_output(6, 5, 1)
                        self.jaka.blinx_set_digital_output(6, 8, 1)
                        time.sleep(0.5)
                        self.jaka.blinx_set_digital_output(6, 5, 0)
                # 判断坐标是否到达，如果到达控制z轴上升
                elif self.public_class.bundle_process == "1-6":
                    error_x = (float(self.public_class.new_data[0]) - float(
                        self.public_class.tcp_pos[0]))
                    error_y = (float(self.public_class.new_data[1]) - float(
                        self.public_class.tcp_pos[1]))
                    error_z = (float(self.public_class.new_data[2]) - float(
                        self.public_class.tcp_pos[2]))
                    if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                        # 控制机械臂退出捆扎机器放置位置
                        data = [float(self.public_class.tcp_pos[0]),
                                float(self.public_class.tcp_pos[1]),
                                float(self.public_class.tcp_pos[2] + 30.00),
                                float(self.public_class.tcp_pos[3]),
                                float(self.public_class.tcp_pos[4]),
                                float(self.public_class.tcp_pos[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.bundle_process = "1-7"
                # 判断坐标是否到达，如果到达控制机械臂到达回收位置
                elif self.public_class.bundle_process == "1-7":
                    error_x = (float(self.public_class.new_data[0]) - float(
                        self.public_class.tcp_pos[0]))
                    error_y = (float(self.public_class.new_data[1]) - float(
                        self.public_class.tcp_pos[1]))
                    error_z = (float(self.public_class.new_data[2]) - float(
                        self.public_class.tcp_pos[2]))
                    if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                        data = [float(self.public_class.initial_angle[0]) - 180.00,
                                float(self.public_class.initial_angle[1]),
                                float(self.public_class.initial_angle[2]), float(self.public_class.initial_angle[3]),
                                float(self.public_class.initial_angle[4]), float(self.public_class.initial_angle[5])]
                        self.jaka.blinx_joint_move(0, data, 50, 50)
                        self.public_class.new_data = data
                        self.public_class.bundle_process = "1-8"
                # 角度是否到达，如果到达控制机械臂回到初始角度
                elif self.public_class.bundle_process == "1-8":
                    error_j1 = (float(self.public_class.new_data[0]) - float(
                        self.public_class.joint_pos[0]))
                    error_j2 = (float(self.public_class.new_data[1]) - float(
                        self.public_class.joint_pos[1]))
                    error_j3 = (float(self.public_class.new_data[2]) - float(
                        self.public_class.joint_pos[2]))
                    error_j4 = (float(self.public_class.new_data[3]) - float(
                        self.public_class.joint_pos[3]))
                    error_j5 = (float(self.public_class.new_data[4]) - float(
                        self.public_class.joint_pos[4]))
                    error_j6 = (float(self.public_class.new_data[5]) - float(
                        self.public_class.joint_pos[5]))
                    if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                            -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                        data = [float(self.public_class.initial_angle[0]),
                                float(self.public_class.initial_angle[1]),
                                float(self.public_class.initial_angle[2]),
                                float(self.public_class.initial_angle[3]),
                                float(self.public_class.initial_angle[4]),
                                float(self.public_class.initial_angle[5])]
                        self.jaka.blinx_joint_move(0, data, 50, 50)
                        self.public_class.new_data = data
                        self.public_class.bundle_process = "1-9"
                # 角度是否到达，如果到达结束流程
                elif self.public_class.bundle_process == "1-9":
                    error_j1 = (float(self.public_class.new_data[0]) - float(
                        self.public_class.joint_pos[0]))
                    error_j2 = (float(self.public_class.new_data[1]) - float(
                        self.public_class.joint_pos[1]))
                    error_j3 = (float(self.public_class.new_data[2]) - float(
                        self.public_class.joint_pos[2]))
                    error_j4 = (float(self.public_class.new_data[3]) - float(
                        self.public_class.joint_pos[3]))
                    error_j5 = (float(self.public_class.new_data[4]) - float(
                        self.public_class.joint_pos[4]))
                    error_j6 = (float(self.public_class.new_data[5]) - float(
                        self.public_class.joint_pos[5]))
                    if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (
                            -0.1 <= error_j3 <= 0.1) and (
                            -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (
                            -0.1 <= error_j6 <= 0.1):
                        self.public_class.new_data = None
                        self.public_class.bundle_process = "0-0"
                        self.public_class.bundle_state = False
        # 判断是否是瓷砖贴片流程
        elif self.public_class.ceramic_process_state:
            # 判断执行器放置区域
            if self.public_class.ceramic_process_node == "0-0":
                # 获取机械臂控制器的数字输入信号
                self.jaka.blinx_get_digital_input_status()
                time.sleep(0.5)
                # 将吸盘捆扎机的放置区的状态提出
                xp_state = self.robot_DI_1
                kzj_state = self.robot_DI_2
                # 判断吸盘与捆扎机是否都在
                if xp_state == 1 and kzj_state == 1:
                    self.public_class.ceramic_process_node = "1-0"  # 如果两个末端执行器都x2在
                # 判断如果吸盘在捆扎机不在
                elif xp_state == 1 and kzj_state != 1:
                    self.public_class.ceramic_process_node = "2-0"  # 如果是吸盘就直接开始下一步
                # 判断如果捆扎机在，吸盘不在
                elif xp_state != 1 and kzj_state == 1:
                    self.public_class.ceramic_process_node = "3-0"  # 需先放置捆扎机，在获取吸盘
                # 如果两个都不在
                elif xp_state != 1 and kzj_state != 1:
                    print("报警，请将末端执行器归位")
                    self.public_class.ceramic_process_node = False
            # 控制机械臂旋转180度
            elif self.public_class.ceramic_process_node == "1-0":
                data = [float(self.public_class.joint_pos[0]) - 180.00, float(self.public_class.joint_pos[1]),
                        float(self.public_class.joint_pos[2]), float(self.public_class.joint_pos[3]),
                        float(self.public_class.joint_pos[4]), float(self.public_class.joint_pos[5])]
                self.jaka.blinx_joint_move(0, data, 50, 50)
                self.public_class.new_data = data
                self.public_class.ceramic_process_node = "1-1"
            # 角度是否到达，如果到达控制J6旋转90都
            elif self.public_class.ceramic_process_node == "1-1":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    # 控制末端移动90度
                    data = [float(self.public_class.joint_pos[0]), float(self.public_class.joint_pos[1]),
                            float(self.public_class.joint_pos[2]), float(self.public_class.joint_pos[3]),
                            float(self.public_class.joint_pos[4]), float(self.public_class.joint_pos[5]) + 90.00]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "1-2"
            # 角度是否到达，如果到达控制机械臂到达吸盘上方
            elif self.public_class.ceramic_process_node == "1-2":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    # 控制机械臂到达吸盘上方
                    data = [float(self.public_class.sucker_actuator_loc[0]),
                            float(self.public_class.sucker_actuator_loc[1]),
                            float(self.public_class.sucker_actuator_loc[2]) + 30.00,
                            float(self.public_class.sucker_actuator_loc[3]),
                            float(self.public_class.sucker_actuator_loc[4]),
                            float(self.public_class.sucker_actuator_loc[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "1-3"
            # 判断坐标是否到达，如果到达先打开快换夹具，在控制机械臂到达吸盘位置
            elif self.public_class.ceramic_process_node == "1-3":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 快换夹具开
                    self.jaka.blinx_set_digital_output(6, 5, 1)
                    self.jaka.blinx_set_digital_output(6, 8, 1)
                    time.sleep(0.5)
                    self.jaka.blinx_set_digital_output(6, 5, 0)
                    # 控制机械臂到达吸盘位置
                    data = [float(self.public_class.sucker_actuator_loc[0]),
                            float(self.public_class.sucker_actuator_loc[1]),
                            float(self.public_class.sucker_actuator_loc[2]),
                            float(self.public_class.sucker_actuator_loc[3]),
                            float(self.public_class.sucker_actuator_loc[4]),
                            float(self.public_class.sucker_actuator_loc[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "1-4"
            # 判断坐标是否到达，如果到达控制快换夹具关，并控制机械臂上升
            elif self.public_class.ceramic_process_node == "1-4":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 快换夹具开
                    self.jaka.blinx_set_digital_output(6, 5, 1)
                    self.jaka.blinx_set_digital_output(6, 8, 0)
                    time.sleep(0.5)
                    self.jaka.blinx_set_digital_output(6, 5, 0)
                    # 控制机械臂到达吸盘位置
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2] + 10.00),
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "1-5"
            # 判断坐标是否到达，如果到达控制Y轴移动到
            elif self.public_class.ceramic_process_node == "1-5":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 控制机械臂退出吸盘放置位置
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1] + 100.00),
                            float(self.public_class.tcp_pos[2]),
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "1-6"
            # 判断坐标是否到达，如果到达控制机械臂到达回收位置
            elif self.public_class.ceramic_process_node == "1-6":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    data = [float(self.public_class.initial_angle[0]) - 180.00,
                            float(self.public_class.initial_angle[1]),
                            float(self.public_class.initial_angle[2]), float(self.public_class.initial_angle[3]),
                            float(self.public_class.initial_angle[4]), float(self.public_class.initial_angle[5])]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "1-7"
            # 角度是否到达，如果到达控制机械臂到达吸盘上方
            elif self.public_class.ceramic_process_node == "1-7":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    data = [float(self.public_class.initial_angle[0]),
                            float(self.public_class.initial_angle[1]),
                            float(self.public_class.initial_angle[2]),
                            float(self.public_class.initial_angle[3]),
                            float(self.public_class.initial_angle[4]),
                            float(self.public_class.initial_angle[5])]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "1-8"
            # 角度是否到达，如果到达控制机械臂到达吸盘上方
            elif self.public_class.ceramic_process_node == "1-8":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (
                        -0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (
                        -0.1 <= error_j6 <= 0.1):
                    data = [float(self.public_class.initial_angle[0]),
                            float(self.public_class.initial_angle[1]),
                            float(self.public_class.initial_angle[2]),
                            float(self.public_class.initial_angle[3]),
                            float(self.public_class.initial_angle[4]),
                            float(self.public_class.initial_angle[5])]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "3-0"
            # 控制机械臂到达过度点位
            elif self.public_class.ceramic_process_node == "2-0":
                # 控制机械臂到达吸盘位置
                data = [float(self.public_class.tcp_pos[0]),
                        float(self.public_class.tcp_pos[1]) - 50.00,
                        float(self.public_class.tcp_pos[2]),
                        float(self.public_class.tcp_pos[3]),
                        float(self.public_class.tcp_pos[4]),
                        float(self.public_class.tcp_pos[5])]
                self.jaka.blinx_moveL(data, 250, 5000, 0)
                self.public_class.new_data = data
                self.public_class.ceramic_process_node = "2-1"
            # 控制机械臂旋转180度
            elif self.public_class.ceramic_process_node == "2-1":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    data = [float(self.public_class.joint_pos[0]) - 180.00, float(self.public_class.joint_pos[1]),
                            float(self.public_class.joint_pos[2]), float(self.public_class.joint_pos[3]),
                            float(self.public_class.joint_pos[4]), float(self.public_class.joint_pos[5])]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "2-2"
            # 角度是否到达，如果到达控制机械臂待放置位置
            elif self.public_class.ceramic_process_node == "2-2":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    # 控制机械臂待放置位置
                    data = [float(self.public_class.bundle_actuator_loc[0]),
                            float(self.public_class.bundle_actuator_loc[1]) + 100.00,
                            float(self.public_class.bundle_actuator_loc[2]) + 10.00,
                            float(self.public_class.bundle_actuator_loc[3]),
                            float(self.public_class.bundle_actuator_loc[4]),
                            float(self.public_class.bundle_actuator_loc[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "2-3"
            # 判断坐标是否到达，如果到达先控制机械臂到达捆扎机放置区上方一公分位置
            elif self.public_class.ceramic_process_node == "2-3":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 控制机械臂到达捆扎机位置
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]) - 100.00,
                            float(self.public_class.tcp_pos[2]),
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "2-4"
            # 判断坐标是否到达，并控制机械臂下降，再释放快换夹具
            elif self.public_class.ceramic_process_node == "2-4":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 控制机械臂到达捆扎机位置
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2]) - 10.00,
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "2-5"
                    time.sleep(1)
                    # 快换夹具释放
                    self.jaka.blinx_set_digital_output(6, 5, 1)
                    self.jaka.blinx_set_digital_output(6, 8, 1)
                    time.sleep(0.5)
                    self.jaka.blinx_set_digital_output(6, 5, 0)
            # 判断坐标是否到达，如果到达控制z轴上升
            elif self.public_class.ceramic_process_node == "2-5":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 控制机械臂退出捆扎机器放置位置
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2] + 30.00),
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "2-6"
            # 判断坐标是否到达，如果到达控制机械臂到达回收位置
            elif self.public_class.ceramic_process_node == "2-6":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    data = [float(self.public_class.initial_angle[0]) - 180.00,
                            float(self.public_class.initial_angle[1]),
                            float(self.public_class.initial_angle[2]), float(self.public_class.initial_angle[3]),
                            float(self.public_class.initial_angle[4]), float(self.public_class.initial_angle[5])]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "1-1"

            # 控制滑轨到达指定位置
            elif self.public_class.ceramic_process_node == "3-0":
                if self.public_class.new_data == None:
                    self.jaka.blinx_set_analog_output(6, 26, 100)  # 滑轨绝对速度
                    self.jaka.blinx_set_analog_output(6, 25, 790.58)  # 滑轨绝对位置
                    self.jaka.blinx_set_digital_output(6, 5, 1)
                    self.jaka.blinx_set_digital_output(6, 4, 1)
                    self.jaka.blinx_set_digital_output(6, 4, 0)
                    self.jaka.blinx_set_digital_output(6, 5, 0)
                    self.public_class.ceramic_process_node = "3-1"
                else:
                    error_j1 = (float(self.public_class.new_data[0]) - float(
                        self.public_class.joint_pos[0]))
                    error_j2 = (float(self.public_class.new_data[1]) - float(
                        self.public_class.joint_pos[1]))
                    error_j3 = (float(self.public_class.new_data[2]) - float(
                        self.public_class.joint_pos[2]))
                    error_j4 = (float(self.public_class.new_data[3]) - float(
                        self.public_class.joint_pos[3]))
                    error_j5 = (float(self.public_class.new_data[4]) - float(
                        self.public_class.joint_pos[4]))
                    error_j6 = (float(self.public_class.new_data[5]) - float(
                        self.public_class.joint_pos[5]))
                    if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                            -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                        self.jaka.blinx_set_analog_output(6, 26, 100)  # 滑轨绝对速度
                        self.jaka.blinx_set_analog_output(6, 25, 418.63)  # 滑轨绝对位置
                        self.jaka.blinx_set_digital_output(6, 5, 1)
                        self.jaka.blinx_set_digital_output(6, 4, 1)
                        self.jaka.blinx_set_digital_output(6, 4, 0)
                        self.jaka.blinx_set_digital_output(6, 5, 0)
                        self.public_class.ceramic_process_node = "3-1"
            # 判断滑轨是否到达，如果到达控制机械臂到达拍照点位
            elif self.public_class.ceramic_process_node == "3-1":
                # 获取滑轨当前位置
                self.jaka.blinx_get_analog_input(6, 25)
                time.sleep(0.5)
                print(self.public_class.ai_value)
                error_ai = abs(float(self.public_class.ai_value) - float(790.58))
                if error_ai <= 1:
                    # 控制机械臂到达拍照位置
                    data = [float(self.public_class.identify_loc2[0]), float(self.public_class.identify_loc2[1]),
                            float(self.public_class.identify_loc2[2]), float(self.public_class.identify_loc2[3]),
                            float(self.public_class.identify_loc2[4]), float(self.public_class.identify_loc2[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "3-2"
            # 判断机械臂是否达到位置，如果到达，拍摄照片进行识别并获取数据
            elif self.public_class.ceramic_process_node == "3-2":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    self.public_class.ceramic_process_result = None
                    # 获取图像数据
                    self.public_class.mech_2d_image, self.public_class.mech_depth_map, self.public_class.mech_point_cloud = self.mechCam.GrabImages()
                    # 图像识别
                    image, depth_rgb, pick_result = self.yolo_iamge.blinx_brick_image_rec(
                        self.public_class.mech_2d_image,
                        self.public_class.mech_depth_map,
                    )

                    # 将图像显示在界面中
                    # 获取图像高度和宽度
                    heigt, width = image.shape[:2]
                    # 显示图像到界面
                    pixmap = QtGui.QImage(image, width, heigt, QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(pixmap)
                    self.Image_Show_1.setPixmap(pixmap)
                    self.Image_Show_1.setScaledContents(True)  # 图像自适应窗口大小

                    # 深度图像显示
                    # 获取图像高度和宽度
                    heigt, width = depth_rgb.shape[:2]
                    # 显示图像到界面首先
                    pixmap = QtGui.QImage(depth_rgb, width, heigt, QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(pixmap)
                    self.Image_Show_2.setPixmap(pixmap)
                    self.Image_Show_2.setScaledContents(True)  # 图像自适应窗口大小
                    self._set_display_image(self.Image_Show_1, image, QtGui.QImage.Format_RGB888)
                    self._set_display_image(self.Image_Show_2, depth_rgb, QtGui.QImage.Format_RGB888)
                    if pick_result is not None:
                        if pick_result.get("decision_status") is not None:
                            print("primary_pick decision:", pick_result.get("decision_status"),
                                  pick_result.get("decision_reason"))
                        self._set_display_image(self.Image_Show_1, image, QtGui.QImage.Format_RGB888)
                        self._set_display_image(self.Image_Show_2, depth_rgb, QtGui.QImage.Format_RGB888)
                        x, y, z = self.image_camera.blinx_image_to_camera(
                            pick_result["pixel_x"],
                            pick_result["pixel_y"],
                            pick_result["depth_mm"],
                        )
                        xy_data = self.conversion_3d.blinx_conversion([x, y, z])
                        pick_result["robot_x"] = xy_data["X"]
                        pick_result["robot_y"] = xy_data["Y"]
                        self.public_class.ceramic_process_result = pick_result
                        # 控制机械臂达到抓取位置上方
                        data = [float(xy_data["X"]),
                                float(xy_data["Y"]),
                                float(self.public_class.tcp_pos[2]),
                                float(self.public_class.tcp_pos[3]),
                                float(self.public_class.tcp_pos[4]),
                                float(self.public_class.tcp_pos[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.ceramic_process_node = "3-3-1"
                    else:
                        # 结束
                        print("结束")
                        # 回到初始位置
                        data = [float(self.public_class.identify_loc1[0]), float(self.public_class.identify_loc1[1]),
                                float(self.public_class.identify_loc1[2]), float(self.public_class.identify_loc1[3]),
                                float(self.public_class.identify_loc1[4]), float(self.public_class.identify_loc1[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = None
                        self.public_class.ceramic_process_state = False  # 瓷砖流程状态
                        self.public_class.ceramic_process_node = "0-0"  # 瓷砖流程节点
                        self.public_class.ceramic_process_num = 0  # 瓷砖流程抓取次数
                        self.public_class.ceramic_process_result = None
                        self.reset_rotation_history("ceramic")
            # 判断机械臂是否达到位置，如果到达，计算角度，并让机械臂按照角度旋转
            elif self.public_class.ceramic_process_node == "3-3-1":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    angle = self.select_shortest_rotation_delta(
                        "ceramic",
                        self.public_class.ceramic_process_result["angle_deg"],
                    )
                    print(angle)
                    time.sleep(0.5)
                    data = [float(self.public_class.joint_pos[0]), float(self.public_class.joint_pos[1]),
                           float(self.public_class.joint_pos[2]), float(self.public_class.joint_pos[3]),
                           float(self.public_class.joint_pos[4]), float(self.public_class.joint_pos[5]) + angle]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.remember_rotation_delta("ceramic", angle)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "3-3"
            # 判断机械臂是否达到位置，如果到达，根据深度进行下降抓取
            elif self.public_class.ceramic_process_node == "3-3":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    z = self.public_class.tcp_pos[2] - self.public_class.ceramic_process_result["depth_mm"] + 166.00
                    # 控制机械臂达到抓取位置上方
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(z),
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "3-4"
            # 判断机械臂是否达到位置，如果到达,打开吸盘，再控制机械臂上升
            elif self.public_class.ceramic_process_node == "3-4":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 打开吸盘
                    self.jaka.blinx_set_digital_output(6, 5, 1)
                    self.jaka.blinx_set_digital_output(6, 6, 1)
                    time.sleep(0.2)
                    self.jaka.blinx_set_digital_output(6, 5, 0)

                    # 控制机械臂上升
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2]) + 50.00,
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "3-5"
            # 判断机械臂是否达到位置，如果到达,回到初始位置
            elif self.public_class.ceramic_process_node == "3-5":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                   # 控制机械臂达到抓取位置上方
                    data = [float(self.public_class.initial_angle[0]),
                            float(self.public_class.initial_angle[1]),
                            float(self.public_class.initial_angle[2]),
                            float(self.public_class.initial_angle[3]),
                            float(self.public_class.initial_angle[4]),
                            float(self.public_class.initial_angle[5])]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "3-6"
            # 判断机械臂是否达到位置，如果到达,回到初始位置
            elif self.public_class.ceramic_process_node == "3-6":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    # 控制进行旋转
                    data = [float(self.public_class.joint_pos[0]) - 180.00, float(self.public_class.joint_pos[1]),
                            float(self.public_class.joint_pos[2]), float(self.public_class.joint_pos[3]),
                            float(self.public_class.joint_pos[4]), float(self.public_class.joint_pos[5])]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "3-7-1"
            # 角度是否到达，到待放置位置
            # elif self.public_class.ceramic_process_node == "3-7":
            #     error_j1 = (float(self.public_class.new_data[0]) - float(
            #         self.public_class.joint_pos[0]))
            #     error_j2 = (float(self.public_class.new_data[1]) - float(
            #         self.public_class.joint_pos[1]))
            #     error_j3 = (float(self.public_class.new_data[2]) - float(
            #         self.public_class.joint_pos[2]))
            #     error_j4 = (float(self.public_class.new_data[3]) - float(
            #         self.public_class.joint_pos[3]))
            #     error_j5 = (float(self.public_class.new_data[4]) - float(
            #         self.public_class.joint_pos[4]))
            #     error_j6 = (float(self.public_class.new_data[5]) - float(
            #         self.public_class.joint_pos[5]))
            #     if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
            #             -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
            #         # 控制机械臂到达放置过度点
            #         data = [float(self.public_class.ceramic_excessive_loc[0]),
            #                 float(self.public_class.ceramic_excessive_loc[1]),
            #                 float(self.public_class.ceramic_excessive_loc[2]),
            #                 float(self.public_class.ceramic_excessive_loc[3]),
            #                 float(self.public_class.ceramic_excessive_loc[4]),
            #                 float(self.public_class.ceramic_excessive_loc[5])]
            #         self.jaka.blinx_moveL(data, 250, 5000, 0)
            #         self.public_class.new_data = data
            #         self.public_class.ceramic_process_node = "3-7-1"

            # 将物料放置二次定位上方
            elif self.public_class.ceramic_process_node == "3-7-1":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    data = [float(self.public_class.secondary_positioning_loc[0]),
                            float(self.public_class.secondary_positioning_loc[1]),
                            float(self.public_class.secondary_positioning_loc[2]) + 20.00,
                            float(self.public_class.secondary_positioning_loc[3]),
                            float(self.public_class.secondary_positioning_loc[4]),
                            float(self.public_class.secondary_positioning_loc[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "3-7-2"
            # 将物料放置二次定位
            elif self.public_class.ceramic_process_node == "3-7-2":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2]) - 20.00,
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "3-7-3"
            # 关闭吸盘，然后在控制机械臂上升
            elif self.public_class.ceramic_process_node == "3-7-3":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 关闭吸盘
                    self.btn_xipan_close_click()
                    time.sleep(1)
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2]) + 20.00,
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "3-7-4"
            # 控制机械臂到达二次拍照点位
            elif self.public_class.ceramic_process_node == "3-7-4":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    data = [float(self.public_class.secondary_photography_loc[0]),
                            float(self.public_class.secondary_photography_loc[1]),
                            float(self.public_class.secondary_photography_loc[2]),
                            float(self.public_class.secondary_photography_loc[3]),
                            float(self.public_class.secondary_photography_loc[4]),
                            float(self.public_class.secondary_photography_loc[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "3-7-5"
            # 进行拍照，对物品进行二次识别
            elif self.public_class.ceramic_process_node == "3-7-5":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    try:
                        # 获取图像数据
                        self.public_class.mech_2d_image, self.public_class.mech_depth_map, self.public_class.mech_point_cloud = self.mechCam.GrabImages()
                        # 图像识别
                        image, data = self.yolo_iamge.blinx_brickandporcelain_image_rec(self.public_class.mech_2d_image)

                        heigt, width = image.shape[:2]
                        # 显示图像到界面
                        pixmap = QtGui.QImage(image, width, heigt, QtGui.QImage.Format_RGB888)
                        pixmap = QtGui.QPixmap.fromImage(pixmap)
                        self.Image_Show_1.setPixmap(pixmap)
                        self.Image_Show_1.setScaledContents(True)  # 图像自适应窗口大小

                        # 深度图像显示
                        # 获取图像高度和宽度
                        heigt, width = depth_rgb.shape[:2]
                        # 显示图像到界面
                        pixmap = QtGui.QImage(depth_rgb, width, heigt, QtGui.QImage.Format_RGB888)
                        pixmap = QtGui.QPixmap.fromImage(pixmap)
                        self.Image_Show_2.setPixmap(pixmap)
                        self.Image_Show_2.setScaledContents(True)  # 图像自适应窗口大小

                        time.sleep(1)
                        # 进行标定转换
                        self._set_display_image(self.Image_Show_1, image, QtGui.QImage.Format_RGB888)
                        self._set_display_image(self.Image_Show_2, depth_rgb, QtGui.QImage.Format_RGB888)
                        test_robot_xy = np.dot(self.m_ini, [data[0], data[1], 1])  # 仿射逆变换，得到坐标（x,y)
                        data = [float(test_robot_xy[0]),
                                float(test_robot_xy[1]),
                                float(self.public_class.tcp_pos[2]),
                                float(self.public_class.tcp_pos[3]),
                                float(self.public_class.tcp_pos[4]),
                                float(self.public_class.tcp_pos[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.ceramic_process_node = "3-7-6"
                    except Exception as e:
                        print(e)
            # 控制Z轴下降
            elif self.public_class.ceramic_process_node == "3-7-6":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2]) - 91.00,
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "3-7-7"
            # 打开吸盘，控制Z轴上升
            elif self.public_class.ceramic_process_node == "3-7-7":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 打开吸盘
                    self.jaka.blinx_set_digital_output(6, 5, 1)
                    self.jaka.blinx_set_digital_output(6, 6, 1)
                    time.sleep(0.2)
                    self.jaka.blinx_set_digital_output(6, 5, 0)

                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2]) + 91.00,
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "3-8"

            # # 回到待放点位
            # elif self.public_class.ceramic_process_node == "3-7-8":
            #     error_x = (float(self.public_class.new_data[0]) - float(
            #         self.public_class.tcp_pos[0]))
            #     error_y = (float(self.public_class.new_data[1]) - float(
            #         self.public_class.tcp_pos[1]))
            #     error_z = (float(self.public_class.new_data[2]) - float(
            #         self.public_class.tcp_pos[2]))
            #     if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
            #         # 控制机械臂到达放置过度点
            #         data = [float(self.public_class.ceramic_excessive_loc[0]),
            #                 float(self.public_class.ceramic_excessive_loc[1]),
            #                 float(self.public_class.ceramic_excessive_loc[2]),
            #                 float(self.public_class.ceramic_excessive_loc[3]),
            #                 float(self.public_class.ceramic_excessive_loc[4]),
            #                 float(self.public_class.ceramic_excessive_loc[5])]
            #         self.jaka.blinx_moveL(data, 250, 5000, 0)
            #         self.public_class.new_data = data
            #         self.public_class.ceramic_process_node = "3-8"

            # 判断机械臂是否达到位置，判断这是第几次放置，根据放置位置，进行放置
            elif self.public_class.ceramic_process_node == "3-8":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 根据抓取次数进行放置
                    group_num = int(self.public_class.ceramic_process_num / 3)
                    c_num = int(self.public_class.ceramic_process_num % 3)
                    if c_num == 0:
                        c_num = 0
                    # 根据抓取数进行xy移动
                    x = self.public_class.ceramic_place_loc[0] - (group_num * 102.00)
                    y = self.public_class.ceramic_place_loc[1] + (c_num * 102.00)
                    # 控制机械臂到达放置过度点
                    data = [float(x) - 20.00,
                            float(y) + 20.00,
                            float(self.public_class.ceramic_place_loc[2]) + 20.00,
                            float(self.public_class.ceramic_place_loc[3]),
                            float(self.public_class.ceramic_place_loc[4]),
                            float(self.public_class.ceramic_place_loc[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "3-9"
            # 判断机械臂是否达到位置，如果到达，下降Z轴
            elif self.public_class.ceramic_process_node == "3-9":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 控制机械臂到达放置过度点
                    data = [float(self.public_class.tcp_pos[0]) + 20.00,
                            float(self.public_class.tcp_pos[1]) - 20.00,
                            float(self.public_class.tcp_pos[2]) - 20.00,
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "3-10"
            # 判断机械臂是否达到位置，如果到达，关闭吸盘，Z轴上升
            elif self.public_class.ceramic_process_node == "3-10":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 关闭吸盘
                    self.btn_xipan_close_click()
                    # 控制机械臂Z轴上升
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2]) + 50.00,
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "3-11"
            # 判断机械臂是否达到位置，如果到达，回到转180位置
            elif self.public_class.ceramic_process_node == "3-11":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 控制末端进行旋转
                    data = [float(self.public_class.initial_angle[0]) - 180.00, float(self.public_class.initial_angle[1]),
                            float(self.public_class.initial_angle[2]), float(self.public_class.initial_angle[3]),
                            float(self.public_class.initial_angle[4]), float(self.public_class.initial_angle[5])]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "3-12"
            # 角度是否到达，如果到达再回到初始姿态
            elif self.public_class.ceramic_process_node == "3-12":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    # 控制末端移动90度
                    data = [float(self.public_class.initial_angle[0]), float(self.public_class.initial_angle[1]),
                            float(self.public_class.initial_angle[2]), float(self.public_class.initial_angle[3]),
                            float(self.public_class.initial_angle[4]), float(self.public_class.initial_angle[5])]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.ceramic_process_node = "3-13"
            # 角度是否到达，如果到达回到拍照姿态
            elif self.public_class.ceramic_process_node == "3-13":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    self.public_class.ceramic_process_num = self.public_class.ceramic_process_num + 1
                    self.public_class.ceramic_process_node = "3-1"


        # 判断是否是砌砖流程
        elif self.public_class.brick_process_state:
            # 判断执行器放置区域
            if self.public_class.brick_process_node == "0-0":
                # 获取机械臂控制器的数字输入信号
                self.jaka.blinx_get_digital_input_status()
                time.sleep(0.5)
                # 将吸盘捆扎机的放置区的状态提出
                xp_state = self.robot_DI_1
                kzj_state = self.robot_DI_2
                # 判断吸盘与捆扎机是否都在
                if xp_state == 1 and kzj_state == 1:
                    self.public_class.brick_process_node = "1-0"  # 如果两个末端执行器都在
                # 判断如果吸盘在捆扎机不在
                elif xp_state == 1 and kzj_state != 1:
                    self.public_class.brick_process_node = "2-0"  # 如果是吸盘就直接开始下一步
                # 判断如果捆扎机在，吸盘不在
                elif xp_state != 1 and kzj_state == 1:
                    self.public_class.brick_process_node = "3-0"  # 需先放置捆扎机，在获取吸盘
                # 如果两个都不在
                elif xp_state != 1 and kzj_state != 1:
                    print("报警，请将末端执行器归位")
                    self.public_class.brick_process_node = False
            # 控制机械臂旋转180度
            elif self.public_class.brick_process_node == "1-0":
                data = [float(self.public_class.joint_pos[0]) - 180.00, float(self.public_class.joint_pos[1]),
                        float(self.public_class.joint_pos[2]), float(self.public_class.joint_pos[3]),
                        float(self.public_class.joint_pos[4]), float(self.public_class.joint_pos[5])]
                self.jaka.blinx_joint_move(0, data, 50, 50)
                self.public_class.new_data = data
                self.public_class.brick_process_node = "1-1"
            # 角度是否到达，如果到达控制J6旋转90都
            elif self.public_class.brick_process_node == "1-1":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    # 控制末端移动90度
                    data = [float(self.public_class.joint_pos[0]), float(self.public_class.joint_pos[1]),
                            float(self.public_class.joint_pos[2]), float(self.public_class.joint_pos[3]),
                            float(self.public_class.joint_pos[4]), float(self.public_class.joint_pos[5]) + 90.00]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "1-2"
            # 角度是否到达，如果到达控制机械臂到达吸盘上方
            elif self.public_class.brick_process_node == "1-2":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    # 控制机械臂到达吸盘上方
                    data = [float(self.public_class.sucker_actuator_loc[0]),
                            float(self.public_class.sucker_actuator_loc[1]),
                            float(self.public_class.sucker_actuator_loc[2]) + 30.00,
                            float(self.public_class.sucker_actuator_loc[3]),
                            float(self.public_class.sucker_actuator_loc[4]),
                            float(self.public_class.sucker_actuator_loc[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "1-3"
            # 判断坐标是否到达，如果到达先打开快换夹具，在控制机械臂到达吸盘位置
            elif self.public_class.brick_process_node == "1-3":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 快换夹具开
                    self.jaka.blinx_set_digital_output(6, 5, 1)
                    self.jaka.blinx_set_digital_output(6, 8, 1)
                    time.sleep(0.5)
                    self.jaka.blinx_set_digital_output(6, 5, 0)
                    # 控制机械臂到达吸盘位置
                    data = [float(self.public_class.sucker_actuator_loc[0]),
                            float(self.public_class.sucker_actuator_loc[1]),
                            float(self.public_class.sucker_actuator_loc[2]),
                            float(self.public_class.sucker_actuator_loc[3]),
                            float(self.public_class.sucker_actuator_loc[4]),
                            float(self.public_class.sucker_actuator_loc[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "1-4"
            # 判断坐标是否到达，如果到达控制快换夹具关，并控制机械臂上升
            elif self.public_class.brick_process_node == "1-4":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 快换夹具开
                    self.jaka.blinx_set_digital_output(6, 5, 1)
                    self.jaka.blinx_set_digital_output(6, 8, 0)
                    time.sleep(0.5)
                    self.jaka.blinx_set_digital_output(6, 5, 0)
                    # 控制机械臂到达吸盘位置
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2] + 10.00),
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "1-5"
            # 判断坐标是否到达，如果到达控制Y轴移动到
            elif self.public_class.brick_process_node == "1-5":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 控制机械臂退出吸盘放置位置
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1] + 100.00),
                            float(self.public_class.tcp_pos[2]),
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "1-6"
            # 判断坐标是否到达，如果到达控制机械臂到达回收位置
            elif self.public_class.brick_process_node == "1-6":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    data = [float(self.public_class.initial_angle[0]) - 180.00,
                            float(self.public_class.initial_angle[1]),
                            float(self.public_class.initial_angle[2]), float(self.public_class.initial_angle[3]),
                            float(self.public_class.initial_angle[4]), float(self.public_class.initial_angle[5])]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "1-7"
            # 角度是否到达，如果到达控制机械臂到达吸盘上方
            elif self.public_class.brick_process_node == "1-7":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    data = [float(self.public_class.initial_angle[0]),
                            float(self.public_class.initial_angle[1]),
                            float(self.public_class.initial_angle[2]),
                            float(self.public_class.initial_angle[3]),
                            float(self.public_class.initial_angle[4]),
                            float(self.public_class.initial_angle[5])]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "1-8"
            # 角度是否到达，如果到达控制机械臂到达吸盘上方
            elif self.public_class.brick_process_node == "1-8":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (
                        -0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (
                        -0.1 <= error_j6 <= 0.1):
                    data = [float(self.public_class.initial_angle[0]),
                            float(self.public_class.initial_angle[1]),
                            float(self.public_class.initial_angle[2]),
                            float(self.public_class.initial_angle[3]),
                            float(self.public_class.initial_angle[4]),
                            float(self.public_class.initial_angle[5])]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "3-0"
            # 控制机械臂到达过度点位
            elif self.public_class.brick_process_node == "2-0":
                # 控制机械臂到达吸盘位置
                data = [float(self.public_class.tcp_pos[0]),
                        float(self.public_class.tcp_pos[1]) - 50.00,
                        float(self.public_class.tcp_pos[2]),
                        float(self.public_class.tcp_pos[3]),
                        float(self.public_class.tcp_pos[4]),
                        float(self.public_class.tcp_pos[5])]
                self.jaka.blinx_moveL(data, 250, 5000, 0)
                self.public_class.new_data = data
                self.public_class.brick_process_node = "2-1"
            # 控制机械臂旋转180度
            elif self.public_class.brick_process_node == "2-1":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    data = [float(self.public_class.joint_pos[0]) - 180.00, float(self.public_class.joint_pos[1]),
                            float(self.public_class.joint_pos[2]), float(self.public_class.joint_pos[3]),
                            float(self.public_class.joint_pos[4]), float(self.public_class.joint_pos[5])]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "2-2"
            # 角度是否到达，如果到达控制机械臂待放置位置
            elif self.public_class.brick_process_node == "2-2":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    # 控制机械臂待放置位置
                    data = [float(self.public_class.bundle_actuator_loc[0]),
                            float(self.public_class.bundle_actuator_loc[1]) + 100.00,
                            float(self.public_class.bundle_actuator_loc[2]) + 10.00,
                            float(self.public_class.bundle_actuator_loc[3]),
                            float(self.public_class.bundle_actuator_loc[4]),
                            float(self.public_class.bundle_actuator_loc[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "2-3"
            # 判断坐标是否到达，如果到达先控制机械臂到达捆扎机放置区上方一公分位置
            elif self.public_class.brick_process_node == "2-3":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 控制机械臂到达捆扎机位置
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]) - 100.00,
                            float(self.public_class.tcp_pos[2]),
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "2-4"
            # 判断坐标是否到达，并控制机械臂下降，再释放快换夹具
            elif self.public_class.brick_process_node == "2-4":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 控制机械臂到达捆扎机位置
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2]) - 10.00,
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "2-5"
                    time.sleep(1)
                    # 快换夹具释放
                    self.jaka.blinx_set_digital_output(6, 5, 1)
                    self.jaka.blinx_set_digital_output(6, 8, 1)
                    time.sleep(0.5)
                    self.jaka.blinx_set_digital_output(6, 5, 0)
            # 判断坐标是否到达，如果到达控制z轴上升
            elif self.public_class.brick_process_node == "2-5":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 控制机械臂退出捆扎机器放置位置
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2] + 30.00),
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "2-6"
            # 判断坐标是否到达，如果到达控制机械臂到达回收位置
            elif self.public_class.brick_process_node == "2-6":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    data = [float(self.public_class.initial_angle[0]) - 180.00,
                            float(self.public_class.initial_angle[1]),
                            float(self.public_class.initial_angle[2]), float(self.public_class.initial_angle[3]),
                            float(self.public_class.initial_angle[4]), float(self.public_class.initial_angle[5])]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "1-1"

            # 控制滑轨到达指定位置
            elif self.public_class.brick_process_node == "3-0":
                if self.public_class.new_data == None:
                    self._brick_slider_move_to(790.58)
                    self.public_class.brick_process_node = "3-1"
                else:
                    error_j1 = (float(self.public_class.new_data[0]) - float(
                        self.public_class.joint_pos[0]))
                    error_j2 = (float(self.public_class.new_data[1]) - float(
                        self.public_class.joint_pos[1]))
                    error_j3 = (float(self.public_class.new_data[2]) - float(
                        self.public_class.joint_pos[2]))
                    error_j4 = (float(self.public_class.new_data[3]) - float(
                        self.public_class.joint_pos[3]))
                    error_j5 = (float(self.public_class.new_data[4]) - float(
                        self.public_class.joint_pos[4]))
                    error_j6 = (float(self.public_class.new_data[5]) - float(
                        self.public_class.joint_pos[5]))
                    if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                            -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                        self._brick_slider_move_to(418.63)
                        self.public_class.brick_process_node = "3-1"
            # 判断滑轨是否到达，如果到达控制机械臂到达拍照点位
            elif self.public_class.brick_process_node == "3-1":
                # 获取滑轨当前位置
                self.jaka.blinx_get_analog_input(6, 25)
                time.sleep(0.5)
                print(self.public_class.ai_value)
                error_ai = abs(float(self.public_class.ai_value) - float(790.58))
                if error_ai <= 1:
                    # 控制机械臂到达拍照位置
                    data = [float(self.public_class.identify_loc2[0]), float(self.public_class.identify_loc2[1]),
                            float(self.public_class.identify_loc2[2]), float(self.public_class.identify_loc2[3]),
                            float(self.public_class.identify_loc2[4]), float(self.public_class.identify_loc2[5])]
                    self._brick_move_linear_fast(data)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "3-2"
            # 判断机械臂是否达到位置，如果到达，拍摄照片进行识别并获取数据
            elif self.public_class.brick_process_node == "3-2":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    self.public_class.brick_process_result = None
                    # 获取图像数据
                    self.public_class.mech_2d_image, self.public_class.mech_depth_map, self.public_class.mech_point_cloud = self.mechCam.GrabImages()

                    # 图像识别
                    image, depth_rgb, pick_result, primary_pick_report = self.yolo_iamge.blinx_brick_primary_pick_fusion(
                        self.public_class.mech_2d_image,
                        self.public_class.mech_depth_map,
                    )
                    fusion_result = primary_pick_report["fusion_report"]

                    # 将图像显示在界面中
                    # 获取图像高度和宽度
                    heigt, width = image.shape[:2]
                    # 显示图像到界面
                    pixmap = QtGui.QImage(image, width, heigt, QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(pixmap)
                    self.Image_Show_1.setPixmap(pixmap)
                    self.Image_Show_1.setScaledContents(True)  # 图像自适应窗口大小

                    # 深度图像显示
                    # 获取图像高度和宽度
                    heigt, width = depth_rgb.shape[:2]
                    # 显示图像到界面
                    pixmap = QtGui.QImage(depth_rgb, width, heigt, QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(pixmap)
                    self.Image_Show_2.setPixmap(pixmap)
                    self.Image_Show_2.setScaledContents(True)  # 图像自适应窗口大小

                    if pick_result is not None:
                        # 进行标定转换
                        x, y, z = self.image_camera.blinx_image_to_camera(
                            pick_result["pixel_x"],
                            pick_result["pixel_y"],
                            pick_result["depth_mm"],
                        )
                        xy_data = self.conversion_3d.blinx_conversion([x, y, z])
                        pick_result["robot_x"] = xy_data["X"]
                        pick_result["robot_y"] = xy_data["Y"]
                        self.public_class.brick_process_result = pick_result
                        # 控制机械臂达到抓取位置上方
                        data = [float(xy_data["X"]),
                                float(xy_data["Y"]),
                                float(self.public_class.tcp_pos[2]),
                                float(self.public_class.tcp_pos[3]),
                                float(self.public_class.tcp_pos[4]),
                                float(self.public_class.tcp_pos[5])]
                        self._set_display_image(self.Image_Show_1, image, QtGui.QImage.Format_RGB888)
                        self._set_display_image(self.Image_Show_2, depth_rgb, QtGui.QImage.Format_RGB888)
                        self._brick_record_start_case(
                            self.public_class.mech_2d_image,
                            self.public_class.mech_depth_map,
                            image,
                            extra={
                                "capture_stage": "primary_pick",
                                "pick_result": self._summarize_pick_result(pick_result),
                                "vision_decision": {
                                    "decision_status": fusion_result.get("decision_status"),
                                    "decision_reason": fusion_result.get("decision_reason"),
                                    "decision_warning": fusion_result.get("decision_warning"),
                                    "rgb_quality": fusion_result.get("rgb_quality"),
                                    "depth_fallback": fusion_result.get("depth_fallback"),
                                    "rgb_selected_candidate": self._summarize_pick_result(
                                        fusion_result.get("rgb_selected_candidate")
                                    ),
                                    "depth_selected_candidate": self._summarize_pick_result(
                                        fusion_result.get("depth_selected_candidate")
                                    ),
                                    "matched_depth_candidate": self._summarize_pick_result(
                                        fusion_result.get("matched_depth_candidate")
                                    ),
                                    "match_metrics": fusion_result.get("match_metrics"),
                                },
                                "camera_xyz": [float(x), float(y), float(z)],
                                "robot_xy": {"X": float(xy_data["X"]), "Y": float(xy_data["Y"])},
                                "command_target": data,
                            },
                        )
                        self._brick_move_linear_fast(data)
                        self.public_class.new_data = data
                        self.public_class.brick_process_node = "3-3-1"
                    else:
                        print("primary_pick failed:", fusion_result.get("decision_reason"))
                        # 结束
                        print("结束")
            # 判断机械臂是否达到位置，如果到达，计算角度，并让机械臂按照角度旋转
            elif self.public_class.brick_process_node == "3-3-1":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    angle = self.select_shortest_rotation_delta(
                        "brick",
                        self.public_class.brick_process_result["angle_deg"],
                    )
                    print(angle)
                    time.sleep(0.5)
                    data = [float(self.public_class.joint_pos[0]), float(self.public_class.joint_pos[1]),
                            float(self.public_class.joint_pos[2]), float(self.public_class.joint_pos[3]),
                            float(self.public_class.joint_pos[4]), float(self.public_class.joint_pos[5]) + angle]
                    self._brick_record_event(
                        "primary_rotation_command",
                        command_target=data,
                        extra={"rotation_angle_deg": float(angle)},
                    )
                    self._brick_joint_move_rotate(data)
                    self.remember_rotation_delta("brick", angle)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "3-3"
            # 判断机械臂是否达到位置，如果到达，根据深度进行下降抓取
            elif self.public_class.brick_process_node == "3-3":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    z = self.public_class.tcp_pos[2] - self.public_class.brick_process_result["depth_mm"] + 166.00
                    # 控制机械臂达到抓取位置上方
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(z),
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self._brick_record_event(
                        "primary_descend_command",
                        command_target=data,
                        extra={
                            "target_z": float(z),
                            "depth_mm": float(self.public_class.brick_process_result["depth_mm"]),
                        },
                    )
                    self._brick_move_linear_pick(data)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "3-4"
            # 判断机械臂是否达到位置，如果到达,打开吸盘，再控制机械臂上升
            elif self.public_class.brick_process_node == "3-4":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 打开吸盘
                    self.jaka.blinx_set_digital_output(6, 5, 1)
                    self.jaka.blinx_set_digital_output(6, 6, 1)
                    time.sleep(0.2)
                    self.jaka.blinx_set_digital_output(6, 5, 0)

                    # 控制机械臂上升
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2]) + 100.00,
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self._brick_record_event(
                        "primary_suction_and_lift_command",
                        command_target=data,
                        extra={
                            "suction_state": "open",
                            "executed_pick_tcp_pose": self._normalize_pose_values(self.public_class.tcp_pos),
                        },
                    )
                    self._brick_move_linear_pick(data)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "3-5"
            # 判断机械臂是否达到位置，如果到达,回到初始位置
            elif self.public_class.brick_process_node == "3-5":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 控制机械臂达到抓取位置上方
                    data = [float(self.public_class.initial_angle[0]),
                            float(self.public_class.initial_angle[1]),
                            float(self.public_class.initial_angle[2]),
                            float(self.public_class.initial_angle[3]),
                            float(self.public_class.initial_angle[4]),
                            float(self.public_class.initial_angle[5])]
                    self._brick_joint_move_rotate(data)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "3-6"
            # 判断机械臂是否达到位置，如果到达,回到初始位置
            elif self.public_class.brick_process_node == "3-6":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    # 控制进行旋转
                    data = [float(self.public_class.joint_pos[0]) - 180.00, float(self.public_class.joint_pos[1]),
                            float(self.public_class.joint_pos[2]), float(self.public_class.joint_pos[3]),
                            float(self.public_class.joint_pos[4]), float(self.public_class.joint_pos[5])]
                    self._brick_joint_move_rotate(data)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "3-7-1"
            # # 角度是否到达，到待放置位置
            # elif self.public_class.brick_process_node == "3-7":
            #     error_j1 = (float(self.public_class.new_data[0]) - float(
            #         self.public_class.joint_pos[0]))
            #     error_j2 = (float(self.public_class.new_data[1]) - float(
            #         self.public_class.joint_pos[1]))
            #     error_j3 = (float(self.public_class.new_data[2]) - float(
            #         self.public_class.joint_pos[2]))
            #     error_j4 = (float(self.public_class.new_data[3]) - float(
            #         self.public_class.joint_pos[3]))
            #     error_j5 = (float(self.public_class.new_data[4]) - float(
            #         self.public_class.joint_pos[4]))
            #     error_j6 = (float(self.public_class.new_data[5]) - float(
            #         self.public_class.joint_pos[5]))
            #     if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
            #             -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
            #         # 控制机械臂到达放置过度点
            #         data = [float(self.public_class.brick_excessive_loc[0]),
            #                 float(self.public_class.brick_excessive_loc[1]),
            #                 float(self.public_class.brick_excessive_loc[2]),
            #                 float(self.public_class.brick_excessive_loc[3]),
            #                 float(self.public_class.brick_excessive_loc[4]),
            #                 float(self.public_class.brick_excessive_loc[5])]
            #         self.jaka.blinx_moveL(data, 250, 5000, 0)
            #         self.public_class.new_data = data
            #         self.public_class.brick_process_node = "3-7-1"

            # 将物料放置二次定位上方
            elif self.public_class.brick_process_node == "3-7-1":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    data = [float(self.public_class.secondary_positioning_loc[0]),
                            float(self.public_class.secondary_positioning_loc[1]),
                            float(self.public_class.secondary_positioning_loc[2]) + 45.00,
                            float(self.public_class.secondary_positioning_loc[3]),
                            float(self.public_class.secondary_positioning_loc[4]),
                            float(self.public_class.secondary_positioning_loc[5])]
                    self._brick_record_event(
                        "secondary_positioning_approach_command",
                        command_target=data,
                    )
                    self._brick_move_linear_fast(data)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "3-7-2"
            # 将物料放置二次定位
            elif self.public_class.brick_process_node == "3-7-2":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2]) - 25.00,
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self._brick_record_event(
                        "secondary_positioning_descend_command",
                        command_target=data,
                    )
                    self._brick_move_linear_pick(data)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "3-7-3"
            # 关闭吸盘，然后在控制机械臂上升
            elif self.public_class.brick_process_node == "3-7-3":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 关闭吸盘
                    self.btn_xipan_close_click()
                    time.sleep(1)
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2]) + 20.00,
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self._brick_record_event(
                        "secondary_release_and_lift_command",
                        command_target=data,
                        extra={"suction_state": "closed"},
                    )
                    self._brick_move_linear_pick(data)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "3-7-4"
            # 控制机械臂到达二次拍照点位
            elif self.public_class.brick_process_node == "3-7-4":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    data = [float(self.public_class.secondary_photography_loc[0]),
                            float(self.public_class.secondary_photography_loc[1]),
                            float(self.public_class.secondary_photography_loc[2]),
                            float(self.public_class.secondary_photography_loc[3]),
                            float(self.public_class.secondary_photography_loc[4]),
                            float(self.public_class.secondary_photography_loc[5])]
                    self._brick_record_event(
                        "secondary_photo_point_command",
                        command_target=data,
                    )
                    self._brick_move_linear_fast(data)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "3-7-5"
            # 进行拍照，对物品进行二次识别
            elif self.public_class.brick_process_node == "3-7-5":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    try:
                        # 获取图像数据
                        self.public_class.mech_2d_image, self.public_class.mech_depth_map, self.public_class.mech_point_cloud = self.mechCam.GrabImages()

                        # 图像识别
                        image, depth_rgb, alignment_result, secondary_report = self.yolo_iamge.blinx_brick_secondary_alignment_depth_first(
                            self.public_class.mech_2d_image,
                            self.public_class.mech_depth_map,
                        )
                        secondary_decision = secondary_report["decision_report"]

                        # 将图像显示在界面中
                        # 获取图像高度和宽度
                        heigt, width = image.shape[:2]
                        # 显示图像到界面
                        pixmap = QtGui.QImage(image, width, heigt, QtGui.QImage.Format_RGB888)
                        pixmap = QtGui.QPixmap.fromImage(pixmap)
                        self.Image_Show_1.setPixmap(pixmap)
                        self.Image_Show_1.setScaledContents(True)  # 图像自适应窗口大小
                        # 深度图像显示
                        # 获取图像高度和宽度
                        heigt, width = depth_rgb.shape[:2]
                        # 显示图像到界面
                        pixmap = QtGui.QImage(depth_rgb, width, heigt, QtGui.QImage.Format_RGB888)
                        pixmap = QtGui.QPixmap.fromImage(pixmap)
                        self.Image_Show_2.setPixmap(pixmap)
                        self.Image_Show_2.setScaledContents(True)  # 图像自适应窗口大小

                        self._set_display_image(self.Image_Show_1, image, QtGui.QImage.Format_RGB888)
                        self._set_display_image(self.Image_Show_2, depth_rgb, QtGui.QImage.Format_RGB888)
                        time.sleep(1)
                        if alignment_result is None:
                            self.public_class.brick_secondary_alignment_result = None
                            failure_extra = {
                                "vision_decision": {
                                    "decision_status": secondary_decision.get("decision_status"),
                                    "source": None,
                                    "depth_candidate": self._summarize_pick_result(
                                        secondary_decision.get("depth_candidate")
                                    ),
                                    "rgb_fallback_used": bool(secondary_decision.get("rgb_fallback_used")),
                                },
                            }
                            self._brick_record_capture(
                                "secondary_alignment",
                                self.public_class.mech_2d_image,
                                self.public_class.mech_depth_map,
                                image,
                                extra={
                                    "alignment_result": None,
                                    "alignment_result_detail": None,
                                    **failure_extra,
                                },
                            )
                            self._brick_record_event(
                                "secondary_alignment_missing",
                                extra={
                                    "reason": "二次识别未返回有效坐标",
                                    **failure_extra,
                                },
                            )
                            self._brick_restart_from_primary_pick(
                                "secondary_alignment_missing",
                                failure_extra,
                            )
                            print("secondary_alignment missing, restart from primary pick")
                            return
                        print(
                            "secondary_alignment decision:",
                            secondary_decision.get("decision_status"),
                            "source:",
                            alignment_result.get("source"),
                        )
                        self.public_class.brick_secondary_alignment_result = alignment_result
                        # 进行标定转换
                        test_robot_xy = np.dot(
                            self.m_ini,
                            [alignment_result["pixel_x"], alignment_result["pixel_y"], 1],
                        )  # 仿射逆变换，得到坐标（x,y)
                        data = [float(test_robot_xy[0]),
                                float(test_robot_xy[1]),
                                float(self.public_class.tcp_pos[2]),
                                float(self.public_class.tcp_pos[3]),
                                float(self.public_class.tcp_pos[4]),
                                float(self.public_class.tcp_pos[5])]
                        self._brick_record_capture(
                            "secondary_alignment",
                            self.public_class.mech_2d_image,
                            self.public_class.mech_depth_map,
                            image,
                            extra={
                                "alignment_result": [
                                    float(alignment_result["pixel_x"]),
                                    float(alignment_result["pixel_y"]),
                                ],
                                "alignment_result_detail": self._summarize_pick_result(alignment_result),
                                "vision_decision": {
                                    "decision_status": secondary_decision.get("decision_status"),
                                    "source": alignment_result.get("source"),
                                    "depth_candidate": self._summarize_pick_result(
                                        secondary_decision.get("depth_candidate")
                                    ),
                                    "rgb_fallback_used": bool(secondary_decision.get("rgb_fallback_used")),
                                },
                                "command_target": data,
                            },
                        )
                        self._brick_record_event(
                            "secondary_alignment_command",
                            command_target=data,
                            extra={
                                "alignment_result": [
                                    float(alignment_result["pixel_x"]),
                                    float(alignment_result["pixel_y"]),
                                ],
                                "alignment_result_detail": self._summarize_pick_result(alignment_result),
                                "vision_decision": {
                                    "decision_status": secondary_decision.get("decision_status"),
                                    "source": alignment_result.get("source"),
                                    "depth_candidate": self._summarize_pick_result(
                                        secondary_decision.get("depth_candidate")
                                    ),
                                    "rgb_fallback_used": bool(secondary_decision.get("rgb_fallback_used")),
                                },
                            },
                        )
                        self._brick_move_linear_fast(data)
                        self.public_class.new_data = data
                        self.public_class.brick_process_node = "3-7-5-1"
                    except Exception as e:
                        print(e)
                        self._brick_restart_from_primary_pick(
                            "secondary_alignment_exception",
                            {"exception": str(e)},
                        )
                        return
            # 二次识别到目标上方后，按角度补一次旋转
            elif self.public_class.brick_process_node == "3-7-5-1":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    alignment_result = getattr(self.public_class, "brick_secondary_alignment_result", None)
                    if not isinstance(alignment_result, dict) or alignment_result.get("angle_deg") is None:
                        self._brick_record_event(
                            "secondary_rotation_skipped",
                            extra={
                                "reason": "secondary_alignment_angle_missing",
                                "alignment_result_detail": self._summarize_pick_result(alignment_result),
                            },
                        )
                        self.public_class.brick_process_node = "3-7-6"
                    else:
                        angle = self.select_shortest_rotation_delta(
                            "brick",
                            alignment_result["angle_deg"],
                        )
                        data = [float(self.public_class.joint_pos[0]), float(self.public_class.joint_pos[1]),
                                float(self.public_class.joint_pos[2]), float(self.public_class.joint_pos[3]),
                                float(self.public_class.joint_pos[4]), float(self.public_class.joint_pos[5]) + angle]
                        self._brick_record_event(
                            "secondary_rotation_command",
                            command_target=data,
                            extra={
                                "rotation_angle_deg": float(angle),
                                "alignment_angle_deg": float(alignment_result["angle_deg"]),
                                "alignment_result_detail": self._summarize_pick_result(alignment_result),
                            },
                        )
                        self._brick_joint_move_rotate(data)
                        self.remember_rotation_delta("brick", angle)
                        self.public_class.new_data = data
                        self.public_class.brick_process_node = "3-7-5-2"
            # 二次旋转完成后，再进入下降取砖
            elif self.public_class.brick_process_node == "3-7-5-2":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    self.public_class.new_data = self._normalize_pose_values(self.public_class.tcp_pos)
                    self.public_class.brick_process_node = "3-7-6"
            # 控制Z轴下降
            elif self.public_class.brick_process_node == "3-7-6":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2]) - 70.00,
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self._brick_record_event(
                        "secondary_descend_command",
                        command_target=data,
                    )
                    self._brick_move_linear_pick(data)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "3-7-7"
            # 打开吸盘，控制Z轴上升
            elif self.public_class.brick_process_node == "3-7-7":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 打开吸盘
                    self.jaka.blinx_set_digital_output(6, 5, 1)
                    self.jaka.blinx_set_digital_output(6, 6, 1)
                    time.sleep(0.2)
                    self.jaka.blinx_set_digital_output(6, 5, 0)

                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2]) + 71.00,
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self._brick_record_event(
                        "secondary_suction_and_lift_command",
                        command_target=data,
                        extra={
                            "suction_state": "open",
                            "executed_secondary_pick_tcp_pose": self._normalize_pose_values(self.public_class.tcp_pos),
                        },
                    )
                    self._brick_move_linear_pick(data)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "3-8"

            # # 回到待放点位
            # elif self.public_class.brick_process_node == "3-7-8":
            #     error_x = (float(self.public_class.new_data[0]) - float(
            #         self.public_class.tcp_pos[0]))
            #     error_y = (float(self.public_class.new_data[1]) - float(
            #         self.public_class.tcp_pos[1]))
            #     error_z = (float(self.public_class.new_data[2]) - float(
            #         self.public_class.tcp_pos[2]))
            #     if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
            #         # 控制机械臂到达放置过度点
            #         data = [float(self.public_class.brick_excessive_loc[0]),
            #                 float(self.public_class.brick_excessive_loc[1]),
            #                 float(self.public_class.brick_excessive_loc[2]),
            #                 float(self.public_class.brick_excessive_loc[3]),
            #                 float(self.public_class.brick_excessive_loc[4]),
            #                 float(self.public_class.brick_excessive_loc[5])]
            #         self.jaka.blinx_moveL(data, 250, 5000, 0)
            #         self.public_class.new_data = data
            #         self.public_class.brick_process_node = "3-8"

            # 判断机械臂是否达到位置，判断这是第几次放置，根据放置位置，进行放置
            elif self.public_class.brick_process_node == "3-8":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 根据抓取次数进行放置

                    data = []
                    if self.public_class.brick_process_num == 0:
                        # 控制机械臂到达放置过度点
                        data = [float(self.public_class.brick_place_loc[0]),
                                float(self.public_class.brick_place_loc[1]),
                                float(self.public_class.brick_place_loc[2]) + 40.00,
                                float(self.public_class.brick_place_loc[3]),
                                float(self.public_class.brick_place_loc[4]),
                                float(self.public_class.brick_place_loc[5])]
                    elif self.public_class.brick_process_num == 1:
                        # 控制机械臂到达放置过度点
                        data = [float(self.public_class.brick_place_loc[0]) + 137.00,
                                float(self.public_class.brick_place_loc[1]),
                                float(self.public_class.brick_place_loc[2]) + 40.00,
                                float(self.public_class.brick_place_loc[3]),
                                float(self.public_class.brick_place_loc[4]),
                                float(self.public_class.brick_place_loc[5])]
                    elif self.public_class.brick_process_num == 2:
                        # 控制机械臂到达放置过度点
                        data = [float(self.public_class.brick_place_loc[0]) + 70.00,
                                float(self.public_class.brick_place_loc[1]) - 103.00,
                                float(self.public_class.brick_place_loc[2]) + 40.00,
                                float(179.694),
                                float(0.567),
                                float(46.242)]
                    elif self.public_class.brick_process_num == 3:
                        # 控制机械臂到达放置过度点
                        data = [float(self.public_class.brick_place_loc[0]) + 172.00,
                                float(self.public_class.brick_place_loc[1]) - 103.00,
                                float(self.public_class.brick_place_loc[2]) + 40.00,
                                float(self.public_class.brick_place_loc[3]),
                                float(self.public_class.brick_place_loc[4]),
                                float(self.public_class.brick_place_loc[5])]
                    elif self.public_class.brick_process_num == 4:
                        # 控制机械臂到达放置过度点
                        data = [float(self.public_class.brick_place_loc[0]) + 137.00,
                                float(self.public_class.brick_place_loc[1]) - 204.00,
                                float(self.public_class.brick_place_loc[2]) + 40.00,
                                float(self.public_class.brick_place_loc[3]),
                                float(self.public_class.brick_place_loc[4]),
                                float(self.public_class.brick_place_loc[5])]
                    elif self.public_class.brick_process_num == 5:
                        # 控制机械臂到达放置过度点
                        data = [float(self.public_class.brick_place_loc[0]),
                                float(self.public_class.brick_place_loc[1]) - 204.00,
                                float(self.public_class.brick_place_loc[2]) + 40.00,
                                float(self.public_class.brick_place_loc[3]),
                                float(self.public_class.brick_place_loc[4]),
                                float(self.public_class.brick_place_loc[5])]
                    elif self.public_class.brick_process_num == 6:
                        # 控制机械臂到达放置过度点
                        data = [float(self.public_class.brick_place_loc[0]) - 35.00,
                                float(self.public_class.brick_place_loc[1]) - 103.00,
                                float(self.public_class.brick_place_loc[2]) + 40.00,
                                float(179.694),
                                float(0.567),
                                float(46.242)]
                    if len(data) > 0:
                        self._brick_record_event(
                            "place_approach_command",
                            command_target=data,
                            extra={"brick_process_num": int(self.public_class.brick_process_num)},
                        )
                        self._brick_move_linear_fast(data)
                        self.public_class.new_data = data
                        self.public_class.brick_process_node = "3-9"
                    else:
                        print("DATA 无数据")
            # 判断机械臂是否达到位置，如果到达，下降Z轴
            elif self.public_class.brick_process_node == "3-9":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 控制机械臂到达放置过度点
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2]) - 40.00,
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self._brick_record_event(
                        "place_descend_command",
                        command_target=data,
                    )
                    self._brick_move_linear_pick(data)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "3-10"
            # 判断机械臂是否达到位置，如果到达，关闭吸盘，Z轴上升
            elif self.public_class.brick_process_node == "3-10":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 关闭吸盘
                    self.btn_xipan_close_click()
                    # 控制机械臂Z轴上升
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2]) + 50.00,
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self._brick_record_event(
                        "place_release_and_lift_command",
                        command_target=data,
                        extra={"suction_state": "closed"},
                    )
                    self._brick_move_linear_pick(data)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "3-11"
            # 判断机械臂是否达到位置，如果到达，回到转180位置
            elif self.public_class.brick_process_node == "3-11":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 控制末端进行旋转
                    data = [float(self.public_class.initial_angle[0]) - 180.00,
                            float(self.public_class.initial_angle[1]),
                            float(self.public_class.initial_angle[2]), float(self.public_class.initial_angle[3]),
                            float(self.public_class.initial_angle[4]), float(self.public_class.initial_angle[5])]
                    self._brick_joint_move_rotate(data)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "3-12"
            # 角度是否到达，如果到达再回到初始姿态
            elif self.public_class.brick_process_node == "3-12":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    # 控制末端进行旋转
                    data = [float(self.public_class.initial_angle[0]),
                            float(self.public_class.initial_angle[1]),
                            float(self.public_class.initial_angle[2]), float(self.public_class.initial_angle[3]),
                            float(self.public_class.initial_angle[4]), float(self.public_class.initial_angle[5])]
                    self._brick_joint_move_rotate(data)
                    self.public_class.new_data = data
                    self.public_class.brick_process_node = "3-13"

            #  角度是否到达，如果到达再回到初始姿态
            elif self.public_class.brick_process_node == "3-13":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    if self.public_class.brick_process_num <= 5:
                        completed_num = int(self.public_class.brick_process_num)
                        self._brick_record_finalize_case(
                            "cycle_complete",
                            extra={
                                "completed_brick_process_num": completed_num,
                                "next_brick_process_num": completed_num + 1,
                            },
                        )
                        self.public_class.brick_process_num = self.public_class.brick_process_num + 1
                        self.public_class.brick_process_node = "3-1"
                    else:
                        # 回到初始位置
                        data = [float(self.public_class.identify_loc1[0]), float(self.public_class.identify_loc1[1]),
                                float(self.public_class.identify_loc1[2]), float(self.public_class.identify_loc1[3]),
                                float(self.public_class.identify_loc1[4]), float(self.public_class.identify_loc1[5])]
                        self._brick_move_linear_fast(data)
                        self._brick_record_finalize_case(
                            "brick_process_complete",
                            extra={"return_target": data},
                        )
                        self.public_class.new_data = None
                        self.public_class.brick_process_state = False  # 墙砖流程状态
                        self.public_class.brick_process_node = "0-0"  # 墙砖流程节点
                        self.public_class.brick_process_num = 0  # 墙砖抓取次数
                        self.public_class.brick_process_result = None
                        self.public_class.brick_secondary_alignment_result = None
                        if self.brick_process_recorder is not None:
                            self.textEdit_log.append(
                                f"墙砖记录完成: {self.brick_process_recorder.session_dir}\n"
                            )
                        self.reset_rotation_history("brick")

        # 判断是否是钢筋捆扎流程
        elif self.public_class.rebar_process_state:
            # 判断执行器放置区域
            if self.public_class.rebar_process_node == "0-0":
                # 获取机械臂控制器的数字输入信号
                self.jaka.blinx_get_digital_input_status()
                time.sleep(0.5)
                # 将吸盘捆扎机的放置区的状态提出
                xp_state = self.robot_DI_1
                kzj_state = self.robot_DI_2

                # 判断吸盘与捆扎机是否都在
                if xp_state == 1 and kzj_state == 1:
                    self.public_class.rebar_process_node = "1-0"   # 如果两个末端执行器都在
                # 判断如果吸盘在捆扎机不在
                elif xp_state == 1 and kzj_state != 1:
                    self.public_class.rebar_process_node = "3-0"   # 如果是捆扎机就直接开始下一步
                # 判断如果捆扎机在，吸盘不在
                elif xp_state != 1 and kzj_state == 1:
                    self.public_class.rebar_process_node = "2-0"   # 需先放置捆扎机，在获取夹爪
                # 如果两个都不在
                elif xp_state != 1 and kzj_state != 1:
                    print("报警，请将末端执行器归位")
                    self.public_class.rebar_process_state = False
            # 控制机械臂旋转180度
            elif self.public_class.rebar_process_node == "1-0":
                data = [float(self.public_class.joint_pos[0]) - 180.00, float(self.public_class.joint_pos[1]),
                        float(self.public_class.joint_pos[2]), float(self.public_class.joint_pos[3]),
                        float(self.public_class.joint_pos[4]), float(self.public_class.joint_pos[5])]
                self.jaka.blinx_joint_move(0, data, 50, 50)
                self.public_class.new_data = data
                self.public_class.rebar_process_node = "1-1"
            # 角度是否到达，如果到达控制J6旋转90都
            elif self.public_class.rebar_process_node == "1-1":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    # 控制末端移动90度
                    data = [float(self.public_class.joint_pos[0]), float(self.public_class.joint_pos[1]),
                            float(self.public_class.joint_pos[2]), float(self.public_class.joint_pos[3]),
                            float(self.public_class.joint_pos[4]),
                            float(self.public_class.joint_pos[5]) - 90.00]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "1-2"
            # 角度是否到达，如果到达控制机械臂到达捆扎机上方
            elif self.public_class.rebar_process_node == "1-2":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    # 控制机械臂到达捆扎机上方
                    data = [float(self.public_class.bundle_actuator_loc[0]),
                            float(self.public_class.bundle_actuator_loc[1]),
                            float(self.public_class.bundle_actuator_loc[2]) + 30.00,
                            float(self.public_class.bundle_actuator_loc[3]),
                            float(self.public_class.bundle_actuator_loc[4]),
                            float(self.public_class.bundle_actuator_loc[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "1-3"
            # 判断坐标是否到达，如果到达先打开快换夹具，在控制机械臂到达捆扎机位置
            elif self.public_class.rebar_process_node == "1-3":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 快换夹具开
                    self.jaka.blinx_set_digital_output(6, 5, 1)
                    self.jaka.blinx_set_digital_output(6, 8, 1)
                    time.sleep(0.5)
                    self.jaka.blinx_set_digital_output(6, 5, 0)
                    # 控制机械臂到达捆扎机位置
                    data = [float(self.public_class.bundle_actuator_loc[0]),
                            float(self.public_class.bundle_actuator_loc[1]),
                            float(self.public_class.bundle_actuator_loc[2]),
                            float(self.public_class.bundle_actuator_loc[3]),
                            float(self.public_class.bundle_actuator_loc[4]),
                            float(self.public_class.bundle_actuator_loc[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "1-4"
            # 判断坐标是否到达，如果到达控制快换夹具关，并控制机械臂上升
            elif self.public_class.rebar_process_node == "1-4":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 快换夹具开
                    self.jaka.blinx_set_digital_output(6, 5, 1)
                    self.jaka.blinx_set_digital_output(6, 8, 0)
                    time.sleep(0.5)
                    self.jaka.blinx_set_digital_output(6, 5, 0)
                    # 控制机械臂到达吸盘位置
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2] + 10.00),
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "1-5"
            # 判断坐标是否到达，如果到达控制Y轴移动到
            elif self.public_class.rebar_process_node == "1-5":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 控制机械臂退出吸盘放置位置
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1] + 100.00),
                            float(self.public_class.tcp_pos[2]),
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "1-6"
            # 判断坐标是否到达，如果到达控制机械臂到达回收位置
            elif self.public_class.rebar_process_node == "1-6":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    data = [float(self.public_class.initial_angle[0]) - 180.00,
                            float(self.public_class.initial_angle[1]),
                            float(self.public_class.initial_angle[2]),
                            float(self.public_class.initial_angle[3]),
                            float(self.public_class.initial_angle[4]),
                            float(self.public_class.initial_angle[5])]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "1-7"
            # 角度是否到达，如果到达控制机械臂到达捆扎机过度位置
            elif self.public_class.rebar_process_node == "1-7":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    # 控制机械臂捆扎机过度位置
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1] + 50.00),
                            float(self.public_class.tcp_pos[2]),
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "1-8"
            # 判断坐标是否到达，如果到达控制机械臂旋转180度
            elif self.public_class.rebar_process_node == "1-8":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    data = [float(self.public_class.joint_pos[0]) + 180.00,
                            float(self.public_class.joint_pos[1]),
                            float(self.public_class.joint_pos[2]),
                            float(self.public_class.joint_pos[3]),
                            float(self.public_class.joint_pos[4]),
                            float(self.public_class.joint_pos[5])]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "1-9"
            # 角度是否到达，如果到达控制机械臂到达初始位置
            elif self.public_class.rebar_process_node == "1-9":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                        self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (
                        -0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (
                        -0.1 <= error_j6 <= 0.1):
                    data = [float(self.public_class.initial_angle[0]),
                            float(self.public_class.initial_angle[1]),
                            float(self.public_class.initial_angle[2]),
                            float(self.public_class.initial_angle[3]),
                            float(self.public_class.initial_angle[4]),
                            float(self.public_class.initial_angle[5])]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "3-0"
            
            # 控制走放置吸盘流程
            # 控制机械臂旋转180度
            elif self.public_class.rebar_process_node == "2-0":
                data = [float(self.public_class.joint_pos[0]) - 180.00, float(self.public_class.joint_pos[1]),
                        float(self.public_class.joint_pos[2]), float(self.public_class.joint_pos[3]),
                        float(self.public_class.joint_pos[4]), float(self.public_class.joint_pos[5])]
                self.jaka.blinx_joint_move(0, data, 50, 50)
                self.public_class.new_data = data
                self.public_class.rebar_process_node = "2-1"
            # 角度是否到达，如果到达控制J6旋转90都
            elif self.public_class.rebar_process_node == "2-1":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    # 控制末端移动90度
                    data = [float(self.public_class.joint_pos[0]), float(self.public_class.joint_pos[1]),
                            float(self.public_class.joint_pos[2]), float(self.public_class.joint_pos[3]),
                            float(self.public_class.joint_pos[4]), float(self.public_class.joint_pos[5]) + 90.00]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "2-2"
            # 角度是否到达，如果到达控制机械臂待放置位置
            elif self.public_class.rebar_process_node == "2-2":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    # 控制机械臂待放置位置
                    data = [float(self.public_class.sucker_actuator_loc[0]),
                            float(self.public_class.sucker_actuator_loc[1]) + 100.00,
                            float(self.public_class.sucker_actuator_loc[2]) + 10.00,
                            float(self.public_class.sucker_actuator_loc[3]),
                            float(self.public_class.sucker_actuator_loc[4]),
                            float(self.public_class.sucker_actuator_loc[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "2-3"
            # 判断坐标是否到达，如果到达先控制机械臂到达吸盘放置区位置上一公分位置
            elif self.public_class.rebar_process_node == "2-3":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 控制机械臂到达吸盘位置
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]) - 100.00,
                            float(self.public_class.tcp_pos[2]),
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "2-4"
            # 判断坐标是否到达，并控制机械臂下降，再释放快换夹具
            elif self.public_class.rebar_process_node == "2-4":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 控制机械臂到达吸盘位置
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2]) - 10.00,
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "2-5"
                    time.sleep(1)
                    # 快换夹具释放
                    self.jaka.blinx_set_digital_output(6, 5, 1)
                    self.jaka.blinx_set_digital_output(6, 8, 1)
                    time.sleep(0.5)
                    self.jaka.blinx_set_digital_output(6, 5, 0)
            # 判断坐标是否到达，如果到达控制z轴上升
            elif self.public_class.rebar_process_node == "2-5":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 控制机械臂退出吸盘放置位置
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2] + 30.00),
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "2-6"
            # 判断坐标是否到达，如果到达控制机械臂到达回收位置
            elif self.public_class.rebar_process_node == "2-6":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    data = [float(self.public_class.initial_angle[0]) - 180.00,
                            float(self.public_class.initial_angle[1]),
                            float(self.public_class.initial_angle[2]), float(self.public_class.initial_angle[3]),
                            float(self.public_class.initial_angle[4]), float(self.public_class.initial_angle[5])]
                    self.jaka.blinx_joint_move(0, data, 50, 50)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "2-7"
            # 角度是否到达，如果到达控制机械臂回到初始角度,并控制滑轨到达拍照点位
            elif self.public_class.rebar_process_node == "2-7":
                error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                error_j2 = (float(self.public_class.new_data[1]) - float(
                    self.public_class.joint_pos[1]))
                error_j3 = (float(self.public_class.new_data[2]) - float(
                    self.public_class.joint_pos[2]))
                error_j4 = (float(self.public_class.new_data[3]) - float(
                    self.public_class.joint_pos[3]))
                error_j5 = (float(self.public_class.new_data[4]) - float(
                    self.public_class.joint_pos[4]))
                error_j6 = (float(self.public_class.new_data[5]) - float(
                    self.public_class.joint_pos[5]))
                if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                        -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                    self.public_class.rebar_process_node = "1-1"
            
            # 开启流程
            # 控制机械臂达到拍照位置
            elif self.public_class.rebar_process_node == "3-0":
                if self.public_class.new_data == None:
                    # 控制机械臂达到钢筋捆扎的拍照位置
                    data = [float(self.public_class.identify_loc1[0]), float(self.public_class.identify_loc1[1]), 
                            float(self.public_class.identify_loc1[2]), float(self.public_class.identify_loc1[3]), 
                            float(self.public_class.identify_loc1[4]), float(self.public_class.identify_loc1[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "3-1"
                else:
                    error_j1 = (float(self.public_class.new_data[0]) - float(
                    self.public_class.joint_pos[0]))
                    error_j2 = (float(self.public_class.new_data[1]) - float(
                        self.public_class.joint_pos[1]))
                    error_j3 = (float(self.public_class.new_data[2]) - float(
                        self.public_class.joint_pos[2]))
                    error_j4 = (float(self.public_class.new_data[3]) - float(
                        self.public_class.joint_pos[3]))
                    error_j5 = (float(self.public_class.new_data[4]) - float(
                        self.public_class.joint_pos[4]))
                    error_j6 = (float(self.public_class.new_data[5]) - float(
                        self.public_class.joint_pos[5]))
                    if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                            -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                        # 控制机械臂达到钢筋捆扎的拍照位置
                        data = [float(self.public_class.identify_loc1[0]), float(self.public_class.identify_loc1[1]), 
                                float(self.public_class.identify_loc1[2]), float(self.public_class.identify_loc1[3]), 
                                float(self.public_class.identify_loc1[4]), float(self.public_class.identify_loc1[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.rebar_process_node = "3-1"
                self.jaka.blinx_set_analog_output(6, 26, 500)  # 滑轨绝对速度
                self.jaka.blinx_set_analog_output(6, 25, -873.00)  # 滑轨绝对位置
                self.jaka.blinx_set_digital_output(6, 5, 1)
                self.jaka.blinx_set_digital_output(6, 4, 1)
                self.jaka.blinx_set_digital_output(6, 4, 0)
                self.jaka.blinx_set_digital_output(6, 5, 0)
            # 识别图像，并将结果进行Y方向的排序
            elif self.public_class.rebar_process_node == "3-1":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 获取滑轨当前位置
                    self.jaka.blinx_get_analog_input(6, 25)
                    time.sleep(0.5)
                    print(self.public_class.ai_value)
                    error_ai = abs(float(self.public_class.ai_value) - float(-873.00))
                    if error_ai <= 1:
                        time.sleep(1)
                        print("进行拍照识别")
                        # 获取图像数据
                        self.public_class.mech_2d_image, self.public_class.mech_depth_map, self.public_class.mech_point_cloud = self.mechCam.GrabImages()
                        # 将2D图像进行图像识别
                        img_rec, data_list = self.yolo_iamge.blinx_rebar_image_rec(self.public_class.mech_2d_image)

                        # 将图像显示在界面中
                        # 获取图像高度和宽度
                        heigt, width = img_rec.shape[:2]
                        # 显示图像到界面
                        pixmap = QtGui.QImage(img_rec, width, heigt, QtGui.QImage.Format_RGB888)
                        pixmap = QtGui.QPixmap.fromImage(pixmap)
                        self.Image_Show_1.setPixmap(pixmap)
                        self.Image_Show_1.setScaledContents(True)  # 图像自适应窗口大小

                        # 深度图像显示
                        # 获取图像高度和宽度
                        depth_8bit = cv2.normalize(self.public_class.mech_depth_map, None, 0, 255,
                                                   cv2.NORM_MINMAX, cv2.CV_8UC1)
                        depth_color = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
                        depth_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
                        heigt, width = depth_rgb.shape[:2]
                        # 显示图像到界面
                        pixmap = QtGui.QImage(depth_rgb, width, heigt, QtGui.QImage.Format_RGB888)
                        pixmap = QtGui.QPixmap.fromImage(pixmap)
                        self.Image_Show_2.setPixmap(pixmap)
                        self.Image_Show_2.setScaledContents(True)  # 图像自适应窗口大小

                        # 将数据进行Y方向的排序
                        self._set_display_image(self.Image_Show_1, img_rec, QtGui.QImage.Format_RGB888)
                        self._set_display_image(self.Image_Show_2, depth_rgb, QtGui.QImage.Format_RGB888)
                        sorted_data = sorted(data_list, key=lambda x: x[1], reverse=True)
                        print(sorted_data)
                        # 将识别数据保存到公共类中
                        self.public_class.rebar_data_list = sorted_data
                        self.public_class.rebar_process_node = "3-2"
            # 控制旋转45度
            elif self.public_class.rebar_process_node == "3-2":
                # 控制末端移动45度
                data = [float(self.public_class.joint_pos[0]), float(self.public_class.joint_pos[1]),
                        float(self.public_class.joint_pos[2]), float(self.public_class.joint_pos[3]),
                        float(self.public_class.joint_pos[4]), float(self.public_class.joint_pos[5]) - 45.00]
                self.jaka.blinx_joint_move(0, data, 50, 50)
                self.public_class.new_data = data
                self.public_class.rebar_process_node = "3-3"
            # 根据识别数据进行标定，并先移动XY轴
            elif self.public_class.rebar_process_node == "3-3" or self.public_class.rebar_process_node == "3-3-1":
                if self.public_class.rebar_process_node == "3-3":
                    error_j1 = (float(self.public_class.new_data[0]) - float(
                        self.public_class.joint_pos[0]))
                    error_j2 = (float(self.public_class.new_data[1]) - float(
                        self.public_class.joint_pos[1]))
                    error_j3 = (float(self.public_class.new_data[2]) - float(
                        self.public_class.joint_pos[2]))
                    error_j4 = (float(self.public_class.new_data[3]) - float(
                        self.public_class.joint_pos[3]))
                    error_j5 = (float(self.public_class.new_data[4]) - float(
                        self.public_class.joint_pos[4]))
                    error_j6 = (float(self.public_class.new_data[5]) - float(
                        self.public_class.joint_pos[5]))
                    if (-0.1 <= error_j1 <= 0.1) and (-0.1 <= error_j2 <= 0.1) and (-0.1 <= error_j3 <= 0.1) and (
                            -0.1 <= error_j4 <= 0.1) and (-0.1 <= error_j5 <= 0.1) and (-0.1 <= error_j6 <= 0.1):
                        # 进行标定转换
                        x = self.public_class.rebar_data_list[int(self.public_class.rebar_process_num)][0]
                        y = self.public_class.rebar_data_list[int(self.public_class.rebar_process_num)][1]
                        test_robot_xy = np.dot(self.m_ini1, [x, y, 1])  # 仿射逆变换，得到坐标（x,y)
                        x = test_robot_xy[0] - 15.00
                        y = test_robot_xy[1] - 10.00

                        # 根据标定结果移动xy轴
                        data = [float(x), float(y),
                                float(self.public_class.tcp_pos[2]), float(self.public_class.tcp_pos[3]),
                                float(self.public_class.tcp_pos[4]), float(self.public_class.tcp_pos[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.rebar_process_node = "3-4"
                else:
                    # 进行标定转换
                    x = self.public_class.rebar_data_list[int(self.public_class.rebar_process_num)][0]
                    y = self.public_class.rebar_data_list[int(self.public_class.rebar_process_num)][1]
                    test_robot_xy = np.dot(self.m_ini1, [x, y, 1])  # 仿射逆变换，得到坐标（x,y)
                    x = test_robot_xy[0] - 15.00
                    y = test_robot_xy[1] - 10.00

                    # 根据标定结果移动xy轴
                    data = [float(x), float(y),
                            float(self.public_class.tcp_pos[2]), float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]), float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "3-4"
            # 根据高度，下降Z轴
            elif self.public_class.rebar_process_node == "3-4":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    if self.public_class.rebar_process_num > 0:
                        # 根据高度下降Z
                        data = [float(self.public_class.tcp_pos[0]), float(self.public_class.tcp_pos[1]),
                                float(self.public_class.tcp_pos[2]) - 136.00 + 36, float(self.public_class.tcp_pos[3]),
                                float(self.public_class.tcp_pos[4]), float(self.public_class.tcp_pos[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.rebar_process_node = "3-5"
                    else:
                        # 根据高度下降Z
                        data = [float(self.public_class.tcp_pos[0]), float(self.public_class.tcp_pos[1]),
                                float(self.public_class.tcp_pos[2]) - 144.00, float(self.public_class.tcp_pos[3]),
                                float(self.public_class.tcp_pos[4]), float(self.public_class.tcp_pos[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = data
                        self.public_class.rebar_process_node = "3-5"
            # 控制机械臂往内移动
            elif self.public_class.rebar_process_node == "3-5":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 根据高度下降Z
                    data = [float(self.public_class.tcp_pos[0]) + 10.00, float(self.public_class.tcp_pos[1]) + 10.00,
                            float(self.public_class.tcp_pos[2]), float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]), float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "3-6"
            # 控制Z轴继续下降一公分
            elif self.public_class.rebar_process_node == "3-6":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 根据高度下降Z
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2]) - 10.00,
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "3-7"
            # 打开捆扎机，进行捆扎，并控制z轴上升
            elif self.public_class.rebar_process_node == "3-7":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 打开捆扎机
                    self.jaka.blinx_set_digital_output(6, 5, 1)
                    self.jaka.blinx_set_digital_output(6, 7, 1)
                    time.sleep(0.2)
                    self.jaka.blinx_set_digital_output(6, 5, 0)
                    # 等待两秒
                    time.sleep(2)
                    # 关闭捆扎机
                    self.jaka.blinx_set_digital_output(6, 5, 1)
                    self.jaka.blinx_set_digital_output(6, 7, 0)
                    time.sleep(0.2)
                    self.jaka.blinx_set_digital_output(6, 5, 0)
                    # 控制Z轴上升
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2]) + 10.00,
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "3-8"
            # 控制XY进行移动
            elif self.public_class.rebar_process_node == "3-8":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 控制
                    data = [float(self.public_class.tcp_pos[0]) - 10.00,
                            float(self.public_class.tcp_pos[1]) - 10.00,
                            float(self.public_class.tcp_pos[2]),
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "3-9"
            # 控制Z轴上升
            elif self.public_class.rebar_process_node == "3-9":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    # 控制Z轴上升
                    data = [float(self.public_class.tcp_pos[0]),
                            float(self.public_class.tcp_pos[1]),
                            float(self.public_class.tcp_pos[2]) + 100.60,
                            float(self.public_class.tcp_pos[3]),
                            float(self.public_class.tcp_pos[4]),
                            float(self.public_class.tcp_pos[5])]
                    self.jaka.blinx_moveL(data, 250, 5000, 0)
                    self.public_class.new_data = data
                    self.public_class.rebar_process_node = "3-10"
            # 判断是否还需要进行捆扎
            elif self.public_class.rebar_process_node == "3-10":
                error_x = (float(self.public_class.new_data[0]) - float(
                    self.public_class.tcp_pos[0]))
                error_y = (float(self.public_class.new_data[1]) - float(
                    self.public_class.tcp_pos[1]))
                error_z = (float(self.public_class.new_data[2]) - float(
                    self.public_class.tcp_pos[2]))
                if (-0.1 <= error_x <= 0.1) and (-0.1 <= error_y <= 0.1) and (-0.1 <= error_z <= 0.1):
                    if self.public_class.rebar_process_num >= len(self.public_class.rebar_data_list) - 1:
                        # 回到初始位置
                        data = [float(self.public_class.identify_loc1[0]), float(self.public_class.identify_loc1[1]),
                                float(self.public_class.identify_loc1[2]), float(self.public_class.identify_loc1[3]),
                                float(self.public_class.identify_loc1[4]), float(self.public_class.identify_loc1[5])]
                        self.jaka.blinx_moveL(data, 250, 5000, 0)
                        self.public_class.new_data = None
                        self.public_class.rebar_process_node = "0-0"
                        self.public_class.rebar_process_state = False
                    else:
                        # 回到流程继续捆扎
                        self.public_class.rebar_process_node = "3-3-1"
                        self.public_class.rebar_process_num = self.public_class.rebar_process_num + 1




    # endregion

    # region page3标定
    def get_image_mech(self):
        self.pixel_point = []
        self.lineEdit_P_x_1.setText("")
        self.lineEdit_P_y_1.setText("")
        self.lineEdit_P_x_2.setText("")
        self.lineEdit_P_y_2.setText("")
        self.lineEdit_P_x_3.setText("")
        self.lineEdit_P_y_3.setText("")
        self.lineEdit_P_x_4.setText("")
        self.lineEdit_P_y_4.setText("")
        if self.public_class.mech_connected:
            self.public_class.mech_2d_image, self.public_class.mech_depth_map, self.public_class.mech_point_cloud = self.mechCam.GrabImages()
            if self.public_class.mech_2d_image is not None:
                self.biaoding_image = self.public_class.mech_2d_image
                # 处理并显示2D图像
                self.process_and_show_biaoding_image()

        else:
            QMessageBox.warning(mainWindow, "Error", "Mech3D相机未连接", QMessageBox.Ok)

    def process_and_show_biaoding_image(self):
        # 清除旧场景
        self.scene_biaoding.clear()

        # 转换图像格式
        height, width = self.biaoding_image.shape[:2]
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(self.biaoding_image.data, width, height,
                             bytes_per_line, QtGui.QImage.Format_RGB888)

        # 创建QPixmap并添加到场景
        pixmap = QtGui.QPixmap.fromImage(q_img)
        self.pixmap_item = self.scene_biaoding.addPixmap(pixmap)
        self.pixmap_item.setTransformationMode(QtCore.Qt.SmoothTransformation)
        # print("pixmap_item 创建成功:", self.pixmap_item)  # 调试输出

        # 设置场景大小
        self.scene_biaoding.setSceneRect(0, 0, width, height)

        # 重置视图变换
        self.graphics_view_biaoding.resetTransform()

        # 新增：初始自适应视图大小
        self.fit_biaoding_image_to_view()

    # 新增方法：自适应2D图像到视图
    def fit_biaoding_image_to_view(self):
        if not self.scene_biaoding.items():
            return

        # 获取视图和场景的尺寸
        view_rect = self.graphics_view_biaoding.viewport().rect()
        scene_rect = self.scene_biaoding.sceneRect()

        # 计算合适的缩放比例
        x_ratio = view_rect.width() / scene_rect.width()
        y_ratio = view_rect.height() / scene_rect.height()
        ratio = min(x_ratio, y_ratio)

        # 应用缩放
        self.graphics_view_biaoding.resetTransform()
        self.graphics_view_biaoding.scale(ratio, ratio)

        # 确保图像居中
        self.center_biaoding_image()

    # 新增方法：居中2D图像
    def center_biaoding_image(self):
        if not self.scene_biaoding.items():
            return
        self.graphics_view_biaoding.setSceneRect(self.scene_biaoding.sceneRect())
        self.graphics_view_biaoding.centerOn(self.scene_biaoding.items()[0])

    # endregion
    # region 主界面按钮
    def btn_systermRun_click(self):
        self.public_class.is_continue_show = False  # 连续模式关闭
        self.tabWidget.setCurrentIndex(0)
        self.btn_systermRun.setStyleSheet(
            "image: url(:/background/实时监测.png);background-color: rgb(0, 0, 150);border-radius: 30px; border: 2px groove rgb(255, 85, 0);border-style: none;color: white;border: 4px double rgb(0, 255, 0);")
        self.btn_camSet.setStyleSheet(
            "image: url(:/background/相机.png);background-color: rgb(0, 0, 150);border-radius: 30px; border: 2px groove rgb(255, 85, 0);border-style: none;color: white;")
        self.btn_calibrate.setStyleSheet(
            "image: url(:/background/坐标标定.png);background-color: rgb(0, 0, 150);border-radius: 30px; border: 2px groove rgb(255, 85, 0);border-style: none;color: white;")
        self.label_systermRun.setStyleSheet("color:rgb(0,255,0);border:none;")
        self.label_camSet.setStyleSheet("color:rgb(0,0,0);border:none;")
        self.label_calibrate.setStyleSheet("color:rgb(0,0,0);border:none;")

    def btn_camSet_click(self):
        self.tabWidget.setCurrentIndex(1)
        self.btn_camSet.setStyleSheet(
            "image: url(:/background/相机.png);background-color: rgb(0, 0, 150);border-radius: 30px; border: 2px groove rgb(255, 85, 0);border-style: none;color: white;border: 4px double rgb(0, 255, 0);")
        self.btn_systermRun.setStyleSheet(
            "image: url(:/background/实时监测.png);background-color: rgb(0, 0, 150);border-radius: 30px; border: 2px groove rgb(255, 85, 0);border-style: none;color: white;")
        self.btn_calibrate.setStyleSheet(
            "image: url(:/background/坐标标定.png);background-color: rgb(0, 0, 150);border-radius: 30px; border: 2px groove rgb(255, 85, 0);border-style: none;color: white;")
        self.label_systermRun.setStyleSheet("color:rgb(0,0,0);border:none;")
        self.label_camSet.setStyleSheet("color:rgb(0,255,0);border:none;")
        self.label_calibrate.setStyleSheet("color:rgb(0,0,0);border:none;")

    def btn_calibrate_click(self):
        self.public_class.is_continue_show = False  # 连续模式关闭
        self.tabWidget.setCurrentIndex(2)
        self.btn_systermRun.setStyleSheet(
            "image: url(:/background/实时监测.png);background-color: rgb(0, 0, 150);border-radius: 30px; border: 2px groove rgb(255, 85, 0);border-style: none;color: white;")
        self.btn_calibrate.setStyleSheet(
            "image: url(:/background/坐标标定.png);background-color: rgb(0, 0, 150);border-radius: 30px; border: 2px groove rgb(255, 85, 0);border-style: none;color: white;border: 4px double rgb(0, 255, 0);")
        self.btn_camSet.setStyleSheet(
            "image: url(:/background/相机.png);background-color: rgb(0, 0, 150);border-radius: 30px; border: 2px groove rgb(255, 85, 0);border-style: none;color: white;")
        self.label_systermRun.setStyleSheet("color:rgb(0,0,0);border:none;")
        self.label_camSet.setStyleSheet("color:rgb(0,0,0);border:none;")
        self.label_calibrate.setStyleSheet("color:rgb(0,255,0);border:none;")

    # endregion

    # region 流程按钮事件
    # 吸盘取
    def blinx_btn_suction_get(self):
        # 获取滑轨当前位置
        self.jaka.blinx_get_analog_input(6, 25)
        time.sleep(0.5)
        print(self.public_class.ai_value)
        error_ai = abs(float(self.public_class.ai_value) - float(-991))
        if error_ai <= 1:
            self.public_class.new_data = None
            self.public_class.sucker_process = "0-0"
            self.public_class.sucker_type = 0  # 0取  1放
            self.public_class.sucker_state = True
        else:
            QMessageBox.warning(mainWindow, "Error", "请将滑轨回到初始位置", QMessageBox.Ok)

    # 吸盘放
    def blinx_btn_suction_set(self):
        # 获取滑轨当前位置
        self.jaka.blinx_get_analog_input(6, 25)
        time.sleep(0.5)
        print(self.public_class.ai_value)
        error_ai = abs(float(self.public_class.ai_value) - float(-991))
        if error_ai <= 1:
            self.public_class.new_data = None
            self.public_class.sucker_process = "0-0"
            self.public_class.sucker_type = 1  # 0取  1放
            self.public_class.sucker_state = True
        else:
            QMessageBox.warning(mainWindow, "Error", "请将滑轨回到初始位置", QMessageBox.Ok)

    # 捆扎机取
    def blinx_btn_bundle_get(self):
        # 获取滑轨当前位置
        self.jaka.blinx_get_analog_input(6, 25)
        time.sleep(0.5)
        print(self.public_class.ai_value)
        error_ai = abs(float(self.public_class.ai_value) - float(-991))
        if error_ai <= 1:
            self.public_class.new_data = None
            self.public_class.bundle_process = "0-0"
            self.public_class.bundle_type = 0  # 0取   1放
            self.public_class.bundle_state = True
        else:
            QMessageBox.warning(mainWindow, "Error", "请将滑轨回到初始位置", QMessageBox.Ok)


    # 捆扎机放
    def blinx_btn_bundle_set(self):
        # 获取滑轨当前位置
        self.jaka.blinx_get_analog_input(6, 25)
        time.sleep(0.5)
        print(self.public_class.ai_value)
        error_ai = abs(float(self.public_class.ai_value) - float(-991))
        if error_ai <= 1:
            self.public_class.new_data = None
            self.public_class.bundle_process = "0-0"
            self.public_class.bundle_type = 1  # 0取   1放
            self.public_class.bundle_state = True
        else:
            QMessageBox.warning(mainWindow, "Error", "请将滑轨回到初始位置", QMessageBox.Ok)


    # 初始化按钮
    def blinx_btn_initialization(self):
        self.jaka.blinx_set_analog_output(6, 26, 100)  # 滑轨绝对速度
        self.jaka.blinx_set_analog_output(6, 25, -991)  # 滑轨绝对位置
        self.jaka.blinx_set_digital_output(6, 5, 1)
        self.jaka.blinx_set_digital_output(6, 4, 1)
        self.jaka.blinx_set_digital_output(6, 4, 0)
        self.jaka.blinx_set_digital_output(6, 5, 0)

        # 控制机械臂回到初始位置
        data = [float(self.public_class.initial_angle[0]), float(self.public_class.initial_angle[1]),
                float(self.public_class.initial_angle[2]), float(self.public_class.initial_angle[3]),
                float(self.public_class.initial_angle[4]), float(self.public_class.initial_angle[5])]
        self.jaka.blinx_joint_move(0, data, 50, 50)

        # 将按钮启用
        self.btn_suction_get.setEnabled(True)
        self.btn_suction_set.setEnabled(True)
        self.btn_bundle_get.setEnabled(True)
        self.btn_bundle_set.setEnabled(True)
        self.btn_porcelain.setEnabled(True)
        self.btn_brick.setEnabled(True)
        self.btn_rebar.setEnabled(True)

    # 瓷砖流程按钮
    def blinx_btn_porcelain(self):
        # 获取滑轨当前位置
        self.jaka.blinx_get_analog_input(6, 25)
        time.sleep(0.5)
        print(self.public_class.ai_value)
        error_ai = abs(float(self.public_class.ai_value) - float(-991))
        if error_ai <= 1:
            self.public_class.new_data = None
            self.public_class.ceramic_process_state = True  # 瓷砖流程状态
            self.public_class.ceramic_process_node = "0-0"  # 瓷砖流程节点
            self.public_class.ceramic_process_num = 0  # 瓷砖流程抓取次数
            self.public_class.ceramic_process_result = None
            self.reset_rotation_history("ceramic")
        else:
            QMessageBox.warning(mainWindow, "Error", "请先初始化，将设备回到初始位置", QMessageBox.Ok)


    # 砌砖流程按钮
    def blinx_btn_brick(self):
        # 获取滑轨当前位置
        self.jaka.blinx_get_analog_input(6, 25)
        time.sleep(0.5)
        print(self.public_class.ai_value)
        error_ai = abs(float(self.public_class.ai_value) - float(-991))
        if error_ai <= 1:
            self.public_class.new_data = None
            self.public_class.brick_process_state = True  # 墙砖流程状态
            self.public_class.brick_process_node = "0-0"  # 墙砖流程节点
            self.public_class.brick_process_num = 0  # 墙砖抓取次数
            self.public_class.brick_process_result = None
            self.public_class.brick_secondary_alignment_result = None
            self.reset_rotation_history("brick")
            self._start_brick_process_recording_session()
        else:
            QMessageBox.warning(mainWindow, "Error", "请先初始化，将设备回到初始位置", QMessageBox.Ok)
    # 钢筋捆扎流程按钮
    def blinx_btn_rebar(self):
        # 获取滑轨当前位置
        self.jaka.blinx_get_analog_input(6, 25)
        time.sleep(0.5)
        print(self.public_class.ai_value)
        error_ai = abs(float(self.public_class.ai_value) - float(-991))
        if error_ai <= 1:
            self.public_class.new_data = None
            self.public_class.rebar_process_state = True  # 钢筋捆扎流程状态
            self.public_class.rebar_process_node = "0-0"  # 钢筋捆扎流程节点
            self.public_class.rebar_process_num = 0  # 钢筋捆扎次数
        else:
            QMessageBox.warning(mainWindow, "Error", "请先初始化，将设备回到初始位置", QMessageBox.Ok)
    # endregion

    # region功能函数:标定
    # ch:存图 | en:save image
    def save_jpg_biaoding(self):
        now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))  # 获取系统时间
        if self.biaoding_image is None:
            strError = "Save jpg failed:"
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            image = cv2.cvtColor(self.biaoding_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite('pic/biaoding1/image_' + str(now_time) + '.jpg', image)  # 保存2D图片
            print("Save image_biaoding success")

    def getPixel(self, event):
        if self.biaoding_image is not None:
            # 1. 获取鼠标点击的视图坐标
            view_pos = event.pos()

            # 2. 转换为场景坐标
            scene_pos = self.graphics_view_biaoding.mapToScene(view_pos)

            # 3. 转换为图像项坐标（考虑缩放、旋转）
            if hasattr(self, 'pixmap_item') and self.pixmap_item:
                item_pos = self.pixmap_item.mapFromScene(scene_pos)

                # 4. 获取逆变换矩阵（处理缩放、旋转）
                transform = self.pixmap_item.transform().inverted()[0]
                pixel_pos = transform.map(item_pos)

                x = int(pixel_pos.x())
                y = int(pixel_pos.y())

                # 5. 确保坐标在图像范围内
                width = self.pixmap_item.pixmap().width()
                height = self.pixmap_item.pixmap().height()
                pixel_x = max(0, min(x, width - 1))
                pixel_y = max(0, min(y, height - 1))

                print(f"真实像素坐标: ({pixel_x}, {pixel_y})")

            if len(self.pixel_point) <= 3:
                self.pixel_point.append((pixel_x, pixel_y))
                print(self.pixel_point)
            for i in range(len(self.pixel_point) - 1, len(self.pixel_point)):
                # if len(self.pixel_point)<=4:
                # image_paint=image
                cv2.circle(self.biaoding_image, (self.pixel_point[i][0], self.pixel_point[i][1]), 5,
                           (255, 0, 0),
                           thickness=-1)
                cv2.putText(self.biaoding_image, "%s,%s" % (
                    str(len(self.pixel_point)) + ":(" + str(self.pixel_point[i][0]), str(self.pixel_point[i][1])) + ")",
                            (self.pixel_point[i][0], self.pixel_point[i][1]), cv2.FONT_HERSHEY_PLAIN,
                            3, (255, 0, 0), thickness=4)
                if len(self.pixel_point) == 1:
                    self.lineEdit_P_x_1.setText(str(self.pixel_point[0][0]))
                    self.lineEdit_P_y_1.setText(str(self.pixel_point[0][1]))
                elif len(self.pixel_point) == 2:
                    self.lineEdit_P_x_2.setText(str(self.pixel_point[1][0]))
                    self.lineEdit_P_y_2.setText(str(self.pixel_point[1][1]))
                elif len(self.pixel_point) == 3:
                    self.lineEdit_P_x_3.setText(str(self.pixel_point[2][0]))
                    self.lineEdit_P_y_3.setText(str(self.pixel_point[2][1]))
                elif len(self.pixel_point) == 4:
                    self.lineEdit_P_x_4.setText(str(self.pixel_point[3][0]))
                    self.lineEdit_P_y_4.setText(str(self.pixel_point[3][1]))
                else:
                    return
            # 转换图像格式
            height, width = self.biaoding_image.shape[:2]
            bytes_per_line = 3 * width
            q_img = QtGui.QImage(self.biaoding_image.data, width, height,
                                 bytes_per_line, QtGui.QImage.Format_RGB888)
            # 创建QPixmap并添加到场景
            pixmap = QtGui.QPixmap.fromImage(q_img)
            pixmap_item = self.scene_biaoding.addPixmap(pixmap)
            pixmap_item.setTransformationMode(QtCore.Qt.SmoothTransformation)

    def pixel_to_object(self):  # 开始标定
        # 坐标转为数组
        if self.lineEdit_P_x_1.text() != "" and self.lineEdit_P_y_1.text() != "" and self.lineEdit_P_x_2.text() != "" and self.lineEdit_P_y_2.text() != "" and self.lineEdit_P_x_3.text() != "" and self.lineEdit_P_y_3.text() != "" and self.lineEdit_R_x_1.text() != "" and self.lineEdit_R_y_1.text() != "" and self.lineEdit_R_x_2.text() != "" and self.lineEdit_R_y_2.text() != "" and self.lineEdit_R_x_3.text() != "" and self.lineEdit_R_y_3.text() != "":
            pts1 = np.float32([[int(self.lineEdit_P_x_1.text()), int(self.lineEdit_P_y_1.text())],
                               [int(self.lineEdit_P_x_2.text()), int(self.lineEdit_P_y_2.text())],
                               [int(self.lineEdit_P_x_3.text()), int(self.lineEdit_P_y_3.text())]])
            pts2 = np.float32([[float(self.lineEdit_R_x_1.text()), float(self.lineEdit_R_y_1.text())],
                               [float(self.lineEdit_R_x_2.text()), float(self.lineEdit_R_y_2.text())],
                               [float(self.lineEdit_R_x_3.text()), float(self.lineEdit_R_y_3.text())]])
            self.M = cv2.getAffineTransform(pts1, pts2)  # 仿射变化，仿射矩阵为2*3
            print("M:", self.M)
            test_robot_xy = np.dot(self.M, [int(self.lineEdit_P_x_4.text()), int(self.lineEdit_P_y_4.text()),
                                            1])  # 仿射逆变换，得到坐标（x,y)
            px = float(test_robot_xy[0])
            py = float(test_robot_xy[1])
            self.label_Ero_x.setText(str(round(float(self.lineEdit_R_x_4.text()) - px, 2)))
            self.label_Ero_y.setText(str(round(float(self.lineEdit_R_y_4.text()) - py, 2)))
            QMessageBox.warning(mainWindow, "OK", "标定成功", QMessageBox.Ok)
            self.biaoding_ok = True
        else:
            QMessageBox.warning(mainWindow, "Error", "输入坐标为空", QMessageBox.Ok)
            self.biaoding_ok = False
            return

    def matrix_saveto_ini(self):  # 保存标定结果
        if self.biaoding_ok:
            # region 配置文件读取
            config = configparser.ConfigParser()
            # -read读取ini文件
            config.read('Config/config.ini', encoding="utf-8")
            config.set("RelationalMatrix", "m1", str(self.M))
            fh = open('Config/config.ini', 'w')
            config.write(fh)
            fh.close()
            QMessageBox.warning(mainWindow, "OK", "保存标定参数成功", QMessageBox.Ok)
        else:
            QMessageBox.warning(mainWindow, "Error", "没有获取到标定参数矩阵", QMessageBox.Ok)

    def lab_R_X_1_clicked(self, event):
        if self.joint_or_xyz:
            self.lineEdit_R_x_1.setText(self.label_j1.text())
        else:
            QMessageBox.warning(mainWindow, "Error", "请将机器人切换到坐标控制", QMessageBox.Ok)

    def lab_R_Y_1_clicked(self, event):
        if self.joint_or_xyz:
            self.lineEdit_R_y_1.setText(self.label_j2.text())
        else:
            QMessageBox.warning(mainWindow, "Error", "请将机器人切换到坐标控制", QMessageBox.Ok)

    def lab_R_X_2_clicked(self, event):
        if self.joint_or_xyz:
            self.lineEdit_R_x_2.setText(self.label_j1.text())
        else:
            QMessageBox.warning(mainWindow, "Error", "请将机器人切换到坐标控制", QMessageBox.Ok)

    def lab_R_Y_2_clicked(self, event):
        if self.joint_or_xyz:
            self.lineEdit_R_y_2.setText(self.label_j2.text())
        else:
            QMessageBox.warning(mainWindow, "Error", "请将机器人切换到坐标控制", QMessageBox.Ok)

    def lab_R_X_3_clicked(self, event):
        if self.joint_or_xyz:
            self.lineEdit_R_x_3.setText(self.label_j1.text())
        else:
            QMessageBox.warning(mainWindow, "Error", "请将机器人切换到坐标控制", QMessageBox.Ok)

    def lab_R_Y_3_clicked(self, event):
        if self.joint_or_xyz:
            self.lineEdit_R_y_3.setText(self.label_j2.text())
        else:
            QMessageBox.warning(mainWindow, "Error", "请将机器人切换到坐标控制", QMessageBox.Ok)

    def lab_R_X_4_clicked(self, event):
        if self.joint_or_xyz:
            self.lineEdit_R_x_4.setText(self.label_j1.text())
        else:
            QMessageBox.warning(mainWindow, "Error", "请将机器人切换到坐标控制", QMessageBox.Ok)

    def lab_R_Y_4_clicked(self, event):
        if self.joint_or_xyz:
            self.lineEdit_R_y_4.setText(self.label_j2.text())
        else:
            QMessageBox.warning(mainWindow, "Error", "请将机器人切换到坐标控制", QMessageBox.Ok)

    def calibration(self, piexl_x, piexl_y):  # 坐标转换：像素坐标转为机器人坐标
        robot_xy = np.dot(self.m_ini, [piexl_x, piexl_y, 1])  # 仿射逆变换，得到坐标（x,y)
        rx = round(float(robot_xy[0]), 1)  # 保留一位小数
        ry = round(float(robot_xy[1]), 1)
        return rx, ry

    def calibration_parm_ini(self):  # 读取配置文件标定参数转换成矩阵参数
        self.m_ini = self.strMatrix_to_Matrix(self.public_class.M)
        self.m_ini1 = self.strMatrix_to_Matrix(self.public_class.M1)
        print("m_ini:", self.m_ini)

    def strMatrix_to_Matrix(self, strM):  # 字符串转2*3矩阵
        out = strM.replace('[', '').replace(']', '')  # 去掉中括号
        str = out.replace('\n', ' ')
        res = re.sub(' +', ' ', str)  # 去掉一个或多个空格
        # print("out:", out)
        dlist = res.strip(' ').split(' ')  # 转换成一个list
        listresult = []
        for i in range(0, len(dlist)):
            listresult.append(float(dlist[i]))  # 将字符串list转为float型的list
        # print("listresult:", listresult)
        darr = np.array(listresult)  # 将list转换为array
        # print("darr:", darr)
        resultM = darr.reshape(2, 3)  # 将array转换为2维(2,3)的矩阵
        # print("result:", resultM)
        return resultM

    # endregion
    # region功能函数:机械臂示教
    # region 机器人控制器
    def btn_controller_power_on_click(self):
        self.plc.controller_power_on()

    def btn_controller_power_off_click(self):
        if self.jaka.blinx_shutdown():
            self.textEdit_log.append("关闭控制器成功!\n")
            self.label_jaka.setText("机器人连接失败")
            self.label_jaka.setStyleSheet("color:rgb(255,0,0);border:none;")
            self.textEdit_log.append("Error：机器人连接失败\n")
        else:
            self.textEdit_log.append("Error:关闭控制器失败\n")

    def btn_robot_power_on_click(self):
        if self.jaka.blinx_power_on():
            self.textEdit_log.append("开启机器人电源成功!\n")
        else:
            self.textEdit_log.append("Error:开启机器人电源失败\n")

    def btn_robot_power_off_click(self):
        if self.jaka.blinx_power_off():
            self.textEdit_log.append("关闭机器人电源成功!\n")
        else:
            self.textEdit_log.append("Error:关闭机器人电源失败\n")

    def btn_enable_robot_click(self):
        if self.jaka.blinx_enable_robot():
            self.textEdit_log.append("开启机器上使能成功!\n")
        else:
            self.textEdit_log.append("Error:开启机器人上使能失败\n")

    def btn_disable_robot_click(self):
        if self.jaka.blinx_disable_robot():
            self.textEdit_log.append("机器人下使能成功!\n")
        else:
            self.textEdit_log.append("Error:机器人下使能失败\n")

    def btn_home_click(self):
        home_point = [-90, 68.096, 108.580, 93.324, -90, 45]
        self.jaka.blinx_joint_move(0, home_point, self.jaka.speeds_joint[1], 100)

    def btn_play_click(self):
        play_point = [-90, 48.804, 115.965, 105.230, -90, 45]
        self.jaka.blinx_joint_move(0, play_point, self.jaka.speeds_joint[1], 100)

    # endregion
    # region 机器人状态获取
    def get_jaka_state(self):
        try:
            self.jaka.blinx_get_joint_pos()
            self.jaka.blinx_get_tcp_pos()
            if self.joint_or_xyz == False:
                if len(self.public_class.joint_pos) > 0:
                    self.label_j1.setText(str(round(self.public_class.joint_pos[0], 3)))
                    self.label_j2.setText(str(round(self.public_class.joint_pos[1], 3)))
                    self.label_j3.setText(str(round(self.public_class.joint_pos[2], 3)))
                    self.label_j4.setText(str(round(self.public_class.joint_pos[3], 3)))
                    self.label_j5.setText(str(round(self.public_class.joint_pos[4], 3)))
                    self.label_j6.setText(str(round(self.public_class.joint_pos[5], 3)))
            elif self.joint_or_xyz:
                if len(self.public_class.tcp_pos) > 0:
                    self.label_j1.setText(str(round(self.public_class.tcp_pos[0], 3)))
                    self.label_j2.setText(str(round(self.public_class.tcp_pos[1], 3)))
                    self.label_j3.setText(str(round(self.public_class.tcp_pos[2], 3)))
                    self.label_j4.setText(str(round(self.public_class.tcp_pos[3], 3)))
                    self.label_j5.setText(str(round(self.public_class.tcp_pos[4], 3)))
                    self.label_j6.setText(str(round(self.public_class.tcp_pos[5], 3)))
            self.jaka.blinx_get_digital_input_status()
            if len(self.public_class.din_status) > 0:
                self.robot_DI_1 = self.public_class.din_status[0]
                self.robot_DI_2 = self.public_class.din_status[1]
                self.robot_DI_3 = self.public_class.din_status[2]
                self.robot_DI_4 = self.public_class.din_status[3]
                self.robot_DI_5 = self.public_class.din_status[4]
                # print(self.robot_DI_1,self.robot_DI_2,self.robot_DI_3,self.robot_DI_4,self.robot_DI_5)
        except Exception as e:
            print("Error:获取机器人状态失败:", e)

    # endregion
    # region  关节控制
    def btn_j1_add_click(self):
        if self.joint_or_xyz == False:
            degree = float(self.label_j1.text()) + float(self.lineEdit_step.text())
            if -360 <= degree <= 360:
                self.jaka.blinx_jog(2, self.jaka.COORD_JOINT, 0, self.lineEdit_speed.text(), self.lineEdit_step.text())
        if self.joint_or_xyz == True:
            distance = float(self.label_j1.text()) + float(self.lineEdit_step.text())
            ref_pos = [0, 0, 90, 0, 90, 180]
            cartesian_pose = [distance, float(self.label_j2.text()), float(self.label_j3.text()),
                              float(self.label_j4.text()), float(self.label_j5.text()), float(self.label_j6.text())]
            ret = self.jaka.blinx_kine_inverse(ref_pos, cartesian_pose)
            if ret:
                self.jaka.blinx_jog(2, self.jaka.COORD_BASE, 0, self.lineEdit_speed.text(), self.lineEdit_step.text())
            else:
                QMessageBox.warning(mainWindow, "Error", "坐标不可达", QMessageBox.Ok)

    def btn_j1_subtract_click(self):
        if self.joint_or_xyz == False:
            degree = float(self.label_j1.text()) - float(self.lineEdit_step.text())
            if -360 <= degree <= 360:
                self.jaka.blinx_jog(2, self.jaka.COORD_JOINT, 0, -float(self.lineEdit_speed.text()),
                                    self.lineEdit_step.text())
        if self.joint_or_xyz == True:
            distance = float(self.label_j1.text()) - float(self.lineEdit_step.text())
            ref_pos = [0, 0, 90, 0, 90, 180]
            cartesian_pose = [distance, float(self.label_j2.text()), float(self.label_j3.text()),
                              float(self.label_j4.text()), float(self.label_j5.text()), float(self.label_j6.text())]
            ret = self.jaka.blinx_kine_inverse(ref_pos, cartesian_pose)
            if ret:
                self.jaka.blinx_jog(2, self.jaka.COORD_BASE, 0, -float(self.lineEdit_speed.text()),
                                    self.lineEdit_step.text())
            else:
                QMessageBox.warning(mainWindow, "Error", "坐标不可达", QMessageBox.Ok)

    def btn_j2_add_click(self):
        if self.joint_or_xyz == False:
            degree = float(self.label_j2.text()) + float(self.lineEdit_step.text())
            if -125 <= degree <= 125:
                self.jaka.blinx_jog(2, self.jaka.COORD_JOINT, 1, self.lineEdit_speed.text(), self.lineEdit_step.text())
        if self.joint_or_xyz == True:
            distance = float(self.label_j2.text()) + float(self.lineEdit_step.text())
            ref_pos = [0, 0, 90, 0, 90, 180]
            cartesian_pose = [float(self.label_j1.text()), distance, float(self.label_j3.text()),
                              float(self.label_j4.text()), float(self.label_j5.text()), float(self.label_j6.text())]
            ret = self.jaka.blinx_kine_inverse(ref_pos, cartesian_pose)
            if ret:
                self.jaka.blinx_jog(2, self.jaka.COORD_BASE, 1, self.lineEdit_speed.text(), self.lineEdit_step.text())
            else:
                QMessageBox.warning(mainWindow, "Error", "坐标不可达", QMessageBox.Ok)

    def btn_j2_subtract_click(self):
        if self.joint_or_xyz == False:
            degree = float(self.label_j2.text()) - float(self.lineEdit_step.text())
            if -125 <= degree <= 125:
                self.jaka.blinx_jog(2, self.jaka.COORD_JOINT, 1, -float(self.lineEdit_speed.text()),
                                    self.lineEdit_step.text())
        if self.joint_or_xyz == True:
            distance = float(self.label_j2.text()) - float(self.lineEdit_step.text())
            ref_pos = [0, 0, 90, 0, 90, 180]
            cartesian_pose = [float(self.label_j1.text()), distance, float(self.label_j3.text()),
                              float(self.label_j4.text()), float(self.label_j5.text()), float(self.label_j6.text())]
            ret = self.jaka.blinx_kine_inverse(ref_pos, cartesian_pose)
            if ret:
                self.jaka.blinx_jog(2, self.jaka.COORD_BASE, 1, -float(self.lineEdit_speed.text()),
                                    self.lineEdit_step.text())
            else:
                QMessageBox.warning(mainWindow, "Error", "坐标不可达", QMessageBox.Ok)

    def btn_j3_add_click(self):
        if self.joint_or_xyz == False:
            degree = float(self.label_j3.text()) + float(self.lineEdit_step.text())
            if -130 <= degree <= 130:
                self.jaka.blinx_jog(2, self.jaka.COORD_JOINT, 2, self.lineEdit_speed.text(), self.lineEdit_step.text())
        if self.joint_or_xyz == True:
            distance = float(self.label_j3.text()) + float(self.lineEdit_step.text())
            ref_pos = [0, 0, 90, 0, 90, 180]
            cartesian_pose = [float(self.label_j1.text()), float(self.label_j2.text()), distance,
                              float(self.label_j4.text()), float(self.label_j5.text()), float(self.label_j6.text())]
            ret = self.jaka.blinx_kine_inverse(ref_pos, cartesian_pose)
            if ret:
                self.jaka.blinx_jog(2, self.jaka.COORD_BASE, 2, self.lineEdit_speed.text(), self.lineEdit_step.text())
            else:
                QMessageBox.warning(mainWindow, "Error", "坐标不可达", QMessageBox.Ok)

    def btn_j3_subtract_click(self):
        if self.joint_or_xyz == False:
            degree = float(self.label_j3.text()) - float(self.lineEdit_step.text())
            if -130 <= degree <= 130:
                self.jaka.blinx_jog(2, self.jaka.COORD_JOINT, 2, -float(self.lineEdit_speed.text()),
                                    self.lineEdit_step.text())
        if self.joint_or_xyz == True:
            distance = float(self.label_j3.text()) - float(self.lineEdit_step.text())
            ref_pos = [0, 0, 90, 0, 90, 180]
            cartesian_pose = [float(self.label_j1.text()), float(self.label_j2.text()), distance,
                              float(self.label_j4.text()), float(self.label_j5.text()), float(self.label_j6.text())]
            ret = self.jaka.blinx_kine_inverse(ref_pos, cartesian_pose)
            if ret:
                self.jaka.blinx_jog(2, self.jaka.COORD_BASE, 2, -float(self.lineEdit_speed.text()),
                                    self.lineEdit_step.text())
            else:
                QMessageBox.warning(mainWindow, "Error", "坐标不可达", QMessageBox.Ok)

    def btn_j4_add_click(self):
        if self.joint_or_xyz == False:
            degree = float(self.label_j4.text()) + float(self.lineEdit_step.text())
            if -360 <= degree <= 360:
                self.jaka.blinx_jog(2, self.jaka.COORD_JOINT, 3, self.lineEdit_speed.text(), self.lineEdit_step.text())
        if self.joint_or_xyz == True:
            distance = float(self.label_j4.text()) + float(self.lineEdit_step.text())
            ref_pos = [0, 0, 90, 0, 90, 180]
            cartesian_pose = [float(self.label_j1.text()), float(self.label_j2.text()), float(self.label_j3.text()),
                              distance, float(self.label_j5.text()), float(self.label_j6.text())]
            ret = self.jaka.blinx_kine_inverse(ref_pos, cartesian_pose)
            if ret:
                self.jaka.blinx_jog(2, self.jaka.COORD_BASE, 3, self.lineEdit_speed.text(), self.lineEdit_step.text())
            else:
                QMessageBox.warning(mainWindow, "Error", "坐标不可达", QMessageBox.Ok)

    def btn_j4_subtract_click(self):
        if self.joint_or_xyz == False:
            degree = float(self.label_j4.text()) - float(self.lineEdit_step.text())
            if -360 <= degree <= 360:
                self.jaka.blinx_jog(2, self.jaka.COORD_JOINT, 3, -float(self.lineEdit_speed.text()),
                                    self.lineEdit_step.text())
        if self.joint_or_xyz == True:
            distance = float(self.label_j4.text()) - float(self.lineEdit_step.text())
            ref_pos = [0, 0, 90, 0, 90, 180]
            cartesian_pose = [float(self.label_j1.text()), float(self.label_j2.text()), float(self.label_j3.text()),
                              distance, float(self.label_j5.text()), float(self.label_j6.text())]
            ret = self.jaka.blinx_kine_inverse(ref_pos, cartesian_pose)
            if ret:
                self.jaka.blinx_jog(2, self.jaka.COORD_BASE, 3, -float(self.lineEdit_speed.text()),
                                    self.lineEdit_step.text())
            else:
                QMessageBox.warning(mainWindow, "Error", "坐标不可达", QMessageBox.Ok)

    def btn_j5_add_click(self):
        if self.joint_or_xyz == False:
            degree = float(self.label_j5.text()) + float(self.lineEdit_step.text())
            if -120 <= degree <= 120:
                self.jaka.blinx_jog(2, self.jaka.COORD_JOINT, 4, self.lineEdit_speed.text(), self.lineEdit_step.text())
        if self.joint_or_xyz == True:
            distance = float(self.label_j5.text()) + float(self.lineEdit_step.text())
            ref_pos = [0, 0, 90, 0, 90, 180]
            cartesian_pose = [float(self.label_j1.text()), float(self.label_j2.text()), float(self.label_j3.text()),
                              float(self.label_j4.text()), distance, float(self.label_j6.text())]
            ret = self.jaka.blinx_kine_inverse(ref_pos, cartesian_pose)
            if ret:
                self.jaka.blinx_jog(2, self.jaka.COORD_BASE, 4, self.lineEdit_speed.text(), self.lineEdit_step.text())
            else:
                QMessageBox.warning(mainWindow, "Error", "坐标不可达", QMessageBox.Ok)

    def btn_j5_subtract_click(self):
        if self.joint_or_xyz == False:
            degree = float(self.label_j5.text()) - float(self.lineEdit_step.text())
            if -120 <= degree <= 120:
                self.jaka.blinx_jog(2, self.jaka.COORD_JOINT, 4, -float(self.lineEdit_speed.text()),
                                    self.lineEdit_step.text())
        if self.joint_or_xyz == True:
            distance = float(self.label_j5.text()) - float(self.lineEdit_step.text())
            ref_pos = [0, 0, 90, 0, 90, 180]
            cartesian_pose = [float(self.label_j1.text()), float(self.label_j2.text()), float(self.label_j3.text()),
                              float(self.label_j4.text()), distance, float(self.label_j6.text())]
            ret = self.jaka.blinx_kine_inverse(ref_pos, cartesian_pose)
            if ret:
                self.jaka.blinx_jog(2, self.jaka.COORD_BASE, 4, -float(self.lineEdit_speed.text()),
                                    self.lineEdit_step.text())
            else:
                QMessageBox.warning(mainWindow, "Error", "坐标不可达", QMessageBox.Ok)

    def btn_j6_add_click(self):
        if self.joint_or_xyz == False:
            degree = float(self.label_j6.text()) + float(self.lineEdit_step.text())
            if -360 <= degree <= 360:
                self.jaka.blinx_jog(2, self.jaka.COORD_JOINT, 5, self.lineEdit_speed.text(), self.lineEdit_step.text())
        if self.joint_or_xyz == True:
            distance = float(self.label_j6.text()) + float(self.lineEdit_step.text())
            ref_pos = [0, 0, 90, 0, 90, 180]
            cartesian_pose = [float(self.label_j1.text()), float(self.label_j2.text()), float(self.label_j3.text()),
                              float(self.label_j4.text()), float(self.label_j5.text()), distance]
            ret = self.jaka.blinx_kine_inverse(ref_pos, cartesian_pose)
            if ret:
                self.jaka.blinx_jog(2, self.jaka.COORD_BASE, 5, self.lineEdit_speed.text(), self.lineEdit_step.text())
            else:
                QMessageBox.warning(mainWindow, "Error", "坐标不可达", QMessageBox.Ok)

    def btn_j6_subtract_click(self):
        if self.joint_or_xyz == False:
            degree = float(self.label_j6.text()) - float(self.lineEdit_step.text())
            if -360 <= degree <= 360:
                self.jaka.blinx_jog(2, self.jaka.COORD_JOINT, 5, -float(self.lineEdit_speed.text()),
                                    self.lineEdit_step.text())
        if self.joint_or_xyz == True:
            distance = float(self.label_j6.text()) - float(self.lineEdit_step.text())
            ref_pos = [0, 0, 90, 0, 90, 180]
            cartesian_pose = [float(self.label_j1.text()), float(self.label_j2.text()), float(self.label_j3.text()),
                              float(self.label_j4.text()), float(self.label_j5.text()), distance]
            ret = self.jaka.blinx_kine_inverse(ref_pos, cartesian_pose)
            if ret:
                self.jaka.blinx_jog(2, self.jaka.COORD_BASE, 5, -float(self.lineEdit_speed.text()),
                                    self.lineEdit_step.text())
            else:
                QMessageBox.warning(mainWindow, "Error", "坐标不可达", QMessageBox.Ok)

    def btn_clip_open_click(self):
        self.plc.clip_open()
        value = self.plc.clip_open_state()
        print("打开状态：", bool(value))

    # 判断夹爪是否打开完成
    def clip_open_isok(self):
        while True:
            time.sleep(0.1)
            try:
                if bool(self.plc.clip_open_state()):
                    print("夹爪打开成功")
                    break
                else:
                    print("夹爪未打开")
                    time.sleep(0.3)
            except Exception as e:
                print("夹爪打开失败")
                continue

    # 判断夹爪是否关闭完成
    def clip_close_isok(self):
        while True:
            time.sleep(0.1)
            try:
                if bool(self.plc.clip_close_state()):
                    print("夹爪关闭成功")
                    break
                else:
                    print("夹爪未关闭")
                    time.sleep(0.3)
            except Exception as e:
                print("夹爪关闭失败")
                continue

    # 判断是否回原点完成
    def servo_reset_isok(self):
        while True:
            time.sleep(0.1)
            try:
                if bool(self.plc.servo_reset_state()):
                    print("回原点完成")
                    break
                else:
                    print("等待回原点")
                    time.sleep(0.3)
            except Exception as e:
                continue

    # 判断滑轨移动是否完成
    def servo_absolute_finish_isok(self):
        while True:
            time.sleep(0.1)
            try:
                if bool(self.plc.servo_absolute_finish_state()):
                    print("滑轨移动完成")
                    break
                else:
                    print("滑轨移动中")
                    time.sleep(0.3)
            except Exception as e:
                continue

    def btn_step_add_click(self):
        if float(self.lineEdit_step.text()) <= 20:
            step = str(float(self.lineEdit_step.text()) + 1)
            self.lineEdit_step.setText(step)

    def btn_step_subtract_click(self):
        if float(self.lineEdit_step.text()) >= 0.1:
            step = str(float(self.lineEdit_step.text()) - 1)
            self.lineEdit_step.setText(step)

    def btn_speed_min_click(self):
        if self.joint_or_xyz == False:
            self.lineEdit_speed.setText('90')
        else:
            self.lineEdit_speed.setText('100')

    def btn_speed_max_click(self):
        if self.joint_or_xyz:
            self.lineEdit_speed.setText('1500')
        else:
            self.lineEdit_speed.setText('180')

    # 扎丝机开/关
    def btn_zsj_open_click(self):
        self.jaka.blinx_set_digital_output(6, 5, 1)
        self.jaka.blinx_set_digital_output(6, 7, 1)
        time.sleep(0.2)
        self.jaka.blinx_set_digital_output(6, 5, 0)

    def btn_zsj_close_click(self):
        self.jaka.blinx_set_digital_output(6, 5, 1)
        self.jaka.blinx_set_digital_output(6, 7, 0)
        time.sleep(0.2)
        self.jaka.blinx_set_digital_output(6, 5, 0)

    # 吸盘打开/关闭
    def btn_xipan_open_click(self):
        self.jaka.blinx_set_digital_output(6, 5, 1)
        self.jaka.blinx_set_digital_output(6, 6, 1)
        time.sleep(0.2)
        self.jaka.blinx_set_digital_output(6, 5, 0)
    def btn_xipan_close_click(self):
        self.jaka.blinx_set_digital_output(6, 5, 1)
        self.jaka.blinx_set_digital_output(6, 6, 0)
        time.sleep(0.2)
        self.jaka.blinx_set_digital_output(6, 5, 0)

    # 快换打开/关闭
    def btn_kh_open_click(self):
        self.jaka.blinx_set_digital_output(6, 5, 1)
        self.jaka.blinx_set_digital_output(6, 8, 1)
        time.sleep(0.2)
        self.jaka.blinx_set_digital_output(6, 5, 0)
    def btn_kh_close_click(self):
        self.jaka.blinx_set_digital_output(6, 5, 1)
        self.jaka.blinx_set_digital_output(6, 8, 0)
        time.sleep(0.2)
        self.jaka.blinx_set_digital_output(6, 5, 0)

    # 滑轨向左/向右
    def btn_servor_left_click(self):
        self.jaka.blinx_get_analog_input(6,25)
        time.sleep(0.01)
        print(self.public_class.ai_value)
        value=round(float(self.public_class.ai_value),2)-10
        self.jaka.blinx_set_analog_output(6, 26, 500)#滑轨绝对速度
        self.jaka.blinx_set_analog_output(6,25,value)#滑轨绝对位置
        self.jaka.blinx_set_digital_output(6, 5, 1)
        self.jaka.blinx_set_digital_output(6, 4, 1)
        self.jaka.blinx_set_digital_output(6, 4, 0)
        self.jaka.blinx_set_digital_output(6, 5, 0)


    def btn_servor_right_click(self):
        self.jaka.blinx_get_analog_input(6, 25)
        time.sleep(0.01)
        print(self.public_class.ai_value)
        value = round(float(self.public_class.ai_value), 2) + 10
        self.jaka.blinx_set_analog_output(6, 26, 500)
        self.jaka.blinx_set_analog_output(6, 25, value)
        self.jaka.blinx_set_digital_output(6, 5, 1)
        self.jaka.blinx_set_digital_output(6, 4, 1)
        self.jaka.blinx_set_digital_output(6, 4, 0)
        self.jaka.blinx_set_digital_output(6, 5, 0)
    # 滑轨启动/停止/回原点
    def btn_servor_run_click(self):
        9
    def btn_servor_reset_click(self):
        10
    def absolute_speed_changed(self):
        print(self.lineEdit_absolute_speed.text())

    def absolute_loaction_changed(self):
        print(self.lineEdit_absolute_loaction.text())

    # endregion
    # region 角度控制按钮
    def btn_joint_click(self):
        self.joint_or_xyz = False
        self.lab1.setText("关节1：")
        self.lab2.setText("关节2：")
        self.lab3.setText("关节3：")
        self.lab4.setText("关节4：")
        self.lab5.setText("关节5：")
        self.lab6.setText("关节6：")
        self.btn_j1_add.setText("J1+")
        self.btn_j1_subtract.setText("J1-")
        self.btn_j2_add.setText("J2+")
        self.btn_j2_subtract.setText("J2-")
        self.btn_j3_add.setText("J3+")
        self.btn_j3_subtract.setText("J3-")
        self.btn_j4_add.setText("J4+")
        self.btn_j4_subtract.setText("J4-")
        self.btn_j5_add.setText("J5+")
        self.btn_j5_subtract.setText("J5-")
        self.btn_j6_add.setText("J6+")
        self.btn_j6_subtract.setText("J6-")
        self.btn_speed_min.setText("90")
        self.btn_speed_max.setText("180")
        self.lineEdit_speed.setText("90")

    # endregion
    # region 坐标控制按钮
    def btn_xyz_click(self):
        self.joint_or_xyz = True
        self.lab1.setText("X  轴：")
        self.lab2.setText("Y  轴：")
        self.lab3.setText("Z  轴：")
        self.lab4.setText("RX 轴：")
        self.lab5.setText("RY 轴：")
        self.lab6.setText("RZ 轴：")
        self.btn_j1_add.setText("X+")
        self.btn_j1_subtract.setText("X-")
        self.btn_j2_add.setText("Y+")
        self.btn_j2_subtract.setText("Y-")
        self.btn_j3_add.setText("Z+")
        self.btn_j3_subtract.setText("Z-")
        self.btn_j4_add.setText("RX+")
        self.btn_j4_subtract.setText("RX-")
        self.btn_j5_add.setText("RY+")
        self.btn_j5_subtract.setText("RY-")
        self.btn_j6_add.setText("RZ+")
        self.btn_j6_subtract.setText("RZ-")
        self.btn_speed_min.setText("100")
        self.btn_speed_max.setText("1500")
        self.lineEdit_speed.setText("800")

    # endregion
    # region 判断字符串是否是数值类型
    def is_number(self, s):
        try:
            float(s)
            return True
        except Exception as e:
            print("数值转换整型错误:", e)
            pass
        try:
            import unicodedata
            unicodedata.numeric(s)
        except Exception as e:
            print("数值转换整型错误:", e)
        return False

    # endregion
    # region退出程序，关闭线程
    # region 关闭按钮
    def closeEvent(self, event):
        reply = QMessageBox.question(self, '本程序', "是否要退出程序？", QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            if self.public_class.mech_connected:
                self.mechCam.DisConnectCamera()
            pid = os.getpid()
            # SIGTERM表示终止进程
            os.kill(pid, signal.SIGTERM)
            event.accept()
        else:
            event.ignore()

    # endregion
    def _async_raise(self, tid, exctype):
        """raises the exception, performs cleanup if needed"""
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if res == 0:
            raise ValueError("invalid thread id")
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")

    def stop_thread(self, thread):
        self._async_raise(thread.ident, SystemExit)
    # endregion

# region 主程序运行
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    blinx_xbmd_robot_vision = Blinx_XXXY_Robot_Vision()
    blinx_xbmd_robot_vision.show()
    sys.exit(app.exec_())
# endregion
