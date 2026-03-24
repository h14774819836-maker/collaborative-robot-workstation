import numpy as np
import math
from scipy.spatial.transform import Rotation

class Blinx_3D_Conversion():
    def blinx_conversion(self, CalObjInCamPose):
        # 初始化位姿数据
        CalObjInCamPose = np.array([CalObjInCamPose[0], CalObjInCamPose[1], CalObjInCamPose[2], 359.155, 359.807,
                                    270.864])  # 从CalObjInCamPose.dat读取的实际数据

        # 机械手在基座坐标系中的位姿（毫米和度）
        CamTransX = 115.190  # mm
        CamTransY = 335.361  # mm
        CamTransZ = 429.177  # mm
        CamRotX = 180.00  # deg
        CamRotY = 0.00  # deg
        CamRotZ = -42.878  # deg

        # 角度规范化到[0, 360)
        CamRotY = CamRotY + 360 if CamRotY < 0 else CamRotY
        CamRotX = CamRotX + 360 if CamRotX < 0 else CamRotX
        CamRotZ = CamRotZ + 360 if CamRotZ < 0 else CamRotZ

        # 将角度转换为弧度
        CamRot_aerfa = math.radians(CamRotZ)
        CamRot_beita = math.radians(CamRotY)
        CamRot_r = math.radians(CamRotX)

        # 计算旋转矩阵元素
        sin_aerfa = math.sin(CamRot_aerfa)
        cos_aerfa = math.cos(CamRot_aerfa)
        sin_beita = math.sin(CamRot_beita)
        cos_beita = math.cos(CamRot_beita)
        sin_r = math.sin(CamRot_r)
        cos_r = math.cos(CamRot_r)

        r11 = cos_aerfa * cos_beita
        r12 = cos_aerfa * sin_beita * sin_r - sin_aerfa * cos_r
        r13 = cos_aerfa * sin_beita * cos_r + sin_aerfa * sin_r
        r21 = sin_aerfa * cos_beita
        r22 = sin_aerfa * sin_beita * sin_r + cos_aerfa * cos_r
        r23 = sin_aerfa * sin_beita * cos_r - cos_aerfa * sin_r
        r31 = -sin_beita
        r32 = cos_beita * sin_r
        r33 = cos_beita * cos_r

        # 构建齐次变换矩阵
        HomMat3DIdent_Rxyz = np.identity(4)
        HomMat3DIdent_Rxyz[0, 0] = r11
        HomMat3DIdent_Rxyz[0, 1] = r12
        HomMat3DIdent_Rxyz[0, 2] = r13
        HomMat3DIdent_Rxyz[1, 0] = r21
        HomMat3DIdent_Rxyz[1, 1] = r22
        HomMat3DIdent_Rxyz[1, 2] = r23
        HomMat3DIdent_Rxyz[2, 0] = r31
        HomMat3DIdent_Rxyz[2, 1] = r32
        HomMat3DIdent_Rxyz[2, 2] = r33
        HomMat3DIdent_Rxyz[0, 3] = CamTransX / 1000.0  # 转换为米
        HomMat3DIdent_Rxyz[1, 3] = CamTransY / 1000.0  # 转换为米
        HomMat3DIdent_Rxyz[2, 3] = CamTransZ / 1000.0  # 转换为米

        # 手眼标定参数
        CamTransX = -0.069025
        CamTransY = -0.127719
        CamTransZ = 0.003279
        q1, q2, q3, q4 = 0.927010, 0.008221, -0.000312, -0.374946

        # 四元数转旋转矩阵
        rot = Rotation.from_quat([q2, q3, q4, q1])  # 注意四元数顺序: [x, y, z, w]
        rotation_matrix = rot.as_matrix()

        # 构建工具在相机中的齐次变换矩阵
        ToolInCam_mat = np.eye(4)
        ToolInCam_mat[:3, :3] = rotation_matrix
        ToolInCam_mat[0, 3] = CamTransX
        ToolInCam_mat[1, 3] = CamTransY
        ToolInCam_mat[2, 3] = CamTransZ

        # 坐标变换计算
        # obj_in_Base = ToolInBase * ToolInCam * CalObjInCam
        compose = HomMat3DIdent_Rxyz @ ToolInCam_mat

        # 标定板在相机中的位姿转换为齐次矩阵
        CalObjInCam_mat = np.eye(4)
        CalObjInCam_mat[:3, 3] = CalObjInCamPose[:3]  # 平移部分
        rx, ry, rz = CalObjInCamPose[3:]
        rot_mat = Rotation.from_euler('xyz', [rx, ry, rz]).as_matrix()
        CalObjInCam_mat[:3, :3] = rot_mat

        # 计算标定板在基座中的位姿
        obj_in_Base_mat = compose @ CalObjInCam_mat

        # 提取位置 (米转毫米)
        position = obj_in_Base_mat[:3, 3] * 1000

        # 提取欧拉角 (ZYX顺序)
        rotation = Rotation.from_matrix(obj_in_Base_mat[:3, :3])
        angles = rotation.as_euler('zyx', degrees=True)

        # 角度规范化到[-180, 180]
        angles = np.where(angles > 180, angles - 360, angles)
        angles = np.where(angles < -180, angles + 360, angles)

        # 添加偏移量 (可选)
        ShiftX, ShiftY, ShiftZ = 0, 0, 0
        position[0] += ShiftX
        position[1] += ShiftY
        position[2] += ShiftZ

        # 最终位姿结果
        Pose_Res = {
            'X': position[0],
            'Y': position[1],
        }

        print("标定板在基座中的位姿:")
        print(f"X: {Pose_Res['X']:.3f} mm")
        print(f"Y: {Pose_Res['Y']:.3f} mm")

        return Pose_Res