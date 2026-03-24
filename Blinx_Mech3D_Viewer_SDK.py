import cv2
import numpy as np
from Blinx_Public_Class import *
import time
from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import *

class Blinx_Mech3D_Viewer_SDK(object):
    def __init__(self, public_class):
        self.public_class = public_class
        self.cam_index = -1
        self.camera = Camera()

    def connect_and_capture(self):
        # Obtain the 2D image resolution and the depth map resolution of the camera.
        # resolution = CameraResolutions()
        # show_error(self.camera.get_camera_resolutions(resolution))
        # print_camera_resolution(resolution)

        time1 = time.time()
        # Obtain the 2D image.
        frame2d = Frame2D()
        show_error(self.camera.capture_2d(frame2d))
        row, col = 222, 222
        color_map = frame2d.get_color_image()
        # print("The size of the 2D image is {} (width) * {} (height).".format(
        #     color_map.width(), color_map.height()))
        rgb = color_map[row * color_map.width() + col]
        # print("The RGB values of the pixel at ({},{}) is R:{},G:{},B{}\n".
        #       format(row, col, rgb.b, rgb.g, rgb.r))

        Image2d = color_map.data()
        time2 = time.time()
        # print('grab 2d image : ' + str((time2 - time1) * 1000) + 'ms')

        # if not confirm_capture_3d():
        #     return

        # Obtain the depth map.
        frame3d = Frame3D()
        show_error(self.camera.capture_3d(frame3d))
        depth_map = frame3d.get_depth_map()
        # print("The size of the depth map is {} (width) * {} (height).".format(
        #     depth_map.width(), depth_map.height()))
        depth = depth_map[row * depth_map.width() + col]
        # print("The depth value of the pixel at ({},{}) is depth :{}mm\n".
        #       format(row, col, depth.z))
        Image3d = depth_map.data()
        time3 = time.time()
        # print('grab depth image : ' + str((time3 - time2) * 1000) + 'ms')

        # Obtain the point cloud.
        point_cloud = frame3d.get_untextured_point_cloud()
        # print("The size of the point cloud is {} (width) * {} (height).".format(
        #     point_cloud.width(), point_cloud.height()))
        point_xyz = point_cloud[row * depth_map.width() + col]
        # print("The coordinates of the point corresponding to the pixel at ({},{}) is X: {}mm , Y: {}mm, Z: {}mm\n".
        #       format(row, col, point_xyz.x, point_xyz.y, point_xyz.z))
        time4 = time.time()
        # print('grab point_cloud image : ' + str((time4 - time3) * 1000) + 'ms')
        return Image2d, Image3d, point_xyz

    def GrabImages(self):
        d2, d3, point_xyz = self.connect_and_capture()
        return d2, d3, point_xyz

    def ConnectCamera(self):
        self.camera_infos = Camera.discover_cameras()
        for i in range(len(self.camera_infos)):
            if self.public_class.cam_sn==self.camera_infos[i].serial_number:
                self.cam_index=i
                self.public_class.mech_connected = True
            else:
                self.public_class.mech_connected = False
                print("查找对于序列号相机失败，检查相机连接")
                return
        error_status = self.camera.connect(self.camera_infos[self.cam_index])
        if not error_status.is_ok():
            show_error(error_status)
            self.public_class.mech_connected = False
            return
        else:
            self.public_class.mech_connected = True

    def main(self):
        # List all available cameras and connect to a camera by the displayed index.
        if find_and_connect(self.camera):
            d2, d3, point_xyz = self.connect_and_capture()
            self.camera.disconnect()
            print("Disconnected from the camera successfully.")
        return d2, d3, point_xyz
    def DisConnectCamera(self):
        self.camera.disconnect()
        print("Disconnected from the camera successfully.")
# if __name__ == "__main__":
#     public_class=Blinx_Public_Class()
#     print('初始化相机对象')
#     mech=Blinx_Mech3D_Viewer_SDK(public_class)
#     print('连接相机')
#     mech.ConnectCamera()
#     print(mech.public_class.mech_connected)
#     d2,d3,point_xyz=mech.GrabImages()
#     cv2.imwrite("1.jpg",d2)
