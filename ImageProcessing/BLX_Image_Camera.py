class Blinx_Image_Camera:
    def __init__(self):
        # 相机内参
        self.camera_fx = 2275.5910981121006  # x轴焦距
        self.camera_fy = 2275.5910981121006  # y轴焦距
        self.camera_cx = 1351.0913172620878  # 主点x坐标
        self.camera_cy = 884.5576714397653  # 主点y坐标

    """将图像坐标和深度转换为相机坐标系下的3D坐标

    参数:
        x (float/ndarray): 图像x坐标（像素）
        y (float/ndarray): 图像y坐标（像素）
        z (float/ndarray): 深度值（与Z坐标相同）

    返回:
        tuple/ndarray: 相机坐标系下的X, Y, Z坐标
    """

    def blinx_image_to_camera(self, x, y, z):

        z = z / 1000
        X = (x - self.camera_cx) * z / self.camera_fx
        Y = (y - self.camera_cy) * z / self.camera_fy
        Z = z
        return (X, Y, Z)