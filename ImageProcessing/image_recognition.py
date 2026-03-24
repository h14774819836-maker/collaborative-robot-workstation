import argparse
from ImageProcessing.yolov8_onnx import *
from ImageProcessing.yolov8_onnx_seg import *
from ImageProcessing.yolov8_onnx2 import *

class Blinx_image_rec():
    def __init__(self, public_class):
        self.public_class = public_class

        # 目标检测
        self.model_path = "ImageProcessing/gangjin2-m.onnx"
        # 加载模型
        self.detection = YOLOv8(self.model_path)
        # 模型初始化
        self.session, self.model_inputs = self.detection.init_detect_model()
        # 目标检测
        self.model_path1 = "ImageProcessing/hongzhuan-detect.onnx"
        # 加载模型
        self.detection1 = YOLOv82(self.model_path1)
        # 模型初始化
        self.session1, self.model_inputs1 = self.detection1.init_detect_model()



        # 实例分割
        # 模型路径
        self.model_path2 = "ImageProcessing/hongzhuan-seg2.onnx"
        # 实例化模型
        self.model2 = YOLOv8Seg(self.model_path2)
        # 置信度
        self.conf = 0.65
        self.iou = 0.45


    # 钢筋图像识别
    def blinx_rebar_image_rec(self, image):
        output_image, result_list = self.detection.detect(self.session, self.model_inputs, image)
        print(result_list)
        data = []
        for i in range(len(result_list)):
            res = []
            res.append(result_list[i][2])
            res.append(result_list[i][3])
            data.append(res)
        
        return output_image, data

    # 瓷砖与墙砖二次识别
    def blinx_brickandporcelain_image_rec(self, image):
        output_image, result_list = self.detection1.detect(self.session1, self.model_inputs1, image)
        print(result_list)
        data = []
        data.append(result_list[0][2])
        data.append(result_list[0][3])
        return output_image, data

    # 砖块图像识别
    def blinx_brick_image_rec(self, image, mech_depth_map):
        # 推理
        boxes, segments, _ = self.model2(image, conf_threshold=self.conf, iou_threshold=self.iou)
        # 画图
        if len(boxes) > 0:
            # 单次拍照同种类别只有一个
            # output_image, dict_label_seg = model.draw_and_visualize(img, boxes, segments, vis=False, save=True)
            # print(dict_label_seg)
            # 单次拍照同种类别存在多个
            output_image, data_list, vertices_list = self.model2.draw_and_visualize_seg(
                image, boxes, segments, vis=False, save=True)
            print(data_list)
            data = []
            data.append(data_list[0][1][0])
            data.append(data_list[0][1][1])
            depth_num = mech_depth_map[int(data_list[0][1][1]), int(data_list[0][1][0])]
            data.append(depth_num)
            data.append(data_list[0][2])
            return output_image, data
        else:
            return image, None

    # 瓷砖二次识别
    def blinx_image_gain(self, img):
        try:
            # 保存原始图像引用
            orig_img = img.copy()

            # 缩放处理
            scale_percent = 25
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            # 图像预处理
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)

            # 轮廓检测
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 存储矩形信息：中心点、角度、轮廓
            rectangles_info = []

            for cnt in contours:
                # 轮廓近似
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

                # 四边形筛选
                if len(approx) == 4:
                    # 计算面积
                    area = cv2.contourArea(approx)

                    # 排除太小或太大的轮廓
                    if area < 100 or area > (width * height * 0.5):
                        continue

                    # 计算凸性，确保是凸四边形
                    if not cv2.isContourConvex(approx):
                        continue

                    # 计算最小外接矩形
                    rect = cv2.minAreaRect(approx)
                    (cx, cy), (w, h), angle = rect

                    # 计算宽高比
                    aspect_ratio = max(w, h) / (min(w, h) + 1e-5)

                    # 几何特征验证
                    if 0.7 < aspect_ratio < 1.4 and area > 1000:
                        # 还原原始坐标
                        scale_factor = 100 / scale_percent
                        cx_orig = int(cx * scale_factor)
                        cy_orig = int(cy * scale_factor)
                        angle_orig = angle

                        # 获取旋转矩形的四个顶点（原始尺寸）
                        box = cv2.boxPoints(((cx_orig, cy_orig), (w * scale_factor, h * scale_factor), angle_orig))
                        box = np.int0(box)

                        # 存储矩形信息
                        rectangles_info.append({
                            'center': (cx_orig, cy_orig),
                            'angle': angle_orig,
                            'contour': box,
                            'size': (int(w * scale_factor), int(h * scale_factor))
                        })

            # 绘制结果
            for info in rectangles_info:
                # 绘制旋转矩形
                cv2.drawContours(orig_img, [info['contour']], 0, (0, 255, 0), 2)

                # 绘制中心点
                cv2.circle(orig_img, info['center'], 5, (0, 0, 255), -1)

                # 绘制角度指示线
                angle_rad = math.radians(info['angle'])
                line_length = 50
                end_x = int(info['center'][0] + line_length * math.cos(angle_rad))
                end_y = int(info['center'][1] + line_length * math.sin(angle_rad))
                cv2.line(orig_img, info['center'], (end_x, end_y), (255, 0, 0), 2)

                # 显示角度和中心点信息
                text = f"Angle: {info['angle']:.1f}°"
                cv2.putText(orig_img, text, (info['center'][0] + 10, info['center'][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            print(rectangles_info[0]['center'][0], rectangles_info[0]['center'][1], rectangles_info[0]['angle'])
            # 打印检测到的矩形信息
            for i, info in enumerate(rectangles_info):
                print(f"矩形 {i + 1}:")
                print(f"  中心点: ({info['center'][0]}, {info['center'][1]})")
                print(f"  角度: {info['angle']:.2f}°")
                print(f"  尺寸: {info['size'][0]}x{info['size'][1]} 像素")

            return orig_img, rectangles_info[0]['center'][0], rectangles_info[0]['center'][1], rectangles_info[0]['angle']
        except Exception as e:
            print(e)

    # 红砖二次识别
    def blinx_image_gain2(self, image):
        # 调整图像大小（可根据实际情况调整尺寸）
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

        # 预设ROI区域（x, y, width, height）
        # 如需手动选择，将此部分注释掉
        roi = (212, 60, 939, 484)  # 示例预设区域，根据实际图像调整

        # 手动选择ROI（取消下面注释启用）
        # print("请在图像上框选感兴趣区域，选好后按Enter确认")
        # roi = cv2.selectROI("选择ROI区域", image, False)
        # cv2.destroyWindow("选择ROI区域")

        # 提取ROI
        x, y, w, h = roi
        roi_image = image[y:y + h, x:x + w]

        # 图像去噪处理
        roi_image = cv2.GaussianBlur(roi_image, (5, 5), 0)
        roi_image = cv2.medianBlur(roi_image, 5)
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        # 二值化处理
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 寻找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            if area < 100000:
                continue
            # 计算最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # 将ROI内的坐标转换为原图坐标
            for i in range(len(box)):
                box[i][0] += x
                box[i][1] += y

            # 获取矩形的中心点坐标（转换为原图坐标）
            center = (int(rect[0][0] + x), int(rect[0][1] + y))
            # 获取矩形的角度
            angle = rect[2]

            # 在原图上绘制矩形和中心点
            cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
            cv2.circle(image, center, 5, (0, 255, 0), -1)

            # 在中心点上方显示角度信息
            cv2.putText(image, f"Angle: {angle:.1f} deg",
                        (center[0], center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            print(f"矩形中心点：({center[0]}, {center[1]}), 角度：{angle} 度")
            return image, center[0], center[1], angle

    # 砖块图像识别
    def blinx_brick_image_rec2(self, image, mech_depth_map):
        # 推理
        boxes, segments, _ = self.model2(image, conf_threshold=self.conf, iou_threshold=self.iou)
        # 画图
        if len(boxes) > 0:
            output_image, data_list, vertices_list = self.model2.draw_and_visualize_seg(
                image, boxes, segments, vis=False, save=True)
            print(data_list)
            print(vertices_list[0][0][0])
            dataS = []
            for i in range(len(vertices_list)):
                # 角度技计算
                corners = [
                    (vertices_list[i][0][0], vertices_list[i][0][1]), (vertices_list[i][1][0], vertices_list[i][1][1]),
                    (vertices_list[i][2][0], vertices_list[i][2][1]), (vertices_list[i][3][0], vertices_list[i][3][1])
                ]
                angle = self.calculate_rectangle_angle(corners)
                data = []
                data.append(data_list[i][1][0])
                data.append(data_list[i][1][1])
                depth_num = mech_depth_map[int(data_list[i][1][1]), int(data_list[i][1][0])]
                data.append(depth_num)
                data.append(angle)
                dataS.append(data)
            if len(dataS) > 0:
                sorted(dataS, key=lambda x: x[2], reverse=True)
            return output_image, dataS[0]
        else:
            return image, None
    # 红砖角度转型
    def calculate_rectangle_angle(self, corners):
        """
        计算长方形长边与图像水平方向的平行角度

        参数:
        corners: 包含四个角点的列表，每个角点是一个二元组 (x, y)
                 角点的顺序可以是任意的

        返回:
        angle: 长边与图像水平方向的夹角，单位为度，范围从 -45 到 45 度
        """
        # 如果角点数量不足4个，抛出异常
        if len(corners) != 4:
            raise ValueError("需要提供四个角点")

        # 将角点转换为 numpy 数组
        corners = np.array(corners, dtype=np.float32)

        # 计算所有边的长度
        edges = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            edge_length = np.sqrt(np.sum((p1 - p2) ** 2))
            edges.append((edge_length, (p1, p2)))

        # 按边长排序，长边在前
        edges.sort(key=lambda x: x[0], reverse=True)

        # 获取最长的边
        longest_edge = edges[0][1]
        p1, p2 = longest_edge

        # 计算长边的向量
        vector = p2 - p1

        # 计算向量与水平方向的夹角（弧度）
        angle_rad = np.arctan2(vector[1], vector[0])

        # 将弧度转换为度
        angle_deg = np.degrees(angle_rad)

        # 将角度归一化到 -45 到 45 度之间
        angle_deg = angle_deg % 180
        if angle_deg >= 90:
            angle_deg -= 180



        return angle_deg
