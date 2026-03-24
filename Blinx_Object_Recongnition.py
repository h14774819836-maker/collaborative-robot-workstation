import configparser
import cv2
import numpy as np
import time
import math
class Blinx_Object_Recongnition():
    def __init__(self,public_class):
        self.public_class = public_class
    def recognition(self, image):
        all_result_list = []
        result_list = []
        src = image.copy()
        top_left_pt=(302,289)#矩形左上角
        bottom_right_pt=(2381,1843)#矩形右下角
        src_img_roi=image[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]]#ROI
        # 高斯滤波
        gauss = cv2.GaussianBlur(src_img_roi, (11, 11), 2)
        # 边缘保留滤波EPF  去噪
        blur = cv2.pyrMeanShiftFiltering(gauss, sp=15, sr=25)
        blur_img_roi=blur.copy()
        r, g, b = cv2.split(blur)  # RGB分割
        rg_diff = cv2.absdiff(r, g)  # 图像做差
        gb_diff = cv2.absdiff(g, b)
        rb_diff = cv2.absdiff(r, b)
        v1 = cv2.mean(rg_diff)[0]  # 求平均值
        v2 = cv2.mean(gb_diff)[0]
        v3 = cv2.mean(rb_diff)[0]
        v_max = max(v1, max(v2, v3))  # 求最大值
        print(v_max)
        if v_max > 2:
            if v1 == v_max:
                image_diff = rg_diff
            elif v2 == v_max:
                image_diff = gb_diff
            elif v3 == v_max:
                image_diff = rb_diff
        else:
            image_diff = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
        # cv2.namedWindow("image_diff", 0)
        # cv2.resizeWindow("image_diff", 680, 400)  # 设置窗口大小
        # cv2.imshow('image_diff', image_diff)

        # 得到二值图像区间阈值
        ret, binary = cv2.threshold(image_diff, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
        # cv2.namedWindow("thres image", 0)
        # cv2.resizeWindow("thres image", 680, 400)  # 设置窗口大小
        # cv2.imshow('thres image', binary)

        # 距离变换
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        dist_out = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)
        # cv2.namedWindow("distance-Transform", 0)
        # cv2.resizeWindow("distance-Transform", 680, 400)  # 设置窗口大小
        # cv2.imshow('distance-Transform', dist_out * 100)
        ret, surface = cv2.threshold(dist_out, 0.5 * dist_out.max(), 255, cv2.THRESH_BINARY)
        # cv2.namedWindow("surface", 0)
        # cv2.resizeWindow("surface", 680, 400)  # 设置窗口大小
        # cv2.imshow('surface', surface)
        sure_fg = np.uint8(surface)  # 转成8位整型
        # cv2.namedWindow("Sure foreground", 0)
        # cv2.resizeWindow("Sure foreground", 680, 400)  # 设置窗口大小
        # cv2.imshow('Sure foreground', sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)  # 连通区域
        # print(ret)
        markers = markers + 1  # 整个图+1，使背景不是0而是1值

        # 未知区域标记(不能确定是前景还是背景)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=1)
        unknown = binary - sure_fg
        # cv2.namedWindow("unknown", 0)
        # cv2.resizeWindow("unknown", 680, 400)  # 设置窗口大小
        # cv2.imshow('unknown', unknown)

        # 未知区域标记为0
        markers[unknown == 255] = 0
        # 区域标记结果
        markers_show = np.uint8(markers)
        # cv2.namedWindow("markers", 0)
        # cv2.resizeWindow("markers", 680, 400)  # 设置窗口大小
        # cv2.imshow('markers', markers_show * 100)

        # 分水岭算法分割
        markers = cv2.watershed(blur_img_roi, markers=markers)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(markers)
        markers_8u = np.uint8(markers)
        # print(max_val)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255), (255, 128, 0), (255, 0, 128),
                  (128, 255, 0), (128, 0, 255), (255, 128, 128), (128, 255, 255)]
        for i in range(2, int(max_val + 1)):
            ret, thres1 = cv2.threshold(markers_8u, i - 1, 255, cv2.THRESH_BINARY)
            ret2, thres2 = cv2.threshold(markers_8u, i, 255, cv2.THRESH_BINARY)
            mask = thres1 - thres2
            # cv2.namedWindow('mask', 0)
            # cv2.resizeWindow('mask', 680, 400)  # 设置窗口大小
            # cv2.imshow('mask', mask)
            # color = (rd.randint(0,255), rd.randint(0,255), rd.randint(0,255))
            # image[markers == i] = [rd.randint(0,255), rd.randint(0,255), rd.randint(0,255)]
            # image[markers == i] = [colors[i-2]]
            # 定义结构元素
            kernel = np.ones((11, 11), np.uint8)
            # 执行腐蚀操作
            erosion = cv2.erode(mask, kernel, iterations=1)
            contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # cv2.drawContours(image, contours, -1, colors[(i - 2) % 12], 2)
            for contour in contours:
                contour = contour + np.array([[top_left_pt[0], top_left_pt[1]]])
                cv2.drawContours(src, [contour], -1, colors[(i - 2) % 12], 2)
            # 循环轮廓，判断每一个形状
            for cnt in contours:
                # 获取轮廓面积
                area = cv2.contourArea(cnt)
                print("轮廓像素面积:", area)  # 打印所有轮廓面积
                # 去掉没有分割的区域
                if int(self.public_class.minArea) < area < int(self.public_class.maxArea):
                    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)  # 拟合的多边形的边数
                    if len(approx)==3:#三角形
                        vertices = approx.reshape(-1, 2)
                        print("Vertices of the triangle:",vertices)
                        max_x_point = vertices[0]
                        max_y_point = vertices[0]
                        # 遍历所有顶点
                        for point in vertices:
                            # 检查X坐标最大的点
                            if point[0] > max_x_point[0]:
                                max_x_point = point
                            # 检查Y坐标最大的点
                            if point[1] > max_y_point[1]:
                                max_y_point = point
                        theta_degrees_thri=self.calculate_angle(max_x_point,max_y_point)
                        print("正三角形右上和右下与水平线的夹角:", theta_degrees_thri, "度")
                    # print("几个角：", len(approx))
                    rect = cv2.minAreaRect(approx)  # 最小外接矩形
                    # box = cv2.boxPoints(rect)  # boxPoints返回四个点顺序：右下→左下→左上→右上
                    # box = np.intp(box)
                    # # 将ROI的左上角坐标(x, y)加到每个顶点的坐标上
                    # box[:, 0] += top_left_pt[0]
                    # box[:, 1] += top_left_pt[1]
                    # cv2.drawContours(src, [box], 0, (255, 255, 255), 5)  # 画出多边形形状
                    angle = rect[2]  # 旋转角度
                    if len(approx)==3:#三角形
                        angle=theta_degrees_thri
                    elif len(approx)==6:
                        # 如果角度大于45度，则减去90度，因为正六边形的长边和短边比例是固定的
                        if angle > 45:
                            angle = 90 - angle
                    # 获取轮廓重心
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:  # 轮廓重心
                        cx = int(M['m10'] / M['m00'])+top_left_pt[0]
                        cy = int(M['m01'] / M['m00'])+top_left_pt[1]
                    else:
                        # 避免除以零
                        continue
                    # print("M:", cx, cy)
                    hsvRoi = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                    mean = hsvRoi[cy, cx]
                    print("mean:", mean)
                    H = mean[0]
                    S = mean[1]
                    V = mean[2]
                    color=None
                    if (((H >= 0 and H <= 10) or (
                            H >= 80 and H <= 180)) and S >= 43 and S <= 255 and V >= 46 and V <= 255):
                        color = "red"
                    elif (H >= 20 and H <= 34 and S >= 43 and S <= 255 and V >= 46 and V <= 255):
                        color = "yellow"
                    else:
                        clolr = None
                    cv2.putText(src, color, (cx + 10, cy + 30), 0, 1, (0, 255, 0), 2)  # 标注颜色
                    cv2.drawMarker(src, (cx, cy), (255, 255, 255), 1, 10, 2)  # 标注中心点
                    cv2.putText(src, str(len(approx)), (cx + 10, cy - 30), 0, 1, (0, 255, 0), 2)  # 标注形状边长
                    cv2.putText(src, "(" + str(cx) + "," + str(cy) + ")", (cx - 200, cy + 10), 0, 1, (0, 255, 0),
                                2)  # 标注中心点坐标像素
                    cv2.putText(src, str(round(angle, 2)), (cx + 10, cy + 90), 0, 1, (0, 255, 0), 2)  # 标注角度
                    result_list = [(cx, cy), color, len(approx), round(angle, 2), area]  # 中心，颜色，形状，角度,像素面积
                    all_result_list.append(result_list)

        cv2.putText(src, "count=%d" % (int(max_val - 1)), (220, 30), 0, 1, (0, 255, 0), 2)  # 标注识别个数
        # cv2.putText(image, "count=%d" % (int(max_val - 1)), (220, 30), 0, 1, (0, 255, 0), 2)
        # cv2.namedWindow('regions', 0)
        # cv2.resizeWindow('regions', 680, 400)  # 设置窗口大小
        # cv2.imshow('regions', image)
        result_image = cv2.addWeighted(src, 0.6, image, 0.5, 0)  # 图像权重叠加
        now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))  # 获取系统时间
        if len(str(all_result_list))>0:
            print("all_result_list_ori:",all_result_list)
            img=cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
            cv2.imwrite('pic/result/result_' + str(now_time) + '.jpg', img)  # 保存图片
        return all_result_list, result_image

    def calculate_angle(self,line_start, line_end):
        """
        计算两点之间连线与水平线的夹角。
        参数:
        line_start (tuple): 连线的起点坐标 (x1, y1)。
        line_end (tuple): 连线的终点坐标 (x2, y2)。
        返回:
        float: 两点连线与水平线的夹角，范围在0到180度。
        """
        # 计算两点之间的向量
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        # 计算与水平线的夹角（使用atan2函数，返回值是弧度）
        angle_rad = math.atan2(dy, dx)
        # 将弧度转换为度
        angle_deg = math.degrees(angle_rad)
        # 确保角度在0到180度之间
        if angle_deg <=0:
            angle_deg += 180
        elif angle_deg >=180:
            angle_deg = angle_deg - 180
        return angle_deg

if __name__ == '__main__':
    obj=Blinx_Object_Recongnition()
    img=cv2.imread('2.bmp')
    list,result=obj.recognition(img)
    cv2.namedWindow('result', 0)
    cv2.resizeWindow('result', 680, 400)  # 设置窗口大小
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()