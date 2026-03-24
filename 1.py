import cv2
import numpy as np


def detect_rectangles(image_path):
    # 读取灰度图像
    img_orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_orig is None:
        print("无法读取图像")
        return

    orig_height, orig_width = img_orig.shape[:2]
    scale_factor = 0.5  # 缩放因子，用于加速处理

    # 缩小图像进行处理
    img_resized = cv2.resize(img_orig, (0, 0), fx=scale_factor, fy=scale_factor)

    # 图像预处理
    blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学操作（可选，根据图像质量调整）
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []

    for cnt in contours:
        # 过滤小轮廓
        if cv2.contourArea(cnt) < 1000 * scale_factor:
            continue

        # 轮廓近似
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # 检测四边形
        if len(approx) == 4:
            # 获取最小外接矩形（带旋转）
            rect = cv2.minAreaRect(approx)
            (cx, cy), (w, h), angle = rect

            # 转换为原始图像坐标
            cx_orig = int(cx / scale_factor)
            cy_orig = int(cy / scale_factor)
            w_orig = int(w / scale_factor)
            h_orig = int(h / scale_factor)

            # 过滤非长方形（长宽比阈值）
            aspect_ratio = max(w_orig, h_orig) / (min(w_orig, h_orig) + 1e-5)
            if aspect_ratio < 1.2:  # 正方形过滤阈值
                continue

            # 存储结果（原始图像坐标）
            rectangles.append({
                'center': (cx_orig, cy_orig),
                'size': (w_orig, h_orig),
                'angle': angle
            })

    # 在原始图像上绘制结果
    result_img = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2BGR)
    for rect in rectangles:
        cx, cy = rect['center']
        w, h = rect['size']
        angle = rect['angle']

        # 绘制中心点
        cv2.circle(result_img, (int(cx), int(cy)), 10, (0, 0, 255), -1)

        # 绘制旋转矩形
        box = cv2.boxPoints(((cx, cy), (w, h), angle))
        box = np.int0(box)
        cv2.drawContours(result_img, [box], 0, (0, 255, 0), 3)

        # 显示角度和坐标
        text = f"Angle: {angle:.1f}°"
        cv2.putText(result_img, text, (int(cx) - 100, int(cy) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 创建自适应显示窗口
    window_name = "Rectangle Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 显示结果
    cv2.imshow(window_name, result_img)

    # 窗口自适应调整
    screen_res = 1920, 1080  # 根据你的屏幕分辨率调整
    scale_width = screen_res[0] / result_img.shape[1]
    scale_height = screen_res[1] / result_img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(result_img.shape[1] * scale)
    window_height = int(result_img.shape[0] * scale)
    cv2.resizeWindow(window_name, window_width, window_height)

    # 输出结果
    print(f"检测到 {len(rectangles)} 个长方形:")
    for i, rect in enumerate(rectangles, 1):
        print(f"矩形 {i}: 中心({rect['center'][0]}, {rect['center'][1]}), "
              f"角度: {rect['angle']:.2f}°, "
              f"尺寸: {rect['size'][0]:.1f}x{rect['size'][1]:.1f}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 使用示例
detect_rectangles("222.png")