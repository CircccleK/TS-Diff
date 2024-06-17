import os

import cv2

# pic_path = "/home/ly/RepDiff/experiments/pic/pic_QID_finetune/10-3_s12_iso16000_exp0.04_2/"
pic_path = "/home/ly/RepDiff/experiments/pic/pic_QID/10-3_s12_iso16000_exp0.04_2/"


for pic in os.listdir(pic_path):
    if "zoomed" in pic:
        continue
    image_path = os.path.join(pic_path, pic)  # 替换为你的图像路径
    image = cv2.imread(image_path)

    # roi_x, roi_y, roi_w, roi_h = 505, 207, 100, 100 # 10-1_s1_iso2000_exp0.01_5_show.png
    # roi_x, roi_y, roi_w, roi_h = 648, 195, 100, 100 # 10-2_s17_iso16000_exp0.03_2
    roi_x, roi_y, roi_w, roi_h = 595, 256, 100, 100 # 100_scene-5_IMG_009

    roi = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # 放大区域：使用resize放大，例如放大2倍
    scale_factor = 3
    roi_resized = cv2.resize(roi, (roi_w * scale_factor, roi_h * scale_factor), interpolation=cv2.INTER_CUBIC)

    # 在原图上绘制绿色边框标记放大区域
    color = (0, 255, 0)  # 绿色边框
    thickness = 2
    cv2.rectangle(image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), color, thickness)

    # 在放大后的区域添加到原图上，选择左下角或右上角进行嵌入
    image_h, image_w = image.shape[:2]

    # 计算放大区域嵌入的起始坐标（左下角）
    embed_x = 0  # 距离左边缘 10 像素
    embed_y = image_h - roi_resized.shape[0]  # 距离底部 10 像素

    # 在左下角嵌入放大区域
    image[embed_y:embed_y + roi_resized.shape[0], embed_x:embed_x + roi_resized.shape[1]] = roi_resized

    # 保存结果
    cv2.imwrite(os.path.join(pic_path, os.path.basename(pic).split('.png')[0] + "_zoomed.png"), image)