import os

from PIL import Image


# 定义 local_run 函数
def local_run():
    color_example_rgb = Image.open(
        r'/Users/ayi/Downloads/P1034285.JPG')
    color_example_rgb = np.array(color_example_rgb)
    block_positions = select_block_positions(color_example_rgb)
    save_file_name = f'canon200D2_block_pos.npy'
    if not os.path.exists('../calib/pos'):
        os.mkdir('../calib/pos')
    np.save(os.path.join('../calib/pos', save_file_name), block_positions)


# select each block and ret the positions
import cv2
import numpy as np


def select_block_positions(rgb_image, calib_num=24, block_width=100, block_height=100):
    """
    选择一个点，基于固定宽度和高度生成矩形。

    Args:
        rgb_image: 输入的RGB图像。
        calib_num: 要选择的点的数量。
        block_width: 固定的矩形宽度。
        block_height: 固定的矩形高度。

    Returns:
        positions: 包含所有矩形框的左上角点(x, y)及宽度和高度。
    """
    positions = []
    cv2.namedWindow("Selectable Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Selectable Image", 1000, 800)

    point_data = {"point": None, "clicked": False}

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 捕获鼠标左键点击
            print(f"点击点坐标: ({x}, {y})")
            param["point"] = (x, y)
            param["clicked"] = True

    cv2.setMouseCallback("Selectable Image", mouse_callback, point_data)

    print(f"请依次选择 {calib_num} 个点，用于生成固定宽高的矩形框。")

    while len(positions) < calib_num:
        cv2.imshow("Selectable Image", rgb_image)
        key = cv2.waitKey(1)

        if point_data["clicked"]:  # 如果用户点击了点
            x, y = point_data["point"]
            rect = (int(x - block_width / 2), int(y - block_height / 2), block_width, block_height)

            # 在图像上绘制矩形框
            cv2.rectangle(rgb_image, (rect[0], rect[1]),
                          (rect[0] + rect[2], rect[1] + rect[3]),
                          color=(255, 255, 0), thickness=2)

            print(f"生成矩形框: (x={rect[0]}, y={rect[1]}, w={block_width}, h={block_height})")
            print('---------------')

            positions.append(rect)
            point_data["clicked"] = False  # 重置点击状态

    cv2.destroyAllWindows()  # 关闭所有窗口
    return np.array(positions)

if __name__ == "__main__":
    local_run()