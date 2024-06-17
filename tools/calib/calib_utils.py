import os

import cv2
import numpy as np
import rawpy
from matplotlib import pyplot as plt


def pack_raw_bayer(raw: np.ndarray, raw_pattern: np.ndarray):
    #pack Bayer image to 4 channels
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)

    raw = raw.astype(np.uint16)
    out = np.stack((raw[R[0][0]::2,  R[1][0]::2], #RGBG
                    raw[G1[0][0]::2, G1[1][0]::2],
                    raw[B[0][0]::2,  B[1][0]::2],
                    raw[G2[0][0]::2, G2[1][0]::2]), axis=0).astype(np.uint16)

    return out


# def bayer2rggb(bayer):
#     H, W = bayer.shape
#     return bayer.reshape(H//2, 2, W//2, 2).transpose(0, 2, 1, 3).reshape(H//2, W//2, 4)

# read raw and return rgb
def raw2rgb(raw_path):
    raw_image = rawpy.imread(raw_path)
    rgb_image = raw_image.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
    return rgb_image
    # imageio.imsave('output_rgb_image.jpg', rgb_image)



# def load_rgb_image(rgb_root):
#     rgb_path = os.path.join(rgb_root,"raw_visual")
#     filenames = os.listdir(rgb_path)
#     return cv2.imread(os.path.join(rgb_path,filenames[0]))


# Obtain and store the color block positions for a specific camera's color image
def get_block_positoins(cam_dir,iso=100):
    raw_path = os.path.join(os.path.join(cam_dir,'color'),str(iso))
    raw_imgs = sorted(os.listdir(raw_path))
    if '.DS_Store' in raw_imgs:
        raw_imgs.remove('.DS_Store')
    example_raw_img = os.path.join(raw_path,raw_imgs[0])
    color_example_rgb = raw2rgb(example_raw_img)
    block_positions = select_block_positions(color_example_rgb,calib_num=24)
    cam_name = cam_dir.split('/')[-1]
    save_file_name = f'{cam_name}_block_pos.npy'
    if not os.path.exists('./pos'):
        os.mkdir('./pos')
    np.save(os.path.join('./pos',save_file_name),block_positions)


# select each block and ret the positions
def select_block_positions(rgb_image,calib_num=24):
    positions = []
    for i in range(calib_num):
        # rect = cv2.selectROI(rgb_image,False,False)
        rect = cv2.selectROI(rgb_image,showCrosshair=True)
        cv2.rectangle(rgb_image,rect,color=(23,128,62))
        print(i)
        print(rect)
        print('---------------')
        positions.append(list(rect))

    return np.array(positions)

def select_whole_positions(rgb_image):
    rect = cv2.selectROI(rgb_image,showCrosshair=True)
    cv2.rectangle(rgb_image,rect,color=(23,128,62))

    return rect


def linear_regression(x, y): 
    return np.polyfit(x, y, 1)


from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import numpy as np


def regr_plot(x, y, log=True, ax=None, xlabel="", ylabel="", title="", c1=None, c2=None, label=False, noise_type="r"):
    """
    绘制线性回归散点图及拟合直线，支持动态调整噪声类型参数名。

    Args:
        x (list/np.array): 自变量数据
        y (list/np.array): 因变量数据
        log (bool): 是否对数据取对数
        xlabel (str): 横坐标标签
        ylabel (str): 纵坐标标签
        title (str): 图标题
        c1 (str): 拟合直线颜色
        c2 (str): 置信区间颜色
        label (bool): 是否显示图例
        noise_type (str): 噪声类型 ('r' 表示行噪声，'TL' 表示读出噪声)

    Returns:
        ax (matplotlib.axes._axes.Axes): 图形对象
        data (dict): 拟合参数和评估指标
    """
    x = np.array(x)
    y = np.array(y)
    if log:
        x = np.log(x)
        y = np.log(y)

    if ax is None:
        fig, ax = plt.subplots()

    # Scatter plot
    ax.scatter(x, y, label="Data", color="blue")

    # Linear regression
    regr = LinearRegression()
    regr.fit(x.reshape(-1, 1), y)
    a, b = float(regr.coef_), float(regr.intercept_)
    x_range = np.linspace(np.min(x), np.max(x), 100)
    y_pred = regr.predict(x.reshape(-1, 1))
    r2 = r2_score(y, y_pred)
    std = np.mean((a * x + b - y) ** 2) ** 0.5

    # 动态调整参数名
    if noise_type == "r":
        k_label = f"$k_r$"
        b_label = f"$b_r$"
        sigma_label = f"$\\sigma_r$"
    elif noise_type == "TL":
        k_label = f"$k_{{TL}}$"
        b_label = f"$b_{{TL}}$"
        sigma_label = f"$\\sigma_{{TL}}$"
    else:
        raise ValueError("Invalid noise_type. Use 'r' for row noise or 'TL' for readout noise.")

    # Regression line
    if c1 is not None:
        label_text = f"{k_label}={a:.5f}\n{b_label}={b:.5f}\n{sigma_label}={std:.5f}" if label else None
        ax.plot(x_range, a * x_range + b, linewidth=2, color=c1, label=label_text)

    # Standard deviation bounds
    if c2 is not None:
        ax.plot(x_range, a * x_range + b + std, color=c2, linewidth=2, linestyle="--")
        ax.plot(x_range, a * x_range + b - std, color=c2, linewidth=2, linestyle="--")

    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add legend
    if label:
        ax.legend(loc="upper left")

    # Annotate R² in the bottom-right corner
    ax.text(0.95, 0.05, f"$R^2 = {r2:.5f}$", transform=ax.transAxes,
            fontsize=12, ha="right", va="bottom", bbox=dict(facecolor="white", alpha=0.8))

    data = {'k': a, 'b': b, 'sig': std, 'R2': r2}
    return ax, data
# raw2rgb('/Users/hyx/Code/CV/raw_denoising/calib/r10/color/100/IMG_0483.CR3')

# data = np.load('/Users/hyx/Code/CV/raw_denoising/pos/r10_block_pos.npy')
# print(data)

# block_positoins = np.load('/Users/hyx/Code/CV/raw_denoising/pos/r10_block_pos.npy')

# raw_path = os.path.join(os.path.join('/Users/hyx/Code/CV/raw_denoising/calib/r10','color'),str(100))
# raw_imgs = sorted(os.listdir(raw_path))
# raw_imgs.remove('.DS_Store')
# example_raw_img = os.path.join(raw_path,raw_imgs[0])
# color_example_rgb = raw2rgb(example_raw_img)

# for i in range(24):
#     # rect = cv2.selectROI(rgb_image,False,False)
#     rect = block_positoins[i]
#     cv2.rectangle(color_example_rgb,rect,color=(23,128,62))
# cv2.imshow('example',color_example_rgb)
# cv2.waitKey(0)

