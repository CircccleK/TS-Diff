import json

from calib_utils import *


# Fit K and Var(No) using the formula Var(D) = K(KI) + Var(No)


def calib_color_per_iso(cam_dir, iso, camera):
    cur_iso_path = os.path.join(os.path.join(cam_dir, 'color'), str(iso))
    raw_imgs = os.listdir(cur_iso_path)
    if '.DS_Store' in raw_imgs:
        raw_imgs.remove('.DS_Store')

    # Retrieve previously stored position information
    block_position_path = f'{cam_dir}/{camera}_block_pos.npy'
    block_positions = np.load(block_position_path)
    KI = np.zeros((4, len(block_positions)))
    var_D = np.zeros((4, len(block_positions)))
    for raw_img in raw_imgs:
        cur_raw_path = os.path.join(cur_iso_path, raw_img)
        raw = rawpy.imread(cur_raw_path)
        black_level = raw.black_level_per_channel
        white_point = raw.camera_white_level_per_channel
        raw_vis = raw.raw_image_visible.copy()
        raw_pattern = raw.raw_pattern
        raw = raw.raw_image_visible.astype(np.float32)
        rggb_img = pack_raw_bayer(raw_vis, raw_pattern)
        rggb_img = rggb_img.transpose(1, 2, 0).astype(np.int64)

        rggb_img -= black_level

        for i, pos in enumerate(block_positions):
            minx, miny, w, h = pos
            maxx, maxy = minx + w, miny + h

            KI[:, i] += np.mean(rggb_img[miny:maxy, minx:maxx, :], axis=(0, 1))
            var_D[:, i] += np.var(rggb_img[miny:maxy, minx:maxx, :], axis=(0, 1))

    KI /= len(raw_imgs)
    var_D /= len(raw_imgs)

    K, var_No, R2_scores = np.zeros((4)), np.zeros((4)), np.zeros((4))
    for i in range(4):
        K[i], var_No[i] = linear_regression(KI[i], var_D[i])
        # Calculate predictions and R^2
        predicted_var_D = K[i] * KI[i] + var_No[i]
        R2_scores[i] = r2_score(var_D[i], predicted_var_D)
        print(f"Channel {i}, ISO {iso}, R^2: {R2_scores[i]}")

    for i in range(4):
        # 绘制数据点，调整圆径大小
        plt.scatter(KI[i], var_D[i], color="blue", label="Data", s=10)  # 参数 `s` 控制数据点大小

        # 绘制拟合直线，调整线条粗细
        plt.plot(KI[i], K[i] * KI[i] + var_No[i], color="red", label="Fit")  # 参数 `linewidth` 控制线条粗细

        plt.xlabel("$KI$")
        plt.ylabel("$\mathrm{Var}(D)$")
        plt.legend()

        # 添加标题
        plt.title(f"ISO {iso} - Channel {i}")

        # 添加右下角文本，保留小数点后4位
        plt.text(
            0.95, 0.05,
            f"$R^2$: {R2_scores[i]:.4f}\n$K_c$: {K[i]:.4f}",
            fontsize=10,
            ha="right", va="bottom",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
        )

        # 保存图像
        fig_dir = os.path.join(cam_dir, 'figs')
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        fig_name = os.path.join(fig_dir, f'{iso}_K{i}.png')
        plt.savefig(fig_name)
        plt.close()

    return K, var_No, R2_scores

def calib_color_per_camera(cam_dir, camera):
    color_dir = os.path.join(cam_dir, 'color')
    iso_list = sorted(os.listdir(color_dir))
    if '.DS_Store' in iso_list:
        iso_list.remove('.DS_Store')
    color_calib_params = dict()
    color_calib_params['K_list'], color_calib_params['var_No_list'] = dict(), dict()
    for iso in iso_list:
        K, var_No, _ = calib_color_per_iso(cam_dir, iso=int(iso), camera=camera)
        # K, var_No = calib_color_per_iso(cam_dir, iso=6400, camera=camera)
        color_calib_params['K_list'][iso] = K.tolist()
        color_calib_params['var_No_list'][iso] = var_No.tolist()

    color_calib_params_dir = cam_dir + "/color_calib_params/"
    if not os.path.exists(color_calib_params_dir):
        os.makedirs(color_calib_params_dir)
    with open(os.path.join(color_calib_params_dir, 'color_calib_params.json'), 'w') as json_file:
        json.dump(color_calib_params, json_file)
