import json

import scipy
import scipy.stats

from calib_utils import *


def calib_dark_per_iso(cam_dir, iso):
    cur_iso_path = os.path.join(os.path.join(cam_dir, 'dark'), str(iso))
    raw_imgs = os.listdir(cur_iso_path)
    if '.DS_Store' in raw_imgs:
        raw_imgs.remove('.DS_Store')

    r = 400
    # 行噪声变量
    sigma_row = np.zeros(len(raw_imgs), dtype=np.float32)
    mean_row = np.zeros(len(raw_imgs), dtype=np.float32)
    r2_row = np.zeros(len(raw_imgs), dtype=np.float32)
    # TL 分布变量
    sigma_TL = np.zeros((4, len(raw_imgs)), dtype=np.float32)
    mean_TL = np.zeros((4, len(raw_imgs)), dtype=np.float32)
    r2_TL = np.zeros((4, len(raw_imgs)), dtype=np.float32)
    lamda = np.zeros((4, len(raw_imgs)), dtype=np.float32)

    for i, raw_img in enumerate(raw_imgs):
        # 为当前图像创建文件夹
        img_output_dir = os.path.join(cam_dir, f'iso_{iso}', f'img_{i}')
        if not os.path.exists(img_output_dir):
            os.makedirs(img_output_dir)

        cur_raw_path = os.path.join(cur_iso_path, raw_img)
        raw = rawpy.imread(cur_raw_path)
        black_level = raw.black_level_per_channel

        raw_vis = raw.raw_image_visible.copy()
        raw_pattern = raw.raw_pattern
        raw = raw.raw_image_visible.astype(np.float32)
        raw -= np.mean(black_level)
        row_all = np.mean(
            raw[raw.shape[0] // 2 - r * 2:raw.shape[0] // 2 + r * 2, raw.shape[1] // 2 - r * 2:raw.shape[1] // 2 + r * 2],
            axis=1
        )
        # 行噪声高斯分布检验
        _, (sig_row, u_row, r_row) = scipy.stats.probplot(row_all, rvalue=True)
        sigma_row[i] = sig_row
        mean_row[i] = u_row
        r2_row[i] = r_row**2

        # 保存行噪声 Q-Q 图并标注 \( R^2 \) 和 \(\sigma_r\)
        plt.figure()
        res = scipy.stats.probplot(row_all, dist="norm", plot=None)
        plt.scatter(res[0][0], res[0][1], s=10, color="blue", label="Data")  # 数据点
        plt.plot(res[0][0], res[1][0] * res[0][0] + res[1][1], color="red", label="Fit")  # 拟合线
        plt.title(f"ISO {iso}")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Actual Quantiles")
        plt.legend()
        plt.text(0.95, 0.05, f"$R^2$: {r2_row[i]:.4f}\n$\\sigma_{{rb}}$: {sig_row:.4f}",
                 transform=plt.gca().transAxes, fontsize=10, ha="right", va="bottom",
                 bbox=dict(facecolor="white", alpha=0.8))
        plt.savefig(os.path.join(img_output_dir, f'gaussian_row_fit.png'))
        plt.close()

        # 处理 RGGB 数据
        rggb_img = pack_raw_bayer(raw_vis, raw_pattern)
        rggb_img = rggb_img.transpose(1, 2, 0).astype(np.int64)
        rggb_img -= black_level
        H, W = rggb_img.shape[:2]
        rggb_img = rggb_img[H // 2 - r:H // 2 + r, W // 2 - r:W // 2 + r, :]

        for c in range(4):
            cur_channel_img = rggb_img[:, :, c]
            row_all_cur_channel = np.mean(cur_channel_img, axis=1)
            cur_channel_img = cur_channel_img.astype(np.float32)
            cur_channel_img -= row_all_cur_channel.reshape(-1, 1)
            X = cur_channel_img.reshape(-1)
            lam = scipy.stats.ppcc_max(X)
            lamda[c][i] = lam
            _, (sig_TL, u_TL, r_TL) = scipy.stats.probplot(X, dist=scipy.stats.tukeylambda(lam), rvalue=True)
            sigma_TL[c][i] = sig_TL
            mean_TL[c][i] = u_TL
            r2_TL[c][i] = r_TL**2

            # 保存 TL 分布 Q-Q 图并标注 \( R^2 \) 和 \(\sigma_{TL}\)
            plt.figure()
            res = scipy.stats.probplot(X, dist=scipy.stats.tukeylambda(lam), plot=None)
            plt.scatter(res[0][0], res[0][1], s=10, color="blue", label="Data")  # 数据点
            plt.plot(res[0][0], res[1][0] * res[0][0] + res[1][1], color="red", label="Fit")  # 拟合线
            plt.title(f"Channel {c} ISO {iso}")
            plt.xlabel("Theoretical Quantiles")
            plt.ylabel("Actual Quantiles")
            plt.legend()
            plt.text(0.95, 0.05, f"$R^2$: {r2_TL[c][i]:.4f}\n$\\sigma_{{TLc}}$: {sig_TL:.4f}",
                     transform=plt.gca().transAxes, fontsize=10, ha="right", va="bottom",
                     bbox=dict(facecolor="white", alpha=0.8))
            plt.savefig(os.path.join(img_output_dir, f'tl_fit_channel_{c}.png'))
            plt.close()

            # 保存 PPCC 图
            plt.figure()
            ppcc_lam, ppcc_vals = scipy.stats.ppcc_plot(X, -5, 5)
            plt.plot(ppcc_lam, ppcc_vals, label="PPCC")
            plt.axvline(x=lam, color="red", linestyle="--", label=f"Optimal $λ_c$={lam:.4f}")
            plt.legend()
            plt.title(f"Channel {c} ISO {iso}")
            plt.savefig(os.path.join(img_output_dir, f'ppcc_curve_channel_{c}.png'))
            plt.close()

    param = {
        'black_level': black_level,
        'lam': lamda.tolist(),
        'sigmaR': sigma_row.tolist(),
        'meanR': mean_row.tolist(),
        'r2R': r2_row.tolist(),
        'sigmaTL': sigma_TL.tolist(),
        'meanTL': mean_TL.tolist(),
        'r2TL': r2_TL.tolist(),
    }

    param_channel_mean = {
        'black_level':black_level,
        'lam':np.mean(lamda,axis=0).tolist(),
        'sigmaR':sigma_row.tolist(), 'meanR':mean_row.tolist(), 'r2R':r2_row.tolist(),
        'sigmaTL':np.mean(sigma_TL,axis=0).tolist(), 'meanTL':np.mean(mean_TL,axis=0).tolist(), 'r2TL':np.mean(r2_TL,axis=0).tolist(),
    }

    return param, param_channel_mean


# get noise params from dark imgs per camera
def calib_dark_per_camera(cam_dir):
    dark_dir = os.path.join(cam_dir, 'dark')
    iso_list = sorted(os.listdir(dark_dir))
    if '.DS_Store' in iso_list:
        iso_list.remove('.DS_Store')
    # if '400' in iso_list:
    #     iso_list.remove('400')
    dark_calib_params = dict()
    dark_calib_params_channel_mean = dict()
    for iso in iso_list:
        print(iso)
        param, param_channel_mean = calib_dark_per_iso(cam_dir,iso=int(iso))
        dark_calib_params[iso] = param
        dark_calib_params_channel_mean[iso] = param_channel_mean

    dark_calib_params_dir = cam_dir + "/dark_calib_params/"
    if not os.path.exists(dark_calib_params_dir):
        os.mkdir(dark_calib_params_dir)

    with open(os.path.join(dark_calib_params_dir,'dark_calib_params.json'),'w') as json_file:
        json.dump(dark_calib_params,json_file)
    with open(os.path.join(dark_calib_params_dir,'dark_calib_params_channel_mean.json'),'w') as json_file:
        json.dump(dark_calib_params_channel_mean,json_file)