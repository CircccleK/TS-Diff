from calib_color import *
from calib_dark import *

# fit (log) var_TL, var_gauss, var_row and K
def fit_log(cam_name, color_num, dark_num, cam_dir):
    sig_gauss = []
    sig_row = []
    sig_TL = []
    K = []
    K_points = []

    dir = os.path.join(cam_dir, color_num)

    with open(f'{dir}/color_calib_params/color_calib_params.json', 'r') as f:
        data = json.load(f)
        K_list, var_No_list = data['K_list'], data['var_No_list']
    with open(f'{cam_dir}/{dark_num}/dark_calib_params/dark_calib_params_channel_mean.json', 'r') as f:
        dark_calib_params_channel_mean = json.load(f)

    # 初始化 lamda 最大值和最小值
    lamda_means = []

    for iso, param in dark_calib_params_channel_mean.items():
        if int(iso) in [100, 200, 25600]:
            del K_list[iso]
            continue
        # 采集时按iso线性增长采集好点，避免过拟合
        cur_k = np.mean(K_list[iso])
        K.append(cur_k)
        sig_row.append(np.mean(param['sigmaR']))
        sig_TL.append(np.mean(param['sigmaTL']))

        # 计算 lamda 平均值并存储
        lamda_means.append(np.mean(param['lam']))

    fig = plt.figure(figsize=(20, 8))
    axsig_row = fig.add_subplot(1, 2, 1)
    axsig_TL = fig.add_subplot(1, 2, 2)

    axsig_row.set_title('log(sig_row) - log(K)')
    axsig_TL.set_title('log(sig_TL) - log(K)')

    # 绘制回归图
    axsig_row, data_row = regr_plot(
        K, sig_row, ax=axsig_row,
        xlabel="log($K$)", ylabel="log($\\sigma_r$)",
        title="log($\\sigma_r$) - log($K$)",
        c1='red', c2='orange', label=True, noise_type="r"
    )
    axsig_TL, data_TL = regr_plot(
        K, sig_TL, ax=axsig_TL,
        xlabel="log($K$)", ylabel="log($\\sigma_{TL}$)",
        title="log($\\sigma_{TL}$) - log($K$)",
        c1='red', c2='orange', label=True, noise_type="TL"
    )

    # 保存参数
    params = {
        'Kmin': min(min(values) for values in K_list.values()),
        'Kmax': max(max(values) for values in K_list.values()),
        'Row': {
            'slope': data_row['k'],
            'bias': data_row['b'],
            'std': data_row['sig']
        },
        'TL': {
            'slope': data_TL['k'],
            'bias': data_TL['b'],
            'std': data_TL['sig']
        },
        'Lamda': {
            'max': max(lamda_means),
            'min': min(lamda_means)
        }
    }

    cam_param_dir = dir + '/cam_log_params/'
    if not os.path.exists(cam_param_dir):
        os.mkdir(cam_param_dir)
    cam_param_file = os.path.join(cam_param_dir, cam_name + '.json')
    with open(cam_param_file, 'w') as json_file:
        json.dump(params, json_file, indent=4)
    tmp_path = dir + '/log_figs/'
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    plt.savefig(tmp_path + cam_name + '.png')

if __name__ == "__main__":
    # fit_log('6d2')
    # calib_dark()
    # get_block_positoins(cam_dir= '/Users/hyx/Code/CV/raw_denoising/calib/550d')
    # calib_color_per_iso_whole(cam_dir='/Users/hyx/Code/CV/raw_denoising/calib/6d2',iso=1600)
    # calib_dark_per_iso('/Users/hyx/Code/CV/raw_denoising/calib/d5200',1600)

    # calib_dark_per_camera('/Users/hyx/Code/CV/raw_denoising/calib/d5200')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',type=str,default='/data/ly/calib_data/')
    parser.add_argument('--mode',type=str,default='calib')
    directory = parser.parse_args().dir
    # cam_tar = "canon200D2"
    cam_tar = "LUMIX"
    color_num = "color_1"
    cam_list = os.listdir(directory)
    cam_dir = os.path.join(directory, cam_tar)
    dark_num = "dark_2"

    if '.DS_Store' in cam_list:
        cam_list.remove('.DS_Store')
    print(cam_list)
    if parser.parse_args().mode == 'get_pos':
        for cam in cam_list:
            get_block_positoins(cam_dir=os.path.join(directory,cam))
    elif parser.parse_args().mode == 'calib':
        for cam in cam_list:
            if cam == cam_tar:
                cam_dir = os.path.join(directory,cam)
                calib_dark_per_camera(os.path.join(cam_dir, dark_num))
                print('-------')
    elif parser.parse_args().mode == 'fig_log':
        for cam in cam_list:
            if cam == cam_tar:
                fit_log(cam, color_num, dark_num,cam_dir)
    elif parser.parse_args().mode == 'color':
        for cam in cam_list:
            if cam == cam_tar:
                calib_color_per_camera(os.path.join(cam_dir, color_num), camera=cam_tar)
