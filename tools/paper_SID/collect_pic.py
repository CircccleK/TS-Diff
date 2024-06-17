import os
import shutil

BM3D_path = "/home/ly/BM3D_raw/res/Sony_pic_ELD/"
MCDM_path = "/home/ly/RepDiff/experiments/test/final_test/Sony_raw_pic/visualization/"
N2N_path = "/home/ly/ELD_mine/result/N2N/"
SID_path = "/home/ly/ELD_mine/result/SID/"
pg_sonya7s2_path = "/home/ly/ELD_mine/result/pg_sonya7s2/"
prtq_sonya7s2_path = "/home/ly/ELD_mine/result/prtq_sonya7s2_SID/"
LRD_path = "/home/ly/LRD/results/"
ExposureDiffusion_path = "/home/ly/ExposureDiffusion/images/checkpoints/"

target_path = "/home/ly/RepDiff/experiments/pic"
ratio_dict_ELD = {"300": "0.033", "250": "0.04", "100": "0.1"}
ratio_dict_ELD_rev = {"0.033": "300", "0.04": "250", "0.1": "100"}

for file_name in os.listdir(MCDM_path):
    parts = file_name.split("_")
    cur_ratio = parts[0]
    cur_scene = parts[1]

    cur_path = os.path.join(os.path.join(target_path, cur_ratio), cur_scene)

    if not os.path.exists(cur_path):
        os.makedirs(cur_path)

    shutil.copy(os.path.join(MCDM_path, file_name), cur_path)

for file_name in os.listdir(BM3D_path):
    parts = file_name.split("_")
    cur_exp = parts[2].split('s')[0]
    if cur_exp not in ratio_dict_ELD_rev or len(parts) < 4:
        continue
    cur_scene = parts[0]

    cur_ratio = ratio_dict_ELD_rev[cur_exp]
    cur_path = os.path.join(os.path.join(target_path, cur_ratio), cur_scene)

    shutil.copy(os.path.join(BM3D_path, file_name), cur_path)

for file_name in os.listdir(N2N_path):
    parts = file_name.split("_")
    cur_exp = parts[2].split('s')[0]
    cur_scene = parts[0]

    cur_ratio = ratio_dict_ELD_rev[cur_exp]
    cur_path = os.path.join(os.path.join(target_path, cur_ratio), cur_scene)

    keyword = "N2N"
    N2N_sub_path = os.path.join(N2N_path, file_name)

    for file in os.listdir(N2N_sub_path):
        if keyword in file:
            shutil.copy(os.path.join(N2N_sub_path, file), cur_path)

for file_name in os.listdir(SID_path):
    parts = file_name.split("_")
    cur_exp = parts[2].split('s')[0]
    cur_scene = parts[0]

    cur_ratio = ratio_dict_ELD_rev[cur_exp]
    cur_path = os.path.join(os.path.join(target_path, cur_ratio), cur_scene)

    keyword = "sid-paired"
    SID_sub_path = os.path.join(SID_path, file_name)

    for file in os.listdir(SID_sub_path):
        if keyword in file:
            shutil.copy(os.path.join(SID_sub_path, file), cur_path)

for file_name in os.listdir(pg_sonya7s2_path):
    parts = file_name.split("_")
    cur_exp = parts[2].split('s')[0]
    cur_scene = parts[0]

    cur_ratio = ratio_dict_ELD_rev[cur_exp]
    cur_path = os.path.join(os.path.join(target_path, cur_ratio), cur_scene)

    keyword = "pg_sonya7s2"
    pg_sonya7s2_sub_path = os.path.join(pg_sonya7s2_path, file_name)

    for file in os.listdir(pg_sonya7s2_sub_path):
        if keyword in file:
            shutil.copy(os.path.join(pg_sonya7s2_sub_path, file), cur_path)

for file_name in os.listdir(prtq_sonya7s2_path):
    parts = file_name.split("_")
    cur_exp = parts[2].split('s')[0]
    cur_scene = parts[0]

    cur_ratio = ratio_dict_ELD_rev[cur_exp]
    cur_path = os.path.join(os.path.join(target_path, cur_ratio), cur_scene)

    keyword = "prtq_sonya7s2"
    prtq_sonya7s2_sub_path = os.path.join(prtq_sonya7s2_path, file_name)

    for file in os.listdir(prtq_sonya7s2_sub_path):
        if keyword in file:
            shutil.copy(os.path.join(prtq_sonya7s2_sub_path, file), cur_path)

LRD_txt = "./Sony_test_list.txt"
LRD_dic = {"100": 0, "250": 0, "300": 0}
with open(LRD_txt, 'r', encoding='utf-8') as file:
    for line_number, line in enumerate(file, start=1):
        parts = line.split(" ")
        name = parts[0].split("/")
        cur_exp = name[3].split('_')[2].split('s')[0]
        cur_scene = name[3].split('_')[0]
        cur_ratio = ratio_dict_ELD_rev[cur_exp]
        LRD_dic[cur_ratio] += 1

        cur_path = os.path.join(os.path.join(target_path, cur_ratio), cur_scene)

        LRD_cur_path = os.path.join(LRD_path, "SID_denoise_results_ratio" + cur_ratio)
        LRD_cur_path = os.path.join(LRD_cur_path, "test1")
        LRD_cur_path = os.path.join(LRD_cur_path, "denoised")

        keyword = str(LRD_dic[cur_ratio]) + "_"

        for LRD_file in os.listdir(LRD_cur_path):
            if keyword in LRD_file:
                shutil.copy(os.path.join(LRD_cur_path, LRD_file), cur_path)
                break

# ratios = ["100", "250", "300"]
# for ratio in ratios:
#     ExposureDiffusion_cur_path = os.path.join(ExposureDiffusion_path, ratio)
#     for file_name in os.listdir(ExposureDiffusion_cur_path):
#         parts = file_name.split("_")
#         cur_scene = parts[0]
#
#         cur_ratio = ratio
#         cur_path = os.path.join(os.path.join(target_path, cur_ratio), cur_scene)
#
#         keyword = "checkpoints"
#         checkpoints_sub_path = os.path.join(ExposureDiffusion_cur_path, file_name)
#
#         for file in os.listdir(checkpoints_sub_path):
#             if keyword in file:
#                 shutil.copy(os.path.join(checkpoints_sub_path, file), cur_path)