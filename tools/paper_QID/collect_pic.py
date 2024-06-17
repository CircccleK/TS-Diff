import os
import shutil

MCDM_path = "/home/ly/RepDiff/experiments/L118_test_Sony_raw_pic/visualization/"
prtq_sonya7s2_path = "/home/ly/ELD_mine/result/prtq_sonya7s2_QID/"
TMCDM_path = "/home/ly/RepDiff/experiments/L118_finetune_test_pic/visualization/"

target_path = "/home/ly/RepDiff/experiments/pic/pic_QID/"

for file_name in os.listdir(MCDM_path):
    parts = file_name.split("_")
    cur_illumination = parts[1]
    cur_scene = parts[2]
    cur_iso = parts[3]
    cur_exp = parts[4]
    cur_num = parts[5]

    tar_path = os.path.join(target_path,
                            cur_illumination + "_" + cur_scene + "_" + cur_iso + "_" + cur_exp + "_" + cur_num)

    if not os.path.exists(tar_path):
        os.makedirs(tar_path)

    if "sr" in file_name:
        shutil.copy(os.path.join(MCDM_path, file_name), os.path.join(tar_path, "MCDM_" + file_name))
    else:
        shutil.copy(os.path.join(MCDM_path, file_name), tar_path)

# for file_name in os.listdir(TMCDM_path):
#     parts = file_name.split("_")
#     cur_illumination = parts[1]
#     cur_scene = parts[2]
#     cur_iso = parts[3]
#     cur_exp = parts[4]
#     cur_num = parts[5]
#
#     tar_path = os.path.join(target_path,
#                             cur_illumination + "_" + cur_scene + "_" + cur_iso + "_" + cur_exp + "_" + cur_num)
#
#     if not os.path.exists(tar_path):
#         os.makedirs(tar_path)
#
#     if "sr" in file_name:
#         shutil.copy(os.path.join(TMCDM_path, file_name), os.path.join(tar_path, "TMCDM_" + file_name))
#     else:
#         shutil.copy(os.path.join(TMCDM_path, file_name), tar_path)

for file_name in os.listdir(prtq_sonya7s2_path):
    parts = file_name.split("_")
    cur_illumination = parts[0]
    cur_scene = parts[1]
    cur_iso = parts[2]
    cur_exp = parts[3]
    cur_num = parts[4]

    tar_path = os.path.join(target_path,
                            cur_illumination + "_" + cur_scene + "_" + cur_iso + "_" + cur_exp + "_" + cur_num)

    keyword = "prtq_sonya7s2"
    prtq_sonya7s2_sub_path = os.path.join(prtq_sonya7s2_path, file_name)

    for file in os.listdir(prtq_sonya7s2_sub_path):
        if keyword in file:
            shutil.copy(os.path.join(prtq_sonya7s2_sub_path, file), tar_path)