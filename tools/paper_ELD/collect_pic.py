import os
import shutil

cameras = {"1": 'CanonEOS70D', "2": 'NikonD850', "3": 'SonyA7S2', "4": 'CanonEOS700D'}

MCDM_path = "/home/ly/RepDiff/experiments/test/final_test/"
ELD_path = "/home/ly/ELD_mine/result/prtq_sonya7s2_ELD/"

target_path = "/home/ly/RepDiff/experiments/pic/ELD_finetune"

for key in cameras:
    cur_path = os.path.join(MCDM_path, f'ELD_Sony_raw_{key}_test_pic')
    cur_path = os.path.join(cur_path, "visualization")

    for file_name in os.listdir(cur_path):
        parts = file_name.split("_")
        cur_ratio = parts[0]
        cur_scene = parts[1]
        cur_num = parts[3]

        tar_path = os.path.join(target_path, cameras[key] + "_" + cur_scene + "_" + cur_num)

        if not os.path.exists(tar_path):
            os.makedirs(tar_path)

        if "sr" in file_name:
            shutil.copy(os.path.join(cur_path, file_name), os.path.join(tar_path, "MCDM_" + file_name))
        else:
            shutil.copy(os.path.join(cur_path, file_name), tar_path)

for key in cameras:
    cur_path = os.path.join(MCDM_path, f'ELD_finetune_{key}_test_pic')
    cur_path = os.path.join(cur_path, "visualization")

    for file_name in os.listdir(cur_path):
        parts = file_name.split("_")
        cur_ratio = parts[0]
        cur_scene = parts[1]
        cur_num = parts[3]

        tar_path = os.path.join(target_path, cameras[key] + "_" + cur_scene + "_" + cur_num)

        if not os.path.exists(tar_path):
            os.makedirs(tar_path)

        if "sr" in file_name:
            shutil.copy(os.path.join(cur_path, file_name), os.path.join(tar_path, "finetune_" + file_name))
        else:
            shutil.copy(os.path.join(cur_path, file_name), tar_path)

for key in cameras:
    cur_path_camera = os.path.join(ELD_path, cameras[key])
    for scene in os.listdir(cur_path_camera):
        cur_path_scene = os.path.join(cur_path_camera, scene)
        for file_name in os.listdir(cur_path_scene):
            parts = file_name.split("_")
            cur_num = parts[1].split('.')[0]

            tar_path = os.path.join(target_path, cameras[key] + "_" + scene + "_" + cur_num)

            shutil.copy(os.path.join(cur_path_scene, file_name), tar_path)
