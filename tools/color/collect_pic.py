import os
import shutil

no_cs_path = "/home/ly/RepDiff/experiments/with_no_cs/visualization"
cs_path = "/home/ly/RepDiff/experiments/with_cs/visualization/"

target_path = "/home/ly/RepDiff/experiments/pic/color/"

for file_name in os.listdir(no_cs_path):
    if "sr" in file_name:
        shutil.copy(os.path.join(no_cs_path, file_name), os.path.join(target_path, "no_cs_" + file_name))

for file_name in os.listdir(cs_path):
    if "sr" in file_name:
        shutil.copy(os.path.join(no_cs_path, file_name), os.path.join(target_path, "cs_" + file_name))
