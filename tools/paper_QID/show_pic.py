import os

from PIL import Image

pic_path = "/home/ly/RepDiff/experiments/pic/pic_QID_finetune"
target_path = "/home/ly/RepDiff/experiments/pic/QID_all_finetune/"

order_dic1 = {"lq": "lq", "prtq": "prtq", "MCDM": "MCDM", "gt": "gt"}
order_dic2 = {"lq": "lq", "prtq": "prtq", "MCDM": "MCDM", "TMCDM": "TMCDM", "gt": "gt"}

for dir in os.listdir(pic_path):
    parts = dir.split("_")
    cur_illumination = parts[0]
    cur_scene = parts[1]
    cur_iso = parts[2]
    cur_exp = parts[3]
    cur_num = parts[4]
    cur_path = os.path.join(pic_path, dir)

    target_dir = cur_illumination + "_" + cur_scene + "_" + cur_iso + "_" + cur_exp + "_" + cur_num

    spacing = 10
    flag = False

    result_image = None
    x_offset = 0

    # show zoomed
    for key in order_dic2.keys():
        for file_name in os.listdir(cur_path):
            if order_dic2[key] in file_name:
                img = Image.open(os.path.join(cur_path, file_name))
                if not flag:
                    total_width = img.width * 5 + spacing * 4
                    result_image = Image.new("RGB", (total_width, img.height), (255, 255, 255))
                    flag = True
                    result_image.paste(img, (x_offset, 0))
                else:
                    result_image.paste(img, (x_offset, 0))
                x_offset += img.width + spacing
                break

    result_image.save(target_path + target_dir + "_show" + ".png")
