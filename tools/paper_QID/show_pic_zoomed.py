import os

from PIL import Image

pic_path = "/home/ly/RepDiff/experiments/pic/pic_QID/"
target_path = "/home/ly/RepDiff/experiments/pic/show_pic_QID/"

dir = '10-3_s12_iso16000_exp0.04_2'
parts = dir.split("_")
cur_illumination = parts[0]
cur_scene = parts[1]
cur_iso = parts[2]
cur_exp = parts[3]
cur_num = parts[4]

target_dir = cur_illumination + "_" + cur_scene + "_" + cur_iso + "_" + cur_exp + "_" + cur_num

pic_path = os.path.join(pic_path, dir)

keywords = "zoomed"

order_dic1 = {"lq": "lq", "prtq": "prtq", "MCDM": "MCDM", "gt": "gt"}

spacing = 10

flag = False
result_image = None
x_offset = 0

# # show zoomed
# for key in order_dic1.keys():
#     for file_name in os.listdir(pic_path):
#         if keywords in file_name and order_dic1[key] in file_name:
#             img = Image.open(os.path.join(pic_path, file_name))
#             if not flag:
#                 total_width = img.width * 4 + spacing * 3
#                 result_image = Image.new("RGB", (total_width, img.height), (255, 255, 255))
#                 flag = True
#                 result_image.paste(img, (x_offset, 0))
#             else:
#                 result_image.paste(img, (x_offset, 0))
#             x_offset += img.width + spacing
#             break
# result_image.save(target_path + target_dir + "_show" + ".png")



flag = False
pic_path = "/home/ly/RepDiff/experiments/pic/pic_QID_finetune/"
target_path = "/home/ly/RepDiff/experiments/pic/show_pic_QID_finetune_paper/"
pic_path = os.path.join(pic_path, dir)

result_image = None
x_offset = 0
# order_dic2 = {"lq": "lq", "prtq": "prtq", "MCDM": "MCDM", "TMCDM": "TMCDM", "gt": "gt"}
order_dic2 = {"lq": "lq", "prtq": "prtq", "TMCDM": "TMCDM", "gt": "gt"}


for key in order_dic2.keys():
    for file_name in os.listdir(pic_path):
        if keywords in file_name and order_dic2[key] in file_name:
            img = Image.open(os.path.join(pic_path, file_name))
            if not flag:
                total_width = img.width * 4 + spacing * 3
                result_image = Image.new("RGB", (total_width, img.height), (255, 255, 255))
                flag = True
                result_image.paste(img, (x_offset, 0))
            else:
                result_image.paste(img, (x_offset, 0))
            x_offset += img.width + spacing
            break
result_image.save(target_path + target_dir + "_show" + ".png")
