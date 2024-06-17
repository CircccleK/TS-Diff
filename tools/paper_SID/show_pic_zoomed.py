import os

from PIL import Image

pic_path = "/home/ly/RepDiff/experiments/pic/"
target_path = "/home/ly/RepDiff/experiments/pic/show_pic_SID_finetune_paper/"

ratio = "300"
scene = "10227"

cur_path = os.path.join(pic_path, ratio)
cur_path = os.path.join(cur_path, scene)
keywords = "zoomed"

# order_dic1 = {"lq": "lq", "denoised": "denoised", "N2N": "N2N", "sid": "sid", "pg": "pg",}
# order_dic2 = {"prtq": "prtq", "lrd": "lrd", "MCDM": "MCDM", "finetune": "finetune", "gt": "gt"}

order_dic1 = {"lq": "lq", "denoised": "denoised", "N2N": "N2N", "sid": "sid", "pg": "pg", "prtq": "prtq", "lrd": "lrd", "finetune": "finetune", "gt": "gt"}

spacing = 10
flag = False

result_image = None
x_offset = 0

# show zoomed
for key in order_dic1.keys():
    for file_name in os.listdir(cur_path):
        if keywords in file_name and key in file_name:
            img = Image.open(os.path.join(cur_path, file_name))
            if not flag:
                # total_width = img.width * 4 + spacing * 3
                total_width = img.width * 9 + spacing * 8
                result_image = Image.new("RGB", (total_width, img.height), (255, 255, 255))
                flag = True
                result_image.paste(img, (x_offset, 0))
            else:
                result_image.paste(img, (x_offset, 0))
            x_offset += img.width + spacing

result_image.save(target_path + ratio + "_" + scene + "_show1" + ".png")

# flag = False
# result_image = None
# x_offset = 0
#
# for key in order_dic2.keys():
#     for file_name in os.listdir(cur_path):
#         if keywords in file_name and key in file_name:
#             img = Image.open(os.path.join(cur_path, file_name))
#             if not flag:
#                 # total_width = img.width * 4 + spacing * 3
#                 total_width = img.width * 5 + spacing * 4
#                 result_image = Image.new("RGB", (total_width, img.height), (255, 255, 255))
#                 flag = True
#                 result_image.paste(img, (x_offset, 0))
#             else:
#                 result_image.paste(img, (x_offset, 0))
#             x_offset += img.width + spacing
#
# result_image.save(target_path + ratio + "_" + scene + "_show2" + ".png")