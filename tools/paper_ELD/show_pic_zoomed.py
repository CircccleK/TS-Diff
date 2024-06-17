import os

from PIL import Image

pic_path = "/home/ly/RepDiff/experiments/pic/pic_ELD/"
target_path = "/home/ly/RepDiff/experiments/pic/show_pic_ELD/"

camera = "CanonEOS700D"
scene = "scene-1"
num = "0015"

suffix = {"CanonEOS70D": "CR2", "CanonEOS700D": "CR2", "SonyA7S2": "ARW", "NikonD850": "nef"}

pic_path = os.path.join(pic_path, camera + "_" + scene + "_" + num)

keywords = "zoomed"

order_dic = {"lq": "lq", "prtq": suffix[camera], "sr": "sr", "gt": "gt"}

spacing = 10

flag = False
result_image = None
x_offset = 0

# # show zoomed
# for key in order_dic.keys():
#     for file_name in os.listdir(pic_path):
#         if keywords in file_name and order_dic[key] in file_name:
#             img = Image.open(os.path.join(pic_path, file_name))
#             if not flag:
#                 total_width = img.width * 4 + spacing * 3
#                 result_image = Image.new("RGB", (total_width, img.height), (255, 255, 255))
#                 flag = True
#                 result_image.paste(img, (x_offset, 0))
#             else:
#                 result_image.paste(img, (x_offset, 0))
#             x_offset += img.width + spacing
#
# result_image.save(target_path + camera + "_" + scene + "_" + num + "_show" + ".png")



# flag = False
target_path = "/home/ly/RepDiff/experiments/pic/show_pic_ELD_finetune_paper/"
pic_path = "/home/ly/RepDiff/experiments/pic/pic_ELD_finetune/"
#
# result_image = None
# x_offset = 0
# order_dic = {"lq": "lq", "prtq": suffix[camera], "MCDM": "MCDM", "finetune": "finetune", "gt": "gt"}
order_dic = {"lq": "lq", "prtq": suffix[camera], "finetune": "finetune", "gt": "gt"}
pic_path = os.path.join(pic_path, camera + "_" + scene + "_" + num)

for key in order_dic.keys():
    for file_name in os.listdir(pic_path):
        if keywords in file_name and order_dic[key] in file_name:
            img = Image.open(os.path.join(pic_path, file_name))
            if not flag:
                total_width = img.width * 4 + spacing * 3
                result_image = Image.new("RGB", (total_width, img.height), (255, 255, 255))
                flag = True
                result_image.paste(img, (x_offset, 0))
            else:
                result_image.paste(img, (x_offset, 0))
            x_offset += img.width + spacing

result_image.save(target_path + camera + "_" + scene + "_" + num + "_show" + ".png")