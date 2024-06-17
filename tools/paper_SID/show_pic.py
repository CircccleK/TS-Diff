import os

from PIL import Image

pic_path = "/home/ly/RepDiff/experiments/pic/"
target_path = "/home/ly/RepDiff/experiments/pic/all/"

ratios = ["100", "250", "300"]
scenes = [
    "20188", "20201", "20208", "20210", "20211",
    "10185", "10187", "10191", "10193", "10198",
    "10213", "10217", "10226", "10227", "10228"
]

for ratio in ratios:
    for scene in scenes:

        cur_path = os.path.join(pic_path, ratio)
        cur_path = os.path.join(cur_path, scene)

        order_dic1 = {"lq": "lq", "denoised": "denoised", "N2N": "N2N", "sid": "sid"}
        order_dic2 = {"prtq": "prtq", "lrd": "lrd", "sr": "sr", "gt": "gt"}

        spacing = 10
        flag = False

        result_image = None
        x_offset = 0

        # show zoomed
        for key in order_dic1.keys():
            for file_name in os.listdir(cur_path):
                if key in file_name:
                    img = Image.open(os.path.join(cur_path, file_name))
                    if not flag:
                        total_width = img.width * 4 + spacing * 3
                        result_image = Image.new("RGB", (total_width, img.height), (255, 255, 255))
                        flag = True
                        result_image.paste(img, (x_offset, 0))
                    else:
                        result_image.paste(img, (x_offset, 0))
                    x_offset += img.width + spacing

        result_image.save(target_path + ratio + "_" + scene + "_show1" + ".png")

        flag = False
        result_image = None
        x_offset = 0

        for key in order_dic2.keys():
            for file_name in os.listdir(cur_path):
                if key in file_name:
                    img = Image.open(os.path.join(cur_path, file_name))
                    if not flag:
                        total_width = img.width * 4 + spacing * 3
                        result_image = Image.new("RGB", (total_width, img.height), (255, 255, 255))
                        flag = True
                        result_image.paste(img, (x_offset, 0))
                    else:
                        result_image.paste(img, (x_offset, 0))
                    x_offset += img.width + spacing

        result_image.save(target_path + ratio + "_" + scene + "_show2" + ".png")