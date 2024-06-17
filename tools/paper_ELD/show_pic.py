import os

from PIL import Image

pic_path = "/home/ly/RepDiff/experiments/pic/pic_ELD_finetune/"
target_path = "/home/ly/RepDiff/experiments/pic/ELD_all_finetune/"

cameras = {"1": 'CanonEOS70D', "2": 'NikonD850', "3": 'SonyA7S2', "4": 'CanonEOS700D'}
scenes = [f"{i}" for i in range(1, 11)]
nums = ["0004", "0005", "0009", "0010", "0014", "0015"]

suffix = {"CanonEOS70D": "CR2", "CanonEOS700D": "CR2", "SonyA7S2": "ARW", "NikonD850": "nef"}


for key_cam in cameras:
    camera = cameras[key_cam]
    order_dic = {"lq": "lq", "prtq": suffix[camera], "MCDM": "MCDM", "finetune": "finetune", "gt": "gt"}

    for scene in scenes:
        for num in nums:
            target_dir = camera + "_scene-" + scene + "_" + num
            if target_dir == "CanonEOS70D_scene-3_0014":
                continue

            cur_path = os.path.join(pic_path, target_dir)

            spacing = 10
            flag = False

            result_image = None
            x_offset = 0

            # show zoomed
            for key in order_dic.keys():
                for file_name in os.listdir(cur_path):
                    if order_dic[key] in file_name:
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

            result_image.save(target_path + camera + "_" + scene + "_" + num + "_show" + ".png")
