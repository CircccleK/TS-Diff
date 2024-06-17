import os
import shutil

from PIL import Image
from PIL.ExifTags import TAGS


def get_iso_from_jpg(image_path):
    img = Image.open(image_path)
    exif_data = img._getexif()

    # 查找ISO信息
    for tag, value in exif_data.items():
        decoded = TAGS.get(tag, tag)
        if decoded == "ISOSpeedRatings":
            return value
    return None


def move_file(src_file, dest_folder):
    # 如果目标文件夹不存在，则创建它
    os.makedirs(dest_folder, exist_ok=True)

    # 移动文件到目标文件夹
    shutil.move(src_file, dest_folder)


def mov_raw(cur_raw_path, cur_rgb_path, suffix):
    raw_imgs = os.listdir(cur_raw_path)
    for i, raw_img in enumerate(raw_imgs):
        file_name = os.path.splitext(raw_img)[0]
        file_extension = os.path.splitext(raw_img)[1]
        if file_extension == '.JPG':
            cur_JPG_path = os.path.join(cur_raw_path, raw_img)
            iso = get_iso_from_jpg(cur_JPG_path)
            move_file(cur_JPG_path, os.path.join(cur_rgb_path, str(iso)))
            move_file(os.path.join(cur_raw_path, file_name + suffix), os.path.join(cur_raw_path, str(iso)))


def in_raw_move_jpg(cur_raw_path):
    iso_list = sorted(os.listdir(cur_raw_path))
    for iso in iso_list:
        cur_iso_path = os.path.join(cur_raw_path, str(iso))
        raw_imgs = os.listdir(cur_iso_path)
        for i, raw_img in enumerate(raw_imgs):
            file_extension = os.path.splitext(raw_img)[1]
            if file_extension == '.JPG':
                cur_JPG_path = os.path.join(cur_iso_path, raw_img)
                target = os.path.join(cur_rgb_path, str(iso))
                move_file(cur_JPG_path, target)


cur_raw_path = '/data/ly/calib_data/LUMIX/dark_2/'
cur_rgb_path = '/data/ly/calib_data/LUMIX/dark_2/dark_rgb'
suffix = '.RW2'

mov_raw(cur_raw_path, cur_rgb_path, suffix)
