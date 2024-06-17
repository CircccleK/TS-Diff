import os

# 定义要处理的文件夹路径
# folder_path = '/data/ly/SID/Sony_rgb/ratio300/train/clean_cropped/'
# folder_path = '/data/ly/SID/Sony_rgb/ratio300/train/noisy_cropped/'
# folder_path = '/data/ly/SID/Sony_rgb/ratio300/val/noisy_cropped/'
folder_path = '/data/ly/SID/Sony_rgb/ratio300/val/clean_cropped/'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 获取文件的完整路径
    file_path = os.path.join(folder_path, filename)

    # 检查文件是否是一个普通文件
    if os.path.isfile(file_path):
        # 分割文件名和扩展名
        name, extension = os.path.splitext(filename)

        # 根据下划线和点进行分割
        parts = name.split('_')
        parts = [part.split('.')[0] for part in parts]  # 去掉最后一个点后面的部分

        # 构建新的文件名
        new_filename = '_'.join(parts) + extension

        # 构建新的文件路径
        new_file_path = os.path.join(folder_path, new_filename)

        # 重命名文件
        os.rename(file_path, new_file_path)

