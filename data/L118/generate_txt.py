import os
import random
import re

from utils.raw_data import read_paired_fns


def extract_s_number(filename):
    """从文件名中提取 's*' 部分"""
    match = re.search(r's\d+', filename)
    if match:
        return match.group(0)
    return None


def extract_last_number(filename):
    """从文件名中提取最后一个数字"""
    match = re.search(r'(\d+)(?=\D*$)', filename)
    if match:
        return match.group(0)
    return None

def short_paired_list(short_dir, output_file):
    short_files = sorted(os.listdir(short_dir))

    list_txt = '//home/ly/RepDiff/data/L118/L118_test.txt'
    fns = read_paired_fns(list_txt)
    input_list = [fn[0] for fn in fns]

    short_mapping = {}
    for filename in short_files:
        if filename in input_list:
            continue
        s_number = extract_s_number(filename)
        if s_number:
            if s_number not in short_mapping:
                short_mapping[s_number] = []
            short_mapping[s_number].append(filename)

    with open(output_file, 'w') as f:
        for s_number in short_mapping:
            lst = short_mapping[s_number]
            while len(lst) >= 2:
                selected = random.sample(lst, 2)
                f.write(f"{selected[0]} {selected[1]}\n")

                for item in selected:
                    lst.remove(item)


def map_files(long_dir, short_dir, output_file):
    # 获取 long 文件夹中的文件列表，并按文件名排序
    long_files = sorted(os.listdir(long_dir))
    # 获取 short 文件夹中的文件列表，并按文件名排序
    short_files = sorted(os.listdir(short_dir))

    # 构建 long 文件的映射关系，键为 's*'，值为相应的文件名列表
    long_mapping = {}
    for filename in long_files:
        s_number = extract_s_number(filename)
        if s_number:
            if s_number not in long_mapping:
                long_mapping[s_number] = []
            long_mapping[s_number].append(filename)

    # 构建 short 文件的映射关系，键为 's*'，值为相应的文件名列表
    short_mapping = {}
    for filename in short_files:
        s_number = extract_s_number(filename)
        if s_number:
            if s_number not in short_mapping:
                short_mapping[s_number] = []
            short_mapping[s_number].append(filename)

    # 将映射关系写入输出文件
    with open(output_file, 'w') as f:
        for s_number in short_mapping:
            if s_number in long_mapping:
                # 获取 short 和 long 对应的文件列表
                short_files_list = short_mapping[s_number]
                long_files_list = long_mapping[s_number]

                # 对每个 short 文件，根据最后的数字在 long 中查找匹配项
                for short_file in short_files_list:
                    short_file_last_num = extract_last_number(short_file)

                    # 在 long 文件列表中查找与最后数字匹配的文件
                    matched_long_file = None
                    for long_file in long_files_list:
                        long_file_last_num = extract_last_number(long_file)
                        if long_file_last_num == short_file_last_num:
                            matched_long_file = long_file
                            break

                    if matched_long_file:
                        f.write(f"{short_file} {matched_long_file}\n")
                    else:
                        continue


if __name__ == "__main__":
    # 设置文件夹路径和输出文件路径
    long_directory = "/data/ly/L118/raw/long"
    short_directory = "/data/ly/L118/raw/short"
    output_txt_file = "L118_train_debug.txt"

    # 生成文件对应关系
    # map_files(long_directory, short_directory, output_txt_file)
    short_paired_list(short_directory, output_txt_file)
    print(f"文件对应关系已写入 {output_txt_file}")