# # 读取文件中的内容
# with open('L118_train_debug.txt', 'r') as file:
#     file_pairs = file.readlines()
#
# # 去掉每行文本的换行符
# file_pairs = [line.strip() for line in file_pairs]
#
# # 打乱文件顺序
# random.shuffle(file_pairs)
#
# # 计算分割索引
# total_files = len(file_pairs)
# # train_split = int(0.8 * total_files)
# # val_split = int(0.9 * total_files)
# train_split = int(0.9 * total_files)
# # val_split = int(0.9 * total_files)
#
# # 划分数据集
# train_files = file_pairs[:train_split]
# val_files = file_pairs[train_split:]
# # test_files = file_pairs[val_split:]
#
# # 保存到文件或者输出结果
# with open('L118_train_paired_noise.txt', 'w') as train_file:
#     for file in train_files:
#         train_file.write(file + '\n')
#
# with open('L118_val_paired_noise.txt', 'w') as val_file:
#     for file in val_files:
#         val_file.write(file + '\n')
#
# # with open('L118_test.txt', 'w') as test_file:
# #     for file in test_files:
# #         test_file.write(file + '\n')
#
# print("数据集已划分并保存至 train.txt, val.txt, test.txt")

def count_rows():
    count_10_1 = 0
    count_10_2 = 0
    count_10_3 = 0

    # 读取文件
    with open('L118_test.txt', 'r') as file:
        for line in file:
            if line.startswith('10-1'):
                count_10_1 += 1
            elif line.startswith('10-2'):
                count_10_2 += 1
            elif line.startswith('10-3'):
                count_10_3 += 1

    # 输出统计结果
    print(f"10-1 出现的行数: {count_10_1}")
    print(f"10-2 出现的行数: {count_10_2}")
    print(f"10-3 出现的行数: {count_10_3}")


def split_10_txt():
    with open('L118_train.txt', 'r') as file:
        with open('./L118_train_light_10-1.txt', 'w') as train_file:
            for line in file:
                if line.startswith('10-1'):
                    train_file.write(line)

    with open('L118_train.txt', 'r') as file:
        with open('./L118_train_light_10-2.txt', 'w') as train_file:
            for line in file:
                if line.startswith('10-2'):
                    train_file.write(line)

    with open('L118_train.txt', 'r') as file:
        with open('./L118_train_light_10-3.txt', 'w') as train_file:
            for line in file:
                if line.startswith('10-3'):
                    train_file.write(line)


# count_rows()
split_10_txt()