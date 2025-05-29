# import os


# def delete_files_by_condition(folder_path):
#     for root, dirs, files in os.walk(folder_path):
#         for file_name in files:
#             parts = file_name.split('@')
#             if len(parts) >= 5:
#                 try:
#                     num = float(parts[4])
#                     if num < 135 or num > 635:
#                         file_full_path = os.path.join(root, file_name)
#                         os.remove(file_full_path)
#                         print(f"已删除文件: {file_full_path}")
#                 except ValueError:
#                     continue


# folder_path = "/root/workspace/ctf53sc7v38s73e0mksg/maps/qd"
# delete_files_by_condition(folder_path)


# import os


# def get_max_min_values(folder_path):
#     nums_second_third = []
#     nums_third_fourth = []
#     for root, dirs, files in os.walk(folder_path):
#         for file_name in files:
#             parts = file_name.split('@')
#             if len(parts) >= 4:
#                 try:
#                     num_second_third = float(parts[2])
#                     num_third_fourth = float(parts[3])
#                     nums_second_third.append(num_second_third)
#                     nums_third_fourth.append(num_third_fourth)
#                 except ValueError:
#                     continue

#     if nums_second_third:
#         max_second_third = max(nums_second_third)
#         min_second_third = min(nums_second_third)
#     else:
#         max_second_third = None
#         min_second_third = None

#     if nums_third_fourth:
#         max_third_fourth = max(nums_third_fourth)
#         min_third_fourth = min(nums_third_fourth)
#     else:
#         max_third_fourth = None
#         min_third_fourth = None

#     return max_second_third, min_second_third, max_third_fourth, min_third_fourth


# folder_path = "/root/workspace/ctf53sc7v38s73e0mksg/maps/qd"
# max_second_third, min_second_third, max_third_fourth, min_third_fourth = get_max_min_values(folder_path)
# print(f"第二个@和第三个@之间的数的最大值: {max_second_third}")
# print(f"第二个@和第三个@之间的数的最小值: {min_second_third}")
# print(f"第三个@和第四个@之间的数的最大值: {max_third_fourth}")
# print(f"第三个@和第四个@之间的数的最小值: {min_third_fourth}")

# ( 120.4529197 , 36.5921062 )    -    ( 120.4150817 , 36.5921062 )    之间的距离为    3.378119342033595 km
# ( 120.4150817 , 36.6025564 )    -    ( 120.4150817 , 36.5921062 )    之间的距离为    1.162010822239352 km


import os
import random
import shutil


def split_images_randomly(source_folder, target_folder, num_to_keep):
    # 获取源文件夹下所有的图像文件列表
    image_files = [os.path.join(source_folder, file) for file in os.listdir(source_folder)
                   if file.endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_files) <= num_to_keep:
        print("图像文件数量小于或等于要保留的数量，无需移动操作。")
        return

    # 随机打乱图像文件列表的顺序
    random.shuffle(image_files)

    # 挑选出要保留在原文件夹的文件
    files_to_keep = image_files[:num_to_keep]
    # 剩下要移动的文件
    files_to_move = image_files[num_to_keep:]

    # 将需要保留的文件留在原文件夹（其实这里不需要额外操作，因为本身就在原文件夹）
    for file in files_to_keep:
        continue

    # 移动要移除的文件到目标文件夹
    for file in files_to_move:
        file_name = os.path.basename(file)
        target_file_path = os.path.join(target_folder, file_name)
        shutil.move(file, target_file_path)
        print(f"已将文件 {file} 移动到 {target_file_path}")


source_folder = "/root/workspace/ctf53sc7v38s73e0mksg/maps/qd"
target_folder = "/root/workspace/ctf53sc7v38s73e0mksg/maps/qd_train"
num_to_keep = 500
split_images_randomly(source_folder, target_folder, num_to_keep)