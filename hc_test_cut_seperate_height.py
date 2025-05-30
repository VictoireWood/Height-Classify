import glob
from os import listdir
import os
import copy
from pickle import FALSE

import haversine
from haversine import haversine, Unit
import numpy
import cv2
from fractions import Fraction
from tqdm import tqdm, trange
import sys
import platform
import math
import utm
import pandas as pd
import random




def bool_based_on_probability(true_probability=0.5):
    import random
    return random.random() < true_probability

h_max = 700
h_min = 100
class_interval = 50

flight_test_range = range(h_min, h_max+1, class_interval) # 高度范围，5米一取，stop需要取大一点，否则range到不了h_max
class_number = len(flight_test_range) - 1   # 相当于有13个分割线
times = 20

flight_heights = []

for idx in range(class_number):
    range_start = flight_test_range[idx]
    range_end = flight_test_range[idx+1]
    for time in range(times):
        h = random.randint(range_start, range_end-1)    # range_end可以被取到，所以要减去1
        flight_heights.append(h)
print(flight_heights)

# TODO: 
# 分辨率
resolution_w = 2048
resolution_h = 1536
# 焦距
focal_length = 1200  # TODO: the intrinsics of the camera

def photo_area_meters(flight_height):
    # 默认width更长
    map_tile_meters_w = resolution_w / focal_length * flight_height   # 相机内参矩阵里focal_length的单位是像素
    map_tile_meters_h = resolution_h / focal_length * flight_height # NOTE w768*h576
    return map_tile_meters_h, map_tile_meters_w

if platform.system() == "Windows":
    slash = '\\'
else:
    slash = '/'

def crop_rot_img_wo_border(image, crop_width, crop_height, crop_center_x, crop_center_y, angle):
    # 裁剪并旋转图像
    half_crop_width = (crop_width / 2)
    half_crop_height = (crop_height / 2)
    # 矩形四个顶点的坐标
    x1, y1 = crop_center_x - half_crop_width, crop_center_y - half_crop_height  # 顶点A的坐标
    x2, y2 = crop_center_x - half_crop_width, crop_center_y + half_crop_height  # 顶点B的坐标
    x3, y3 = crop_center_x + half_crop_width, crop_center_y + half_crop_height  # 顶点C的坐标
    x4, y4 = crop_center_x + half_crop_width, crop_center_y - half_crop_height  # 顶点D的坐标

    # 矩形中心点坐标
    Ox = (x1 + x2 + x3 + x4) / 4
    Oy = (y1 + y2 + y3 + y4) / 4

    # 角度转换为弧度
    alpha_rad = angle * math.pi / 180

    # 旋转矩阵
    cos_alpha = math.cos(alpha_rad)
    sin_alpha = math.sin(alpha_rad)

    # 计算新坐标
    def rotate_point(x, y, Ox, Oy, cos_alpha, sin_alpha):
        return (
            Ox + (x - Ox) * cos_alpha - (y - Oy) * sin_alpha,
            Oy + (x - Ox) * sin_alpha + (y - Oy) * cos_alpha
        )

    # 新的四个顶点坐标
    new_x1, new_y1 = rotate_point(x1, y1, Ox, Oy, cos_alpha, sin_alpha)
    new_x2, new_y2 = rotate_point(x2, y2, Ox, Oy, cos_alpha, sin_alpha)
    new_x3, new_y3 = rotate_point(x3, y3, Ox, Oy, cos_alpha, sin_alpha)
    new_x4, new_y4 = rotate_point(x4, y4, Ox, Oy, cos_alpha, sin_alpha)
    start_x = int(min((new_x1, new_x2, new_x3, new_x4)))
    end_x = int(max((new_x1, new_x2, new_x3, new_x4)))
    start_y = int(min((new_y1, new_y2, new_y3, new_y4)))
    end_y = int(max((new_y1, new_y2, new_y3, new_y4)))

    if start_x < 0 or start_y < 0:
        return None
    elif end_x > image.shape[1] or end_y > image.shape[0]:
        return None
    else:
        cropped_image = image[start_y:end_y, start_x:end_x]

    def rotate_image(image, angle, new_w, new_h):
        (h, w) = image.shape[:2]
        (cx, cy) = (w // 2, h // 2)

        # 计算旋转矩阵
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        # 调整旋转矩阵的平移部分
        M[0, 2] += (new_w / 2) - cx
        M[1, 2] += (new_h / 2) - cy

        # 执行旋转并返回新图像
        rotated = cv2.warpAffine(image, M, (new_w, new_h))
        return rotated

    result = rotate_image(cropped_image, angle, crop_width, crop_height)
    return result

def generate_map_tiles(raw_map_path:str, patches_save_dir:str, rotation_angles=[0]):

    # 飞行高度
    # flight_height = 150
    # flight_height = int(patches_save_dir.split('_')[-1])

    #TODO: 
    target_w = 480                  # TODO: set the width corresponding to the shape of your query image
    # target_h = 360
    # w_h_factor = target_h / target_w
    # map_tile_width_meters = 300     # TODO: set the meters in width the you want to crop

    # 这是指地图切片的宽度对应到地面上是多长（米为单位）
    # map_tile_heigth_meters = map_tile_width_meters * w_h_factor


    for flight_height in flight_heights:
    # 对应地面宽高（像素为单位）
    #LINK: https://cameraharmony.com/wp-content/uploads/2020/03/focal-length-graphic-1-2048x1078.png

        flight_class = (flight_height - h_min) // class_interval

        map_tile_meters_w = resolution_w / focal_length * flight_height   # 相机内参矩阵里focal_length的单位是像素
        map_tile_meters_h = resolution_h / focal_length * flight_height # NOTE w768*h576

        w_h_factor = resolution_w / resolution_h
        target_h = round(target_w / w_h_factor)         # NOTE 最后要resize的高度(h360,w480)

        map_data = cv2.imread(raw_map_path)

        map_w = map_data.shape[1]   # 大地图像素宽度
        map_h = map_data.shape[0]   # 大地图像素高度

        gnss_data = raw_map_path.split(slash)[-1]

        LT_lon = float(gnss_data.split('@')[2]) # left top 左上
        LT_lat = float(gnss_data.split('@')[3])
        RB_lon = float(gnss_data.split('@')[4]) # right bottom 右下
        RB_lat = float(gnss_data.split('@')[5])

        lon_res = (RB_lon - LT_lon) / map_w     # 大地图的纬线方向每像素代表的经度跨度
        lat_res = (RB_lat - LT_lat) / map_h     # 大地图的经线方向每像素代表的纬度跨度

        # map_width_meters = abs(LT_e - RB_e)
        # map_height_meters = abs(LT_n - RB_n)

        mid_lat = (LT_lat + RB_lat) / 2
        mid_lon = (LT_lon + RB_lon) / 2

        map_width_meters = haversine((mid_lat, LT_lon), (mid_lat, RB_lon), unit=Unit.METERS)
        map_height_meters = haversine((LT_lat, mid_lon), (RB_lat, mid_lon), unit=Unit.METERS)
        pixel_per_meter_factor = ((map_w / map_width_meters) + (map_h / map_height_meters)) / 2     # 得出来是像素/米，每米对应多少像素

        pixel_per_meter_factor = ((map_w / map_width_meters) + (map_h / map_height_meters)) / 2     # 得出来是像素/米，每米对应多少像素

        # stride_x = round(pixel_per_meter_factor * map_tile_meters_w * stride_ratio)
        # stride_y = round(pixel_per_meter_factor * map_tile_meters_h * stride_ratio)

        img_w = round(pixel_per_meter_factor * map_tile_meters_w)
        img_h = round(pixel_per_meter_factor * map_tile_meters_h)

        # 计算要切多少个tile
        # iter_w = int((map_w - img_w) / stride_x) + 1
        # iter_h = int((map_h - img_h) / stride_y) + 1
        # iter_total = iter_w * iter_h * len(rotation_angles)

        loc_x = random.randint(0, map_w - img_w)
        loc_y = random.randint(0, map_h - img_h)

        # with trange(iter_total, desc=gnss_data) as tbar:
        #     i = 0
        #     loc_x = 0
        #     # LINK: https://blog.csdn.net/winter2121/article/details/111356587
        #     while loc_x < map_w - img_w:    # 已分割像素宽度<大地图宽度-地图切片宽度
        #         loc_y = 0
        #         while loc_y < map_h - img_h:
        LT_cur_lon = str(loc_x * lon_res + LT_lon)
        LT_cur_lat = str(loc_y * lat_res + LT_lat)
        RB_cur_lon = str((loc_x + img_w) * lon_res + LT_lon)
        RB_cur_lat = str((loc_y + img_h) * lat_res + LT_lat)
        CT_cur_lon = str((loc_x + img_w / 2) * lon_res + LT_lon)    # centre
        CT_cur_lat = str((loc_y + img_h / 2) * lat_res + LT_lat)
        CT_cur_lon_ = (loc_x + img_w / 2) * lon_res + LT_lon    # centre
        CT_cur_lat_ = (loc_y + img_h / 2) * lat_res + LT_lat
        # CT_utm_e, CT_utm_n = S51_UTM(CT_cur_lon_, CT_cur_lat_)
        CT_utm_e, CT_utm_n, _, _ = utm.from_latlon(CT_cur_lat_, CT_cur_lon_)

        crop_center_x = loc_x + img_w / 2
        crop_center_y = loc_y + img_h / 2

        # for rotation_angle in rotation_angles:

            # NOTE 如果不想全部生成
            # random_cut = bool_based_on_probability(0.4)
            # if not random_cut:
            #     i += 1
            #     tbar.set_postfix(rate=i/iter_total)
            #     tbar.update()
            #     continue

        alpha = 90
        filename = f'@{year}@{flight_height:.2f}@{flight_class:02d}@{alpha:.2f}@{loc_x}@{loc_y}@.png'
        # filename = f'@{rotation_angle}@{flight_height}@{CT_utm_e}@{CT_utm_n}@.png'
        # @角度@高度@utm_e@utm_n@.png

        # if os.path.exists(filename):
        #     i += 1
        #     tbar.set_postfix(rate=i/iter_total, tiles=i)
        #     tbar.update()
        #     continue

        img_seg_pad = map_data[loc_y:loc_y + img_h, loc_x:loc_x + img_w]
        # print(img_seg_pad.shape)

        # img_seg_pad = crop_rot_img_wo_border(map_data, img_w, img_h, crop_center_x, crop_center_y, rotation_angle)

        if img_seg_pad is None:
            pass
        else:
            # img_seg_pad = map_data[loc_y:loc_y + img_h, loc_x:loc_x + img_w]
            # img_seg_pad = cv2.resize(img_seg_pad, (target_w, target_h), interpolation = cv2.INTER_LINEAR)
            img_seg_pad = cv2.resize(img_seg_pad, (target_w, target_h), interpolation = cv2.INTER_LANCZOS4)

            # data_line = pd.DataFrame([[year, gnss_data, flight_height, flight_class, alpha, loc_x, loc_y]], columns=['year', 'origin_img', 'flight_height', 'flight_class', 'rotation_angle','loc_x', 'loc_y'])
            # data_line.to_csv(csv_path, mode='a', index=False, header=False)
            
            # 决定是否要旋转
            # img_seg_pad = numpy.clip(numpy.rot90(img_seg_pad, 1), 0, 255).astype(numpy.uint8)  # rotate if necessary

            # cv2.imwrite(patches_save_dir + '@map%s.png' % (
            #         '@' + LT_cur_lon + '@' + LT_cur_lat + '@' + RB_cur_lon + '@' + RB_cur_lat + '@'), img_seg_pad)
            # print('%s.png' % ('@' + LT_cur_lon + '@' + LT_cur_lat + '@' + RB_cur_lon + '@' + RB_cur_lat + '@'))
            
            patches_save_dir_height = os.path.join(patches_save_dir, f'{flight_height:04d}')

            save_file_path = os.path.join(f'{patches_save_dir_height}',filename)

            cv2.imwrite(save_file_path, img_seg_pad)
            # cv2.imshow('image',img_seg_pad)
                    

            # i += 1
            # tbar.set_postfix(rate=i/iter_total)
            # tbar.update()


            #     loc_y = loc_y + stride_y

            # loc_x = loc_x + stride_x

        print(f"Finishing {year}-{flight_height}m")  


if __name__ == '__main__':

    # TODO 分数据集
    stage = "train"

    basedir = r'/root/workspace/maps/QDRaw/'
    basedir = r'D:\.cache\QDRaw'
    basedir = r'F:\.cache\QDRaw'
    basedir = r'/root/workspace/crikff47v38s73fnfgdg/maps/QDRaw/'

    cities_dir = r'/root/workspace/crikff47v38s73fnfgdg/maps/Cities'

    basedir = r'/root/workspace/ctf53sc7v38s73e0mksg/maps/QDRaw/'

    cities_dir = r'/root/workspace/ctf53sc7v38s73e0mksg/maps/Cities'

    # map_dirs = {
    #     "2012": rf"{basedir}201209{slash}@rot90map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.jpg",
    #     "2013": rf"{basedir}201310{slash}@rot90map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.jpg",  
    #     "2017": rf"{basedir}201710{slash}@rot90map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.jpg",
    #     "2019": rf"{basedir}201911{slash}@rot90map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.jpg",
    #     "2020": rf"{basedir}202002{slash}@rot90map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.jpg",  
    #     "2022": rf"{basedir}202202{slash}@rot90map@120.42118549346924@36.60643328438966@120.4841423034668@36.573836401969416@.jpg"  
    # }
    map_dirs = {
        #` "2012": os.path.join(basedir, '201209', '@rot90map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.jpg'),
        # "2013": os.path.join(basedir, '201310', '@rot90map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.jpg'),  
        # "2017": os.path.join(basedir, '201710', '@rot90map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.jpg'),
        # "2019": os.path.join(basedir, '201911', '@rot90map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.jpg'),
        # "2020": os.path.join(basedir, '202002', '@rot90map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.jpg'),  
        # "2022": os.path.join(basedir, '202202', '@rot90map@120.42118549346924@36.60643328438966@120.4841423034668@36.573836401969416@.jpg'), 
        "ct01": os.path.join(cities_dir, 'ct01', '2022', '@map@116.35551452636719@40.09815882135811@116.44632339477539@40.15118932709900@.jpg'),
        "ct02": os.path.join(cities_dir, 'ct01', '2022','@map@121.34485244750977@31.08564938117820@121.43737792968750@31.12900748947799@.jpg'),
    }

    patches_save_root_dir = r'/root/workspace/maps/HE-100-700-test'
    patches_save_root_dir = r'D:\.cache\HE-100-700-test'
    patches_save_root_dir = r'F:\.cache\HE-100-700-test'
    patches_save_root_dir = r'/root/workspace/crikff47v38s73fnfgdg/maps/HE-100-700-test'
    patches_save_root_dir = r'/root/workspace/crikff47v38s73fnfgdg/maps/HC-100-700-test'
    patches_save_root_dir = r'/root/workspace/ctf53sc7v38s73e0mksg/maps/HC-100-700-test'

    # alpha_list = range(0, 360, 30)
    alpha_list = [0]


    total_iterations = len(map_dirs)*len(flight_heights)  # Total iterations  
    current_iteration = 0  # To keep track of progress  
    
    for year, map_dir in map_dirs.items():

        header = pd.DataFrame(columns=['year', 'origin_img', 'flight_height', 'flight_class', 'rotation_angle','loc_x', 'loc_y'])
        # csv_dir = os.path.join(patches_save_root_dir, 'Dataframes')
        # os.makedirs(csv_dir, exist_ok=True)
        # csv_path = os.path.join(csv_dir, f'{year}.csv')
        # header.to_csv(csv_path, mode='w', index=False, header=True)
        
        save_dir_year = os.path.join(patches_save_root_dir, f'{year}')

        if not os.path.exists(save_dir_year):  
            os.makedirs(save_dir_year)  

        patches_save_dir = save_dir_year  # Save directory for the current year  
        print(f"Saving tiles to: {patches_save_dir} ")  
            
            # if not os.path.exists(patches_save_dir):  
            #     os.mkdir(patches_save_dir)
        # for flight_height in flight_heights:
            
        # stride_ratio_str = f'1/{stride_ratio}'

        generate_map_tiles(map_dir, patches_save_dir, alpha_list)
    
        # current_iteration += 1  # Increment the progress counter  

        # # Calculate and display progress  
        # progress = (current_iteration / total_iterations) * 100  
        # print(f"[Progress] {progress:.2f}% complete")  
