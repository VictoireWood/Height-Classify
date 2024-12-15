from glob import glob
import shutil
import os
from decimal import Decimal
import utm

src_dir = '/root/shared-storage/shaoxingyu/QingDaoUAVPic'
# dst_dir = '/root/shared-storage/shaoxingyu/workspace_backup/dcqd_test'
dst_dir = '/root/workspace/maps/real_photos'

copy = False

# NOTE: 原始格式@120.4280576000@36.5972545000@.png
# filename = f'@{year_str}@{flight_height:.2f}@{alpha:.2f}@{loc_w}@{loc_h}@.png'

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

if len(os.listdir(dst_dir)) == 0 or not os.path.exists(dst_dir):
    copy = True

copy = False

if copy:
    shutil.copytree(src_dir,dst_dir,dirs_exist_ok=True)

scr_images_paths = sorted(glob(f"{dst_dir}/**/*.png", recursive=True))
for img_path in scr_images_paths:
    img_info = img_path.split('@')
    if len(img_info) == 4:
        CT_lon = float(img_info[1])
        CT_lat = float(img_info[2])
        CT_lon = Decimal(img_info[1])
        CT_lat = Decimal(img_info[2])
        # utm_e, utm_n = S51_UTM(CT_lon, CT_lat)
        utm_e, utm_n, _, _ = utm.from_latlon(float(img_info[2]), float(img_info[1]))
        path_str_list = ['VPR/',"VPR2/","VPR_h400/","VPR_h630/"]
        if path_str_list[0] in img_path:
            flight_height = 175 # 这只是个近似值，实际是150-200之间
        elif path_str_list[1] in img_path:
            flight_height = 200
        elif path_str_list[2] in img_path:
            flight_height = 400
        elif path_str_list[3] in img_path:
            flight_height = 630
        # photo_meters_w = height_to_width(flight_height)
        filename = f'@none@{flight_height:3d}@{utm_e:017.10f}@{utm_n:017.10f}@.png'
        print(filename)
        # filename = f'@none@{photo_meters_w:012.7f}@{flight_height:3d}@{utm_e:017.10f}@{utm_n:017.10f}@.png'
        new_path = img_info[0] + filename
        os.rename(img_path, new_path)
    elif len(img_info) == 7:
        pass
    else:
        print(f'Delete File for NameError: {img_path}')
        # os.remove(img_path)


    
