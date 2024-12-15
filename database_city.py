from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import os
import pandas as pd
from glob import glob
from tqdm import tqdm, trange

torch.manual_seed(1234)

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("/root/workspace/VL/Qwen-VL-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained("/root/workspace/VL/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained("/root/workspace/VL/Qwen-VL-Chat", trust_remote_code=True)

# 1st dialogue turn
# query = tokenizer.from_list_format([
#     {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'}, # Either a local path or an url
#     {'text': '这是什么?'},
# ])
# response, history = model.chat(tokenizer, query=query, history=None)
# print(response)


percentage = 80
scr_dir = '/root/workspace/maps/HE_Train_qdcity'
Images_dir = os.path.join(scr_dir, 'Images')
# root, dirs, files = os.walk(Images_dir, topdown=False)
# dirs = os.listdir(Images_dir)
contents = os.listdir(Images_dir)
# 过滤出只包含子文件夹名称的列表  
subfolders = [f for f in contents if os.path.isdir(os.path.join(Images_dir, f))]
dirs = subfolders
header = pd.DataFrame(columns=['year', 'flight_height', 'rotation_angle', 'loc_x', 'loc_y'])

for year_str in dirs:
    img_dir = os.path.join(Images_dir, year_str)
    df_dir = os.path.join(scr_dir, 'Dataframes')
    os.makedirs(df_dir, exist_ok=True)
    csv_path = os.path.join(df_dir, f'{year_str}.csv')
    header.to_csv(csv_path, mode='w', index=False, header=True)
    images_paths = sorted(glob(f"{img_dir}/**/*.png", recursive=True))
    img_num = len(images_paths)
    print(f'{year_str}: {img_num} images')
    tbar = trange(img_num, desc = year_str)
    idx = 0
    # for idx in range(img_num):
    for _ in tbar:
        image_path = images_paths[idx]
        query = tokenizer.from_list_format([
            {'image': image_path}, # Either a local path or an url
            # {'text': f'请只回答是或否，这张图中是否只有水体和农田？'},
            {'text': f'请只回答是或否，这张图中是否包含建筑物？'},
        ])
        response, history = model.chat(tokenizer, query=query, history=None)
        if response != '是' and response != '否':
            print(response)
            import sys
            sys.exit()
        if '否' in response:
            os.remove(image_path)
        elif '是' in response:
            # f'@{year_str}@{flight_height:.2f}@{alpha:.2f}@{loc_w}@{loc_h}@.png'
            meta = image_path.split('@')
            # flight_height = round(float(meta[-5]), 2)
            # rotation_angle = round(float(meta[-4]), 2)
            # loc_x = int(meta[-3])
            # loc_y = int(meta[-2])

            flight_height = meta[-5]
            rotation_angle = meta[-4]
            loc_x = meta[-3]
            loc_y = meta[-2]
            data_line = pd.DataFrame([[year_str, flight_height, rotation_angle, loc_x, loc_y]], columns=['year', 'flight_height', 'rotation_angle', 'loc_x', 'loc_y'])
            data_line.to_csv(csv_path, mode='a', index=False, header=False)

        idx += 1