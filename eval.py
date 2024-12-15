import warnings

# 忽略UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image
import os
import torchvision.transforms as T
import logging
from datetime import datetime
import sys
import torchmetrics
from tqdm import tqdm
from math import sqrt
import numpy as np
import platform

from dataloaders.HCDataset import HCDataset, realHCDataset, InfiniteDataLoader, testHCDataset
from hc_db_cut import h_max, h_min, class_interval, classes_num
from models import helper, regression
import commons

from utils.checkpoint import save_checkpoint, resume_model, resume_train_with_params
from utils.inference import inference
from utils.losses import CoLoss
from utils.utils import move_to_device
from models.classifiers import AAMC
import parser

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
args = parser.parse_arguments()

# Parser变量
foldernames=['2013', '2017', '2019', '2020', '2022', '0001', '0002', '0003', '0004', 'real_photo']
# train_dataset_folders = ['2013', '2017', '2019', '2020', '2022', '0001', '0002', '0003', '0004']
train_dataset_folders = ['2022']
# test_datasets = ['real_photo', '2022', '2020']
test_datasets = ['2022', '2020']

if 'dinov2' in args.backbone.lower():
    backbone_info = {
        'input_size': args.train_resize,
        'num_trainable_blocks': args.num_trainable_blocks,
    }
elif 'efficientnet_v2' in args.backbone.lower():
    backbone_info = {
        'input_size': args.train_resize,
        'layers_to_freeze': args.layers_to_freeze,
    }
elif 'efficientnet' in args.backbone.lower():
    backbone_info = {
        'input_size': args.train_resize,
        'layers_to_freeze': args.layers_to_freeze,
    }
elif 'resnet' in args.backbone.lower():
    backbone_info = {
        'input_size': args.train_resize,
        'layers_to_freeze': args.layers_to_freeze,
        'layers_to_crop': list(args.layers_to_crop),
    }
agg_config = {}


train_transform = T.Compose([
    T.Resize(args.train_resize, antialias=True),
    T.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform =T.Compose([
    T.Resize(args.test_resize, antialias=True),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#### 初始化
commons.make_deterministic(args.seed)
commons.setup_logging(args.save_dir, console="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

#### Dataset & Dataloader
test_dataset_list = []
test_datasets_load = test_datasets
if 'real_photo' in test_datasets:
    real_photo_dataset = realHCDataset(transform=test_transform)
    test_datasets_load.remove('real_photo')
    test_dataset_list.append(real_photo_dataset)
if len(test_datasets_load) != 0:
    fake_photo_dataset = testHCDataset(base_path=args.test_set_path, foldernames=test_datasets_load, random_sample_from_each_place=False,transform=test_transform)
    test_dataset_list.append(fake_photo_dataset)
if len(test_dataset_list) > 1:
    test_dataset = ConcatDataset(test_dataset_list)
else:
    test_dataset = test_dataset_list[0]
test_img_num = len(test_dataset)
logging.info(f'Found {test_img_num} images in the test set.' )
test_num_workers = 2 if (args.device == "cuda" and platform.system() == "Linux") else 0
test_dl = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=test_num_workers, pin_memory=(args.device == "cuda"))

#### model
model = helper.HeightFeatureNet(backbone_arch=args.backbone, backbone_info=backbone_info, agg_arch=args.aggregator, agg_config=agg_config, regression_ratio=args.regression_ratio)
classifier = AAMC(in_features=model.feature_dim, out_features=classes_num, s=args.aamc_s, m=args.aamc_m)
# NOTE 分类器输出为类的数量的2倍，这里类的数量为12。

#### OPTIMIZER & SCHEDULER
classifier_optimizer = optim.Adam(classifier.parameters(), lr=args.classifier_lr)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.scheduler_patience, verbose=True) #NOTE: 学习率变化设置

#### Resume
if args.resume_model is not None:
    model = resume_model(model, classifier)

# if resume_info['resume_train']:
if args.resume_train is not None:
    model, optimizer, best_loss, start_epoch_num = resume_train_with_params(model, optimizer, scheduler)
else:
    start_epoch_num = 0
    best_loss = float('inf')

model = model.to(args.device)
# 训练模型
model.eval()
#### Validation
correct_1st_recall, in_thresh_1st_recall, in_thresh_1_3_recall = inference(args=args, model=model, classifier=classifier, test_dl=test_dl, test_img_num=test_img_num)


logging.info(f"{correct_1st_recall}")
logging.info(f"{in_thresh_1st_recall}")
logging.info(f"{in_thresh_1_3_recall}")
