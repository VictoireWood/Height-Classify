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

from dataloaders.HCDataset import realHCDataset_N, InfiniteDataLoader, HCDataset_shN, TestDataset
from models import helper, regression
import commons

from utils.checkpoint import save_checkpoint_with_groups, resume_model_with_classifiers, resume_train_with_groups
from utils.inference import inference_with_groups
from utils.losses import CoLoss
from utils.utils import move_to_device
from models.classifiers import AAMC
import parser

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
args = parser.parse_arguments()

assert args.train_set_path is not None, 'you must specify the train set path'
assert os.path.exists(args.train_set_path), 'train set path must exist'
# assert args.val_set_path is not None, 'you must specify the val set path'
assert (args.test_set_path is not None) or (args.val_set_path is not None), 'you must specify the test set path'
if args.test_set_path is not None:
    assert os.path.exists(args.test_set_path), 'test set path must exist'
if args.val_set_path is not None:
    assert os.path.exists(args.val_set_path), 'val set path must exist (real photo)'



# Parser变量
foldernames=['2013', '2017', '2019', '2020', '2022', '0001', '0002', '0003', '0004', 'real_photo', 'ct01', 'ct02']
# train_dataset_folders = ['2013', '2017', '2019', '2020', '2022', '0001', '0002', '0003', '0004']
# train_dataset_folders = ['2022']
# train_dataset_folders = ['ct01']
train_dataset_folders = ['ct02']
# test_datasets = ['real_photo', '2022', '2020']
# test_datasets = ['2022', '2020']
test_datasets = train_dataset_folders


if args.dataset_name == 'ct01':
    train_dataset_folders = ['ct01']
    test_datasets = ['ct01']
elif args.dataset_name == 'ct02':
    train_dataset_folders = ['ct02']
    test_datasets = ['ct02']


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
# if args.train_photos:

groups = []
for n in range(args.N):
    group = HCDataset_shN(group_num=n, dataset_name=args.dataset_name,train_path=args.train_set_path, train_dataset_folders=train_dataset_folders, M=args.M, N=args.N,min_images_per_class=args.min_images_per_class,transform=train_transform)
    groups.append(group)

# train_dataset = HCDataset(train_dataset_folders, random_sample_from_each_place=True, transform=train_transform, base_path=args.train_set_path)
# train_dataloader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=(args.device == "cuda"), drop_last=False, persistent_workers=args.num_workers>0)
# train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=(args.device == "cuda"), drop_last=False, persistent_workers=args.num_workers>0)

# ANCHOR 把一个分类器增加到N个分类器



# train_dl = DataLoader(train_dataset, batch_size=, num_workers=args.num_workers, shuffle=True, pin_memory=(args.device == "cuda"), drop_last=False, persistent_workers=args.num_workers>0)
# iterations_num = len(train_dataset) // train_batch_size
# logging.info(f'Found {len(train_dataset)} images in the training set.' )


test_dataset_list = []
test_datasets_load = test_datasets
if 'real_photo' in test_datasets:
    real_photo_dataset = realHCDataset_N(base_path=args.val_set_path, M=args.M, N=args.N, transform=test_transform)
    test_datasets_load.remove('real_photo')
    test_dataset_list.append(real_photo_dataset)
if len(test_datasets_load) != 0:
    fake_photo_dataset = TestDataset(test_folder=args.test_set_path, test_datasets=test_datasets, M=args.M, N=args.N, image_size=args.test_resize)
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
# model = helper.HeightFeatureNet(backbone_arch=args.backbone, backbone_info=backbone_info, agg_arch=args.aggregator, agg_config=agg_config, regression_ratio=args.regression_ratio)
model = helper.HeightFeatureNet(backbone_arch=args.backbone, backbone_info=backbone_info, agg_arch=args.aggregator, agg_config=agg_config)
# classifier = AAMC(in_features=model.feature_dim, out_features=classes_num, s=args.aamc_s, m=args.aamc_m)
# NOTE 分类器输出为类的数量的2倍，这里类的数量为12。

model = model.to(args.device)

# TODO 多个分类器进行训练

classifiers = [AAMC(in_features=model.feature_dim, out_features=group.get_classes_num(), s=args.aamc_s, m=args.aamc_m) for group in groups]

classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifier_lr) for classifier in classifiers]

logging.info(f"Using {len(groups)} groups")
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[g.get_classes_num() for g in groups]}")
logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")
logging.info(f"Feature dim: {model.feature_dim}")
logging.info(f"resume_model: {args.resume_model}")


# 看model的可训练参数多少
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
logging.info(f'Trainable parameters: {params/1e6:.4}M')

#### OPTIMIZER & SCHEDULER
# classifier_optimizer = optim.Adam(classifier.parameters(), lr=args.classifier_lr)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.scheduler_patience, verbose=True) #NOTE: 学习率变化设置

#### Resume
if args.resume_model is not None:
    # model, classifier = resume_model(model, classifier)
    model, classifiers = resume_model_with_classifiers(model, classifiers)

if args.resume_train is not None:
    # model, optimizer, best_loss, start_epoch_num = resume_train_with_params(model, optimizer, scheduler)

    model, model_optimizer, classifiers, classifiers_optimizers, best_train_loss, start_epoch_num = \
        resume_train_with_groups(args.save_dir, model, optimizer, classifiers, classifiers_optimizers)
    epoch_num = start_epoch_num - 1
    best_loss = best_train_loss
    logging.info(f"Resuming from epoch {start_epoch_num} with best train loss {best_train_loss:.2f} " +
                 f"from checkpoint {args.resume_train}")
else:
    best_valid_acc = 0
    start_epoch_num = 0
    best_loss = float('inf')



### Train&Loss
# 初始化模型、损失函数和优化器
cross_entropy_loss = torch.nn.CrossEntropyLoss()    #NOTE: 交叉熵损失，应该是softmax通用的loss形式


torch.cuda.empty_cache()

