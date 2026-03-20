import warnings
import os
import sys
import platform
import psutil
import time
import logging
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T

# 忽略UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

import parser
import commons
from dataloaders.HCDataset import HCDataset_shN, TestDataset, TestDatasetNew
from utils.checkpoint import resume_model_with_classifiers, resume_train_with_groups
from utils.inference import inference_latency_memory, inference_with_groups_csv
from utils.utils import get_utm_from_path
from models.classifiers import AAMC, QAMC, LMCC, LinearLayer
from models import helper



os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
args = parser.parse_arguments()

assert args.train_set_path is not None, 'you must specify the train set path'
assert os.path.exists(args.train_set_path), 'train set path must exist'
assert (args.test_set_path is not None) or (args.val_set_path is not None), 'you must specify the test set path'
if args.test_set_path is not None:
    assert os.path.exists(args.test_set_path), 'test set path must exist'



# Parser变量

if args.dataset_name == 'ct01':
    train_dataset_folders = ['ct01']
elif args.dataset_name == 'ct02':
    train_dataset_folders = ['ct02']
else:
    train_dataset_folders = args.dataset_name

test_datasets = args.test_set_list

if 'resnet' in args.backbone.lower():
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


test_dataset_list = []
test_datasets_load = test_datasets
for dataset_name in test_datasets:
    if 'qd_test' in dataset_name:
        new_photo_dataset = TestDatasetNew(test_folder=args.test_set_path, M=args.M, N=args.N, image_size=args.test_resize)
        test_datasets_load.remove(dataset_name)
        test_dataset_list.append(new_photo_dataset)
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
model = helper.HeightFeatureNet(backbone_arch=args.backbone, backbone_info=backbone_info, agg_arch=args.aggregator, agg_config=agg_config)
# classifier = AAMC(in_features=model.feature_dim, out_features=classes_num, s=args.aamc_s, m=args.aamc_m)
# NOTE 分类器输出为类的数量的2倍，这里类的数量为12。

model = model.to(args.device)

# TODO 多个分类器进行训练

if args.classifier == 'AAMC':
    classifiers = [AAMC(in_features=model.feature_dim, out_features=group.get_classes_num(), s=args.aamc_s, m=args.aamc_m) for group in groups]
elif args.classifier == 'QAMC':
    classifiers = [QAMC(embedding_size=model.feature_dim, classnum=group.get_classes_num(), m=args.aamc_m, s=args.aamc_s) for group in groups]
elif args.classifier == 'LMCC':
    classifiers = [LMCC(embedding_size=model.feature_dim, classnum=group.get_classes_num(), m=args.aamc_m, s=args.aamc_s) for group in groups]
elif args.classifier == 'LinearLayer':
    classifiers = [LinearLayer(embedding_size=model.feature_dim, classnum=group.get_classes_num()) for group in groups]


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

#### Resume
if args.resume_model is not None:
    model, classifiers = resume_model_with_classifiers(model, classifiers)
else:
    raise ValueError("No model to resume.")

# correct_class_recall, threshold_recall = inference_latency_memory(args=args, model=model, classifiers=classifiers, test_dl=test_dl, groups=groups, num_test_images=test_img_num)
avg_latency, max_latency, cpu_peak, cpu_delta, gpu_peak = inference_latency_memory(
    args=args,
    model=model,
    classifiers=classifiers,
    test_dl=test_dl,
    groups=groups,
    num_test_images=test_img_num
)


logging.info(f"Performance Metrics:")
logging.info(f"Avg latency: {avg_latency:.2f}ms")
logging.info(f"Max latency: {max_latency:.2f}ms")
logging.info(f"Peak CPU ABS: {cpu_peak:.2f}MB")
logging.info(f"CPU DELTA: {cpu_delta:.2f}MB")
logging.info(f"Peak GPU: {gpu_peak:.2f}MB")

# logging.info(f"Test LR: {correct_class_recall}, {threshold_recall}")