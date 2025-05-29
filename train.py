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

from dataloaders.HCDataset import HCDataset, realHCDataset, InfiniteDataLoader, testHCDataset, HCDataset_shN
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
# if args.train_photos:

train_dataset = HCDataset(train_dataset_folders, random_sample_from_each_place=True, transform=train_transform, base_path=args.train_set_path)
train_dataloader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=(args.device == "cuda"), drop_last=False, persistent_workers=args.num_workers>0)
# train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=(args.device == "cuda"), drop_last=False, persistent_workers=args.num_workers>0)

# ANCHOR 把一个分类器增加到N个分类器

groups = []
for n in range(args.N * args.N):
    group = HCDataset_shN(args.train_set_path, dataset_name=args.dataset_name, group_num=n, M=args.M, N=args.N,
                       min_images_per_class=args.min_images_per_class,
                       transform=train_transform,
                       )
    groups.append(group)

# train_dl = DataLoader(train_dataset, batch_size=, num_workers=args.num_workers, shuffle=True, pin_memory=(args.device == "cuda"), drop_last=False, persistent_workers=args.num_workers>0)
# iterations_num = len(train_dataset) // train_batch_size
logging.info(f'Found {len(train_dataset)} images in the training set.' )

test_dataset_list = []
test_datasets_load = test_datasets
if 'real_photo' in test_datasets:
    real_photo_dataset = realHCDataset(transform=test_transform)
    test_datasets_load.remove('real_photo')
    test_dataset_list.append(real_photo_dataset)
if len(test_datasets_load) != 0:
    fake_photo_dataset = testHCDataset(base_path=args.test_set_path, random_sample_from_each_place=False,transform=test_transform)
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


# TODO 多个分类器进行训练

classifiers = [AAMC(in_features=model.feature_dim, out_features=classes_num, s=args.aamc_s, m=args.aamc_m) for group in groups]

# 看model的可训练参数多少
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
logging.info(f'Trainable parameters: {params/1e6:.4}M')

#### OPTIMIZER & SCHEDULER
classifier_optimizer = optim.Adam(classifier.parameters(), lr=args.classifier_lr)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.scheduler_patience, verbose=True) #NOTE: 学习率变化设置

#### Resume
if args.resume_model is not None:
    model, classifier = resume_model(model, classifier)

# if resume_info['resume_train']:
if args.resume_train is not None:
    model, optimizer, best_loss, start_epoch_num = resume_train_with_params(model, optimizer, scheduler)
else:
    start_epoch_num = 0
    best_loss = float('inf')

model = model.to(args.device)

### Train&Loss
# 初始化模型、损失函数和优化器
cross_entropy_loss = torch.nn.CrossEntropyLoss()    #NOTE: 交叉熵损失，应该是softmax通用的loss形式


# 训练模型
scaler = torch.GradScaler('cuda')
for epoch in range(start_epoch_num, args.epochs_num):
    if optimizer.param_groups[0]['lr'] < 1e-6:
        logging.info('LR dropped below 1e-6, stopping training...')
        break

    train_loss = torchmetrics.MeanMetric().to(args.device)

    # Select classifier and dataloader according to epoch
    train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=classes_num).to(args.device) # EDIT 3 train_acc
    classifier = classifier.to(args.device)
    move_to_device(classifier_optimizer, args.device)



    #### Train
    # train_dataloader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=(args.device == "cuda"), drop_last=False)
    dataloader_iterator = iter(train_dataloader)
    model = model.train()

    tqdm_bar = tqdm(range(args.iterations_per_epoch), ncols=100, desc="")
    #NOTE: tqmd.tqmd修饰一个可迭代对象，返回一个与原始可迭代对象完全相同的迭代器，但每次请求值时都会打印一个动态更新的进度条。
    for iteration in tqdm_bar:
        images, labels, _ = next(dataloader_iterator)   # NOTE return tensor_image, class编号
        images, labels = images.to(args.device), labels.to(args.device)

        optimizer.zero_grad()
        classifier_optimizer.zero_grad()

        # with torch.cuda.amp.autocast('cuda',enabled=True): # NOTE: 加上了'cuda'参数和enabled  
        # ORIGION 
        with torch.autocast('cuda'):    # EDIT
            # LINK: https://pytorch.org/docs/stable/amp.html
            descriptors = model(images)
            # 1) 'output' is respectively the angular or cosine margin, of the AMCC or LMCC.
            # 2) 'logits' are the logits obtained multiplying the embedding for the
            # AMCC/LMCC weights. They are used to compute tha accuracy on the train batches 
            output, logits = classifier(descriptors, labels)
            loss = cross_entropy_loss(output, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.step(classifier_optimizer)
        scaler.update()

        train_acc.update(logits, labels)    # ORIGION
        train_loss.update(loss.item())
        tqdm_bar.set_description(f"{loss.item():.1f}")
        del loss, images, output
        _ = tqdm_bar.refresh()  # ORIGION
        _ = tqdm_bar.update()   # EDIT

    # classifier = classifier.cpu()
    # move_to_device(classifier_optimizer, 'cpu') 

    #### Validation
    correct_1st_recall, in_thresh_1st_recall, in_thresh_1_3_recall = inference(args=args, model=model, classifier=classifier, test_dl=test_dl, test_img_num=test_img_num)

    train_acc = train_acc.compute() * 100   # train_
    train_loss = train_loss.compute()

    if train_loss < best_loss:
        is_best = True
        best_loss = train_loss
    else:
        is_best = False


    logging.info(f"E{epoch: 3d}, " + 
                #  f"train_acc: {train_acc.item():.1f}, " +
                 f"train_loss: {train_loss.item():.2f}, best_train_loss: {scheduler.best:.2f}, " +
                 f"not improved for {scheduler.num_bad_epochs}/{args.scheduler_patience} epochs, " +
                 f"lr: {round(optimizer.param_groups[0]['lr'], 21)}")
    logging.info(f"E{epoch: 3d}, {correct_1st_recall}")
    logging.info(f"E{epoch: 3d}, {in_thresh_1st_recall}")
    logging.info(f"E{epoch: 3d}, {in_thresh_1_3_recall}")

    scheduler.step(train_loss)
    
    save_checkpoint({"epoch_num": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "classifier_state_dict": classifier.state_dict(),
        # "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "classifier_optimizer_state_dict": classifier_optimizer.state_dict(),
        "args": args,
        "best_train_loss": best_loss
    }, is_best, args.save_dir)


print("Training complete.")