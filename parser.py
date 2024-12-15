
import os
import argparse
from datetime import datetime
import platform

# from he_database_generate import patches_save_root_dir    # EDIT

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # EDIT
    # parser.add_argument("--H", type=int, default=100)   # NOTE: 按照高度分组

    # Training parameters
    parser.add_argument("--seed", type=int, default=0, help="_")
    parser.add_argument("-ipe", "--iterations_per_epoch", type=int, default=2000, help="_")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="_")
    parser.add_argument('--num_workers', type=int, default=4, help="_")
    parser.add_argument("--scheduler_patience", type=int, default=10, help="_")
    parser.add_argument("--epochs_num", type=int, default=500, help="_")
    # parser.add_argument("--train_resize", type=int, default=(224, 224), help="_") # ANCHOR
    # parser.add_argument("--train_resize", type=tuple, default=(360, 480), help="_") # REVIEW version 1
    # parser.add_argument("--train_resize", type=int, default=(222, 296), help="_")   # REVIEW    如果用DINOv2，就改成210*280
    parser.add_argument("--train_resize", type=int, nargs=2, default=(336, 448), help="_")   # REVIEW    如果用DINOv2，就改成210*280
    # parser.add_argument("--test_resize", type=int, default=256, help="_")           # ANCHOR
    # parser.add_argument("--test_resize", type=int, default=222, help="_")           # REVIEW
    parser.add_argument("--test_resize", type=int, nargs="+", default=336, help="_")           # REVIEW

    parser.add_argument("--N", type=int, default=1, help="_")   # 分类器的个数
    parser.add_argument("--M", type=int, default=50, help="_")  # 分类区间的宽度

    parser.add_argument("--min_images_per_class", type=int, default=15, help="每个class至少要有多少张用于训练的照片")

    # parser.add_argument('-rt', '--random_transform', action='store_true', help='_')
    # parser.add_argument('--lambda1', type=float, default= 0.3, help='图像变换后估计损失的权重')
    # parser.add_argument('--lambda2', type=float, default= 0.2, help='一致性损失的权重')

    parser.add_argument('--aamc_m', type=float, default=0.2, help='margin m in LMCC or AAMC layer')   # NOTE 原始的是LMCC
    parser.add_argument('--aamc_s', type=float, default=100.0, help='margin s in LMCC or AAMC layer')   # NOTE 原始的是LMCC

    # parser.add_argument("--lr", type=float, default=0.0001, help="_")
    parser.add_argument("--lr", type=float, default=0.0001, help="_")
    parser.add_argument("--classifier_lr", type=float, default=0.01, help="_")
    parser.add_argument("-bb", "--backbone", type=str, default="EfficientNet_B0",
                        # choices=["EfficientNet_B0", "EfficientNet_B5", "EfficientNet_B7"],    # ANCHOR 原始
                        # choices=["EfficientNet_B0", "EfficientNet_B5", "EfficientNet_B7","EfficientNet_V2_M"],  # REVIEW 邵星雨改
                        help="_")
    parser.add_argument('-agg','--aggregator',type=str, default='MixVPR',
                        choices=['MixVPR', 'SALAD', 'ConvAP', 'CosPlace', 'GeMPool','AvgPool'], 
                        help="_")
    parser.add_argument('-ntb', '--num_trainable_blocks', type=int, default=2, help='DINOv2最后可训练的层数')
    parser.add_argument('-ltf', '--layers_to_freeze', type=int, default=5, help='CNN需要冻结的层数')
    parser.add_argument('-ltc', '--layers_to_crop', type=int, nargs="+", default=[], choices=[3,4], help='resnet需要裁剪的层数')
    # parser.add_argument('--regression_ratio', type=float, default=0.5, help="_")
    parser.add_argument('-moc','--mixvpr_out_channels', type=int, default=None, help="_")

    # parser.add_argument('--train_dir', type=str, default='/root/workspace/maps/HE-100-200/', help='gsv-cities的格式')
    # parser.add_argument('--test_dir', type=str, default='/root/workspace/maps/HE_Test/')
    parser.add_argument('--train_photos', action='store_true', help='是否train real photos', default=False)

    parser.add_argument('--fft', action='store_true', help='是否对输入图像使用fft变换', default=False)
    parser.add_argument('-flb', '--fft_log_base', type=float, default=None, help='fft变换后对模长进行变换时的log底数')



    # EDIT
    # Test parameters
    parser.add_argument('--threshold', type=int, default=None, help="验证是否成功召回的可容许偏差的距离，单位为米，默认情况下，程序中设置为30m")    # REVIEW M自适应的话可以不设置


    # Init parameters
    parser.add_argument("--resume_train", type=str, default=None, help="path with *_ckpt.pth checkpoint")
    parser.add_argument("--resume_model", type=str, default=None, help="path with *_model.pth model")

    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_")

    # Paths parameters
    parser.add_argument("--exp_name", type=str, default="default",
                        help="name of experiment. The logs will be saved in a folder with such name")
    parser.add_argument("--dataset_name", type=str, default="qd3.5x5",
                        choices=["qd3.5x5", "UAV-VisLoc", "ct01", "ct02", "2022"], 
                        help="_")   # REVIEW
                        # choices=["sf_xl", "tokyo247", "pitts30k", "pitts250k"], help="_") # ANCHOR
    parser.add_argument("--test_set_list", nargs='+', type=str, default=None,
                        # choices=["sf_xl", "tokyo247", "pitts30k", "pitts250k","QingDao_Flight"], 
                        help="_")   # REVIEW
    parser.add_argument("--train_set_path", type=str, default=None,
                        help="path to folder of train set")
    parser.add_argument("--val_set_path", type=str, default=None,
                        help="真实照片的路径，path to folder of val set")   # NOTE 真实照片的路径
    parser.add_argument("--test_set_path", type=str, default=None,
                        help="path to folder of test set")
    
    args = parser.parse_args()

    # EDIT
    if args.exp_name == "default":
        args.exp_name = f'hc-{args.backbone}-{args.aggregator}'

    args.save_dir = os.path.join("logs", args.exp_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    if args.device == 'cpu' or platform.system() == 'windows':
        args.num_workers = 0

    if args.test_set_list is None:
        args.test_set_list = args.dataset_name

    return args

