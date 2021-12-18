import warnings
warnings.filterwarnings('ignore')

import os
import time
import json
import datetime
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import models as vits
from models import DINOHead

from train import train_one_epoch
from data_augmentation import DataAugmentationDINO

from helpers.loss import DINOLoss
from helpers.utils import (MultiCropWrapper, get_params_groups, 
                           cosine_scheduler, restart_from_checkpoint)



torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)
    # Dataset
    parser.add_argument('--data_path', default='datasets/tiny-imagenet-200/train/', type=str,
                         help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, 
                         help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, 
                         help='Save checkpoint every x epochs.')
    
    # Data Augmentation
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                         help="Scale range of the cropped image before resizing")
    parser.add_argument('--local_crops_number', type=int, default=8, 
                         help="Number of small local views to generate.")
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="Scale range of the cropped image before resizing for small local view cropping.")

    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int, help='nb of dataloading workers')

    # Training/Optimization parameters
    parser.add_argument('--weight_decay', type=float, default=0.04, help="weight decay.")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, 
                      help="Final value of the weight decay. We use a cosine schedule for WD.")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="gradient clipping.")
    parser.add_argument('--batch_size', default=32, type=int, help='batch-size.')
    parser.add_argument('--epochs', default=100, type=int, help='nb of epochs for training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, 
                         help="nb of epochs during which we keep the output layer fixed.")
    parser.add_argument("--lr", default=0.0005, type=float, help="Learning rate")
    parser.add_argument("--warmup_epochs", default=10, type=int,
                         help="nb of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, 
                         help="We use a cosine LR schedule with linear warmup.")
    parser.add_argument('--optimizer', default='adamw', type=str,
                         choices=['adamw', 'sgd', 'lars'], help="Type of optimizer")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, 
                         help="stochastic depth rate")

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--out_dim', default=65536, type=int, 
                         help="Dimensionality of the DINO head output.")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                         help="Initial value for the teacher temperature.")
    parser.add_argument('--teacher_temp', default=0.04, type=float, 
                         help="Final value (after linear warmup) of the teacher temperature.")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                         help='nb of warmup epochs for the teacher temperature.')

    # Scheduling
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="parameter for teacher update. Setting a higher value with small batches: for example use 0.9995 with batch size of 256.")

    return parser


def main(args):
    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    cudnn.benchmark = True

    # ============ preparing data ... ===============
    transform = DataAugmentationDINO( args.global_crops_scale,
                                      args.local_crops_scale,
                                      args.local_crops_number,
                                    )
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    indices = list(range(1, len(dataset), 200))
    subset = torch.utils.data.Subset(dataset, indices)
    data_loader = torch.utils.data.DataLoader(subset, shuffle=True,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              pin_memory=True, drop_last=True)
    print(f"Data loaded: there are {len(subset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch]( patch_size=16, drop_path_rate=args.drop_path_rate)
        teacher = vits.__dict__[args.arch](patch_size=16)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = MultiCropWrapper(student, DINOHead(embed_dim, args.out_dim, use_bn=False,
                                                       norm_last_layer=True)
                              )
    teacher = MultiCropWrapper(teacher, DINOHead(embed_dim, args.out_dim, use_bn=False))
    
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    
    # teacher and student start with the same weights
    teacher.load_state_dict(student.state_dict())
    
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")
    
    # ============ preparing loss ... ============
    dino_loss = DINOLoss(args.out_dim,
                         args.local_crops_number + 2,  # total number of crops = 2 global crops 
                                                       # + local_crops_number
                         args.warmup_teacher_temp,
                         args.teacher_temp,
                         args.warmup_teacher_temp_epochs,
                         args.epochs
                        ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    else:
        raise NotImplemented

    # ============ init schedulers ... ============
    lr_schedule = cosine_scheduler( args.lr * args.batch_size / 256.,  # linear scaling rule
                                    args.min_lr, args.epochs, len(data_loader),
                                    warmup_epochs=args.warmup_epochs
                                  )
    wd_schedule = cosine_scheduler( args.weight_decay,
                                    args.weight_decay_end,
                                    args.epochs, len(data_loader),
                                  )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")
    
    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    restart_from_checkpoint(os.path.join(args.output_dir, "checkpoint.pth"),
                            run_variables=to_restore,
                            student=student,
                            teacher=teacher,
                            optimizer=optimizer,
                            dino_loss=dino_loss,
                           )
    start_epoch = to_restore["epoch"]
    
    start_time = time.time()
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(args, data_loader, student, teacher, dino_loss, optimizer, 
                                      lr_schedule, wd_schedule, momentum_schedule, epoch)
        
        # ============ writing logs ... ============
        save_dict = { 'student': student.state_dict(),
                      'teacher': teacher.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'epoch': epoch + 1,
                      'args': args,
                      'dino_loss': dino_loss.state_dict(),
                    }

        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            torch.save(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        with (Path(args.output_dir) / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
