import warnings
warnings.filterwarnings('ignore')

import json
import time
import torch
import argparse
import datetime
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn

from timm.data import Mixup
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import SoftTargetCrossEntropy

from losses import MTL_loss
from train import train_SSL, train_finetune
from eval import evaluate_SSL, evaluate_finetune

from model.network import *
from datasets.prepare_data import build_dataset
from utils import collate_fn, requires_grad, str2bool


# Setting Parameters
def get_args_parser():
    parser = argparse.ArgumentParser('SiT training and evaluation script', add_help=False)
    
    # Dataset parameters   
    parser.add_argument('--dataset_location', default='downloaded_datasets', type=str, 
                        help='dataset location - dataset will be downloaded to this folder')   
    parser.add_argument('--num_imgs_per_cat', default=None, type=int, 
                        help='Number of images per training category')
    parser.add_argument('--ratio', default=0.8, type=float, help='ratio of training set')

    # Dataloading parameters
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    
    # Training parameters
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--epochs', default=501, type=int)
    parser.add_argument('--start_epoch', default=1, type=int, help='start epoch')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--device', default='cuda', help='device to use for training/testing')
    parser.add_argument('--output_dir', default='', help='path where to save, empty-no saving')
    
    # Evaluation parameter
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--val_step', default=1,type=int,help='validate model every val_step')  
    
    # Training SSl or Fine-tuning
    parser.add_argument('--training_mode', default='SSL', type=str, help='SSL or finetune')
    
    # Model parameters
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--net', default=True, type=str2bool, help='Choose an implementation')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, help='Optimizer')
    parser.add_argument('--opt-eps', default=1e-8, type=float, help='Optimizer Epsilon')
    parser.add_argument('--opt-betas',default=None,type=float,nargs='+',help='Optimizer Betas')
    parser.add_argument('--clip-grad', type=float, default=None, help='Clip gradient norm')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, help='LR scheduler')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None,
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float,default=0.67,help='lr noise limit percent')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, help='lr noise std-dev')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, help='warmup lr')
    parser.add_argument('--min-lr', type=float, default=1e-5,
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--decay-epochs',type=float,default=30,help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='epochs to warmup LR')
    parser.add_argument('--cooldown-epochs', type=int, default=10,
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10,
                        help='patience epochs for Plateau LR scheduler')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, help='LR decay rate')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, help='Color jitter factor')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--train-interpolation', type=str, default='bicubic', 
                        choices=['random','bilinear','bicubic'], help='Training interpolation')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, help='Random erase prob')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8, help='mixup alpha')
    parser.add_argument('--cutmix', type=float, default=1.0, help='cutmix alpha')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                  help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                  help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch', 
                        choices=["batch","pair","elem"], 
                        help='How to apply mixup/cutmix params. Per ')

    return parser


# Main Funtion for training and evaluation
def main(args):
    # Set device and seeds
    device = torch.device(args.device)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Prepare datasets
    print("Loading dataset ....")
    dataset, num_classes = build_dataset(args)   

    train_size = int(args.ratio * len(dataset))
    val_size = len(dataset) - train_size
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Dataloading
    train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True,
                                               batch_size=args.batch_size, 
                                               num_workers=args.num_workers,
                                               pin_memory=True, drop_last=True, 
                                               collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset_val, shuffle=False,
                                             batch_size=int(1.5 * args.batch_size), 
                                             num_workers=args.num_workers,
                                             pin_memory=True, drop_last=False, 
                                             collate_fn=collate_fn)
    print()
    print("Train dataset contains " + str(len(train_loader.dataset)) + " images.")
    print("Validation dataset contains " + str(len(val_loader.dataset)) + " images.")
    print("Number of batches in train set: ", len(train_loader))
    print()
    
    # Mixup: Data Augmentation
    mixup_fn = None
    mixup_active = False
    if mixup_active:
        mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, 
                         cutmix_minmax=args.cutmix_minmax, prob=args.mixup_prob, 
                         switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                         label_smoothing=args.smoothing, num_classes=num_classes)
    
    # Defining model
    print("Creating model...")
    if args.net:
        model = SiT(img_size=args.input_size, patch_size=16, in_chans=3,   
                     num_classes=num_classes, training_mode=args.training_mode)
    else:            
        model = SiT_2(image_size=args.input_size, patch_size=16, in_channels=3, 
                      num_classes=num_classes, training_mode = args.training_mode)
    model.to(device)
    
    # Number of parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params: %.3fM' % (n_parameters/(1e+6)))
    print()
    
    # Optimizer
    args.lr = args.lr * args.batch_size / 512.0
    optimizer = create_optimizer(args, model)

    # Scheduling
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # Set loss function
    if args.training_mode == 'SSL':
        criterion = MTL_loss(device, args.batch_size)
    else: # fine-tuning
        # criterion = SoftTargetCrossEntropy()
        criterion = torch.nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    
    # Load checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch']

    # Training / Evaluation
    print("Start training...")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):

        # Training per epoch
        if args.training_mode == 'SSL':
            train_stats = train_SSL(model, criterion, train_loader, optimizer, device, epoch)
        else:
            train_stats = train_finetune(model, criterion, train_loader, optimizer, device, 
                                                                             epoch, mixup_fn)
        
        # Scheduling
        lr_scheduler.step(epoch)
        
        # Evaluation per epoch
        if args.eval and (epoch % args.val_step == 0):   
            if args.training_mode == 'SSL': 
                test_stats = evaluate_SSL(val_loader, model, device, epoch, args.output_dir)
            else:
                test_stats = evaluate_finetune(val_loader, model, device)
                
            print(f"Accuracy of the network on test images: {test_stats['acc1']:.1f}%")
            
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')

        # Logging statistics
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        # Saving weights and statistics
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                torch.save({
                             'model': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'lr_scheduler': lr_scheduler.state_dict(),
                             'epoch': epoch,
                           }, checkpoint_path)
        
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SiT: Training and Evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    if args.dataset_location:
        Path(args.dataset_location).mkdir(parents=True, exist_ok=True)
        
    main(args)
