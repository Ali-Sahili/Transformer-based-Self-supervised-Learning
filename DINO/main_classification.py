import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from torch import nn
import torch.backends.cudnn as cudnn

from torchvision import datasets
from torchvision import transforms
from torchvision import models as torchvision_models

import models
from Classification.eval import evaluate
from Classification.train import train_one_epoch
from Classification.classifier import LinearClassifier
from helpers.load_weights import load_pretrained_weights
from helpers.utils import str2bool, restart_from_checkpoint


def main(args):
    
    cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in models.__dict__.keys():
        model = models.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
        
    model.cuda()
    model.eval()
    
    # load weights to evaluate
    load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, 
                                   args.patch_size)
    print(f"Model {args.arch} built.")

    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()


    # ============ preparing validation data ... ============
    val_transform = transforms.Compose([
                        transforms.Resize(256, interpolation=3),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                      ])
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), 
                                       transform=val_transform)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size,
                                             num_workers=args.num_workers, pin_memory=True)

    # ============ preparing training data ... ============
    train_transform = transforms.Compose([
                           transforms.RandomResizedCrop(224),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                       ])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), 
                                         transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               pin_memory=True)
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")
    
    # set optimizer
    optimizer = torch.optim.SGD(linear_classifier.parameters(),
                                args.lr * args.batch_size / 256., # linear scaling rule
                                momentum=0.9,
                                weight_decay=0, # we do not apply weight decay
                               )
    # Scheduling
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 1, "best_acc": 0.}
    restart_from_checkpoint(os.path.join(args.output_dir, "checkpoint.pth.tar"),
                            run_variables=to_restore,
                            state_dict=linear_classifier,
                            optimizer=optimizer,
                            scheduler=scheduler,
                          )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]
    
    print("Start training ...")
    for epoch in range(start_epoch, args.epochs):
        # ==============   Training  ================
        train_stats = train_one_epoch(args, train_loader, model, linear_classifier, 
                                                          optimizer, epoch)
        # Scheduling
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        
        # ==============  Validation ================
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = evaluate(args, val_loader, model, linear_classifier)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
        
        # Saving logs and checkpoints               
        with (Path(args.output_dir) / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")
            
        save_dict = {  "epoch": epoch + 1,
                       "state_dict": linear_classifier.state_dict(),
                       "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict(),
                       "best_acc": best_acc
                    }
        torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
        
    print("Training of the supervised linear classifier on frozen features completed.")
    print("Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    # Dataloading
    parser.add_argument('--data_path', default='datasets/tiny-imagenet-200', type=str)
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--pretrained_weights', default='', type=str, 
                        help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, 
                        help='Key to use in the checkpoint (example: "teacher")')
                        
    # Architecture parameters
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution.')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-
        Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=str2bool,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] 
              token. We typically set this to False for ViT-Small and to True with ViT-Base.""")
                  
    # Training / Evaluation parameters
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument('--batch_size', default=128, type=int, help='batch-size')
    parser.add_argument('--num_workers', default=10, type=int, help='nb of dataloading workers')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency.")
    parser.add_argument('--eval', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--num_labels', default=1000, type=int, 
                        help='Number of labels for linear classifier')

    args = parser.parse_args()
    main(args)
