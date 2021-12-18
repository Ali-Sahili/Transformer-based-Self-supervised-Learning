import sys
import math
import torch
from utils import MetricLogger, SmoothedValue
from datasets.eval_transforms import *



# Train per one epoch
def train_SSL(model, criterion, data_loader, optimizer, device, epoch):
    
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    for imgs1, rots1, imgs2, rots2 in metric_logger.log_every(data_loader, print_freq, header):        
        imgs1 = imgs1.to(device, non_blocking=True)
        imgs1_aug = distortImages(imgs1) # Apply distortion
        rots1 = rots1.to(device, non_blocking=True)
        
        imgs2 = imgs2.to(device, non_blocking=True)
        imgs2_aug = distortImages(imgs2)
        rots2 = rots2.to(device, non_blocking=True)


        with torch.cuda.amp.autocast():
            
            rot1_p, contrastive1_p, imgs1_recon, r_w, cn_w, rec_w = model(imgs1_aug)
            rot2_p, contrastive2_p, imgs2_recon, _, _, _ = model(imgs2_aug)
            
            rot_p = torch.cat([rot1_p, rot2_p], dim=0) 
            rots = torch.cat([rots1, rots2], dim=0) 
            
            imgs_recon = torch.cat([imgs1_recon, imgs2_recon], dim=0) 
            imgs = torch.cat([imgs1, imgs2], dim=0) 
            
            loss, (loss1, loss2, loss3) = criterion(rot_p, rots, 
                                                        contrastive1_p, contrastive2_p, 
                                                        imgs_recon, imgs, r_w, cn_w, rec_w)
                                                        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Backprop
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()


        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_finetune(model, criterion, data_loader, optimizer, device, epoch, mixup_fn):
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        with torch.cuda.amp.autocast():       
            rot_p, contrastive_p = model(images)
            loss = criterion(rot_p, targets) + criterion(contrastive_p, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # Backprop
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
