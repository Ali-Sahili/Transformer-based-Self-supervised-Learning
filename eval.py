import os
import torch
import torchvision
from pathlib import Path

from timm.utils import accuracy
from datasets.eval_transforms import *
from utils import MetricLogger


@torch.no_grad()
def evaluate_SSL(data_loader, model, device, epoch, output_dir):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    save_recon = os.path.join(output_dir, 'reconstruction_samples')
    Path(save_recon).mkdir(parents=True, exist_ok=True)
    
    # switch to evaluation mode
    model.eval()
    print_freq = 50
    i = 0
    for imgs1, rots1, imgs2, rots2 in metric_logger.log_every(data_loader, print_freq, header):
        imgs1 = imgs1.to(device, non_blocking=True) 
        imgs1_aug = distortImages(imgs1) # Apply distortion
        rots1 = rots1.to(device, non_blocking=True)
        
        imgs2 = imgs2.to(device, non_blocking=True)
        imgs2_aug = distortImages(imgs2)
        rots2 = rots2.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            rot1_p, contrastive1_p, imgs1_recon, r_w, cn_w, rec_w = model(imgs1_aug)
            rot2_p, contrastive2_p, imgs2_recon, _, _, _ = model(imgs2_aug)
            
            rot_p = torch.cat([rot1_p, rot2_p], dim=0) 
            rots = torch.cat([rots1, rots2], dim=0) 
            
            loss = criterion(rot_p, rots)

        acc1, acc5 = accuracy(rot_p, rots, topk=(1, 4))

        batch_size = imgs1.shape[0]*2
        
        if i%print_freq==0:

            print_out = save_recon + '/Test_epoch_' + str(epoch)  + '_Iter' + str(i) + '.jpg' 
            imagesToPrint = torch.cat([imgs1[0:min(15, batch_size)].cpu(), 
                                       imgs1_aug[0:min(15, batch_size)].cpu(),
                                       imgs1_recon[0:min(15, batch_size)].cpu(),
                                       imgs2[0:min(15, batch_size)].cpu(), 
                                       imgs2_aug[0:min(15, batch_size)].cpu(),
                                       imgs2_recon[0:min(15, batch_size)].cpu()], dim=0)
            torchvision.utils.save_image(imagesToPrint, print_out, nrow=min(15, batch_size), normalize=True, range=(-1, 1))
            
            
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
        i = i + 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

   
@torch.no_grad()
def evaluate_finetune(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    print_freq = 50
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = images.to(device, non_blocking=True) 
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            rot_p, contrastive_p = model(images)
            loss = criterion(rot_p, targets) + criterion(contrastive_p, targets)

        acc1, acc5 = accuracy((rot_p+contrastive_p)/2., targets, topk=(1, 5))

        batch_size = images.shape[0]
            
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
