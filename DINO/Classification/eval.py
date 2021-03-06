import torch
import torch.nn as nn

from helpers.metrics import accuracy
from helpers.utils import MetricLogger

@torch.no_grad()
def evaluate(args, val_loader, model, classifier):
    classifier.eval()
    
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, args.n_last_blocks)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if args.avgpool_patchtokens:
                    output = torch.cat((output.unsqueeze(-1), 
                                        torch.mean(intermediate_output[-1][:, 1:], 
                                                                 dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        output = classifier(output)
        loss = nn.CrossEntropyLoss()(output, target)

        if classifier.num_labels >= 5:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1, = accuracy(output, target, topk=(1,))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        
        if classifier.num_labels >= 5:
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            
    if classifier.num_labels >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'.format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'.format(top1=metric_logger.acc1, losses=metric_logger.loss))
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
