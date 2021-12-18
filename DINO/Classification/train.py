import torch
import torch.nn as nn
from helpers.utils import SmoothedValue, MetricLogger

# Train per epoch
def train_one_epoch(args, data_loader, model, classifier, optimizer, epoch):

    classifier.train()
    
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(data_loader, 20, header):
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

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
