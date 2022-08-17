import torch
import utils.utils as utils

def validate_model(model, val_loader):
    model.eval()
    testacc = utils.AverageMeter()
    with torch.no_grad():
        for i, (images, target, _) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            output = model(images)
            acc, = utils.accuracy(output.detach(), target)
            testacc.update(acc)
    return testacc.avg    