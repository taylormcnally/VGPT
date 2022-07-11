import torch

loss_obj = torch.nn.CrossEntropyLoss(reduction='none')

def loss_fn(pred, target):
    return loss_obj(pred, target)



