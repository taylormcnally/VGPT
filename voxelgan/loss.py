import torch

loss_obj = torch.nn.BinaryCrossEntropyLoss()

def discriminator_loss(real, generated):
    real_loss = loss_obj(torch.ones_like(real), real)
    generated_loss = loss_obj(torch.zeros_like(generated), generated)
    total_loss = real_loss + generated_loss
    return total_loss

def generator_loss(generated):
    return loss_obj(torch.ones_like(generated), generated)



