import torch
import os


def save_checkpoint(epoch, models, optimizers, losses=None):
    os.makedirs("checkpoint", exist_ok=True)
    torch.save({
        'epoch': epoch,
        'models_state_dict': [model.state_dict() for model in models],
        'optimizers_state_dict': [optimizer.state_dict() for optimizer in optimizers],
        'losses': losses,
    }, f'checkpoint/epoch_{epoch}.pth')
    print(f'Checkpoint {epoch} saved', flush=True)


def load_checkpoint(epoch, models, optimizers):
    checkpoint = torch.load(f'checkpoint/epoch_{epoch}.pth')
    [model.load_state_dict(checkpoint['models_state_dict'][idx]) for idx, model in enumerate(models)]
    [optimizer.load_state_dict(checkpoint['optimizers_state_dict'][idx]) for idx, optimizer in enumerate(optimizers)]
    epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    print(f'Checkpoint {epoch} loaded', flush=True)
    return losses
