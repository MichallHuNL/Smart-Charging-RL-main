import torch
import os
import re


def save_checkpoint(epoch, models, optimizers, env_steps, losses, returns, lengths, checkpoint_name):
    os.makedirs("checkpoint", exist_ok=True)
    os.makedirs(f"checkpoint/{checkpoint_name}", exist_ok=True)
    torch.save({
        'epoch': epoch,
        'models_state_dict': [model.state_dict() for model in models],
        'optimizers_state_dict': [optimizer.state_dict() for optimizer in optimizers],
        'env_steps': env_steps,
        'losses': losses,
        'returns': returns,
        'lengths': lengths
    }, f'checkpoint/{checkpoint_name}/epoch_{epoch}.pth')
    print(f'Checkpoint {epoch} saved', flush=True)


def load_checkpoint(epoch, models, optimizers, checkpoint_name):
    if epoch is None:
        files = os.listdir(f'checkpoint/{checkpoint_name}')

        print(files)

        epoch = max([int(re.search(r"[0-9]+", file).group()) for file in files])
    checkpoint = torch.load(f'checkpoint/{checkpoint_name}/epoch_{epoch}.pth')
    [model.load_state_dict(checkpoint['models_state_dict'][idx]) for idx, model in enumerate(models)]
    [optimizer.load_state_dict(checkpoint['optimizers_state_dict'][idx]) for idx, optimizer in enumerate(optimizers)]
    epoch = checkpoint['epoch']
    env_steps = checkpoint['env_steps']
    losses = checkpoint['losses']
    returns = checkpoint['returns']
    lengths = checkpoint['lengths']
    print(f'Checkpoint {epoch} loaded', flush=True)
    return env_steps, losses, returns, lengths
