
import argparse

import torch
from torch import nn, optim
from torchvision import transforms

from utils import get_cosine_schedule_with_warmup
from data import ChineseFontDataset
from model import UNet
from trainer import DDPMTrainer

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--image-size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--ttf-path', type=str, default='fonts/kaiu.ttf', help='Path of ttf font file.')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--use-amp', type=bool, default=True)
    parser.add_argument('--use-cuda', type=bool, default=True)
    parser.add_argument('--use-ema', type=bool, default=False)
    parser.add_argument('--save-path', type=str, default='results/model.pt', help='The trained model will be saved.')
    parser.add_argument('--restart-path', type=str, default=None, help='The model will be restarted.')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.0438, std=0.1850)
    ])
    dataset = ChineseFontDataset(args.ttf_path, font_size=args.image_size, 
                            image_size=(args.image_size, args.image_size), 
                            transform=img_transform)

    train_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=3000),
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True)
    test_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=1000),
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True)


    unet = UNet(1, 32, 1, [1, 1, 2], [2, 2, 2], [False, False, False, False])
    if args.restart_path is not None:
        unet.load_state_dict(torch.load(args.restart_path))

    optimizer = optim.Adam(unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                   int(args.epochs / 10 * len(train_loader)), 
                                                   args.epochs * len(train_loader), min_lr=1e-3 * args.lr)

    criterion = nn.L1Loss()

    trainer = DDPMTrainer(unet, optimizer, criterion, lr_scheduler=lr_scheduler, 
                          use_amp=args.use_amp, use_cuda=args.use_cuda, use_ema=args.use_ema)
    trainer.fit(train_loader, epochs=args.epochs)


    if args.save_path is not None:
        torch.save(trainer.ema_model.state_dict() if args.use_ema else trainer.model.state_dict() , 
                   args.save_path)


if __name__ == "__main__":
    main()