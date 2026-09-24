"""Bonus: fine-tune an ImageNet-pretrained ResNet18 on CIFAR-10.

- Load torchvision's ResNet18 with ImageNet weights.
- Replace the final 1000-way fully connected layer with a new 10-way layer.
- Upsample the 32x32 images (ResNet18 downsamples by 32x, so a raw 32x32 image
  would shrink to a single 1x1 feature map) and normalize with ImageNet statistics,
  since those are the input statistics the pretrained weights expect.
- Fine-tune ALL layers with a small learning rate for a few epochs.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from common import (CIFAR_MEAN, CIFAR_STD, FIG_DIR, RESULTS_DIR, count_params, evaluate, get_device,
                    load_cifar10, set_seed)
from models import fit

EPOCHS = 3
IMG_SIZE = 128
device = get_device()
print(f"Using device: {device}")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FineTunedResNet18(nn.Module):
    def __init__(self, num_classes=10, img_size=IMG_SIZE):
        super().__init__()
        self.img_size = img_size
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # pretrained on ImageNet
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)  # new 512 -> 10 head
        # The shared loaders normalize with CIFAR statistics; convert to ImageNet statistics here.
        self.register_buffer('cifar_mean', CIFAR_MEAN.clone())
        self.register_buffer('cifar_std', CIFAR_STD.clone())
        self.register_buffer('in_mean', IMAGENET_MEAN.clone())
        self.register_buffer('in_std', IMAGENET_STD.clone())

    def forward(self, x):
        x = x * self.cifar_std + self.cifar_mean          # back to [0, 1] pixels
        x = (x - self.in_mean) / self.in_std              # ImageNet normalization
        x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
        return self.backbone(x)


set_seed(42)
train_loader, val_loader, test_loader = load_cifar10(device, batch_size=64)

set_seed(42)
model = FineTunedResNet18().to(device)
print(f"Trainable parameters: {count_params(model):,}")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # small lr: adjust, don't overwrite, the pretrained features

hist, _ = fit(model, train_loader, val_loader, optimizer, EPOCHS, device, tag='ResNet18')
test_acc = evaluate(model, test_loader)
print(f"\nResNet18 fine-tuned test accuracy: {test_acc * 100:.2f}%  (training time {hist['train_time_s']:.0f}s)")

ref_path = os.path.join(RESULTS_DIR, 'p3a_basic_cnn.json')
ref = None
if os.path.exists(ref_path):
    with open(ref_path) as f:
        ref = json.load(f)
    print("\n| Model | Parameters | Epochs | Training time (s) | Test accuracy |")
    print("|---|---|---|---|---|")
    print(f"| Basic CNN (3A, from scratch) | {ref['params']:,} | {len(ref['val_acc'])} | "
          f"{ref['train_time_s']:.0f} | {ref['test_acc'] * 100:.2f}% |")
    print(f"| ResNet18 (ImageNet, fine-tuned) | {count_params(model):,} | {EPOCHS} | "
          f"{hist['train_time_s']:.0f} | {test_acc * 100:.2f}% |")

# ---- validation accuracy per epoch, compared with the 3A CNN
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(range(1, EPOCHS + 1), [a * 100 for a in hist['val_acc']], marker='o', label='ResNet18 fine-tuned')
if ref is not None:
    ax.plot(range(1, len(ref['val_acc']) + 1), [a * 100 for a in ref['val_acc']], marker='o', ms=3,
            label='Basic CNN from scratch (3A)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation accuracy (%)')
ax.set_title('Transfer learning vs training from scratch')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'bonus_transfer_learning.png'), dpi=150)

with open(os.path.join(RESULTS_DIR, 'bonus_transfer_learning.json'), 'w') as f:
    json.dump({**hist, 'params': count_params(model), 'test_acc': test_acc, 'img_size': IMG_SIZE}, f, indent=2)
