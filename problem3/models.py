"""CNN architectures for Problem 3 plus a shared training loop."""
import copy
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import Timer, evaluate, train_one_epoch


class BasicCNN(nn.Module):
    """Problem 3A architecture.

    32x32x3 -> conv3x3(32) ReLU -> maxpool -> 16x16x32
            -> conv3x3(64) ReLU -> maxpool -> 8x8x64 -> flatten (4096)
            -> FC 128 ReLU -> FC 10 (logits; softmax is inside CrossEntropyLoss)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # padding=1 keeps 32x32 ("same")
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def conv_block(in_ch, out_ch, kernel, n_convs, pool):
    """n_convs conv+ReLU layers followed by one 2x2 pooling layer ('max' or 'avg')."""
    layers = []
    for i in range(n_convs):
        layers += [nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel, padding=kernel // 2), nn.ReLU()]
    layers.append(nn.MaxPool2d(2) if pool == 'max' else nn.AvgPool2d(2))
    return layers


class ConfigurableCNN(nn.Module):
    """CNN built from a list of blocks: [(out_channels, kernel_size, n_convs, pool), ...].

    Every block halves the spatial size, followed by FC 128 -> FC 10.
    """
    def __init__(self, blocks, fc_hidden=128):
        super().__init__()
        layers, in_ch = [], 3
        for out_ch, kernel, n_convs, pool in blocks:
            layers += conv_block(in_ch, out_ch, kernel, n_convs, pool)
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        spatial = 32 // (2 ** len(blocks))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * spatial * spatial, fc_hidden), nn.ReLU(),
            nn.Linear(fc_hidden, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def fit(model, train_loader, val_loader, optimizer, epochs, device, tag=''):
    """Train for a fixed number of epochs, tracking train/val loss and accuracy.

    Also keeps a copy of the weights from the epoch with the best validation
    accuracy (returned as best_state) so callers can report an early-stopped model too.
    """
    criterion = nn.CrossEntropyLoss()
    hist = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_acc, best_state, best_epoch, train_time = -1, None, 0, 0.0
    for epoch in range(epochs):
        with Timer(device) as t:
            tr_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        train_time += t.elapsed
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        hist['train_loss'].append(tr_loss)
        hist['val_loss'].append(val_loss)
        hist['val_acc'].append(val_acc)
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        print(f"[{tag}] epoch {epoch + 1:2d}  train loss {tr_loss:.4f}  val loss {val_loss:.4f}  "
              f"val acc {val_acc:.4f}  ({t.elapsed:.1f}s)")
    hist['train_time_s'] = train_time
    hist['best_epoch'] = best_epoch
    return hist, best_state
