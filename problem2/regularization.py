"""Problem 2C: baseline vs L2 weight decay vs dropout on the 784-128-64-10 MLP.

All three use Adam (lr=0.001) for 20 epochs with identical initial weights.
Training accuracy is measured in eval mode (dropout switched off) after every
epoch, so train and test accuracy are computed the same way and the gap between
them is a fair measure of overfitting.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import torch.nn as nn

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'problem1'))
from common import FIG_DIR, RESULTS_DIR, evaluate, get_device, load_mnist, set_seed, train_one_epoch
from mlp import MLP

EPOCHS = 20
device = get_device(prefer_gpu=False)

set_seed(42)
train_loader, test_loader = load_mnist(device, batch_size=64)

CONFIGS = {
    'Baseline':          {'weight_decay': 0.0,   'dropout': 0.0},
    'L2 (wd=0.001)':     {'weight_decay': 0.001, 'dropout': 0.0},
    'Dropout (p=0.5)':   {'weight_decay': 0.0,   'dropout': 0.5},
}

results = {}
for name, cfg in CONFIGS.items():
    set_seed(42)
    model = MLP(784, [128, 64], 10, 'relu', dropout=cfg['dropout']).to(device)
    # Adam's weight_decay adds wd * w to the gradient, i.e. an L2 penalty (wd/2)*||w||^2 on the loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=cfg['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    train_accs, test_accs = [], []
    for epoch in range(EPOCHS):
        train_one_epoch(model, train_loader, optimizer, criterion)
        train_accs.append(evaluate(model, train_loader))
        test_accs.append(evaluate(model, test_loader))
        print(f"[{name}] epoch {epoch + 1:2d}  train {train_accs[-1]:.4f}  test {test_accs[-1]:.4f}")
    results[name] = {'train_acc': train_accs, 'test_acc': test_accs}

# ---- plot: train vs test accuracy, one panel per configuration
epochs = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
for ax, (name, r) in zip(axes, results.items()):
    ax.plot(epochs, [a * 100 for a in r['train_acc']], marker='o', ms=3, label='Train')
    ax.plot(epochs, [a * 100 for a in r['test_acc']], marker='o', ms=3, label='Test')
    gap = (r['train_acc'][-1] - r['test_acc'][-1]) * 100
    ax.set_title(f'{name}\nfinal gap (train - test) = {gap:.2f} pp')
    ax.set_xlabel('Epoch')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel('Accuracy (%)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'p2c_regularization.png'), dpi=150)

print("\n| Configuration | Final train acc | Final test acc | Gap (pp) | Best test acc |")
print("|---|---|---|---|---|")
for name, r in results.items():
    tr, te = r['train_acc'][-1] * 100, r['test_acc'][-1] * 100
    print(f"| {name} | {tr:.2f}% | {te:.2f}% | {tr - te:.2f} | {max(r['test_acc']) * 100:.2f}% |")

with open(os.path.join(RESULTS_DIR, 'p2c_regularization.json'), 'w') as f:
    json.dump(results, f, indent=2)
