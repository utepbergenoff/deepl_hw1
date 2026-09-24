"""Problem 2A: compare SGD, SGD+momentum, RMSprop and Adam on the 784-128-64-10 MLP.

Each optimizer trains a freshly initialised model with the SAME initial weights
(same seed) for 20 epochs, so differences come only from the optimizer.
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
from common import FIG_DIR, RESULTS_DIR, Timer, evaluate, get_device, load_mnist, set_seed, train_one_epoch
from mlp import MLP

EPOCHS = 20
device = get_device(prefer_gpu=False)  # tiny MLP: CPU is faster than the GPU here
print(f"Using device: {device}")

set_seed(42)
train_loader, test_loader = load_mnist(device, batch_size=64)

OPTIMIZERS = {
    'SGD':          lambda p: torch.optim.SGD(p, lr=0.01),
    'SGD+Momentum': lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9),
    'RMSprop':      lambda p: torch.optim.RMSprop(p, lr=0.001),
    'Adam':         lambda p: torch.optim.Adam(p, lr=0.001),
}

results = {}
for name, make_opt in OPTIMIZERS.items():
    set_seed(42)  # identical initial weights and batch order for every optimizer
    model = MLP(784, [128, 64], 10, 'relu').to(device)
    optimizer = make_opt(model.parameters())
    criterion = nn.CrossEntropyLoss()

    train_losses, test_accs, train_time = [], [], 0.0
    for epoch in range(EPOCHS):
        with Timer(device) as t:  # only the training pass is timed, not evaluation
            loss = train_one_epoch(model, train_loader, optimizer, criterion)
        train_time += t.elapsed
        acc = evaluate(model, test_loader)
        train_losses.append(loss)
        test_accs.append(acc)
        print(f"[{name}] epoch {epoch + 1:2d}  loss {loss:.4f}  test acc {acc:.4f}")

    results[name] = {'train_loss': train_losses, 'test_acc': test_accs,
                     'final_test_acc': test_accs[-1], 'train_time_s': train_time}

# ---- plots: one shared graph per metric
epochs = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for name, r in results.items():
    axes[0].plot(epochs, r['train_loss'], marker='o', ms=3, label=name)
    axes[1].plot(epochs, [a * 100 for a in r['test_acc']], marker='o', ms=3, label=name)
axes[0].set_title('Training loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Cross-entropy loss')
axes[0].set_yscale('log')
axes[1].set_title('Test accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
for ax in axes:
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'p2a_optimizers.png'), dpi=150)

print("\n| Optimizer | Final test accuracy | Training time (s) |")
print("|---|---|---|")
for name, r in results.items():
    print(f"| {name} | {r['final_test_acc'] * 100:.2f}% | {r['train_time_s']:.1f} |")

with open(os.path.join(RESULTS_DIR, 'p2a_optimizers.json'), 'w') as f:
    json.dump(results, f, indent=2)
