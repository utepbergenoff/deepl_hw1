"""Problem 3B: compare three CNN architectures on CIFAR-10.

The architectures differ in depth (1 / 3 / 6 conv layers), width (16 to 256
filters), kernel size (5x5 vs 3x3) and pooling (max vs average). Everything else
(optimizer, lr, batch size, epochs, data split, FC head of 128 units) is fixed,
and the basic CNN from 3A is included as a reference row if its results exist.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from common import FIG_DIR, RESULTS_DIR, count_params, evaluate, get_device, load_cifar10, set_seed
from models import ConfigurableCNN, fit

EPOCHS = 20
device = get_device()
print(f"Using device: {device}")

# block = (filters, kernel size, conv layers in the block, pooling type)
ARCHS = {
    'A: Shallow (1 conv, 5x5, max)': {
        'blocks': [(16, 5, 1, 'max')],
        'desc': 'conv5x5(16) -> maxpool',
    },
    'B: Medium (3 conv, 3x3, avg)': {
        'blocks': [(32, 3, 1, 'avg'), (64, 3, 1, 'avg'), (128, 3, 1, 'avg')],
        'desc': 'conv3x3(32) -> avgpool -> conv3x3(64) -> avgpool -> conv3x3(128) -> avgpool',
    },
    'C: Deep VGG-style (6 conv, 3x3, max)': {
        'blocks': [(64, 3, 2, 'max'), (128, 3, 2, 'max'), (256, 3, 2, 'max')],
        'desc': '[conv3x3(64) x2 -> maxpool] -> [conv3x3(128) x2 -> maxpool] -> [conv3x3(256) x2 -> maxpool]',
    },
}

set_seed(42)
train_loader, val_loader, test_loader = load_cifar10(device, batch_size=64)

results = {}
for name, arch in ARCHS.items():
    set_seed(42)
    model = ConfigurableCNN(arch['blocks']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    hist, best_state = fit(model, train_loader, val_loader, optimizer, EPOCHS, device, tag=name[:1])
    test_acc = evaluate(model, test_loader)
    model.load_state_dict(best_state)
    best_test_acc = evaluate(model, test_loader)
    results[name] = {**hist, 'desc': arch['desc'], 'params': count_params(model),
                     'test_acc': test_acc, 'best_val_checkpoint_test_acc': best_test_acc}
    print(f"{name}: {results[name]['params']:,} params, {hist['train_time_s']:.0f}s, "
          f"test acc {test_acc * 100:.2f}% (best-val checkpoint {best_test_acc * 100:.2f}%)")

# reference: the 3A basic CNN
ref_path = os.path.join(RESULTS_DIR, 'p3a_basic_cnn.json')
if os.path.exists(ref_path):
    with open(ref_path) as f:
        ref = json.load(f)
    results['3A basic CNN (reference)'] = {**{k: ref[k] for k in ['train_loss', 'val_loss', 'val_acc', 'train_time_s',
                                                                   'best_epoch', 'params', 'test_acc',
                                                                   'best_val_checkpoint_test_acc']},
                                          'desc': 'conv3x3(32) -> maxpool -> conv3x3(64) -> maxpool'}

# ---- plots
epochs = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
for name, r in results.items():
    ls = '--' if 'reference' in name else '-'
    axes[0].plot(epochs, [a * 100 for a in r['val_acc']], ls=ls, marker='o', ms=3, label=name)
    axes[1].plot(epochs, r['val_loss'], ls=ls, marker='o', ms=3, label=name)
axes[0].set_title('Validation accuracy')
axes[0].set_ylabel('Accuracy (%)')
axes[1].set_title('Validation loss')
axes[1].set_ylabel('Cross-entropy loss')
for ax in axes:
    ax.set_xlabel('Epoch')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'p3b_architectures.png'), dpi=150)

print("\n| Architecture | Layers | Parameters | Training time (s) | Test acc (epoch 20) | Test acc (best-val epoch) |")
print("|---|---|---|---|---|---|")
for name, r in results.items():
    print(f"| {name} | {r['desc']} | {r['params']:,} | {r['train_time_s']:.0f} | {r['test_acc'] * 100:.2f}% | "
          f"{r['best_val_checkpoint_test_acc'] * 100:.2f}% (ep {r['best_epoch']}) |")

with open(os.path.join(RESULTS_DIR, 'p3b_architectures.json'), 'w') as f:
    json.dump(results, f, indent=2)
