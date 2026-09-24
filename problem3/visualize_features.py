"""Problem 3C: visualize the first conv layer of the trained 3A CNN.

1. All 32 learned 3x3x3 kernels of conv1, shown as tiny RGB images.
2. The 32 conv1 activation maps (after ReLU) for three test images.
Run problem3/basic_cnn.py first -- it saves the weights this script loads.
"""
import os
import sys

import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from common import CIFAR_CLASSES, FIG_DIR, RESULTS_DIR, get_device, load_cifar10, set_seed
from models import BasicCNN

device = get_device()
set_seed(42)
_, _, test_loader = load_cifar10(device)

model = BasicCNN().to(device)
model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'p3a_basic_cnn.pt'), map_location=device))
model.eval()

# ---- 1. filters: weight shape is (32 out channels, 3 in channels, 3, 3)
w = model.conv1.weight.detach().cpu()
fig, axes = plt.subplots(4, 8, figsize=(10, 5.4))
for i, ax in enumerate(axes.flat):
    k = w[i]
    k = (k - k.min()) / (k.max() - k.min())  # rescale each filter to [0, 1] so it can be shown as RGB
    ax.imshow(k.permute(1, 2, 0).numpy(), interpolation='nearest')
    ax.set_title(f'#{i}', fontsize=8)
    ax.axis('off')
fig.suptitle('Basic CNN: all 32 conv1 filters (3x3 RGB, each min-max scaled)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'p3c_conv1_filters.png'), dpi=150)

# ---- 2. activation maps for three test images of different classes
raw = test_loader.x  # uint8 images, not yet normalized
labels = test_loader.y.cpu()
wanted = ['cat', 'ship', 'automobile']
idx = [int((labels == CIFAR_CLASSES.index(c)).nonzero()[0]) for c in wanted]

with torch.no_grad():
    x = test_loader.transform(raw[idx])
    acts = torch.relu(model.conv1(x)).cpu()  # (3, 32, 32, 32): the output of conv1 + ReLU
    preds = model(x).argmax(1).cpu()

fig = plt.figure(figsize=(16, 13))
subfigs = fig.subfigures(len(idx), 1, hspace=0.05)
for sf, i, a, p in zip(subfigs, idx, acts, preds):
    gs = sf.add_gridspec(4, 10, width_ratios=[4, 0.3] + [1] * 8)
    ax = sf.add_subplot(gs[:, 0])
    ax.imshow(raw[i].permute(1, 2, 0).cpu().numpy())
    ax.set_title(f'true: {CIFAR_CLASSES[labels[i]]}\npredicted: {CIFAR_CLASSES[p]}', fontsize=10)
    ax.axis('off')
    for f in range(32):
        ax = sf.add_subplot(gs[f // 8, 2 + f % 8])
        ax.imshow(a[f].numpy(), cmap='viridis')
        ax.set_title(f'#{f}', fontsize=7, pad=2)
        ax.axis('off')
fig.suptitle('conv1 activation maps (after ReLU) for three test images', fontsize=14, y=1.03)
plt.savefig(os.path.join(FIG_DIR, 'p3c_activation_maps.png'), dpi=130, bbox_inches='tight')
print("Saved figures/p3c_conv1_filters.png and figures/p3c_activation_maps.png")
