"""Problem 3A: train the basic CNN on CIFAR-10 for 20 epochs.

Data: 45k train / 5k validation (split from the official training set) / 10k test.
Outputs: loss curves, final test accuracy, confusion matrix, saved weights (used by 3C).
"""
import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from common import (CIFAR_CLASSES, FIG_DIR, RESULTS_DIR, count_params, evaluate, get_device,
                    load_cifar10, predict, set_seed)
from models import BasicCNN, fit

EPOCHS = 20
device = get_device()
print(f"Using device: {device}")

set_seed(42)
train_loader, val_loader, test_loader = load_cifar10(device, batch_size=64)

set_seed(42)
model = BasicCNN().to(device)
print(model)
print(f"Trainable parameters: {count_params(model):,}")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

hist, best_state = fit(model, train_loader, val_loader, optimizer, EPOCHS, device, tag='BasicCNN')

train_acc = evaluate(model, train_loader)
test_acc = evaluate(model, test_loader)
torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'p3a_basic_cnn.pt'))

# For reference: the checkpoint with the best validation accuracy (early stopping)
best_model = BasicCNN().to(device)
best_model.load_state_dict(best_state)
best_test_acc = evaluate(best_model, test_loader)

print(f"\nTraining time: {hist['train_time_s']:.1f}s")
print(f"Final (epoch {EPOCHS}) train accuracy: {train_acc * 100:.2f}%")
print(f"Final (epoch {EPOCHS}) test accuracy:  {test_acc * 100:.2f}%")
print(f"Best-validation checkpoint (epoch {hist['best_epoch']}) test accuracy: {best_test_acc * 100:.2f}%")

# ---- loss curves
epochs = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(epochs, hist['train_loss'], marker='o', ms=3, label='Train')
axes[0].plot(epochs, hist['val_loss'], marker='o', ms=3, label='Validation')
axes[0].axvline(hist['best_epoch'], color='gray', ls='--', lw=1, label=f"best val acc (epoch {hist['best_epoch']})")
axes[0].set_title('Basic CNN: training vs validation loss')
axes[0].set_ylabel('Cross-entropy loss')
axes[1].plot(epochs, [a * 100 for a in hist['val_acc']], marker='o', ms=3, color='tab:orange')
axes[1].set_title('Validation accuracy')
axes[1].set_ylabel('Accuracy (%)')
for ax in axes:
    ax.set_xlabel('Epoch')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3)
axes[0].legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'p3a_loss_curves.png'), dpi=150)

# ---- confusion matrix on the test set (rows = true class, columns = predicted)
preds, labels = predict(model, test_loader)
cm = torch.zeros(10, 10, dtype=torch.long)
for t, p in zip(labels, preds):
    cm[t, p] += 1
cm = cm.numpy()

fig, ax = plt.subplots(figsize=(8.5, 7.5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(10), CIFAR_CLASSES, rotation=45, ha='right')
ax.set_yticks(range(10), CIFAR_CLASSES)
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=8,
                color='white' if cm[i, j] > cm.max() / 2 else 'black')
ax.set_xlabel('Predicted class')
ax.set_ylabel('True class')
ax.set_title(f'Basic CNN confusion matrix (test accuracy {test_acc * 100:.2f}%)')
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'p3a_confusion_matrix.png'), dpi=150)

per_class = cm.diagonal() / cm.sum(1)
print("\nPer-class test accuracy:")
for c, a in zip(CIFAR_CLASSES, per_class):
    print(f"  {c:<11s} {a * 100:.1f}%")
off = cm.copy()
np.fill_diagonal(off, 0)
i, j = np.unravel_index(off.argmax(), off.shape)
print(f"Most common confusion: {CIFAR_CLASSES[i]} predicted as {CIFAR_CLASSES[j]} ({off[i, j]} times)")

with open(os.path.join(RESULTS_DIR, 'p3a_basic_cnn.json'), 'w') as f:
    json.dump({**hist, 'params': count_params(model), 'train_acc': train_acc, 'test_acc': test_acc,
               'best_val_checkpoint_test_acc': best_test_acc, 'confusion_matrix': cm.tolist()}, f, indent=2)
