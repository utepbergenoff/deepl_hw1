"""Problem 2B: the vanishing gradient problem in a deep MLP (sigmoid vs ReLU).

Network: 784 -> 6 hidden layers of 128 -> 10 (7 Linear layers in total).
After every backward pass (every GRAD_EVERY steps) we record the L2 norm of the
weight gradient of each Linear layer. Plain SGD is used so the recorded gradient
is exactly what moves the weights (adaptive optimizers like Adam would rescale
small gradients and hide the effect in the updates).
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'problem1'))
from common import FIG_DIR, RESULTS_DIR, evaluate, get_device, load_mnist, set_seed, train_one_epoch
from mlp import MLP

HIDDEN = [128] * 6
EPOCHS = 5
GRAD_EVERY = 20  # record gradients every 20 mini-batches
device = get_device(prefer_gpu=False)

set_seed(42)
train_loader, test_loader = load_mnist(device, batch_size=64)


def run(activation):
    set_seed(42)
    model = MLP(784, HIDDEN, 10, activation).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    grad_log = []  # one row per recorded step, one column per layer

    # ---- gradient tracking: a hook that runs between backward() and step()
    def record_grads(step):
        if step % GRAD_EVERY == 0:
            grad_log.append([layer.weight.grad.norm().item() for layer in model.layers])

    losses, accs = [], []
    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, after_backward=record_grads)
        acc = evaluate(model, test_loader)
        losses.append(loss)
        accs.append(acc)
        print(f"[{activation}] epoch {epoch + 1}  loss {loss:.4f}  test acc {acc:.4f}")
    return np.array(grad_log), losses, accs


results = {act: run(act) for act in ['sigmoid', 'relu']}

n_layers = len(HIDDEN) + 1
layer_names = [f'L{i + 1}' for i in range(n_layers)]  # L1 = closest to input, L7 = output layer
colors = {'sigmoid': 'tab:red', 'relu': 'tab:blue'}

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

# (1) average gradient norm per layer over all of training
ax = axes[0]
for act, (g, _, _) in results.items():
    ax.plot(range(1, n_layers + 1), g.mean(0), marker='o', color=colors[act], label=act)
ax.set_yscale('log')
ax.set_xticks(range(1, n_layers + 1), layer_names)
ax.set_xlabel('Layer (L1 = input side, L7 = output side)')
ax.set_ylabel('Mean ||dL/dW|| (log scale)')
ax.set_title('Gradient magnitude per layer (averaged over training)')
ax.grid(alpha=0.3, which='both')
ax.legend()

# (2) and (3) gradient norm of each layer over the course of training
steps = np.arange(len(results['relu'][0])) * GRAD_EVERY
cmap = plt.get_cmap('viridis')
for ax, act in zip(axes[1:], ['sigmoid', 'relu']):
    g = results[act][0]
    for i in range(n_layers):
        ax.plot(steps, g[:, i], color=cmap(i / (n_layers - 1)), label=layer_names[i], lw=1)
    ax.set_yscale('log')
    ax.set_xlabel('Training step (mini-batch)')
    ax.set_ylabel('||dL/dW|| (log scale)')
    ax.set_title(f'{act}: per-layer gradient norm during training')
    ax.grid(alpha=0.3, which='both')
    ax.legend(ncol=2, fontsize=8)
# same y-range on both panels so sigmoid and ReLU are directly comparable
all_g = np.concatenate([results['sigmoid'][0].ravel(), results['relu'][0].ravel()])
for ax in axes[1:]:
    ax.set_ylim(all_g.min() / 2, all_g.max() * 2)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'p2b_gradients.png'), dpi=150)

print("\nMean gradient norm per layer:")
print("| Layer | " + " | ".join(results) + " |")
print("|---|---|---|")
for i in range(n_layers):
    print(f"| {layer_names[i]} | " + " | ".join(f"{results[a][0][:, i].mean():.2e}" for a in results) + " |")
for act, (g, _, accs) in results.items():
    m = g.mean(0)
    print(f"{act}: output-layer / first-layer gradient ratio = {m[-1] / m[0]:.1f}x, "
          f"final test acc {accs[-1] * 100:.2f}%")

with open(os.path.join(RESULTS_DIR, 'p2b_gradients.json'), 'w') as f:
    json.dump({a: {'mean_grad_per_layer': g.mean(0).tolist(), 'train_loss': l, 'test_acc': c}
               for a, (g, l, c) in results.items()}, f, indent=2)
