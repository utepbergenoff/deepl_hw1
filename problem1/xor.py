"""Problem 1C: a 2-4-1 MLP that learns XOR.

XOR is not linearly separable, so a single neuron cannot solve it. One hidden
layer lets the network bend the input space so the two classes become separable.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import FIG_DIR, set_seed

set_seed(42)

# The four XOR cases: output is 1 when exactly one input is 1
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])


class XORNet(nn.Module):
    def __init__(self, hidden=4):
        super().__init__()
        self.hidden = nn.Linear(2, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h = torch.tanh(self.hidden(x))
        return torch.sigmoid(self.out(h))  # probability that the output is 1


model = XORNet(hidden=4)
criterion = nn.BCELoss()  # binary cross-entropy, matches the sigmoid output
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

losses = []
for epoch in range(1000):  # full-batch training: all 4 points every step
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}, loss {loss.item():.5f}")

with torch.no_grad():
    probs = model(X)
preds = (probs > 0.5).float()
acc = (preds == y).float().mean().item()

print("\n x1 x2 | target | P(y=1)  | pred")
for xi, yi, pi, pr in zip(X, y, probs, preds):
    print(f"  {int(xi[0])}  {int(xi[1])} |   {int(yi)}    | {pi.item():.4f}  |  {int(pr)}")
print(f"Accuracy on the 4 XOR cases: {acc * 100:.0f}%")

# ---- decision boundary: evaluate the network on a dense grid over the input plane
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 300), np.linspace(-0.5, 1.5, 300))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
with torch.no_grad():
    zz = model(grid).numpy().reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
cf = ax.contourf(xx, yy, zz, levels=20, cmap='RdBu_r', alpha=0.8)
ax.contour(xx, yy, zz, levels=[0.5], colors='black', linewidths=2)  # the 0.5 boundary
for xi, yi in zip(X, y):
    ax.scatter(xi[0], xi[1], s=250, c='white' if yi == 0 else 'black',
               edgecolors='black', linewidths=2, zorder=3)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title(f'XOR decision boundary (2-4-1 MLP), accuracy {acc * 100:.0f}%')
fig.colorbar(cf, ax=ax, label='P(y = 1)')

axes[1].plot(losses)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('BCE loss')
axes[1].set_title('XOR training loss')
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'p1c_xor_decision_boundary.png'), dpi=150)
print("Saved figures/p1c_xor_decision_boundary.png")
