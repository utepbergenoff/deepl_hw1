"""Shared helpers for all problems: seeding, devices, data loading, train/eval loops.

The datasets are small enough to fit in memory, so instead of a torch DataLoader
(which decodes every image again each epoch) we load them once into tensors and
slice mini-batches from those. This makes the MLP experiments several times faster.
"""
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
FIG_DIR = os.path.join(ROOT, 'figures')
RESULTS_DIR = os.path.join(ROOT, 'results')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']


def set_seed(seed=42):
    """Seed every random number generator we use so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(prefer_gpu=True):
    """CUDA > Apple MPS > CPU. Small MLPs are faster on CPU, so they pass prefer_gpu=False."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    if prefer_gpu and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class TensorLoader:
    """Minimal DataLoader replacement that yields mini-batches from in-memory tensors."""

    def __init__(self, x, y, batch_size=64, shuffle=False, transform=None):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform  # applied per batch, e.g. uint8 -> normalized float

    def __len__(self):
        return (len(self.x) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.x)
        idx = torch.randperm(n) if self.shuffle else torch.arange(n)
        for start in range(0, n, self.batch_size):
            batch = idx[start:start + self.batch_size].to(self.x.device)
            xb = self.x[batch]
            if self.transform is not None:
                xb = self.transform(xb)
            yield xb, self.y[batch]


# ---------------------------------------------------------------- MNIST
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def load_mnist(device, batch_size=64):
    """Returns (train_loader, test_loader) with normalized 1x28x28 float images."""
    train = datasets.MNIST(DATA_DIR, train=True, download=True)
    test = datasets.MNIST(DATA_DIR, train=False, download=True)

    def prep(ds):
        x = ds.data.float().div(255).sub(MNIST_MEAN).div(MNIST_STD).unsqueeze(1)
        return x.to(device), ds.targets.to(device)

    x_tr, y_tr = prep(train)
    x_te, y_te = prep(test)
    return (TensorLoader(x_tr, y_tr, batch_size, shuffle=True),
            TensorLoader(x_te, y_te, 1000, shuffle=False))


# ---------------------------------------------------------------- CIFAR-10
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


HF_CIFAR10 = 'https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{}-00000-of-00001.parquet'


def _cifar10_arrays():
    """Returns ((x_train, y_train), (x_test, y_test)) as uint8 NHWC / int64 numpy arrays.

    Uses torchvision's copy if it is already in data/. Otherwise it downloads the
    official Hugging Face mirror (uoft-cs/cifar10, same images and label ids),
    because the original Toronto server is often extremely slow. The decoded
    arrays are cached in data/cifar10.npz so this only happens once.
    """
    try:
        tr = datasets.CIFAR10(DATA_DIR, train=True, download=False)
        te = datasets.CIFAR10(DATA_DIR, train=False, download=False)
        return ((tr.data, np.array(tr.targets)), (te.data, np.array(te.targets)))
    except RuntimeError:
        pass  # torchvision files not present

    cache = os.path.join(DATA_DIR, 'cifar10.npz')
    if not os.path.exists(cache):
        import io
        import urllib.request
        import pyarrow.parquet as pq
        from PIL import Image
        os.makedirs(DATA_DIR, exist_ok=True)
        arrays = {}
        for split in ['train', 'test']:
            print(f"Downloading CIFAR-10 {split} split from Hugging Face...")
            with urllib.request.urlopen(HF_CIFAR10.format(split)) as r:
                table = pq.read_table(io.BytesIO(r.read())).to_pydict()
            arrays[f'x_{split}'] = np.stack([np.array(Image.open(io.BytesIO(img['bytes'])).convert('RGB'))
                                             for img in table['img']])
            arrays[f'y_{split}'] = np.array(table['label'], dtype=np.int64)
        np.savez(cache, **arrays)
    d = np.load(cache)
    return ((d['x_train'], d['y_train']), (d['x_test'], d['y_test']))


def load_cifar10(device, batch_size=64, val_size=5000, seed=42):
    """Returns (train_loader, val_loader, test_loader).

    The 50k official training images are split into 45k train / 5k validation
    (fixed seed). Images stay uint8 on the device and are normalized per batch.
    """
    (x_all, y_all), (x_te, y_te) = _cifar10_arrays()

    def to_tensors(x, y):
        x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()  # N,3,32,32 uint8
        return x.to(device), torch.from_numpy(y).long().to(device)

    x_all, y_all = to_tensors(x_all, y_all)
    x_te, y_te = to_tensors(x_te, y_te)

    perm = torch.randperm(len(x_all), generator=torch.Generator().manual_seed(seed)).to(device)
    val_idx, tr_idx = perm[:val_size], perm[val_size:]

    mean, std = CIFAR_MEAN.to(device), CIFAR_STD.to(device)

    def normalize(xb):
        return (xb.float() / 255 - mean) / std

    return (TensorLoader(x_all[tr_idx], y_all[tr_idx], batch_size, shuffle=True, transform=normalize),
            TensorLoader(x_all[val_idx], y_all[val_idx], 1000, transform=normalize),
            TensorLoader(x_te, y_te, 1000, transform=normalize))


# ---------------------------------------------------------------- training
def train_one_epoch(model, loader, optimizer, criterion, after_backward=None):
    """One pass over the training set. Returns the average training loss.

    after_backward(step) is an optional hook called after loss.backward() and
    before optimizer.step() -- used in Problem 2B to record gradients.
    """
    model.train()
    total_loss = 0.0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        if after_backward is not None:
            after_backward(step)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion=None):
    """Returns accuracy, or (loss, accuracy) if a criterion is given. Uses eval mode (no dropout)."""
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        out = model(x)
        if criterion is not None:
            loss_sum += criterion(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    acc = correct / total
    return (loss_sum / total, acc) if criterion is not None else acc


@torch.no_grad()
def predict(model, loader):
    """Returns (all predictions, all labels) as CPU tensors."""
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x).argmax(1).cpu())
        labels.append(y.cpu())
    return torch.cat(preds), torch.cat(labels)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sync(device):
    """Wait for queued GPU work so wall-clock timings are honest."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


class Timer:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        sync(self.device)
        self.start = time.time()
        return self

    def __exit__(self, *exc):
        sync(self.device)
        self.elapsed = time.time() - self.start
