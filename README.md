# Deep Learning — Homework 1

PyTorch implementation of all parts of Homework 1 (MLPs on MNIST, optimization, CNNs on CIFAR-10, transfer-learning bonus).

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Datasets download automatically into `data/` on first run. MNIST comes from torchvision. For CIFAR-10, the loader uses torchvision's copy if it is already in `data/`. Otherwise it downloads the official Hugging Face mirror (`uoft-cs/cifar10`, identical images and labels), because the original Toronto server is often very slow.

## Running

Run every script from the repository root:

| Part | Command | Outputs |
|---|---|---|
| 1A + 1B | `python problem1/train_mnist.py` | printed accuracies, `figures/p1b_activation_comparison.png` |
| 1C | `python problem1/xor.py` | `figures/p1c_xor_decision_boundary.png` |
| 2A | `python problem2/optimizers.py` | `figures/p2a_optimizers.png` |
| 2B | `python problem2/gradients.py` | `figures/p2b_gradients.png` |
| 2C | `python problem2/regularization.py` | `figures/p2c_regularization.png` |
| 3A | `python problem3/basic_cnn.py` | `figures/p3a_loss_curves.png`, `figures/p3a_confusion_matrix.png`, `results/p3a_basic_cnn.pt` |
| 3B | `python problem3/architectures.py` | `figures/p3b_architectures.png` |
| 3C | `python problem3/visualize_features.py` (needs 3A's saved weights) | `figures/p3c_conv1_filters.png`, `figures/p3c_activation_maps.png` |
| Bonus | `python problem3/transfer_learning.py` | `figures/bonus_transfer_learning.png` |

Every script prints its results tables in Markdown and saves the raw numbers to `results/*.json`. Console logs from the reference run are in `logs/`.

## Code layout

- `common.py`: seeding, device selection, in-memory data loaders, and the shared train/evaluate loops.
- `problem1/mlp.py`: the flexible `MLP` class (any number of hidden layers; ReLU/Sigmoid/Tanh; optional dropout). Problem 2 reuses it.
- `problem3/models.py`: `BasicCNN` (3A), `ConfigurableCNN` (3B), and the CNN training loop with validation tracking.

## Reproducibility notes

- All scripts seed Python, NumPy and PyTorch (seed 42). Within an experiment, every model starts from the same initial weights.
- MLPs run on the CPU (faster than a GPU for such small models). CNNs use CUDA or Apple MPS when available. GPU kernels are not bit-for-bit deterministic, so CNN numbers can vary by a few tenths of a percent between runs or machines.
- CIFAR-10 split: 45,000 train / 5,000 validation (fixed seed, taken from the official training set) / 10,000 official test images.
- Reference run hardware: Apple M2 (8 GB), PyTorch 2.14.0. Training times depend on the hardware.
