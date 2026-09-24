# Report material (results + talking points)

Numbers come from the reference run (logs in `logs/`, raw data in `results/`). The tables and figures can go straight into the PDF report. The **analysis sections are bullet points to write up in your own words**, not finished text.

---

## Problem 1A: MLP implementation
- Code: `problem1/mlp.py` (784 → 128 → 64 → 10, raw logits; softmax is inside `CrossEntropyLoss`).
- ReLU, Adam lr=0.001, batch 64, 5 epochs: **train 98.52%, test 97.42%**.

## Problem 1B: activation functions (5 epochs, Adam 0.001) → `figures/p1b_activation_comparison.png`

| Activation | Final train loss | Train acc | Test acc |
|---|---|---|---|
| ReLU | 0.0493 | 98.52% | **97.42%** |
| Sigmoid | 0.0619 | 98.59% | 97.14% |
| Tanh | 0.0505 | 98.68% | 97.18% |

Talking points:
- ReLU is best on test, but by only ~0.25 pp. With 2 hidden layers and Adam, all three work fine.
- Sigmoid has the highest loss and the slowest start (epoch-1 loss 0.57 vs 0.26 ReLU / 0.28 tanh): its max derivative is 0.25, it saturates, and its outputs aren't zero-centred.
- Tanh is zero-centred (derivative up to 1), so it sits between the two.
- ReLU: gradient is 1 for active units, so it doesn't saturate and is cheap to compute. This matters much more in deep nets (see 2B).

## Problem 1C: XOR → `figures/p1c_xor_decision_boundary.png`
- 2 → 4 (tanh) → 1 (sigmoid), BCE loss, Adam 0.05, 1000 full-batch epochs.
- Outputs: (0,0)→0.0000, (0,1)→0.9992, (1,0)→0.9992, (1,1)→0.0014. **Accuracy 4/4 = 100%.**
- The boundary is two curves that separate the diagonal pairs, something a single linear neuron can't do.

## Problem 2A: optimizers (20 epochs) → `figures/p2a_optimizers.png`

| Optimizer | Final test acc | Training time (s) |
|---|---|---|
| SGD (lr 0.01) | 97.47% | 5.4 |
| SGD + momentum 0.9 | **98.12%** | 6.4 |
| RMSprop (lr 0.001) | 98.03% | 8.1 |
| Adam (lr 0.001) | 97.78% | 9.7 |

Talking points:
- Plain SGD is by far the slowest to converge: 90.4% after epoch 1 vs ~96% for the others, and it is still improving at epoch 20.
- RMSprop and Adam are fastest early (epoch 1: 96.2% / 95.9%), but their test accuracy is noisier and plateaus around 97.5–98%.
- Momentum ends with the lowest training loss (~5e-4, ~25× lower than Adam/RMSprop) and the best test accuracy.
- Time per epoch grows with optimizer state: SGD keeps none, momentum keeps 1 buffer, RMSprop 1, Adam 2 (plus bias correction). On a model this small the gap is small in absolute terms.
- Adam/RMSprop's training loss bumps up late on (per-parameter adaptive steps with a fixed lr).

## Problem 2B: vanishing gradients → `figures/p2b_gradients.png`
- 784 → 6×128 → 10 (7 linear layers), plain SGD lr 0.01, 5 epochs. The ‖∂L/∂W‖ of each layer is recorded every 20 steps via a hook between `backward()` and `step()` (`after_backward` in `common.train_one_epoch`).

| Layer | Sigmoid mean ‖grad‖ | ReLU mean ‖grad‖ |
|---|---|---|
| L1 (input side) | 3.6e-05 | 0.90 |
| L2 | 4.0e-05 | 0.58 |
| L3 | 2.7e-04 | 0.56 |
| L4 | 1.9e-03 | 0.52 |
| L5 | 1.3e-02 | 0.50 |
| L6 | 9.1e-02 | 0.49 |
| L7 (output) | 0.68 | 0.65 |

- Sigmoid: the output layer's gradient is **~19,000× larger** than the first layer's, shrinking by ~7× per layer over L2–L7 (sigmoid' ≤ 0.25, multiplied at each layer by the chain rule). Test accuracy is **10.1%** after 5 epochs, i.e. chance level: the network never learned.
- ReLU: gradients are roughly the same size in every layer (ratio 0.7×). Test accuracy is **91.1%**.
- Worth noting: ReLU also sat at loss ≈ 2.30 for ~2 epochs before taking off, visible in the right panel. The cause is PyTorch's default init, which isn't He init. It shows depth + initialization matter even with ReLU, but the gradients were never 10⁴× apart.

## Problem 2C: regularization (Adam 0.001, 20 epochs) → `figures/p2c_regularization.png`

| Config | Train acc | Test acc | Gap (pp) |
|---|---|---|---|
| Baseline | 99.46% | 97.51% | 1.95 |
| L2 (wd=0.001) | 98.84% | **97.67%** | 1.17 |
| Dropout p=0.5 | 98.30% | 97.23% | **1.07** |

Talking points:
- Both regularizers roughly halve the train–test gap. Dropout gives the smallest gap, L2 the best test accuracy.
- Dropout 0.5 on such narrow layers (128/64) slightly *underfits*: lowest train accuracy, and still climbing at epoch 20, so more epochs would probably help.
- The baseline's test accuracy plateaus from epoch ~5 while train keeps rising (the textbook overfitting signature). The effect is mild because MNIST with 60k samples is easy for this model.
- Train accuracy is measured in eval mode (dropout off), so the comparison is fair.

## Problem 3A: basic CNN → `figures/p3a_loss_curves.png`, `figures/p3a_confusion_matrix.png`
- 545,098 params, Adam 0.001, batch 64, 20 epochs, 45k/5k/10k split, 108 s on M2 (MPS).
- **Test accuracy (epoch 20): 69.59%**; train 98.19%. The best-validation checkpoint (epoch 7) gives 71.19% on test.
- Strong overfitting: val loss bottoms at epoch 4 (0.84) and climbs to 2.15 by epoch 20, while train loss → 0.07. There's no augmentation or regularization.
- Confusion matrix: best classes frog 80.5%, ship 80.1%, truck 79.5%; worst cat 47.1%, dog 60.9%, bird 57.9%. The top confusion is cat→dog (193), then dog→cat (170) and automobile→truck (125): visually similar classes.

## Problem 3B: architectures (same training setup, 20 epochs) → `figures/p3b_architectures.png`

| Architecture | Conv layers / filters / kernel / pooling | Params | Train time (s) | Test acc (ep 20) | Test acc (best-val ep) |
|---|---|---|---|---|---|
| A: Shallow | 1 conv: 16, 5×5, max | 526,922 | 80 | 62.96% | 65.94% (ep 8) |
| B: Medium | 3 conv: 32→64→128, 3×3, avg | **356,810** | 115 | 74.67% | 74.67% (ep 20) |
| C: Deep VGG-style | 6 conv: 64,64→128,128→256,256, 3×3, max | 1,671,114 | 563 | **77.47%** | **79.24%** (ep 5) |
| 3A (reference) | 2 conv: 32→64, 3×3, max | 545,098 | 108 | 69.59% | 71.19% (ep 7) |

Talking points:
- Parameter count ≠ capacity that matters: B has the *fewest* params but beats A and 3A by 5–12 pp. In A and 3A, 96–99% of the parameters sit in the first FC layer (large flattened map), while B spends them on depth and a small 4×4×128 map.
- Depth gives hierarchical features and larger receptive fields. C is best (79% at its best epoch), but costs 3× B's params and ~5× its training time.
- C overfits fastest (val loss minimum at epoch 5). B's average pooling plus its small FC layer seems to overfit latest (best epoch 20): a smoother downsampling that keeps less of the exact peak response.
- Diminishing returns: B→C is 4.7× params and 4.9× time for +2.8 pp (+4.6 at the best epoch).
- Every model overfits by epoch 20 without augmentation, so early stopping on validation gives better test numbers (last column).
- Confound to acknowledge: several factors change at once between architectures, so effects aren't fully isolated.

## Problem 3C: features → `figures/p3c_conv1_filters.png`, `figures/p3c_activation_maps.png`
- All 32 conv1 filters shown (3×3 RGB). Many are colour-opponent or oriented-edge detectors, e.g. #17 (green), #20 (blue→orange gradient), #18 (red horizontal band).
- Activation maps for cat / ship / automobile, all classified correctly:
  - Some maps fire on edges/outlines (#3, #11, #19 on the cat's fur stripes; the car's contour).
  - Some fire on large colour regions (#10, #18 light up the background: the orange wall behind the cat, the red wall behind the car).
  - The ship shows many horizontal-edge responses (#8, #26, #29), matching the hull and deck lines.
- Takeaway: the first layer learns low-level edges and colour blobs; class-specific structure only appears in deeper layers.

## Bonus: transfer learning → `figures/bonus_transfer_learning.png`

| Model | Params | Epochs | Train time (s) | Test acc |
|---|---|---|---|---|
| Basic CNN (3A, scratch) | 545,098 | 20 | 108 | 69.59% |
| ResNet18, ImageNet-pretrained, fine-tuned | 11,181,642 | 3 | 407 | **93.02%** |

- Setup: final fc 512→1000 replaced by 512→10. Images upsampled 32→128 px and ImageNet-normalized; all layers fine-tuned with Adam lr=1e-4.
- 93.8% validation accuracy after only **one** epoch, which a from-scratch CNN never reaches.
- Why it helps: ImageNet's 1.2M images already taught general edge→texture→part features, so we only adapt them rather than learn from 45k images. That's a better starting point and acts like a strong prior against overfitting.
- Caveat for fairness: much larger model and higher input resolution, so it isn't purely a "pretraining" effect. (A from-scratch ResNet18 run would isolate that, if you want to add it.)
