# Chapter 08b: Domain Adaptation

[Back to Paradigms Overview](08_paradigms.md) | [中文文档](08b_domain_adaptation_CN.md) | [Previous: Semi-Supervised](08a_semi_supervised.md) | [Next: Knowledge Distillation](08c_distillation.md)

---

## 1. When Should You Use Domain Adaptation?

You trained a liver segmentation model on data from Hospital A's Siemens CT scanner. It works beautifully — 92% Dice. Then you deploy it at Hospital B with a GE scanner, and Dice drops to 71%. Same anatomy, same task, but the images look different: different contrast, noise texture, intensity range. This is **domain shift**, and it's one of the biggest obstacles to deploying medical AI.

**Domain adaptation** lets you take your Hospital-A-trained model and make it work on Hospital-B images — **without** needing to label Hospital-B data from scratch.

### Real-World Scenarios

| Scenario | Source Domain | Target Domain | Adaptation Approach |
|----------|--------------|---------------|-------------------|
| Multi-scanner deployment | Scanner A (labeled) | Scanner B (unlabeled) | Traditional UDA |
| Multi-center study | Center 1 (labeled) | Center 2–5 (unlabeled) | Multi-source DA |
| Already deployed, performance drift | Original data | New patient population | Test-time adaptation |
| Source data unavailable (privacy) | Pretrained model only | Target images | Source-free DA |

---

## 2. Core Concepts

### 2.1 The Problem: Distribution Shift

In standard supervised learning, we assume training and test data come from the same distribution:

$$P_{\text{train}}(X, Y) = P_{\text{test}}(X, Y)$$

Domain adaptation relaxes this assumption:

$$P_S(X, Y) \neq P_T(X, Y)$$

The shift can manifest in several ways:
- **Covariate shift**: $P(X)$ changes (different image appearance), but $P(Y|X)$ stays the same.
- **Label shift**: $P(Y)$ changes (different disease prevalence), but $P(X|Y)$ stays the same.
- **Concept shift**: $P(Y|X)$ changes (the same image features map to different labels).

In medical imaging, covariate shift is the most common — the anatomy is the same, but the image appearance differs.

### 2.2 Adversarial Alignment (DANN, AdvEnt)

The core idea: train a **domain discriminator** to distinguish source from target features, while simultaneously training the **feature extractor** to fool the discriminator.

```
Source images ──┐
                ├──▶ Feature Extractor G ──▶ Features ──┐
Target images ──┘                                        │
                                                         ▼
                                              Domain Discriminator D
                                              (source or target?)
                                                         │
           G tries to fool D          D tries to tell them apart
           ←── adversarial game ──→
```

**The adversarial objective:**

$$\min_G \max_D \; \mathbb{E}_{x \sim P_S}[\log D(G(x))] + \mathbb{E}_{x \sim P_T}[\log(1 - D(G(x)))]$$

At the saddle point, $G$ produces **domain-invariant features** — the discriminator cannot tell source from target, which means the features capture anatomy rather than scanner-specific artifacts.

**Gradient Reversal Layer (GRL)**: In practice, the adversarial game is implemented by reversing the gradient flowing from $D$ to $G$:

$$\text{GRL}(x) = x \quad \text{(forward)}, \qquad \frac{\partial \text{GRL}}{\partial x} = -\lambda I \quad \text{(backward)}$$

This makes $G$ learn features that *increase* $D$'s loss, while $D$ learns to *decrease* it.

### 2.3 Entropy Minimization (AdvEnt)

A complementary idea: in the target domain, the model's predictions should be **confident** (low entropy). If the model is unsure about a target image, it's likely confused by the domain shift.

$$\mathcal{L}_{\text{entropy}} = -\sum_c p_c \log p_c$$

Minimizing this loss pushes the model toward decisive predictions on target data. AdvEnt combines adversarial alignment with entropy minimization, and uses an adversarial training on the entropy map itself — encouraging the spatial entropy pattern to look like the source domain's.

### 2.4 Test-Time Adaptation (TENT)

TENT (Wang et al., ICLR 2021) is a radical simplification: **adapt the model during inference**, using only the incoming test images — no source data, no target labels.

```
Pretrained model ──▶ Receive test image ──▶ Update BatchNorm stats ──▶ Predict
                          ↑                        │
                          └── minimize entropy ────┘
```

**How it works:**
1. Load a model trained on the source domain.
2. For each test batch, compute the prediction entropy.
3. Backpropagate the entropy loss to update **only the BatchNorm affine parameters** (γ and β).
4. Make predictions with the adapted parameters.

**Why only BatchNorm?** BatchNorm statistics (mean, variance) are highly sensitive to distribution shift. Updating them is cheap (few parameters) and effective — it recalibrates the feature distributions to match the target domain.

**Key advantage**: TENT requires absolutely no labeled data — not even source data. It works with just a pretrained checkpoint and the test images.

### 2.5 Other Methods

| Method | Approach | Data Required |
|--------|----------|--------------|
| DANN | Adversarial alignment with GRL | Source (labeled) + Target (unlabeled) |
| AdvEnt | Adversarial entropy minimization | Source (labeled) + Target (unlabeled) |
| TENT | Test-time BatchNorm adaptation | Pretrained model + Target (test images only) |
| FDA | Fourier-domain style transfer | Source (labeled) + Target (unlabeled) |
| DPL | Denoised pseudo-labeling | Source (labeled) + Target (unlabeled) |
| CBMT | Class-balanced mean teacher | Source (labeled) + Target (unlabeled) |

---

## 3. How It Works in APRIL-MedSeg

### 3.1 Training Script

All domain adaptation methods use `train_domain_adaptation.py`:

```bash
# Traditional UDA (e.g., AdvEnt)
python train_domain_adaptation.py --config configs/training_paradigms/domain_adaptation/advent.yaml

# Test-time adaptation (e.g., TENT)
python train_domain_adaptation.py --config configs/training_paradigms/domain_adaptation/tent.yaml
```

### 3.2 YAML Configuration Walkthrough — AdvEnt

```yaml
model:
  num_classes: 4          # Segmentation classes
  img_size: 224
  encoder:
    name: timm_resnet34
    pretrained: true
    in_channels: 3
  decoder:
    name: bilinear
  bottleneck:
    name: none

data:
  img_size: 224
  # Source domain (labeled — this is where you have annotations)
  source:
    image_dir: ./data/source/images
    mask_dir: ./data/source/masks
  # Target domain (unlabeled — this is the new domain you want to adapt to)
  target:
    image_dir: ./data/target/images
  # Validation (target domain WITH labels — for evaluation only, NOT used in training)
  val:
    image_dir: ./data/target_val/images
    mask_dir: ./data/target_val/masks
  # Test set
  test:
    image_dir: ./data/target_test/images
    mask_dir: ./data/target_test/masks

domain_adaptation:
  method: advent                    # Which DA algorithm to use
  params:
    entropy_weight: 0.1             # Weight of entropy minimization loss
    adversarial_weight: 0.1         # Weight of adversarial alignment loss
    num_classes: 4                  # Must match model.num_classes

training:
  epochs: 100
  batch_size: 8                     # Total batch (source + target mixed)
  loss:
    name: compound
    params:
      losses:
        - name: ce
          weight: 0.4              # Segmentation loss (source only)
        - name: dice
          weight: 0.6
  optimizer:
    name: adamw
    lr: 0.0001
  scheduler:
    name: cosine
```

### 3.3 YAML Configuration Walkthrough — TENT

TENT is simpler because it only needs the target domain and a pretrained model:

```yaml
data:
  img_size: 224
  target:
    image_dir: ./data/target/images     # Target images (unlabeled)
  val:
    image_dir: ./data/target_val/images
    mask_dir: ./data/target_val/masks
  test:
    image_dir: ./data/target_test/images
    mask_dir: ./data/target_test/masks
  pretrained_model: ./checkpoints/source_model.pth   # Source-trained model

domain_adaptation:
  method: tent
  params:
    entropy_weight: 1.0              # Entropy minimization strength

training:
  epochs: 50                         # TENT needs fewer epochs (adaptation, not full training)
  optimizer:
    lr: 0.00001                      # Very small LR (only updating BatchNorm)
```

### 3.4 Available Methods

| Method | Config File | Type | Source Data Needed? |
|--------|------------|------|-------------------|
| AdvEnt | `advent.yaml` | Traditional UDA | Yes |
| DANN | `dann.yaml` | Traditional UDA | Yes |
| TENT | `tent.yaml` | Test-time adaptation | No (checkpoint only) |
| FDA | `fda.yaml` | Style transfer | Yes |
| DPL | `dpl.yaml` | Pseudo-labeling | Yes |
| CBMT | `cbmt.yaml` | Mean teacher + DA | Yes |
| Source Only | `source_only.yaml` | Baseline (no adaptation) | Yes |

### 3.5 Data Preparation

Domain adaptation requires **source and target data organized separately**:

```
data/
├── source/                  # Source domain (labeled)
│   ├── images/
│   │   ├── src_001.npy
│   │   └── ...
│   └── masks/
│       ├── src_001_mask.npy
│       └── ...
├── target/                  # Target domain (unlabeled — NO masks)
│   └── images/
│       ├── tgt_001.npy
│       └── ...
├── target_val/              # Target validation (labeled, for evaluation)
│   ├── images/
│   └── masks/
└── target_test/             # Target test (labeled, for final evaluation)
    ├── images/
    └── masks/
```

**Important**: The target_val and target_test directories need ground truth masks for evaluation, but these are NOT used during training — only for measuring adaptation success.

---

## 4. Step-by-Step: Your First Domain Adaptation Run

### Step 1: Train a source model first

Before adapting, you need a model trained on the source domain:

```bash
python train.py --config configs/architectures/combinations/general/unet_resnet34.yaml \
    --override data.train_dir=./data/source/images data.train_list=... \
    --output_dir output/source_model
```

### Step 2: Choose a DA method

For your first run:
- If you have **source data available during adaptation**: Use **AdvEnt** — it's well-studied and effective.
- If you only have a **pretrained checkpoint**: Use **TENT** — it's the simplest and requires no source data.

### Step 3: Adapt

```bash
# AdvEnt (traditional UDA)
python train_domain_adaptation.py \
    --config configs/training_paradigms/domain_adaptation/advent.yaml \
    --output_dir output/da_advent

# TENT (test-time adaptation)
python train_domain_adaptation.py \
    --config configs/training_paradigms/domain_adaptation/tent.yaml \
    --output_dir output/da_tent
```

### Step 4: Evaluate on target domain

```bash
python test.py \
    --config configs/training_paradigms/domain_adaptation/advent.yaml \
    --checkpoint output/da_advent/best_model.pth
```

Compare with the source-only baseline to measure adaptation gain.

---

## 5. Parameter Tuning Guide

### AdvEnt Parameters

| Parameter | Effect | Tuning Advice |
|-----------|--------|--------------|
| `entropy_weight` | How strongly to minimize target prediction entropy | Start at 0.1. If target predictions are too uncertain, increase to 0.5. |
| `adversarial_weight` | How strongly to align source/target features | Start at 0.1. If features aren't aligning, increase to 0.5. |
| `grl_lambda` | GRL gradient reversal strength | Usually auto-scheduled; manual override rarely needed. |

### TENT Parameters

| Parameter | Effect | Tuning Advice |
|-----------|--------|--------------|
| `entropy_weight` | Entropy minimization strength | Default 1.0. If adaptation is too aggressive, reduce to 0.1. |
| `lr` (optimizer) | Learning rate for BatchNorm update | Must be very small (1e-5). Larger LR causes model collapse. |
| `epochs` | Adaptation epochs | 10–50 is usually enough. More epochs risk overfitting to test data. |

### General Advice

- **Always run a source-only baseline first** to quantify the domain gap.
- **Start with small DA weights** (0.01–0.1) and increase — too-large weights cause the model to forget source knowledge.
- **Monitor target validation Dice** during training, not source loss.

---

## 6. Common Pitfalls

### Pitfall 1: Domain adaptation makes things worse

**Symptom**: Adapted model performs worse than source-only on target data.

**Fix**:
- Reduce `entropy_weight` and `adversarial_weight` — too-strong adaptation can destroy source knowledge.
- Check that source and target tasks are actually the same (same organs, same view).
- Try TENT instead — it's gentler because it only updates BatchNorm.

### Pitfall 2: Mode collapse in adversarial training

**Symptom**: The discriminator accuracy stays near 50% (random) but segmentation quality is terrible.

**Fix**:
- The adversarial loss may be overwhelming the segmentation loss. Reduce `adversarial_weight`.
- Use a gradient penalty or spectral normalization on the discriminator.
- Warm up the segmentation loss for 10–20 epochs before enabling the adversarial loss.

### Pitfall 3: TENT overfits to a small test set

**Symptom**: TENT-adapted model works well on one test batch but poorly on the next.

**Fix**:
- Use a smaller learning rate (1e-6 instead of 1e-5).
- Reduce adaptation epochs (5–10 instead of 50).
- Reset model between test batches (online vs. continual adaptation).

### Pitfall 4: Source and target have different class distributions

**Symptom**: Source data has mostly class A, target has mostly class B. Adaptation aligns overall features but per-class performance varies wildly.

**Fix**:
- Use class-balanced methods like CBMT.
- Add class-level adversarial alignment (per-class discriminators).
- Ensure your source data covers all classes that appear in the target.

---

## 7. Recommended Experiments

### Experiment 1: Quantify the Domain Gap

Before attempting adaptation, measure how bad the domain shift is:

```bash
# Train on source
python train.py --config configs/architectures/combinations/general/unet_resnet34.yaml \
    --output_dir output/source_only

# Test on target (no adaptation)
python test.py --config configs/training_paradigms/domain_adaptation/source_only.yaml \
    --checkpoint output/source_only/best_model.pth
```

If the gap between source Dice and target Dice is <3%, domain adaptation may not be worth the effort.

### Experiment 2: Method Comparison

```bash
# AdvEnt
python train_domain_adaptation.py --config configs/training_paradigms/domain_adaptation/advent.yaml

# DANN
python train_domain_adaptation.py --config configs/training_paradigms/domain_adaptation/dann.yaml

# TENT (from pretrained checkpoint)
python train_domain_adaptation.py --config configs/training_paradigms/domain_adaptation/tent.yaml
```

**Expected results:**

| Method | Source Dice | Target Dice | Adaptation Gain |
|--------|-----------|-------------|----------------|
| Source Only | ~90% | ~65–75% | — |
| AdvEnt | ~90% | ~78–85% | +8–15% |
| DANN | ~90% | ~75–82% | +5–12% |
| TENT | ~90% | ~75–82% | +5–10% |

### Experiment 3: How Much Target Data Helps

Vary the amount of target unlabeled data and measure adaptation quality:

| Target Data | AdvEnt Dice | TENT Dice |
|------------|-------------|-----------|
| 100 images | ~75% | ~73% |
| 500 images | ~80% | ~77% |
| 1000 images | ~83% | ~80% |
| 5000 images | ~85% | ~82% |

---

## 8. Further Reading

### Key Papers

| Paper | Year | Venue | Key Idea |
|-------|------|-------|----------|
| [DANN](https://arxiv.org/abs/1505.07818) | 2016 | JMLR | Domain-adversarial training with GRL |
| [AdvEnt](https://arxiv.org/abs/1811.12833) | 2019 | CVPR | Adversarial entropy minimization |
| [TENT](https://arxiv.org/abs/2006.10726) | 2021 | ICLR | Test-time BatchNorm adaptation |
| [FDA](https://arxiv.org/abs/2004.05498) | 2020 | NeurIPS | Fourier-domain style augmentation |
| [DAFormer](https://arxiv.org/abs/2111.14887) | 2022 | CVPR | Transformer-based DA |

### Related Documentation

- [All Domain Adaptation Methods](../paradigms/domain_adaptation.md) — Complete method catalog
- [ADA4MIA Benchmark](https://github.com/whq-xxh/ADA4MIA) — Reference implementation for many DA methods

---

[Back to Paradigms Overview](08_paradigms.md) | [Previous: Semi-Supervised](08a_semi_supervised.md) | [Next: Knowledge Distillation](08c_distillation.md)
