# Chapter 08a: Semi-Supervised Learning

[Back to Paradigms Overview](08_paradigms.md) | [中文文档](08a_semi_supervised_CN.md) | [Next: Domain Adaptation](08b_domain_adaptation.md)

---

## 1. When Should You Use Semi-Supervised Learning?

Imagine you have a hospital dataset of 10,000 CT scans, but only 200 have been carefully labeled by a radiologist. Labeling the rest would take months and cost tens of thousands of dollars. **Semi-supervised learning** lets you use those 9,800 unlabeled scans to improve your model — often recovering 85–95% of the performance you'd get with full labeling, at a fraction of the cost.

**In short**: if you have lots of images but few labels, start here.

### Real-World Scenarios

| Scenario | Labeled | Unlabeled | Semi-Supervised Benefit |
|----------|---------|-----------|------------------------|
| New hospital, rare disease | 50 scans | 5,000 scans | Huge — labels are extremely scarce |
| Existing dataset, want to improve | 500 scans | 2,000 scans | Moderate — marginal gain from extra data |
| Active learning pipeline | 100 scans | 10,000 scans | Large — selectively label the hardest cases |

---

## 2. Core Concepts

### 2.1 The Fundamental Idea

Semi-supervised learning exploits one observation: **even without labels, images carry useful information**. The model can learn the *structure* of the data — what a "liver" looks like in general — from unlabeled images. Then, the few labeled images teach it the *specific boundaries*.

Think of it like learning a language: you can pick up vocabulary and grammar patterns by listening to conversations (unlabeled), even if nobody explicitly teaches you every word (labeled).

### 2.2 Mean Teacher — Consistency Regularization

The Mean Teacher framework (Tarvainen & Valpola, 2017) is the most widely used semi-supervised approach. It is built on a simple but powerful intuition:

> **A good model should give the same prediction whether or not you add noise to the input.**

If you slightly rotate, flip, or add noise to a CT scan, the liver is still in the same place. The model's prediction shouldn't change.

**How it works:**

```
                  ┌──────────────────────┐
                  │    Student Model     │
                  │    (trained with     │
                  │     gradients)       │
                  └──────────┬───────────┘
                             │
              EMA update:    │  θ_teacher = α·θ_teacher + (1-α)·θ_student
              (no gradient)  │
                             ▼
                  ┌──────────────────────┐
                  │    Teacher Model     │
                  │    (EMA of student,  │
                  │     more stable)     │
                  └──────────────────────┘
```

1. **Student** trains normally on labeled data with a segmentation loss (CE + Dice).
2. **Teacher** is an exponential moving average (EMA) of the student — it doesn't receive gradients directly, but slowly tracks the student's weights.
3. On **unlabeled data**, both models make predictions. The consistency loss encourages the student to match the teacher:

$$\mathcal{L}_{\text{consistency}} = \text{MSE}(f_{\text{student}}(x + \epsilon),\; f_{\text{teacher}}(x))$$

**Why EMA?** A single student can be noisy during training. The EMA averages out this noise, giving the teacher smoother, more reliable predictions. It's like asking a panel of experts (averaged) rather than one overconfident junior doctor.

**Key parameters:**

| Parameter | Meaning | Typical Range |
|-----------|---------|--------------|
| `ema_decay` (α) | How slowly the teacher updates | 0.99–0.999 (higher = more stable) |
| `consistency_weight` | How much the consistency loss matters | 0.1–1.0 |
| `rampup_epochs` | How many epochs to slowly increase consistency_weight from 0 | 20–80 |

### 2.3 Cross Pseudo Supervision (CPS)

CPS (Chen et al., CVPR 2021) takes a different approach: instead of one model with an EMA teacher, it uses **two independently initialized networks** that teach each other.

```
        Model A                          Model B
   ┌──────────────┐                ┌──────────────┐
   │  predicts    │──pseudo-label──▶│  learns      │
   │  on unlabeled│                │  from A's    │
   │              │◀──pseudo-label──│  predictions │
   └──────────────┘                └──────────────┘
```

The key insight: two networks with different random initializations will make **different mistakes**. By cross-supervising, each network corrects the other's blind spots.

**Pseudo-label generation:**

$$\hat{y}_A = \arg\max f_A(x), \qquad \hat{y}_B = \arg\max f_B(x)$$

$$\mathcal{L}_{\text{CPS}} = \mathcal{L}_{\text{CE}}(\hat{y}_A,\; f_B(x)) + \mathcal{L}_{\text{CE}}(\hat{y}_B,\; f_A(x))$$

**Trade-off vs. Mean Teacher:**
- CPS needs ~2× the GPU memory (two full models in memory).
- CPS can be more accurate because the two models genuinely disagree, providing richer learning signals.
- Mean Teacher is simpler and more memory-efficient.

### 2.4 UniMatch — Weak-to-Strong Consistency

UniMatch (Yang et al., CVPR 2023) is a modern single-model approach that combines the best of both worlds:

1. Apply a **weak augmentation** (flip + crop) to generate stable pseudo-labels.
2. Apply a **strong augmentation** (color jitter, CutMix) to the same image and train the model to match those pseudo-labels.
3. Add **feature-level noise** (dropout + multiplicative noise) for extra regularization.

This is more sample-efficient than Mean Teacher because the strong augmentations force the model to learn more robust features.

---

## 3. How It Works in APRIL-MedSeg

### 3.1 Training Script

All semi-supervised methods use `semi_train.py`:

```bash
python semi_train.py --config configs/training_paradigms/semi_supervision/mean_teacher.yaml
```

### 3.2 YAML Configuration Walkthrough

Let's examine a Mean Teacher config step by step:

```yaml
model:
  num_classes: 4          # Number of segmentation classes (e.g., background + 3 organs)
  img_size: 224           # Input image resolution
  encoder:
    name: timm_resnet34   # Backbone encoder (ResNet-34 via timm)
    pretrained: true      # Use ImageNet-pretrained weights
    in_channels: 3        # RGB input
  decoder:
    name: bilinear        # Simple bilinear upsampling decoder
  bottleneck:
    name: none            # No bottleneck module

data:
  img_size: 224
  labeled_dir: ./data/labeled        # Directory with labeled images + masks
  unlabeled_dir: ./data/unlabeled    # Directory with unlabeled images only
  val_dir: ./data/val                # Validation set (with masks, for evaluation)
  test_dir: ./data/test              # Test set
  test_list: ./data/test/list.txt
  labeled_ratio: 0.1                 # Use 10% of data as labeled
  split_mode: dir                    # 'dir' = separate dirs; 'ratio' = auto-split

semi:
  method: mean_teacher               # Which semi-supervised algorithm to use
  params:
    ema_decay: 0.999                 # Teacher EMA decay (higher = slower update)
    consistency_weight: 1.0          # Weight of the consistency loss
    rampup_epochs: 40                # Gradually increase consistency_weight over 40 epochs

training:
  epochs: 200
  batch_size: 16                     # Total batch size
  labeled_batch_size: 8              # How many labeled images per batch
  unlabeled_batch_size: 8            # How many unlabeled images per batch
  loss:
    name: compound
    params:
      losses:
        - name: ce
          weight: 0.4               # Cross-entropy loss for labeled data
        - name: dice
          weight: 0.6               # Dice loss for labeled data
  optimizer:
    name: adamw
    lr: 0.0001
  scheduler:
    name: cosine
    min_lr: 0.000001
```

### 3.3 Available Methods

| Method | Config File | Key Innovation | GPU Memory |
|--------|------------|----------------|------------|
| Mean Teacher | `mean_teacher.yaml` | EMA teacher + MSE consistency | Normal (1 model) |
| CPS | `cps.yaml` | Two networks, cross pseudo-labels | ~2× (2 models) |
| UniMatch | `unimatch.yaml` | Weak-to-strong + feature noise | Normal (1 model) |
| CCT | `cct.yaml` | Cross-consistency training | ~2× |
| FixMatch | `fixmatch.yaml` | Weak/strong augmentation | Normal |
| UA-MT | `ua_mt.yaml` | Uncertainty-aware Mean Teacher | ~1.5× |
| Pseudo-Label | `pseudo_label.yaml` | Self-training with confidence threshold | Normal |

### 3.4 Data Preparation

Semi-supervised training requires **two separate data directories**:

```
data/
├── labeled/              # Images WITH pixel-level masks
│   ├── img001.npy
│   ├── img001_mask.npy
│   └── ...
├── unlabeled/            # Images WITHOUT masks
│   ├── img101.npy
│   ├── img102.npy
│   └── ...
├── val/                  # Validation (always with masks)
│   ├── val001.npy
│   └── val001_mask.npy
└── test/                 # Test set
    ├── list.txt
    └── ...
```

If you have one big dataset and want to auto-split by ratio, use `split_mode: ratio` with `labeled_ratio: 0.1` to use 10% as labeled.

---

## 4. Step-by-Step: Your First Semi-Supervised Run

### Step 1: Prepare your data

Make sure your labeled and unlabeled data are in separate directories (or use ratio-based splitting).

### Step 2: Choose a method

For your first run, **Mean Teacher** is the safest choice — it's simple, memory-efficient, and well-studied.

### Step 3: Train

```bash
python semi_train.py \
    --config configs/training_paradigms/semi_supervision/mean_teacher.yaml \
    --output_dir output/semi_mean_teacher
```

### Step 4: Monitor training

Watch the logs for these key metrics:
- **Supervised loss** (on labeled data) — should decrease normally.
- **Consistency loss** (on unlabeled data) — should decrease as student learns to match teacher.
- **Validation Dice** — the main metric you care about.

### Step 5: Evaluate

```bash
python test.py \
    --config configs/training_paradigms/semi_supervision/mean_teacher.yaml \
    --checkpoint output/semi_mean_teacher/best_model.pth
```

---

## 5. Parameter Tuning Guide

### How to Choose `ema_decay`

| Value | Behavior | When to Use |
|-------|----------|-------------|
| 0.99 | Teacher updates relatively quickly | Small dataset, fast convergence |
| 0.999 | Teacher is very stable | Large dataset, long training (recommended default) |
| 0.9999 | Teacher barely changes | Very noisy data, need maximum stability |

### How to Choose `consistency_weight`

Start with **1.0** and adjust based on the loss ratio:
- If consistency loss dominates: reduce to 0.1.
- If supervised loss dominates and unlabeled data isn't helping: increase to 2.0–5.0.

### How to Choose `labeled_ratio`

| Ratio | Expected Dice (relative to full supervision) |
|-------|---------------------------------------------|
| 5% | ~70–85% |
| 10% | ~85–95% |
| 20% | ~92–98% |
| 50% | ~97–100% |

The sweet spot for demonstrating semi-supervised benefit is **10–20%**.

### How to Choose `rampup_epochs`

The rampup prevents the consistency loss from dominating early in training when the model (and teacher) are still poor. A good rule of thumb: **20–30% of total epochs**. For 200 epochs, use `rampup_epochs: 40–60`.

---

## 6. Common Pitfalls

### Pitfall 1: Consistency loss explodes early in training

**Symptom**: The consistency loss is 100× larger than the supervised loss in the first few epochs.

**Fix**: Increase `rampup_epochs` so the consistency weight grows more slowly. The teacher needs time to stabilize before it can provide useful targets.

### Pitfall 2: Teacher and student collapse together

**Symptom**: Both models make the same wrong predictions, reinforcing each other's errors.

**Fix**: 
- Lower `ema_decay` (e.g., 0.99 → 0.999) to slow down teacher updates.
- Increase data augmentation on unlabeled samples.
- Add a confidence threshold to ignore low-confidence predictions.

### Pitfall 3: Unlabeled data hurts performance

**Symptom**: Adding unlabeled data makes validation Dice *worse* than supervised-only training.

**Fix**:
- Check that unlabeled data is from the same distribution as labeled data.
- Reduce `consistency_weight` to limit the influence of unlabeled data.
- Try UniMatch instead of Mean Teacher — its strong augmentations are more robust.

### Pitfall 4: Out of memory with CPS

**Symptom**: CUDA OOM when using `cps.yaml`.

**Fix**: CPS requires two models in memory. Either:
- Reduce `batch_size` (e.g., 16 → 8, with labeled/unlabeled each 4).
- Use a smaller encoder (e.g., `timm_resnet18` instead of `timm_resnet50`).
- Switch to Mean Teacher or UniMatch (single model).

---

## 7. Recommended Experiments

### Experiment 1: Label Efficiency Curve

Train with different labeled ratios and compare:

```bash
# Modify labeled_ratio in mean_teacher.yaml for each run
python semi_train.py --config configs/training_paradigms/semi_supervision/mean_teacher.yaml \
    --override semi.params.labeled_ratio=0.05 data.labeled_ratio=0.05 \
    --output_dir output/semi_mt_5pct

python semi_train.py --config configs/training_paradigms/semi_supervision/mean_teacher.yaml \
    --override semi.params.labeled_ratio=0.10 data.labeled_ratio=0.10 \
    --output_dir output/semi_mt_10pct

python semi_train.py --config configs/training_paradigms/semi_supervision/mean_teacher.yaml \
    --override semi.params.labeled_ratio=0.20 data.labeled_ratio=0.20 \
    --output_dir output/semi_mt_20pct
```

**Expected results:**

| Labeled Ratio | Supervised Dice | Mean Teacher Dice | Improvement |
|--------------|-----------------|-------------------|-------------|
| 5% | ~55–65% | ~70–85% | +15–25% |
| 10% | ~65–75% | ~85–92% | +15–20% |
| 20% | ~78–85% | ~90–95% | +8–12% |
| 100% | ~88–92% | — | Baseline |

### Experiment 2: Method Comparison (same data, same model)

```bash
# Mean Teacher
python semi_train.py --config configs/training_paradigms/semi_supervision/mean_teacher.yaml

# CPS (needs more GPU memory)
python semi_train.py --config configs/training_paradigms/semi_supervision/cps.yaml

# UniMatch
python semi_train.py --config configs/training_paradigms/semi_supervision/unimatch.yaml
```

**Expected ranking** (at 10% labeled): UniMatch ≥ CPS > Mean Teacher > Supervised

---

## 8. Further Reading

### Key Papers

| Paper | Year | Venue | Key Idea |
|-------|------|-------|----------|
| [Mean Teacher](https://arxiv.org/abs/1703.01780) | 2017 | NeurIPS | EMA teacher for consistency regularization |
| [CPS](https://arxiv.org/abs/2106.01226) | 2021 | CVPR | Two networks cross-supervising with pseudo-labels |
| [UniMatch](https://arxiv.org/abs/2208.09910) | 2023 | CVPR | Weak-to-strong consistency with feature augmentation |
| [FixMatch](https://arxiv.org/abs/2001.07685) | 2020 | NeurIPS | Simple weak/strong augmentation pipeline |
| [UA-MT](https://arxiv.org/abs/1907.07034) | 2019 | MICCAI | Uncertainty-aware Mean Teacher |

### Related Documentation

- [All Semi-Supervised Methods](../paradigms/semi_supervised.md) — Complete method catalog
- [SSL4MIS Reference](https://github.com/HiLab-git/SSL4MIS) — Reference implementation for many semi-supervised methods

---

[Back to Paradigms Overview](08_paradigms.md) | [Next: Domain Adaptation](08b_domain_adaptation.md)
