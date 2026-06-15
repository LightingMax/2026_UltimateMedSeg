# Chapter 08c: Knowledge Distillation

[Back to Paradigms Overview](08_paradigms.md) | [中文文档](08c_distillation_CN.md) | [Previous: Domain Adaptation](08b_domain_adaptation.md) | [Next: Weakly Supervised](08d_weakly_supervised.md)

---

## 1. When Should You Use Knowledge Distillation?

You trained a massive UNet with a ResNet-101 encoder and a CASCADE decoder. It achieves 92% Dice — beautiful. But it takes 800ms per image and 2GB of VRAM. You need to deploy it on an edge device that has 4GB total memory and needs real-time inference.

**Knowledge distillation** (KD) lets you train a **small student model** (e.g., ResNet-18 + bilinear decoder) to mimic the large **teacher model** — recovering 90–98% of the teacher's accuracy at a fraction of the compute.

### Real-World Scenarios

| Scenario | Teacher | Student | Why KD? |
|----------|---------|---------|---------|
| Edge deployment | ResNet-101 (2GB) | ResNet-18 (100MB) | Memory constraint |
| Real-time inference | Swin-L (800ms/img) | ResNet-34 (50ms/img) | Speed constraint |
| Model consolidation | 5 specialist models | 1 generalist model | Simplify pipeline |
| Privacy (teacher API) | Cloud API (black box) | Local model | No data leaves device |

---

## 2. Core Concepts

### 2.1 The Key Insight: Soft Labels Carry More Information

A hard label says: "This pixel is liver." That's 1 bit of information.

A soft label from the teacher says: "This pixel is 85% liver, 10% spleen, 4% kidney, 1% background." This tells the student not just *what* the pixel is, but *what it's similar to* — the inter-class relationships.

**Temperature scaling** controls how much of this "dark knowledge" is revealed:

$$q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

| Temperature $T$ | Distribution | Information |
|-----------------|-------------|-------------|
| $T = 1$ | Standard softmax | Only the top class matters |
| $T = 4$ | Softened | Non-top classes reveal relationships |
| $T \to \infty$ | Nearly uniform | All classes equally weighted (too much noise) |

**The sweet spot** is typically $T = 3$–$6$. At $T=4$, the "liver vs. spleen" ratio in the example above might be 10:4, telling the student that a confusing pixel is more like liver than spleen — valuable information absent from a hard label.

### 2.2 The Distillation Loss

The classic Hinton KD loss combines two objectives:

$$\mathcal{L}_{\text{KD}} = (1-\alpha) \cdot \mathcal{L}_{\text{CE}}(y, p^S) + \alpha \cdot T^2 \cdot \text{KL}(q^T \| q^S)$$

| Term | Purpose |
|------|---------|
| $\mathcal{L}_{\text{CE}}(y, p^S)$ | Student learns from **ground truth** labels (hard targets) |
| $T^2 \cdot \text{KL}(q^T \| q^S)$ | Student learns from **teacher's soft predictions** (soft targets) |
| $\alpha$ | Balance between the two objectives |
| $T^2$ | Compensates for the gradient magnitude reduction from temperature scaling |

**Why the $T^2$ factor?** When you divide logits by $T$, the gradient magnitude scales as $1/T^2$. Multiplying the loss by $T^2$ keeps the gradient scale consistent regardless of temperature.

### 2.3 Beyond Logit Matching

Hinton's original KD only matches output distributions. More advanced methods match intermediate representations:

```
Teacher:                    Student:
┌────────────┐              ┌────────────┐
│  Encoder   │              │  Encoder   │
│  features  │───match───▶ │  features  │   Feature distillation
│            │              │            │   (FitNets, AT)
├────────────┤              ├────────────┤
│  Decoder   │              │  Decoder   │
│  features  │───match───▶ │  features  │   Attention distillation
│            │              │            │   (Attention Transfer)
├────────────┤              ├────────────┤
│  Output    │              │  Output    │
│  logits    │───match───▶ │  logits    │   Logit distillation
│            │              │            │   (Hinton KD)
└────────────┘              └────────────┘
```

**Feature distillation (FitNets)**: Match the raw feature maps between teacher and student using an L2 loss. The student may have fewer channels, so a 1×1 convolution (regressor) projects student features to match teacher dimensions.

**Attention distillation (Attention Transfer)**: Instead of matching full feature maps, match the **spatial attention maps** — the sum of squared channels gives a "what the model is looking at" heatmap. Matching these is cheaper and more robust.

**Channel-wise distillation (CWD)**: Normalize each channel of the feature map independently and apply KL divergence per-channel. This captures spatial patterns within each feature channel.

### 2.4 Method Comparison

| Method | What's Matched | Compute Overhead | Typical Recovery |
|--------|---------------|-----------------|-----------------|
| Hinton KD (logit) | Output probabilities | Minimal | 90–95% |
| CWD | Per-channel spatial patterns | Low | 93–97% |
| FitNets | Intermediate features | Low (1×1 conv) | 92–96% |
| Attention Transfer | Spatial attention maps | Low | 93–97% |
| RKD | Pairwise sample relations | Medium | 91–95% |
| DKD (Decoupled KD) | Target + non-target logits separately | Minimal | 94–98% |

---

## 3. How It Works in APRIL-MedSeg

### 3.1 Training Script

Knowledge distillation uses `train_distillation.py` with **two configs** — one for teacher, one for student:

```bash
python train_distillation.py \
    --teacher_config configs/training_paradigms/distillation/teacher_large.yaml \
    --student_config configs/training_paradigms/distillation/vanilla_kd.yaml \
    --distillation_type logit \
    --temperature 4.0 \
    --alpha 0.5
```

### 3.2 Teacher Configuration

The teacher is typically a larger, pre-trained model:

```yaml
# teacher_large.yaml
model:
  num_classes: 4
  img_size: 224
  encoder:
    name: timm_resnet50          # Larger encoder for teacher
    pretrained: true
    in_channels: 3
  decoder:
    name: cascade                # More powerful decoder
    params: {}
  bottleneck:
    name: none

data:
  img_size: 224
  source:
    image_dir: ./data/source/images
    mask_dir: ./data/source/masks
  val:
    image_dir: ./data/target_val/images
    mask_dir: ./data/target_val/masks

training:
  epochs: 200                    # Train teacher longer
  batch_size: 8
  optimizer:
    name: adamw
    lr: 0.0001
  loss:
    name: compound
    params:
      losses:
        - name: ce
          weight: 0.4
        - name: dice
          weight: 0.6
```

### 3.3 Student Configuration (Vanilla KD)

```yaml
# vanilla_kd.yaml (student)
model:
  num_classes: 4
  img_size: 224
  encoder:
    name: timm_resnet34          # Smaller encoder for student
    pretrained: true
    in_channels: 3
  decoder:
    name: bilinear               # Simple decoder
  bottleneck:
    name: none

distillation:
  method: vanilla_kd
  weight: 1.0                    # Overall KD loss weight
  params:
    temperature: 4.0             # Softening temperature T
    alpha: 0.9                   # Weight on KD term (1-alpha on CE)
    ignore_index: -100           # Pixels to ignore in loss

training:
  epochs: 100
  batch_size: 8
  optimizer:
    name: adamw
    lr: 0.0001
  scheduler:
    name: cosine
```

### 3.4 Channel-Wise Distillation (CWD) Configuration

```yaml
# cwd.yaml (student with CWD)
distillation:
  method: cwd
  params:
    temperature: 1.0             # Per-channel normalization temperature
```

CWD is particularly effective for segmentation because it preserves spatial patterns within each feature channel — important when different channels encode different anatomical structures.

### 3.5 Available Methods

| Method | Config File | What's Matched | Best For |
|--------|------------|---------------|----------|
| Vanilla KD | `vanilla_kd.yaml` | Output logits | Simple baseline |
| CWD | `cwd.yaml` | Per-channel features | Segmentation (recommended) |
| DKD | `ctkd.yaml` | Decoupled logits | Hard-to-separate classes |
| RKD | `rkd.yaml` | Sample relations | Metric learning tasks |
| FitNets | `aicsd.yaml` | Intermediate features | Very small students |
| NST | `nst.yaml` | Neural style transfer | Style-preserving compression |
| AT | `at.yaml` | Attention maps | Balanced compression |

---

## 4. Step-by-Step: Your First Distillation Run

### Step 1: Train the teacher first

The teacher must be a well-trained model. Train it until convergence:

```bash
python train.py --config configs/training_paradigms/distillation/teacher_large.yaml \
    --output_dir output/teacher
```

Verify the teacher achieves good Dice (e.g., >90%).

### Step 2: Choose a distillation method

For your first run:
- **Vanilla KD** is simplest and works well.
- **CWD** is recommended for segmentation tasks.

### Step 3: Run distillation

```bash
# Vanilla KD
python train_distillation.py \
    --teacher_config configs/training_paradigms/distillation/teacher_large.yaml \
    --student_config configs/training_paradigms/distillation/vanilla_kd.yaml \
    --distillation_type logit \
    --temperature 4.0 \
    --alpha 0.5 \
    --output_dir output/kd_vanilla

# CWD
python train_distillation.py \
    --teacher_config configs/training_paradigms/distillation/teacher_large.yaml \
    --student_config configs/training_paradigms/distillation/student_small.yaml \
    --distillation_type cwd \
    --output_dir output/kd_cwd
```

### Step 4: Evaluate the student

```bash
python test.py \
    --config configs/training_paradigms/distillation/student_small.yaml \
    --checkpoint output/kd_vanilla/best_student.pth
```

Compare: Teacher Dice vs. Student (no KD) vs. Student (with KD).

---

## 5. Parameter Tuning Guide

### Temperature ($T$)

| Value | Effect | When to Use |
|-------|--------|-------------|
| $T = 1$ | No softening — equivalent to standard CE | Baseline, no distillation |
| $T = 2$ | Mild softening | Teacher is very confident, classes well-separated |
| $T = 4$ | Good balance (recommended default) | Most segmentation tasks |
| $T = 8$ | Heavy softening | Many similar classes (fine-grained segmentation) |

### Alpha ($\alpha$)

| Value | Effect | When to Use |
|-------|--------|-------------|
| $\alpha = 0.1$ | Mostly learning from labels | Teacher is not very good |
| $\alpha = 0.5$ | Balanced | General starting point |
| $\alpha = 0.9$ | Mostly learning from teacher | Teacher is excellent (recommended) |

### Student Size Ratio

| Ratio (Student/Teacher params) | Expected Recovery |
|-------------------------------|-------------------|
| 1/2 (half the parameters) | 96–99% of teacher |
| 1/4 | 93–97% of teacher |
| 1/10 | 88–94% of teacher |

**Rule of thumb**: A student with 1/4 the teacher's parameters typically recovers ~95% accuracy — this is the best compression-to-quality tradeoff.

---

## 6. Common Pitfalls

### Pitfall 1: Student performs worse than training from scratch

**Symptom**: KD student Dice is lower than a student trained without any teacher.

**Fix**:
- The teacher may not be well-trained. Verify teacher Dice >90% before distilling.
- $\alpha$ is too high — the student is over-relying on a mediocre teacher. Reduce $\alpha$ to 0.3.
- Temperature is wrong — try $T=2$ or $T=4$.

### Pitfall 2: Student collapses to predicting background only

**Symptom**: Student Dice drops to near zero, predicts only the majority class.

**Fix**:
- Check that teacher logits are in the correct format (before softmax, not after).
- Ensure $T^2$ compensation is applied — without it, the KD gradient is too small.
- Start with $\alpha = 0.5$ and increase gradually.

### Pitfall 3: Feature distillation dimension mismatch

**Symptom**: Runtime error about tensor shape mismatch between teacher and student features.

**Fix**:
- Teacher and student have different channel counts. Ensure a regressor (1×1 conv) is inserted to match dimensions.
- In the config, set the student's encoder to output compatible feature dimensions.

### Pitfall 4: Diminishing returns with very small students

**Symptom**: A 1/20 size student doesn't improve much with KD over training from scratch.

**Fix**:
- The student capacity may be too low to absorb teacher knowledge. Try a slightly larger student.
- Use hint-based distillation (FitNets) at multiple layers rather than just the output.
- Consider pruning the teacher instead of distillation.

---

## 7. Recommended Experiments

### Experiment 1: Compression Ratio Study

Train students of different sizes with and without KD:

```bash
# Student = 1/4 teacher (with KD)
python train_distillation.py \
    --teacher_config configs/training_paradigms/distillation/teacher_large.yaml \
    --student_config configs/training_paradigms/distillation/student_small.yaml \
    --distillation_type logit --temperature 4.0 --alpha 0.9

# Student = 1/4 teacher (without KD — baseline)
python train.py --config configs/training_paradigms/distillation/student_small.yaml \
    --output_dir output/student_no_kd
```

**Expected results:**

| Configuration | Dice | Inference Time | Memory |
|--------------|------|---------------|--------|
| Teacher (ResNet-101 + CASCADE) | 92% | 800ms | 2.0GB |
| Student (ResNet-34 + bilinear, no KD) | 82% | 50ms | 0.3GB |
| Student (ResNet-34 + bilinear, with KD) | 88% | 50ms | 0.3GB |
| Recovery | — | — | 95% of teacher Dice |

### Experiment 2: Method Comparison

```bash
# Logit KD
python train_distillation.py --teacher_config ... --student_config ... --distillation_type logit

# CWD
python train_distillation.py --teacher_config ... --student_config ... --distillation_type cwd

# Attention Transfer
python train_distillation.py --teacher_config ... --student_config ... --distillation_type attention
```

**Expected ranking**: CWD ≥ AT > Logit KD > No KD

### Experiment 3: Temperature Sweep

```bash
for T in 1 2 4 6 8; do
    python train_distillation.py \
        --teacher_config configs/training_paradigms/distillation/teacher_large.yaml \
        --student_config configs/training_paradigms/distillation/student_small.yaml \
        --distillation_type logit --temperature $T --alpha 0.9 \
        --output_dir output/kd_T${T}
done
```

Plot Dice vs. Temperature to find the optimal $T$ for your task.

---

## 8. Further Reading

### Key Papers

| Paper | Year | Venue | Key Idea |
|-------|------|-------|----------|
| [Hinton KD](https://arxiv.org/abs/1503.02531) | 2015 | NeurIPS WS | Temperature-based logit distillation |
| [FitNets](https://arxiv.org/abs/1412.6550) | 2015 | ICLR | Hint-based intermediate feature matching |
| [Attention Transfer](https://arxiv.org/abs/1612.03928) | 2017 | ICLR | Match spatial attention maps |
| [CWD](https://arxiv.org/abs/2011.13256) | 2021 | ICCV | Channel-wise distillation for segmentation |
| [DKD](https://arxiv.org/abs/2203.08679) | 2022 | CVPR | Decoupled target/non-target KD |
| [RKD](https://arxiv.org/abs/1904.05068) | 2019 | CVPR | Relational knowledge distillation |

### Related Documentation

- [All Distillation Methods](../paradigms/distillation.md) — Complete method catalog (27 methods)

---

[Back to Paradigms Overview](08_paradigms.md) | [Previous: Domain Adaptation](08b_domain_adaptation.md) | [Next: Weakly Supervised](08d_weakly_supervised.md)
