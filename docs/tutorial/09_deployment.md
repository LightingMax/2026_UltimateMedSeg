# Chapter 09: Deployment and Inference

[Previous: Advanced Paradigms](08_paradigms.md) | [中文文档](09_deployment_CN.md) | [Tutorial Index](index.md)

---

## 1. Background and Motivation

A trained segmentation model is only useful if it can make accurate predictions in practice. The gap between training and deployment involves several challenges:

- **Accuracy gap**: A single model's prediction has inherent variance. Can we do better without retraining?
- **Robustness**: Real-world images vary in orientation, brightness, and quality. How to handle this at inference time?
- **Deployment constraints**: Production environments may require specific formats (ONNX), reduced precision, or fast inference.
- **Cost analysis**: Understanding computational requirements (FLOPs, parameters, latency) is essential for planning.

This chapter explains the theoretical foundations behind each deployment technique — why averaging helps, why test-time augmentation approximates Bayesian inference, and how model export preserves accuracy.

---

## 2. Core Concepts

### 2.1 Ensemble Theory — Why Averaging Works

An ensemble combines predictions from multiple models to produce a more accurate and stable result. The theoretical justification comes from the **bias-variance decomposition**:

$$\mathbb{E}[(y - \hat{y})^2] = \text{Bias}^2[\hat{y}] + \text{Var}[\hat{y}] + \sigma^2$$

where $\sigma^2$ is the irreducible noise. Averaging $N$ independent models reduces variance without affecting bias:

$$\text{Var}\left[\frac{1}{N}\sum_{i=1}^{N} \hat{y}_i\right] = \frac{1}{N} \text{Var}[\hat{y}]$$

This means: if each model has the same bias but makes *independent* errors, averaging cancels out the errors while preserving the correct signal.

**Why this matters for segmentation**: A single model may produce noisy boundary predictions. Ensemble averaging smooths these boundaries by suppressing idiosyncratic errors.

**Averaging modes for segmentation**:

| Mode | Formula | Best For |
|------|---------|----------|
| Logit averaging | $\bar{z} = \frac{1}{N}\sum z_i$ | Multi-class (calibrated) |
| Softmax averaging | $\bar{p} = \frac{1}{N}\sum \text{softmax}(z_i)$ | Stable multi-class |
| Sigmoid averaging | $\bar{p} = \frac{1}{N}\sum \sigma(z_i)$ | Binary / multi-label |

**Logit vs. probability averaging**: Averaging logits before applying softmax is equivalent to a geometric mean of probabilities, which tends to be more confident and better calibrated. Averaging after softmax is an arithmetic mean, which is smoother but may be under-confident.

**Ensemble sources**: Models can differ by random seed, data fold, architecture, or training hyperparameters. More diversity between models → more variance reduction from averaging.

### 2.2 Test-Time Augmentation (TTA) — Approximate Bayesian Inference

TTA applies multiple augmentations to the *same test image*, runs the model on each augmented version, then merges the predictions. This is a practical approximation to **Bayesian model averaging**:

$$p(y|x) \approx \frac{1}{|A|} \sum_{a \in A} p(y | a(x))$$

where $A$ is the set of test-time augmentations (rotations, flips, brightness changes).

**Why TTA works**: A single model may produce different predictions for the same image under different orientations or lighting. By averaging over augmentations, TTA marginalizes out these nuisance factors — the prediction becomes invariant to the augmentations applied.

**Connection to Bayesian inference**: In a full Bayesian framework, we would integrate over all possible model parameters: $p(y|x) = \int p(y|x,\theta) p(\theta) d\theta$. TTA instead integrates over input transformations, which is computationally cheap and surprisingly effective.

**Augmentation selection strategy**:

| Augmentation | What It Captures | When to Use |
|-------------|-----------------|-------------|
| Horizontal flip | Left-right symmetry | Anatomically symmetric structures |
| Rotation (90°, 180°, 270°) | Orientation invariance | No fixed orientation (pathology) |
| Brightness/contrast | Intensity variation | Different scanners/protocols |
| Scale (0.75x, 1.25x) | Size variation | Variable lesion sizes |

**Merge strategies**:

| Strategy | Formula | Property |
|----------|---------|----------|
| Mean | $\bar{p} = \frac{1}{|A|}\sum p_a$ | Standard, balanced |
| Geometric mean | $\bar{p} = (\prod p_a)^{1/|A|}$ | Emphasizes agreement |
| Max | $\bar{p} = \max_a p_a$ | Most confident prediction |
| Median | $\bar{p} = \text{median}_a p_a$ | Robust to outlier augmentations |

**Geometric mean** is often superior for probabilities because it penalizes predictions where any augmentation disagrees — if one augmentation gives $p=0.1$ while others give $p=0.9$, the geometric mean is much lower than the arithmetic mean, reflecting uncertainty.

### 2.3 Model Export — ONNX and Quantization

**ONNX (Open Neural Network Exchange)** represents a trained model as a computation graph, enabling deployment in non-Python environments (C++, mobile, edge devices):

```
PyTorch model → torch.onnx.export() → ONNX graph → ONNX Runtime / TensorRT / OpenVINO
```

The export process traces the model with a sample input, recording all operations into a static graph. Key considerations:

- **Operator compatibility**: Not all PyTorch operations have ONNX equivalents. Custom operators need manual registration.
- **Dynamic shapes**: By default, ONNX models have fixed input sizes. Dynamic axes must be explicitly specified.
- **Operator fusion**: ONNX Runtime fuses adjacent operations (e.g., Conv+BN+ReLU) for faster inference.

**Quantization** reduces numerical precision to decrease model size and latency:

| Precision | Bits per weight | Size (vs FP32) | Speed | Accuracy |
|-----------|----------------|----------------|-------|----------|
| FP32 | 32 | 1.0x | 1.0x | Baseline |
| FP16 | 16 | 0.5x | ~1.5-2x | ~same |
| INT8 | 8 | 0.25x | ~2-4x | -0.5-2% Dice |

**Post-training quantization** (PTQ): Quantize weights and activations after training without retraining. Fast but may lose accuracy.

**Quantization-aware training** (QAT): Simulate quantization during training by inserting fake-quantize nodes. The model learns to be robust to reduced precision. More accurate but requires retraining.

### 2.4 Computational Cost — FLOPs, MACs, Parameters

Understanding computational requirements is essential for deployment planning.

**FLOPs** (Floating-Point Operations): The total number of arithmetic operations (additions + multiplications) for a single forward pass. For a convolution:

$$\text{FLOPs}_{\text{conv}} = 2 \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot k^2 \cdot H_{\text{out}} \cdot W_{\text{out}}$$

where the factor 2 accounts for multiply + add.

**MACs** (Multiply-Accumulate Operations): Often reported instead of FLOPs. One MAC = one multiply + one add = 2 FLOPs.

**Parameter count**: Total number of trainable weights. Note that frozen foundation encoder parameters are *loaded* but not *trained* — they count toward model size but not toward gradient computation.

**The accuracy-cost trade-off**:

```
Accuracy ↑
  │           ● Foundation (frozen)
  │        ● Transformer
  │     ● CNN (large)
  │   ● CNN (medium)
  │ ● CNN (small)
  └──────────────────────── FLOPs →
```

Higher accuracy generally requires more computation, but the relationship is sub-linear — doubling FLOPs rarely doubles accuracy. The sweet spot depends on the deployment constraints.

### 2.5 MLLM Inference Pipeline

Multi-modal Large Language Models enable a **detect-then-segment** paradigm for zero-shot segmentation from text prompts:

```
Text: "liver tumor"
       │
       ▼
  MLLM Detector (GroundingDINO / Qwen-VL / InternVL)
       │  → bounding box or point prompt
       ▼
  Segmenter (SAM2 / MedSAM / SAM-Med2D / LiteMedSAM)
       │  → pixel-level mask
       ▼
  Final segmentation
```

This pipeline requires no training on the target task — the MLLM's visual grounding capability locates the object, and the segmenter produces the mask.

---

## 3. Method Details

### 3.1 Inference Pipeline Overview

```
Trained checkpoint(s)
    │
    ├─ Single model → test.py → Dice, IoU, HD95
    │
    ├─ Ensemble (N checkpoints) → weighted logit average → improved Dice
    │
    ├─ TTA (augmentations) → merge predictions → improved robustness
    │
    ├─ ONNX export → model.onnx → deployment in C++/mobile
    │
    └─ Profiling → FLOPs, Params, FPS → deployment feasibility check
```

### 3.2 Ensemble Configuration

| Parameter | Options | Recommendation |
|-----------|---------|----------------|
| Number of checkpoints | 2-10 | 3-5 (diminishing returns beyond 5) |
| Weights | Equal or learned | Equal (simple), validation-optimized (best) |
| Averaging mode | logit, softmax, sigmoid | `logit` for multi-class |

### 3.3 TTA Configuration

| Parameter | Options | Recommendation |
|-----------|---------|----------------|
| Augmentations | Identity, flips, rotations, brightness | Start with identity + hflip + rot90 |
| Merge strategy | mean, gmean, max, median | `mean` (default), `gmean` (probabilities) |
| Number of augmentations | 2-12 | 4-6 (balance speed vs. gain) |

---

## 4. Hands-On with APRIL-MedSeg

```bash
# Single model evaluation
python test.py --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/best_model.pth

# Ensemble (3 checkpoints)
python test.py --config cfg.yaml \
    --checkpoint ckpt_fold0.pth ckpt_fold1.pth ckpt_fold2.pth \
    --ensemble-weights 0.4 0.3 0.3 \
    --ensemble-average logit

# TTA
python test.py --config cfg.yaml --checkpoint best.pth \
    --tta --tta-augs identity rot90 hflip vflip --tta-merge mean

# ONNX export with verification
python scripts/export_onnx.py --config cfg.yaml --checkpoint best.pth \
    --output model.onnx --verify

# Model profiling
python profile_model.py --config cfg.yaml
```

```python
# MLLM pipeline (zero-shot segmentation)
from medseg.inference.mllm import MLLMPipeline

pipeline = MLLMPipeline(
    detector='grounding_dino',    # or qwen2_vl, qwen3_vl, internvl
    segmenter='sam2',             # or medsam, sam_med2d, lite_medsam
    text_prompt='liver tumor',
)
result = pipeline.predict('data/test/image_001.png')
```

---

## 5. Recommended Experiments

### Experiment 1: Ensemble Improvement

Compare single model vs. ensemble on the same test set:

| Setup | Expected Dice Gain | Inference Cost |
|-------|-------------------|----------------|
| Single checkpoint | Baseline | 1x |
| 3-checkpoint ensemble | +1-3% | 3x |
| 5-checkpoint ensemble | +1.5-4% | 5x |

### Experiment 2: TTA Impact

Same model, with and without TTA:

| TTA Augmentations | Expected Dice Gain | Inference Cost |
|-------------------|-------------------|----------------|
| None | Baseline | 1x |
| identity + hflip | +0.5-1% | 2x |
| 4 rotations + 2 flips | +1-2% | 7x |

### Experiment 3: ONNX Verification

Export to ONNX and verify numerical accuracy:

| Check | Expected Result |
|-------|----------------|
| Output shape match | Identical |
| Max absolute error | < 1e-4 (FP32) |
| Dice difference | < 0.1% |
| Inference speed | 1.2-2x faster (ONNX Runtime) |

### Experiment 4: Model Cost Analysis

Compare architectures on computational cost:

| Architecture | Params | FLOPs | Expected FPS (224×224) |
|-------------|--------|-------|----------------------|
| ResNet50 + UNet | ~27M | ~55G | ~50 |
| Swin-T + UNet | ~30M | ~60G | ~35 |
| DINOv2-B (frozen) + UNet | ~88M | ~100G | ~20 |

---

## 6. Further Reading

### Key Papers

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|-----------------|
| [Ensemble Methods](https://link.springer.com/article/10.1023/A:1010950718922) | 2002 | Data Mining & KD | Comprehensive ensemble theory |
| [Deep Ensembles](https://arxiv.org/abs/1612.01474) | 2017 | NeurIPS | Random initialization ensembles for uncertainty |
| [TTA for Segmentation](https://arxiv.org/abs/1910.11190) | 2019 | - | Test-time augmentation for medical images |
| [ONNX Spec](https://onnx.ai/onnx/) | 2024 | - | Open Neural Network Exchange specification |
| [Quantization Survey](https://arxiv.org/abs/2103.13630) | 2021 | - | Comprehensive survey of neural network quantization |
| [Grounding DINO](https://arxiv.org/abs/2303.05499) | 2023 | ECCV | Open-set object detection from text |
| [SAM 2](https://arxiv.org/abs/2408.00714) | 2024 | - | Segment Anything Model 2 |

### Related Documentation

- [Inference Guide](../usage/inference.md) -- Full test.py options and ensemble configuration
- [ONNX Export](../usage/export.md) -- Export and deployment guide
- [MLLM Pipeline](../paradigms/text_guided.md#mllm-pipeline) -- 5 detectors × 4 segmenters

---

[Previous: Advanced Paradigms](08_paradigms.md) | [Tutorial Index](index.md)
