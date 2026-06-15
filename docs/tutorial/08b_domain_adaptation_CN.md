# 第 08b 章：域适应

[返回训练范式总览](08_paradigms_CN.md) | [English](08b_domain_adaptation.md) | [上一章：半监督](08a_semi_supervised_CN.md) | [下一章：知识蒸馏](08c_distillation_CN.md)

---

## 1. 什么时候该用域适应？

你在医院 A 的西门子 CT 扫描仪数据上训练了一个肝脏分割模型，效果很好——Dice 92%。然后你部署到医院 B 的 GE 扫描仪上，Dice 降到了 71%。同样的解剖结构，同样的任务，但图像看起来不同：对比度不同、噪声纹理不同、强度范围不同。这就是**域偏移**，是部署医学 AI 最大的障碍之一。

**域适应**让你把在医院 A 训练好的模型适配到医院 B 的图像上——**无需**从头标注医院 B 的数据。

### 典型场景

| 场景 | 源域 | 目标域 | 适应方法 |
|------|------|--------|---------|
| 多扫描仪部署 | 扫描仪 A（有标注） | 扫描仪 B（无标注） | 传统 UDA |
| 多中心研究 | 中心 1（有标注） | 中心 2–5（无标注） | 多源 DA |
| 已部署，性能下降 | 原始数据 | 新患者群体 | 测试时适应 |
| 源数据不可用（隐私） | 仅预训练模型 | 目标图像 | 无源 DA |

---

## 2. 核心概念

### 2.1 问题本质：分布偏移

在标准监督学习中，我们假设训练和测试数据来自同一分布：

$$P_{\text{train}}(X, Y) = P_{\text{test}}(X, Y)$$

域适应放宽了这一假设：

$$P_S(X, Y) \neq P_T(X, Y)$$

偏移可以表现为多种形式：
- **协变量偏移**：$P(X)$ 变化（图像外观不同），但 $P(Y|X)$ 不变。
- **标签偏移**：$P(Y)$ 变化（不同疾病流行率），但 $P(X|Y)$ 不变。
- **概念偏移**：$P(Y|X)$ 变化（相同图像特征对应不同标签）。

在医学影像中，协变量偏移最常见——解剖结构相同，但图像外观不同。

### 2.2 对抗对齐（DANN, AdvEnt）

核心思想：训练一个**域判别器**来区分源域和目标域特征，同时训练**特征提取器**来欺骗判别器。

```
源域图像 ──┐
            ├──▶ 特征提取器 G ──▶ 特征 ──┐
目标域图像 ──┘                            │
                                          ▼
                               域判别器 D
                               （源域还是目标域？）
                                          │
         G 试图欺骗 D         D 试图区分它们
         ←── 对抗博弈 ──→
```

**对抗目标函数：**

$$\min_G \max_D \; \mathbb{E}_{x \sim P_S}[\log D(G(x))] + \mathbb{E}_{x \sim P_T}[\log(1 - D(G(x)))]$$

在鞍点处，$G$ 产生**域不变特征**——判别器无法区分源域和目标域，这意味着特征捕捉的是解剖结构而非扫描仪特定的伪影。

**梯度反转层（GRL）**：在实践中，对抗博弈通过反转从 $D$ 流向 $G$ 的梯度来实现：

$$\text{GRL}(x) = x \quad \text{(前向)}, \qquad \frac{\partial \text{GRL}}{\partial x} = -\lambda I \quad \text{(反向)}$$

这使得 $G$ 学习*增大* $D$ 损失的梯度，而 $D$ 学习*减小*它。

### 2.3 熵最小化（AdvEnt）

一个互补的思想：在目标域中，模型的预测应当是**自信的**（低熵）。如果模型对目标图像不确定，那很可能是被域偏移搞混了。

$$\mathcal{L}_{\text{entropy}} = -\sum_c p_c \log p_c$$

最小化这个损失推动模型在目标数据上做出确定性预测。AdvEnt 结合了对抗对齐和熵最小化，并在熵图本身上使用对抗训练——鼓励空间熵模式看起来像源域的。

### 2.4 测试时适应（TENT）

TENT（Wang et al., ICLR 2021）是一种激进的简化：**在推理过程中适配模型**，仅使用当前输入的测试图像——无需源数据、无需目标标签。

```
预训练模型 ──▶ 接收测试图像 ──▶ 更新 BatchNorm 参数 ──▶ 预测
                    ↑                    │
                    └── 最小化熵 ────────┘
```

**工作原理：**
1. 加载在源域上训练好的模型。
2. 对每个测试批次，计算预测熵。
3. 将熵损失反向传播以**仅更新 BatchNorm 仿射参数**（γ 和 β）。
4. 用适配后的参数做预测。

**为什么只更新 BatchNorm？** BatchNorm 统计量（均值、方差）对分布偏移高度敏感。更新它们成本低（参数少）且有效——它将特征分布重新校准以匹配目标域。

**关键优势**：TENT 完全不需要标注数据——甚至不需要源数据。它只需要一个预训练 checkpoint 和测试图像即可工作。

### 2.5 其他方法

| 方法 | 方法类型 | 所需数据 |
|------|---------|---------|
| DANN | 带 GRL 的对抗对齐 | 源域（有标注）+ 目标域（无标注） |
| AdvEnt | 对抗熵最小化 | 源域（有标注）+ 目标域（无标注） |
| TENT | 测试时 BatchNorm 适应 | 预训练模型 + 目标域（仅测试图像） |
| FDA | 傅里叶域风格迁移 | 源域（有标注）+ 目标域（无标注） |
| DPL | 去噪伪标签 | 源域（有标注）+ 目标域（无标注） |
| CBMT | 类别平衡 Mean Teacher | 源域（有标注）+ 目标域（无标注） |

---

## 3. 在 APRIL-MedSeg 中使用

### 3.1 训练脚本

所有域适应方法都使用 `train_domain_adaptation.py`：

```bash
# 传统 UDA（如 AdvEnt）
python train_domain_adaptation.py --config configs/training_paradigms/domain_adaptation/advent.yaml

# 测试时适应（如 TENT）
python train_domain_adaptation.py --config configs/training_paradigms/domain_adaptation/tent.yaml
```

### 3.2 YAML 配置逐行解读 — AdvEnt

```yaml
model:
  num_classes: 4          # 分割类别数
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
  # 源域（有标注——这是你有标注数据的地方）
  source:
    image_dir: ./data/source/images
    mask_dir: ./data/source/masks
  # 目标域（无标注——这是你想适配到的新域）
  target:
    image_dir: ./data/target/images
  # 验证集（目标域有标签——仅用于评估，不用于训练）
  val:
    image_dir: ./data/target_val/images
    mask_dir: ./data/target_val/masks
  # 测试集
  test:
    image_dir: ./data/target_test/images
    mask_dir: ./data/target_test/masks

domain_adaptation:
  method: advent                    # 使用哪种 DA 算法
  params:
    entropy_weight: 0.1             # 熵最小化损失权重
    adversarial_weight: 0.1         # 对抗对齐损失权重
    num_classes: 4                  # 须与 model.num_classes 一致

training:
  epochs: 100
  batch_size: 8                     # 总 batch（源域 + 目标域混合）
  loss:
    name: compound
    params:
      losses:
        - name: ce
          weight: 0.4              # 分割损失（仅源域）
        - name: dice
          weight: 0.6
  optimizer:
    name: adamw
    lr: 0.0001
  scheduler:
    name: cosine
```

### 3.3 YAML 配置逐行解读 — TENT

TENT 更简单，因为它只需要目标域和预训练模型：

```yaml
data:
  img_size: 224
  target:
    image_dir: ./data/target/images     # 目标图像（无标注）
  val:
    image_dir: ./data/target_val/images
    mask_dir: ./data/target_val/masks
  test:
    image_dir: ./data/target_test/images
    mask_dir: ./data/target_test/masks
  pretrained_model: ./checkpoints/source_model.pth   # 源域训练好的模型

domain_adaptation:
  method: tent
  params:
    entropy_weight: 1.0              # 熵最小化强度

training:
  epochs: 50                         # TENT 需要更少 epoch（适应而非完整训练）
  optimizer:
    lr: 0.00001                      # 非常小的学习率（仅更新 BatchNorm）
```

### 3.4 可用方法一览

| 方法 | 配置文件 | 类型 | 需要源数据？ |
|------|---------|------|------------|
| AdvEnt | `advent.yaml` | 传统 UDA | 是 |
| DANN | `dann.yaml` | 传统 UDA | 是 |
| TENT | `tent.yaml` | 测试时适应 | 否（仅 checkpoint） |
| FDA | `fda.yaml` | 风格迁移 | 是 |
| DPL | `dpl.yaml` | 伪标签 | 是 |
| CBMT | `cbmt.yaml` | Mean Teacher + DA | 是 |
| Source Only | `source_only.yaml` | 基线（无适应） | 是 |

### 3.5 数据准备

域适应需要**源域和目标域数据分开组织**：

```
data/
├── source/                  # 源域（有标注）
│   ├── images/
│   │   ├── src_001.npy
│   │   └── ...
│   └── masks/
│       ├── src_001_mask.npy
│       └── ...
├── target/                  # 目标域（无标注——无 mask）
│   └── images/
│       ├── tgt_001.npy
│       └── ...
├── target_val/              # 目标验证集（有标注，用于评估）
│   ├── images/
│   └── masks/
└── target_test/             # 目标测试集（有标注，用于最终评估）
    ├── images/
    └── masks/
```

**重要**：target_val 和 target_test 目录需要 ground truth mask 用于评估，但这些**不用于训练**——仅用于衡量适应效果。

---

## 4. 手把手：你的第一次域适应训练

### 第 1 步：先训练源域模型

在适应之前，需要一个在源域上训练好的模型：

```bash
python train.py --config configs/architectures/combinations/general/unet_resnet34.yaml \
    --override data.train_dir=./data/source/images data.train_list=... \
    --output_dir output/source_model
```

### 第 2 步：选择 DA 方法

首次运行：
- 如果适应时**源数据可用**：使用 **AdvEnt**——研究充分且有效。
- 如果只有**预训练 checkpoint**：使用 **TENT**——最简单且无需源数据。

### 第 3 步：适应

```bash
# AdvEnt（传统 UDA）
python train_domain_adaptation.py \
    --config configs/training_paradigms/domain_adaptation/advent.yaml \
    --output_dir output/da_advent

# TENT（测试时适应）
python train_domain_adaptation.py \
    --config configs/training_paradigms/domain_adaptation/tent.yaml \
    --output_dir output/da_tent
```

### 第 4 步：在目标域上评估

```bash
python test.py \
    --config configs/training_paradigms/domain_adaptation/advent.yaml \
    --checkpoint output/da_advent/best_model.pth
```

与仅源域基线比较以衡量适应增益。

---

## 5. 参数调优指南

### AdvEnt 参数

| 参数 | 效果 | 调优建议 |
|------|------|---------|
| `entropy_weight` | 目标域预测熵最小化强度 | 从 0.1 开始。如果目标预测太不确定，增到 0.5。 |
| `adversarial_weight` | 源/目标特征对齐强度 | 从 0.1 开始。如果特征没对齐，增到 0.5。 |
| `grl_lambda` | GRL 梯度反转强度 | 通常自动调度；很少需要手动调。 |

### TENT 参数

| 参数 | 效果 | 调优建议 |
|------|------|---------|
| `entropy_weight` | 熵最小化强度 | 默认 1.0。如果适应太激进，降到 0.1。 |
| `lr`（优化器） | BatchNorm 更新学习率 | 必须非常小（1e-5）。更大的 LR 会导致模型坍塌。 |
| `epochs` | 适应 epoch 数 | 10–50 通常够了。更多 epoch 有过拟合测试数据的风险。 |

### 通用建议

- **始终先跑仅源域基线**以量化域差距。
- **从小 DA 权重开始**（0.01–0.1）再增大——太大的权重会导致模型遗忘源域知识。
- **监控训练期间的目标验证 Dice**，而不是源域损失。

---

## 6. 常见坑

### 坑 1：域适应反而更差

**症状**：适应后的模型在目标数据上比仅源域更差。

**修复**：
- 减小 `entropy_weight` 和 `adversarial_weight`——太强的适应会破坏源域知识。
- 检查源域和目标域任务是否真的一致（相同器官、相同视角）。
- 尝试 TENT——它更温和，因为只更新 BatchNorm。

### 坑 2：对抗训练模式坍塌

**症状**：判别器准确率一直在 50%（随机）附近，但分割质量很差。

**修复**：
- 对抗损失可能压倒了分割损失。减小 `adversarial_weight`。
- 在判别器上使用梯度惩罚或谱归一化。
- 在启用对抗损失前，先预热分割损失 10–20 个 epoch。

### 坑 3：TENT 在小测试集上过拟合

**症状**：TENT 适应后的模型在一个测试批次上表现好，但在下一个批次上表现差。

**修复**：
- 使用更小的学习率（1e-6 而非 1e-5）。
- 减少适应 epoch 数（5–10 而非 50）。
- 在测试批次间重置模型（在线 vs. 持续适应）。

### 坑 4：源域和目标域类别分布不同

**症状**：源域数据主要是类别 A，目标域主要是类别 B。适应对齐了整体特征但各类别性能差异巨大。

**修复**：
- 使用类别平衡方法如 CBMT。
- 添加类别级对抗对齐（每类一个判别器）。
- 确保源域数据覆盖了目标域中出现的所有类别。

---

## 7. 推荐实验

### 实验 1：量化域差距

在尝试适应之前，先衡量域偏移有多严重：

```bash
# 在源域训练
python train.py --config configs/architectures/combinations/general/unet_resnet34.yaml \
    --output_dir output/source_only

# 在目标域测试（无适应）
python test.py --config configs/training_paradigms/domain_adaptation/source_only.yaml \
    --checkpoint output/source_only/best_model.pth
```

如果源域 Dice 和目标域 Dice 的差距 <3%，域适应可能不值得花精力。

### 实验 2：方法对比

```bash
# AdvEnt
python train_domain_adaptation.py --config configs/training_paradigms/domain_adaptation/advent.yaml

# DANN
python train_domain_adaptation.py --config configs/training_paradigms/domain_adaptation/dann.yaml

# TENT（从预训练 checkpoint 开始）
python train_domain_adaptation.py --config configs/training_paradigms/domain_adaptation/tent.yaml
```

**预期结果：**

| 方法 | 源域 Dice | 目标域 Dice | 适应增益 |
|------|----------|-------------|---------|
| 仅源域 | ~90% | ~65–75% | — |
| AdvEnt | ~90% | ~78–85% | +8–15% |
| DANN | ~90% | ~75–82% | +5–12% |
| TENT | ~90% | ~75–82% | +5–10% |

### 实验 3：目标数据量的影响

变化目标域无标注数据量并衡量适应质量：

| 目标数据 | AdvEnt Dice | TENT Dice |
|---------|-------------|-----------|
| 100 张 | ~75% | ~73% |
| 500 张 | ~80% | ~77% |
| 1000 张 | ~83% | ~80% |
| 5000 张 | ~85% | ~82% |

---

## 8. 延伸阅读

### 关键论文

| 论文 | 年份 | 会议 | 核心思想 |
|------|------|------|---------|
| [DANN](https://arxiv.org/abs/1505.07818) | 2016 | JMLR | 带 GRL 的域对抗训练 |
| [AdvEnt](https://arxiv.org/abs/1811.12833) | 2019 | CVPR | 对抗熵最小化 |
| [TENT](https://arxiv.org/abs/2006.10726) | 2021 | ICLR | 测试时 BatchNorm 适应 |
| [FDA](https://arxiv.org/abs/2004.05498) | 2020 | NeurIPS | 傅里叶域风格增强 |
| [DAFormer](https://arxiv.org/abs/2111.14887) | 2022 | CVPR | 基于 Transformer 的 DA |

### 相关文档

- [所有域适应方法](../paradigms/domain_adaptation.md) — 完整方法目录
- [ADA4MIA 基准](https://github.com/whq-xxh/ADA4MIA) — 许多 DA 方法的参考实现

---

[返回训练范式总览](08_paradigms_CN.md) | [上一章：半监督](08a_semi_supervised_CN.md) | [下一章：知识蒸馏](08c_distillation_CN.md)
