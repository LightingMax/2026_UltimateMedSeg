# 第 08a 章：半监督学习

[返回训练范式总览](08_paradigms_CN.md) | [English](08a_semi_supervised.md) | [下一章：域适应](08b_domain_adaptation_CN.md)

---

## 1. 什么时候该用半监督学习？

假设你手头有一个医院的 10,000 张 CT 扫描数据集，但只有 200 张由放射科医师仔细标注过。标注剩余的数据需要数月时间，花费数万元。**半监督学习**让你利用那 9,800 张未标注的扫描来提升模型——往往能恢复到全标注时 85–95% 的性能，而成本只是其零头。

**一句话总结**：图片多、标注少的时候，从这里开始。

### 典型场景

| 场景 | 已标注 | 未标注 | 半监督收益 |
|------|--------|--------|-----------|
| 新医院、罕见病 | 50 张 | 5,000 张 | 巨大——标注极度稀缺 |
| 已有数据集，想再提升 | 500 张 | 2,000 张 | 中等——边际收益 |
| 主动学习管线 | 100 张 | 10,000 张 | 大——可选择性标注最难样本 |

---

## 2. 核心概念

### 2.1 基本思想

半监督学习利用一个观察：**即使没有标注，图像本身也包含有用信息**。模型可以从无标注图像中学习数据的*结构*——"肝脏"一般长什么样——然后由少量标注数据教它*具体边界*。

类比学语言：你可以通过听对话（无标注）来掌握词汇和语法规律，即使没人逐字教你（有标注）。

### 2.2 Mean Teacher — 一致性正则化

Mean Teacher 框架（Tarvainen & Valpola, 2017）是最广泛使用的半监督方法，建立在一个简单而强大的直觉上：

> **好的模型在输入加噪声后应该给出相同的预测。**

对 CT 扫描做轻微旋转、翻转或加噪声，肝脏还在同一个位置。模型的预测不应该改变。

**工作原理：**

```
                  ┌──────────────────────┐
                  │     学生模型          │
                  │   (接受梯度训练)      │
                  └──────────┬───────────┘
                             │
              EMA 更新:      │  θ_教师 = α·θ_教师 + (1-α)·θ_学生
              (无梯度)       │
                             ▼
                  ┌──────────────────────┐
                  │     教师模型          │
                  │   (学生的 EMA,       │
                  │    更稳定)           │
                  └──────────────────────┘
```

1. **学生**在有标注数据上正常训练分割损失（CE + Dice）。
2. **教师**是学生的指数移动平均（EMA）——不直接接收梯度，而是缓慢跟踪学生的权重。
3. 在**无标注数据**上，两个模型都做预测。一致性损失鼓励学生匹配教师：

$$\mathcal{L}_{\text{consistency}} = \text{MSE}(f_{\text{student}}(x + \epsilon),\; f_{\text{teacher}}(x))$$

**为什么用 EMA？** 单个学生在训练过程中可能有噪声。EMA 平均掉了这种噪声，使教师的预测更平滑、更可靠。就像咨询一个专家组（取平均）而不是一个过度自信的初级医生。

**关键参数：**

| 参数 | 含义 | 典型范围 |
|------|------|---------|
| `ema_decay` (α) | 教师更新多慢 | 0.99–0.999（越高越稳定） |
| `consistency_weight` | 一致性损失的重要性 | 0.1–1.0 |
| `rampup_epochs` | consistency_weight 从 0 渐增到目标值的 epoch 数 | 20–80 |

### 2.3 交叉伪监督（CPS）

CPS（Chen et al., CVPR 2021）采取了不同的方法：不用一个模型加 EMA 教师，而是用**两个独立初始化的网络**互相教学。

```
        模型 A                          模型 B
   ┌──────────────┐                ┌──────────────┐
   │  对无标注数据 │──伪标签───────▶│  从 A 的     │
   │  做预测       │               │  预测中学习   │
   │              │◀──伪标签───────│  对无标注数据 │
   └──────────────┘                │  做预测       │
                                   └──────────────┘
```

关键洞察：两个不同随机初始化的网络会犯**不同的错误**。通过交叉监督，每个网络弥补对方的盲区。

**伪标签生成：**

$$\hat{y}_A = \arg\max f_A(x), \qquad \hat{y}_B = \arg\max f_B(x)$$

$$\mathcal{L}_{\text{CPS}} = \mathcal{L}_{\text{CE}}(\hat{y}_A,\; f_B(x)) + \mathcal{L}_{\text{CE}}(\hat{y}_B,\; f_A(x))$$

**与 Mean Teacher 的权衡：**
- CPS 需要约 2× 的显存（两个完整模型同时在内存中）。
- CPS 可以更准确，因为两个模型确实会产生分歧，提供更丰富的学习信号。
- Mean Teacher 更简单、更省显存。

### 2.4 UniMatch — 弱增强到强增强的一致性

UniMatch（Yang et al., CVPR 2023）是一种现代的单模型方法，结合了两者的优点：

1. 对图像施加**弱增强**（翻转 + 裁剪）生成稳定的伪标签。
2. 对同一图像施加**强增强**（颜色抖动、CutMix），训练模型去匹配那些伪标签。
3. 添加**特征级噪声**（dropout + 乘性噪声）作为额外正则化。

这比 Mean Teacher 更具样本效率，因为强增强迫使模型学习更鲁棒的特征。

---

## 3. 在 APRIL-MedSeg 中使用

### 3.1 训练脚本

所有半监督方法都使用 `semi_train.py`：

```bash
python semi_train.py --config configs/training_paradigms/semi_supervision/mean_teacher.yaml
```

### 3.2 YAML 配置逐行解读

以 Mean Teacher 配置为例：

```yaml
model:
  num_classes: 4          # 分割类别数（如 background + 3 个器官）
  img_size: 224           # 输入图像分辨率
  encoder:
    name: timm_resnet34   # 骨干编码器（通过 timm 调用 ResNet-34）
    pretrained: true      # 使用 ImageNet 预训练权重
    in_channels: 3        # RGB 输入
  decoder:
    name: bilinear        # 简单的双线性上采解码器
  bottleneck:
    name: none            # 无 bottleneck 模块

data:
  img_size: 224
  labeled_dir: ./data/labeled        # 有标注图像 + mask 的目录
  unlabeled_dir: ./data/unlabeled    # 仅有图像的无标注目录
  val_dir: ./data/val                # 验证集（必须有 mask，用于评估）
  test_dir: ./data/test              # 测试集
  test_list: ./data/test/list.txt
  labeled_ratio: 0.1                 # 使用 10% 数据作为有标注
  split_mode: dir                    # 'dir' = 分目录; 'ratio' = 自动拆分

semi:
  method: mean_teacher               # 使用哪种半监督算法
  params:
    ema_decay: 0.999                 # 教师 EMA 衰减率（越高 → 更新越慢）
    consistency_weight: 1.0          # 一致性损失权重
    rampup_epochs: 40                # 在 40 个 epoch 内渐增 consistency_weight

training:
  epochs: 200
  batch_size: 16                     # 总 batch 大小
  labeled_batch_size: 8              # 每 batch 中有标注图像数
  unlabeled_batch_size: 8            # 每 batch 中无标注图像数
  loss:
    name: compound
    params:
      losses:
        - name: ce
          weight: 0.4               # 有标注数据的交叉熵损失
        - name: dice
          weight: 0.6               # 有标注数据的 Dice 损失
  optimizer:
    name: adamw
    lr: 0.0001
  scheduler:
    name: cosine
    min_lr: 0.000001
```

### 3.3 可用方法一览

| 方法 | 配置文件 | 关键创新 | 显存需求 |
|------|---------|----------|---------|
| Mean Teacher | `mean_teacher.yaml` | EMA 教师 + MSE 一致性 | 正常（1 个模型） |
| CPS | `cps.yaml` | 双网络交叉伪标签 | ~2×（2 个模型） |
| UniMatch | `unimatch.yaml` | 弱→强增强 + 特征噪声 | 正常（1 个模型） |
| CCT | `cct.yaml` | 交叉一致性训练 | ~2× |
| FixMatch | `fixmatch.yaml` | 弱/强增强管线 | 正常 |
| UA-MT | `ua_mt.yaml` | 不确定性感知的 Mean Teacher | ~1.5× |
| Pseudo-Label | `pseudo_label.yaml` | 带置信度阈值的自训练 | 正常 |

### 3.4 数据准备

半监督训练需要**两个独立的数据目录**：

```
data/
├── labeled/              # 有像素级 mask 的图像
│   ├── img001.npy
│   ├── img001_mask.npy
│   └── ...
├── unlabeled/            # 无 mask 的图像
│   ├── img101.npy
│   ├── img102.npy
│   └── ...
├── val/                  # 验证集（始终有 mask）
│   ├── val001.npy
│   └── val001_mask.npy
└── test/                 # 测试集
    ├── list.txt
    └── ...
```

如果你只有一个大数据集想按比例自动拆分，使用 `split_mode: ratio` 配合 `labeled_ratio: 0.1` 来取 10% 作为有标注。

---

## 4. 手把手：你的第一次半监督训练

### 第 1 步：准备数据

确保有标注和无标注数据在各自目录中（或使用比例拆分）。

### 第 2 步：选择方法

首次运行推荐 **Mean Teacher**——简单、省显存、研究充分。

### 第 3 步：训练

```bash
python semi_train.py \
    --config configs/training_paradigms/semi_supervision/mean_teacher.yaml \
    --output_dir output/semi_mean_teacher
```

### 第 4 步：监控训练

关注日志中的这些关键指标：
- **监督损失**（有标注数据）——应正常下降。
- **一致性损失**（无标注数据）——随学生学习匹配教师而下降。
- **验证 Dice**——你最关心的主指标。

### 第 5 步：评估

```bash
python test.py \
    --config configs/training_paradigms/semi_supervision/mean_teacher.yaml \
    --checkpoint output/semi_mean_teacher/best_model.pth
```

---

## 5. 参数调优指南

### 如何选择 `ema_decay`

| 值 | 行为 | 适用场景 |
|----|------|---------|
| 0.99 | 教师更新较快 | 小数据集、快速收敛 |
| 0.999 | 教师非常稳定 | 大数据集、长训练（推荐默认值） |
| 0.9999 | 教师几乎不变 | 噪声很大的数据，需要最大稳定性 |

### 如何选择 `consistency_weight`

从 **1.0** 开始，根据损失比例调整：
- 如果一致性损失占主导：降到 0.1。
- 如果监督损失占主导且无标注数据没帮上忙：增到 2.0–5.0。

### 如何选择 `labeled_ratio`

| 比例 | 预期 Dice（相对全监督） |
|------|----------------------|
| 5% | ~70–85% |
| 10% | ~85–95% |
| 20% | ~92–98% |
| 50% | ~97–100% |

展示半监督优势的甜区是 **10–20%**。

### 如何选择 `rampup_epochs`

rampup 防止一致性损失在训练初期（模型和教师都还很差时）占主导。经验法则：**总 epoch 数的 20–30%**。200 个 epoch 时使用 `rampup_epochs: 40–60`。

---

## 6. 常见坑

### 坑 1：一致性损失在训练初期爆炸

**症状**：前几个 epoch 的一致性损失比监督损失大 100×。

**修复**：增大 `rampup_epochs`，让一致性权重增长更慢。教师需要时间稳定后才能提供有用的目标。

### 坑 2：教师和学生一起坍塌

**症状**：两个模型做出相同的错误预测，互相强化错误。

**修复**：
- 调高 `ema_decay`（如 0.99 → 0.999）以减缓教师更新。
- 增加无标注样本的数据增强。
- 添加置信度阈值来忽略低置信度预测。

### 坑 3：无标注数据反而拉低性能

**症状**：加入无标注数据后验证 Dice *变差*。

**修复**：
- 检查无标注数据是否与有标注数据来自同一分布。
- 减小 `consistency_weight` 以限制无标注数据的影响。
- 尝试 UniMatch 代替 Mean Teacher——其强增强更鲁棒。

### 坑 4：CPS 显存不足

**症状**：使用 `cps.yaml` 时 CUDA OOM。

**修复**：CPS 需要两个模型同时在显存中。要么：
- 减小 `batch_size`（如 16 → 8，labeled/unlabeled 各 4）。
- 使用更小的编码器（如 `timm_resnet18` 代替 `timm_resnet50`）。
- 切换到 Mean Teacher 或 UniMatch（单模型）。

---

## 7. 推荐实验

### 实验 1：标注效率曲线

用不同标注比例训练并比较：

```bash
# 每次运行修改 mean_teacher.yaml 中的 labeled_ratio
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

**预期结果：**

| 标注比例 | 监督 Dice | Mean Teacher Dice | 提升 |
|---------|----------|-------------------|------|
| 5% | ~55–65% | ~70–85% | +15–25% |
| 10% | ~65–75% | ~85–92% | +15–20% |
| 20% | ~78–85% | ~90–95% | +8–12% |
| 100% | ~88–92% | — | 基线 |

### 实验 2：方法对比（相同数据、相同模型）

```bash
# Mean Teacher
python semi_train.py --config configs/training_paradigms/semi_supervision/mean_teacher.yaml

# CPS（需要更多显存）
python semi_train.py --config configs/training_paradigms/semi_supervision/cps.yaml

# UniMatch
python semi_train.py --config configs/training_paradigms/semi_supervision/unimatch.yaml
```

**预期排名**（10% 标注时）：UniMatch ≥ CPS > Mean Teacher > 监督

---

## 8. 延伸阅读

### 关键论文

| 论文 | 年份 | 会议 | 核心思想 |
|------|------|------|---------|
| [Mean Teacher](https://arxiv.org/abs/1703.01780) | 2017 | NeurIPS | EMA 教师用于一致性正则化 |
| [CPS](https://arxiv.org/abs/2106.01226) | 2021 | CVPR | 双网络交叉伪标签监督 |
| [UniMatch](https://arxiv.org/abs/2208.09910) | 2023 | CVPR | 弱→强一致性 + 特征增强 |
| [FixMatch](https://arxiv.org/abs/2001.07685) | 2020 | NeurIPS | 简洁的弱/强增强管线 |
| [UA-MT](https://arxiv.org/abs/1907.07034) | 2019 | MICCAI | 不确定性感知的 Mean Teacher |

### 相关文档

- [所有半监督方法](../paradigms/semi_supervised.md) — 完整方法目录
- [SSL4MIS 参考实现](https://github.com/HiLab-git/SSL4MIS) — 许多半监督方法的参考实现

---

[返回训练范式总览](08_paradigms_CN.md) | [下一章：域适应](08b_domain_adaptation_CN.md)
