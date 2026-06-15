# 第 08c 章：知识蒸馏

[返回训练范式总览](08_paradigms_CN.md) | [English](08c_distillation.md) | [上一章：域适应](08b_domain_adaptation_CN.md) | [下一章：弱监督](08d_weakly_supervised_CN.md)

---

## 1. 什么时候该用知识蒸馏？

你训练了一个庞大的 UNet——ResNet-101 编码器加 CASCADE 解码器，Dice 达到 92%。但每张图像推理要 800ms，需要 2GB 显存。你需要部署到一个总共 4GB 内存、需要实时推理的边缘设备上。

**知识蒸馏**（KD）让你训练一个**小型学生模型**（如 ResNet-18 + 双线性解码器）去模仿大型**教师模型**——以少量计算代价恢复到教师 90–98% 的精度。

### 典型场景

| 场景 | 教师 | 学生 | 为什么用 KD？ |
|------|------|------|-------------|
| 边缘部署 | ResNet-101（2GB） | ResNet-18（100MB） | 内存限制 |
| 实时推理 | Swin-L（800ms/张） | ResNet-34（50ms/张） | 速度限制 |
| 模型整合 | 5 个专家模型 | 1 个通才模型 | 简化管线 |
| 隐私（教师 API） | 云端 API（黑盒） | 本地模型 | 数据不出设备 |

---

## 2. 核心概念

### 2.1 关键洞察：软标签携带更多信息

硬标签说："这个像素是肝脏。"这是 1 bit 的信息。

教师的软标签说："这个像素 85% 是肝脏，10% 是脾脏，4% 是肾脏，1% 是背景。"这不仅告诉学生这个像素*是什么*，还告诉它*与什么相似*——类间关系。

**温度缩放**控制这些"暗知识"被揭示的程度：

$$q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

| 温度 $T$ | 分布 | 信息 |
|----------|------|------|
| $T = 1$ | 标准 softmax | 只有顶部类别重要 |
| $T = 4$ | 软化 | 非顶部类别揭示关系 |
| $T \to \infty$ | 近乎均匀 | 所有类别同等权重（噪声太多） |

**甜区**通常在 $T = 3$–$6$。在 $T=4$ 时，上例中"肝脏 vs. 脾脏"的比率可能是 10:4，告诉学生一个令人困惑的像素更像肝脏而非脾脏——这些信息在硬标签中完全缺失。

### 2.2 蒸馏损失

经典的 Hinton KD 损失结合两个目标：

$$\mathcal{L}_{\text{KD}} = (1-\alpha) \cdot \mathcal{L}_{\text{CE}}(y, p^S) + \alpha \cdot T^2 \cdot \text{KL}(q^T \| q^S)$$

| 项 | 用途 |
|----|------|
| $\mathcal{L}_{\text{CE}}(y, p^S)$ | 学生从**真实标签**学习（硬目标） |
| $T^2 \cdot \text{KL}(q^T \| q^S)$ | 学生从**教师的软预测**学习（软目标） |
| $\alpha$ | 两个目标之间的平衡 |
| $T^2$ | 补偿温度缩放导致的梯度幅度降低 |

**为什么要乘 $T^2$？** 当 logit 除以 $T$ 时，梯度幅度按 $1/T^2$ 缩放。损失乘以 $T^2$ 保持梯度尺度与温度无关。

### 2.3 超越 Logit 匹配

Hinton 原始 KD 只匹配输出分布。更高级的方法匹配中间表示：

```
教师:                       学生:
┌────────────┐              ┌────────────┐
│  编码器     │              │  编码器     │
│  特征       │───匹配─────▶│  特征       │   特征蒸馏
│            │              │            │   (FitNets, AT)
├────────────┤              ├────────────┤
│  解码器     │              │  解码器     │
│  特征       │───匹配─────▶│  特征       │   注意力蒸馏
│            │              │            │   (Attention Transfer)
├────────────┤              ├────────────┤
│  输出       │              │  输出       │
│  logits    │───匹配─────▶│  logits    │   Logit 蒸馏
│            │              │            │   (Hinton KD)
└────────────┘              └────────────┘
```

**特征蒸馏（FitNets）**：用 L2 损失匹配教师和学生之间的原始特征图。学生可能有更少的通道，所以用 1×1 卷积（回归器）将学生特征投影以匹配教师维度。

**注意力蒸馏（Attention Transfer）**：不匹配完整特征图，而是匹配**空间注意力图**——通道平方和给出一个"模型在看什么"的热力图。匹配这些更廉价且更鲁棒。

**逐通道蒸馏（CWD）**：独立归一化特征图的每个通道，并对每通道施加 KL 散度。这捕捉了每个特征通道内的空间模式。

### 2.4 方法对比

| 方法 | 匹配什么 | 计算开销 | 典型恢复率 |
|------|---------|---------|-----------|
| Hinton KD（logit） | 输出概率 | 最小 | 90–95% |
| CWD | 逐通道空间模式 | 低 | 93–97% |
| FitNets | 中间特征 | 低（1×1 conv） | 92–96% |
| Attention Transfer | 空间注意力图 | 低 | 93–97% |
| RKD | 样本间成对关系 | 中等 | 91–95% |
| DKD（解耦 KD） | 目标 + 非目标 logit 分开 | 最小 | 94–98% |

---

## 3. 在 APRIL-MedSeg 中使用

### 3.1 训练脚本

知识蒸馏使用 `train_distillation.py`，需要**两个配置**——一个教师，一个学生：

```bash
python train_distillation.py \
    --teacher_config configs/training_paradigms/distillation/teacher_large.yaml \
    --student_config configs/training_paradigms/distillation/vanilla_kd.yaml \
    --distillation_type logit \
    --temperature 4.0 \
    --alpha 0.5
```

### 3.2 教师配置

教师通常是一个更大、预训练好的模型：

```yaml
# teacher_large.yaml
model:
  num_classes: 4
  img_size: 224
  encoder:
    name: timm_resnet50          # 教师用更大的编码器
    pretrained: true
    in_channels: 3
  decoder:
    name: cascade                # 更强的解码器
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
  epochs: 200                    # 教师训练更久
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

### 3.3 学生配置（Vanilla KD）

```yaml
# vanilla_kd.yaml（学生）
model:
  num_classes: 4
  img_size: 224
  encoder:
    name: timm_resnet34          # 学生用更小的编码器
    pretrained: true
    in_channels: 3
  decoder:
    name: bilinear               # 简单解码器
  bottleneck:
    name: none

distillation:
  method: vanilla_kd
  weight: 1.0                    # KD 损失总权重
  params:
    temperature: 4.0             # 软化温度 T
    alpha: 0.9                   # KD 项权重（1-alpha 给 CE）
    ignore_index: -100           # 损失中忽略的像素

training:
  epochs: 100
  batch_size: 8
  optimizer:
    name: adamw
    lr: 0.0001
  scheduler:
    name: cosine
```

### 3.4 逐通道蒸馏（CWD）配置

```yaml
# cwd.yaml（CWD 学生）
distillation:
  method: cwd
  params:
    temperature: 1.0             # 逐通道归一化温度
```

CWD 对分割特别有效，因为它保留了每个特征通道内的空间模式——当不同通道编码不同解剖结构时很重要。

### 3.5 可用方法一览

| 方法 | 配置文件 | 匹配什么 | 最适合 |
|------|---------|---------|--------|
| Vanilla KD | `vanilla_kd.yaml` | 输出 logits | 简单基线 |
| CWD | `cwd.yaml` | 逐通道特征 | 分割（推荐） |
| DKD | `ctkd.yaml` | 解耦 logits | 难区分的类别 |
| RKD | `rkd.yaml` | 样本关系 | 度量学习任务 |
| FitNets | `aicsd.yaml` | 中间特征 | 非常小的学生 |
| NST | `nst.yaml` | 神经风格迁移 | 保留风格的压缩 |
| AT | `at.yaml` | 注意力图 | 均衡压缩 |

---

## 4. 手把手：你的第一次蒸馏训练

### 第 1 步：先训练教师

教师必须是一个训练良好的模型。训练到收敛：

```bash
python train.py --config configs/training_paradigms/distillation/teacher_large.yaml \
    --output_dir output/teacher
```

验证教师达到了好的 Dice（如 >90%）。

### 第 2 步：选择蒸馏方法

首次运行：
- **Vanilla KD** 最简单且效果好。
- **CWD** 对分割任务推荐。

### 第 3 步：运行蒸馏

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

### 第 4 步：评估学生

```bash
python test.py \
    --config configs/training_paradigms/distillation/student_small.yaml \
    --checkpoint output/kd_vanilla/best_student.pth
```

对比：教师 Dice vs. 学生（无 KD）vs. 学生（有 KD）。

---

## 5. 参数调优指南

### 温度（$T$）

| 值 | 效果 | 适用场景 |
|----|------|---------|
| $T = 1$ | 无软化——等价于标准 CE | 基线，不做蒸馏 |
| $T = 2$ | 轻度软化 | 教师非常自信，类别区分明显 |
| $T = 4$ | 好的平衡（推荐默认值） | 大多数分割任务 |
| $T = 8$ | 重度软化 | 很多相似类别（细粒度分割） |

### Alpha（$\alpha$）

| 值 | 效果 | 适用场景 |
|----|------|---------|
| $\alpha = 0.1$ | 主要从标签学习 | 教师不太好 |
| $\alpha = 0.5$ | 平衡 | 通用起始点 |
| $\alpha = 0.9$ | 主要从教师学习 | 教师非常好（推荐） |

### 学生规模比

| 比率（学生/教师参数量） | 预期恢复率 |
|----------------------|-----------|
| 1/2（一半参数） | 教师的 96–99% |
| 1/4 | 教师的 93–97% |
| 1/10 | 教师的 88–94% |

**经验法则**：1/4 参数的学生通常恢复约 95% 精度——这是压缩率与质量的最佳权衡点。

---

## 6. 常见坑

### 坑 1：学生比从头训练还差

**症状**：KD 学生的 Dice 低于不用教师从头训练的学生。

**修复**：
- 教师可能没训练好。蒸馏前确认教师 Dice >90%。
- $\alpha$ 太高——学生过度依赖一个平庸的教师。降低 $\alpha$ 到 0.3。
- 温度不对——尝试 $T=2$ 或 $T=4$。

### 坑 2：学生坍塌为只预测背景

**症状**：学生 Dice 降到接近零，只预测多数类别。

**修复**：
- 检查教师 logits 格式是否正确（softmax 之前，不是之后）。
- 确保应用了 $T^2$ 补偿——没有它，KD 梯度太小。
- 从 $\alpha = 0.5$ 开始逐步增大。

### 坑 3：特征蒸馏维度不匹配

**症状**：教师和学生特征之间张量形状不匹配的运行错误。

**修复**：
- 教师和学生的通道数不同。确保插入了回归器（1×1 conv）来匹配维度。
- 在配置中设置学生的编码器输出兼容的特征维度。

### 坑 4：非常小的学生收益递减

**症状**：1/20 大小的学生用 KD 比从头训练没改善多少。

**修复**：
- 学生容量可能太低，无法吸收教师知识。尝试稍大的学生。
- 使用多层 hint-based 蒸馏（FitNets）而非只在输出层。
- 考虑剪枝教师而非蒸馏。

---

## 7. 推荐实验

### 实验 1：压缩比研究

训练不同大小的学生，有 KD 和无 KD 对比：

```bash
# 学生 = 1/4 教师（有 KD）
python train_distillation.py \
    --teacher_config configs/training_paradigms/distillation/teacher_large.yaml \
    --student_config configs/training_paradigms/distillation/student_small.yaml \
    --distillation_type logit --temperature 4.0 --alpha 0.9

# 学生 = 1/4 教师（无 KD——基线）
python train.py --config configs/training_paradigms/distillation/student_small.yaml \
    --output_dir output/student_no_kd
```

**预期结果：**

| 配置 | Dice | 推理时间 | 显存 |
|------|------|---------|------|
| 教师（ResNet-101 + CASCADE） | 92% | 800ms | 2.0GB |
| 学生（ResNet-34 + bilinear, 无 KD） | 82% | 50ms | 0.3GB |
| 学生（ResNet-34 + bilinear, 有 KD） | 88% | 50ms | 0.3GB |
| 恢复率 | — | — | 教师 Dice 的 95% |

### 实验 2：方法对比

```bash
# Logit KD
python train_distillation.py --teacher_config ... --student_config ... --distillation_type logit

# CWD
python train_distillation.py --teacher_config ... --student_config ... --distillation_type cwd

# Attention Transfer
python train_distillation.py --teacher_config ... --student_config ... --distillation_type attention
```

**预期排名**：CWD ≥ AT > Logit KD > 无 KD

### 实验 3：温度扫描

```bash
for T in 1 2 4 6 8; do
    python train_distillation.py \
        --teacher_config configs/training_paradigms/distillation/teacher_large.yaml \
        --student_config configs/training_paradigms/distillation/student_small.yaml \
        --distillation_type logit --temperature $T --alpha 0.9 \
        --output_dir output/kd_T${T}
done
```

绘制 Dice vs. 温度曲线以找到最优 $T$。

---

## 8. 延伸阅读

### 关键论文

| 论文 | 年份 | 会议 | 核心思想 |
|------|------|------|---------|
| [Hinton KD](https://arxiv.org/abs/1503.02531) | 2015 | NeurIPS WS | 基于温度的 logit 蒸馏 |
| [FitNets](https://arxiv.org/abs/1412.6550) | 2015 | ICLR | 基于 hint 的中间特征匹配 |
| [Attention Transfer](https://arxiv.org/abs/1612.03928) | 2017 | ICLR | 匹配空间注意力图 |
| [CWD](https://arxiv.org/abs/2011.13256) | 2021 | ICCV | 逐通道蒸馏用于分割 |
| [DKD](https://arxiv.org/abs/2203.08679) | 2022 | CVPR | 解耦目标/非目标 KD |
| [RKD](https://arxiv.org/abs/1904.05068) | 2019 | CVPR | 关系知识蒸馏 |

### 相关文档

- [所有蒸馏方法](../paradigms/distillation.md) — 完整方法目录（27 个方法）

---

[返回训练范式总览](08_paradigms_CN.md) | [上一章：域适应](08b_domain_adaptation_CN.md) | [下一章：弱监督](08d_weakly_supervised_CN.md)
