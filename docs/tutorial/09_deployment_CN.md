# 第 09 章：部署与推理

[上一章：高级训练范式](08_paradigms_CN.md) | [English](09_deployment.md) | [教程目录](index.md)

---

## 1. 背景与动机

训练好的分割模型只有在能准确进行预测时才有用。从训练到部署之间存在多个挑战：

- **精度差距**：单模型的预测具有固有方差。能否在不重新训练的情况下做得更好？
- **鲁棒性**：真实世界图像在方向、亮度和质量上各不相同。如何在推理时处理？
- **部署约束**：生产环境可能需要特定格式（ONNX）、降低精度或快速推理。
- **成本分析**：理解计算需求（FLOPs、参数量、延迟）对规划至关重要。

本章解释每种部署技术背后的理论基础——为何平均有帮助、为何测试时增强近似贝叶斯推理，以及模型导出如何保持精度。

---

## 2. 核心概念

### 2.1 集成理论——为何平均有效

集成组合多个模型的预测以产生更准确和稳定的结果。理论依据来自**偏差-方差分解**：

$$\mathbb{E}[(y - \hat{y})^2] = \text{Bias}^2[\hat{y}] + \text{Var}[\hat{y}] + \sigma^2$$

其中 $\sigma^2$ 是不可约噪声。平均 $N$ 个独立模型可减少方差而不影响偏差：

$$\text{Var}\left[\frac{1}{N}\sum_{i=1}^{N} \hat{y}_i\right] = \frac{1}{N} \text{Var}[\hat{y}]$$

这意味着：如果每个模型具有相同偏差但犯*独立*的错误，平均可以消除错误同时保留正确信号。

**对分割的意义**：单模型可能产生嘈杂的边界预测。集成平均通过抑制个体化错误来平滑这些边界。

**分割的平均模式**：

| 模式 | 公式 | 最佳场景 |
|------|------|----------|
| Logit 平均 | $\bar{z} = \frac{1}{N}\sum z_i$ | 多类（校准良好） |
| Softmax 平均 | $\bar{p} = \frac{1}{N}\sum \text{softmax}(z_i)$ | 稳定的多类 |
| Sigmoid 平均 | $\bar{p} = \frac{1}{N}\sum \sigma(z_i)$ | 二值/多标签 |

**Logit vs 概率平均**：在 softmax 之前平均 logit 等价于概率的几何平均，往往更自信且校准更好。softmax 之后的平均是算术平均，更平滑但可能不够自信。

**集成来源**：模型可以通过随机种子、数据折叠、架构或训练超参数来产生差异。模型间更多多样性 → 平均带来更大的方差减少。

### 2.2 测试时增强（TTA）——近似贝叶斯推理

TTA 对*同一测试图像*施加多种增强，对每个增强版本运行模型，然后合并预测。这是**贝叶斯模型平均**的实用近似：

$$p(y|x) \approx \frac{1}{|A|} \sum_{a \in A} p(y | a(x))$$

其中 $A$ 是测试时增强集合（旋转、翻转、亮度变化）。

**TTA 为何有效**：单模型对不同方向或光照下的同一图像可能产生不同预测。通过对增强取平均，TTA 边缘化掉这些干扰因素——预测变得对施加的增强不变。

**与贝叶斯推理的联系**：在完整贝叶斯框架中，我们会对所有可能模型参数积分：$p(y|x) = \int p(y|x,\theta) p(\theta) d\theta$。TTA 则对输入变换积分，计算成本低且出奇地有效。

**增强选择策略**：

| 增强 | 捕获什么 | 何时使用 |
|------|---------|---------|
| 水平翻转 | 左右对称性 | 解剖对称结构 |
| 旋转（90°, 180°, 270°） | 方向不变性 | 无固定方向（病理） |
| 亮度/对比度 | 强度变化 | 不同扫描仪/协议 |
| 缩放（0.75x, 1.25x） | 尺寸变化 | 可变病灶大小 |

**合并策略**：

| 策略 | 公式 | 性质 |
|------|------|------|
| 均值 | $\bar{p} = \frac{1}{|A|}\sum p_a$ | 标准，平衡 |
| 几何平均 | $\bar{p} = (\prod p_a)^{1/|A|}$ | 强调一致性 |
| 最大值 | $\bar{p} = \max_a p_a$ | 最自信的预测 |
| 中位数 | $\bar{p} = \text{median}_a p_a$ | 对异常增强鲁棒 |

**几何平均**通常更适合概率，因为它惩罚任何增强不一致的预测——如果一个增强给出 $p=0.1$ 而其他给出 $p=0.9$，几何平均远低于算术平均，反映了不确定性。

### 2.3 模型导出——ONNX 与量化

**ONNX（开放神经网络交换格式）**将训练好的模型表示为计算图，支持在非 Python 环境（C++、移动端、边缘设备）中部署：

```
PyTorch 模型 → torch.onnx.export() → ONNX 图 → ONNX Runtime / TensorRT / OpenVINO
```

导出过程用样本输入追踪模型，将所有操作记录为静态图。关键考虑因素：

- **算子兼容性**：并非所有 PyTorch 操作都有 ONNX 等价物。自定义算子需要手动注册。
- **动态形状**：默认情况下 ONNX 模型有固定输入尺寸。动态轴必须显式指定。
- **算子融合**：ONNX Runtime 融合相邻操作（如 Conv+BN+ReLU）以加速推理。

**量化**降低数值精度以减小模型体积和延迟：

| 精度 | 每权重位数 | 大小（相对 FP32） | 速度 | 精度 |
|------|-----------|-----------------|------|------|
| FP32 | 32 | 1.0x | 1.0x | 基线 |
| FP16 | 16 | 0.5x | ~1.5-2x | ~相同 |
| INT8 | 8 | 0.25x | ~2-4x | -0.5-2% Dice |

**训练后量化**（PTQ）：训练后直接量化权重和激活，无需重新训练。快速但可能损失精度。

**量化感知训练**（QAT）：训练期间通过插入伪量化节点模拟量化。模型学习对降低精度具有鲁棒性。更准确但需要重新训练。

### 2.4 计算成本——FLOPs、MACs、参数量

理解计算需求对部署规划至关重要。

**FLOPs（浮点运算次数）**：单次前向传播的总算术操作数（加法 + 乘法）。对于卷积：

$$\text{FLOPs}_{\text{conv}} = 2 \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot k^2 \cdot H_{\text{out}} \cdot W_{\text{out}}$$

其中因子 2 表示乘法 + 加法。

**MACs（乘累加运算）**：通常代替 FLOPs 报告。一次 MAC = 一次乘法 + 一次加法 = 2 FLOPs。

**参数量**：可训练权重总数。注意冻结的 Foundation 编码器参数被*加载*但不*训练*——它们计入模型大小但不计入梯度计算。

**精度-成本权衡**：

```
精度 ↑
  │           ● Foundation（冻结）
  │        ● Transformer
  │     ● CNN（大）
  │   ● CNN（中）
  │ ● CNN（小）
  └──────────────────────── FLOPs →
```

更高精度通常需要更多计算，但关系是亚线性的——FLOPs 翻倍很少使精度翻倍。最佳点取决于部署约束。

### 2.5 MLLM 推理管线

多模态大语言模型支持**检测-再分割**范式，从文本提示进行零样本分割：

```
文本: "肝脏肿瘤"
       │
       ▼
  MLLM 检测器（GroundingDINO / Qwen-VL / InternVL）
       │  → 边界框或点提示
       ▼
  分割器（SAM2 / MedSAM / SAM-Med2D / LiteMedSAM）
       │  → 像素级 mask
       ▼
  最终分割结果
```

此管线无需在目标任务上训练——MLLM 的视觉定位能力定位物体，分割器产生 mask。

---

## 3. 方法细节

### 3.1 推理管线概览

```
训练好的 checkpoint
    │
    ├─ 单模型 → test.py → Dice, IoU, HD95
    │
    ├─ 集成（N 个 checkpoint） → 加权 logit 平均 → 提升 Dice
    │
    ├─ TTA（增强） → 合并预测 → 提升鲁棒性
    │
    ├─ ONNX 导出 → model.onnx → C++/移动端部署
    │
    └─ 性能分析 → FLOPs, 参数量, FPS → 部署可行性检查
```

### 3.2 集成配置

| 参数 | 选项 | 推荐 |
|------|------|------|
| Checkpoint 数量 | 2-10 | 3-5（超过 5 收益递减） |
| 权重 | 相等或学习 | 相等（简单），验证集优化（最佳） |
| 平均模式 | logit, softmax, sigmoid | 多类用 `logit` |

### 3.3 TTA 配置

| 参数 | 选项 | 推荐 |
|------|------|------|
| 增强 | identity, 翻转, 旋转, 亮度 | 从 identity + hflip + rot90 开始 |
| 合并策略 | mean, gmean, max, median | `mean`（默认），`gmean`（概率） |
| 增强数量 | 2-12 | 4-6（平衡速度与增益） |

---

## 4. 在 APRIL-MedSeg 中实践

```bash
# 单模型评估
python test.py --config configs/architectures/networks/general/transunet.yaml \
    --checkpoint output/best_model.pth

# 集成（3 个 checkpoint）
python test.py --config cfg.yaml \
    --checkpoint ckpt_fold0.pth ckpt_fold1.pth ckpt_fold2.pth \
    --ensemble-weights 0.4 0.3 0.3 \
    --ensemble-average logit

# TTA
python test.py --config cfg.yaml --checkpoint best.pth \
    --tta --tta-augs identity rot90 hflip vflip --tta-merge mean

# ONNX 导出并验证
python scripts/export_onnx.py --config cfg.yaml --checkpoint best.pth \
    --output model.onnx --verify

# 模型性能分析
python profile_model.py --config cfg.yaml
```

```python
# MLLM 管线（零样本分割）
from medseg.inference.mllm import MLLMPipeline

pipeline = MLLMPipeline(
    detector='grounding_dino',    # 或 qwen2_vl, qwen3_vl, internvl
    segmenter='sam2',             # 或 medsam, sam_med2d, lite_medsam
    text_prompt='liver tumor',
)
result = pipeline.predict('data/test/image_001.png')
```

---

## 5. 推荐实验

### 实验 1：集成提升

在同一测试集上比较单模型与集成：

| 设置 | 预期 Dice 增益 | 推理成本 |
|------|--------------|---------|
| 单 checkpoint | 基线 | 1x |
| 3-checkpoint 集成 | +1-3% | 3x |
| 5-checkpoint 集成 | +1.5-4% | 5x |

### 实验 2：TTA 影响

同一模型，有/无 TTA：

| TTA 增强 | 预期 Dice 增益 | 推理成本 |
|---------|--------------|---------|
| 无 | 基线 | 1x |
| identity + hflip | +0.5-1% | 2x |
| 4 旋转 + 2 翻转 | +1-2% | 7x |

### 实验 3：ONNX 验证

导出为 ONNX 并验证数值精度：

| 检查项 | 预期结果 |
|--------|---------|
| 输出形状匹配 | 完全一致 |
| 最大绝对误差 | < 1e-4（FP32） |
| Dice 差异 | < 0.1% |
| 推理速度 | 1.2-2x 更快（ONNX Runtime） |

### 实验 4：模型成本分析

比较架构的计算成本：

| 架构 | 参数量 | FLOPs | 预期 FPS（224×224） |
|------|--------|-------|-------------------|
| ResNet50 + UNet | ~27M | ~55G | ~50 |
| Swin-T + UNet | ~30M | ~60G | ~35 |
| DINOv2-B（冻结）+ UNet | ~88M | ~100G | ~20 |

---

## 6. 延伸阅读

### 关键论文

| 论文 | 年份 | 会议 | 关键贡献 |
|------|------|------|----------|
| [集成方法](https://link.springer.com/article/10.1023/A:1010950718922) | 2002 | Data Mining & KD | 集成理论综述 |
| [Deep Ensembles](https://arxiv.org/abs/1612.01474) | 2017 | NeurIPS | 随机初始化集成用于不确定性 |
| [分割中的 TTA](https://arxiv.org/abs/1910.11190) | 2019 | - | 医学图像的测试时增强 |
| [ONNX 规范](https://onnx.ai/onnx/) | 2024 | - | 开放神经网络交换格式规范 |
| [量化综述](https://arxiv.org/abs/2103.13630) | 2021 | - | 神经网络量化综合综述 |
| [Grounding DINO](https://arxiv.org/abs/2303.05499) | 2023 | ECCV | 从文本进行开放集目标检测 |
| [SAM 2](https://arxiv.org/abs/2408.00714) | 2024 | - | Segment Anything Model 2 |

### 相关文档

- [推理指南](../usage/inference.md) — 完整的 test.py 选项和集成配置
- [ONNX 导出](../usage/export.md) — 导出与部署指南
- [MLLM 管线](../paradigms/text_guided.md#mllm-pipeline) — 5 检测器 × 4 分割器

---

[上一章：高级训练范式](08_paradigms_CN.md) | [教程目录](index.md)
