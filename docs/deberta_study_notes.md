# DeBERTa 论文核心要点笔记

> **论文**: DeBERTa: Decoding-enhanced BERT with Disentangled Attention 
> **作者**: Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen (Microsoft Research) 
> **论文地址**: https://arxiv.org/abs/2006.03654 

---

## 1. 一句话总结

DeBERTa 通过**解耦注意力机制 (Disentangled Attention)** 和**增强型掩码解码器 (Enhanced Mask Decoder)** 两大创新，在多个 NLU 任务上超越了 BERT 和 RoBERTa。

---

## 2. BERT 的注意力机制回顾

在理解 DeBERTa 之前，先回顾 BERT 的标准自注意力（Self-Attention）：

### BERT 的做法
```
每个 token 的表示 = 内容向量 (Content) + 位置向量 (Position)
```

BERT 在输入层就将 **内容嵌入 (Token Embedding)** 和 **位置嵌入 (Position Embedding)** 直接相加：

```
H_i = TokenEmbed(i) + PosEmbed(i)
```

然后在注意力计算中，使用这个**合并后的向量**来计算 Q、K、V：

```
Q = H · W_Q
K = H · W_K
V = H · W_V
Attention = softmax(Q · K^T / √d) · V
```

### BERT 的问题
- 内容信息和位置信息在**输入层就混合**在一起了
- 注意力计算无法区分"因为内容相关所以关注"还是"因为位置接近所以关注"
- 位置信息在深层会逐渐模糊

---

## 3. DeBERTa 核心创新 #1: Disentangled Attention (解耦注意力)

### 核心思想

**不要把内容和位置加在一起！** 而是让它们各自保持独立的向量，在注意力计算中分别交互。

### 具体做法

每个 token 用**两个独立的向量**表示：
```
H_i = {H_i^content, P_{i|j}^position}
```

其中：
- `H_i^content`: 内容向量（来自 Token Embedding）
- `P_{i|j}^position`: 相对位置向量（token i 和 token j 之间的相对距离）

### 注意力分数的分解

DeBERTa 将注意力分数分解为 **3 个分量**（原本 BERT 是 1 个混合分量）：

```
A_{i,j} = A_{i,j}^{c2c} + A_{i,j}^{c2p} + A_{i,j}^{p2c}
```

| 分量 | 含义 | 解释 |
|------|------|------|
| **Content-to-Content (c2c)** | 内容 → 内容 | "这两个词的语义有多相关？" |
| **Content-to-Position (c2p)** | 内容 → 位置 | "这个词倾向于关注多远的位置？" |
| **Position-to-Content (p2c)** | 位置 → 内容 | "这个位置倾向于关注什么内容？" |

> **注意**: 没有 Position-to-Position (p2p) 分量，因为作者发现它不提供额外信息。

### 计算公式

```
A_{i,j}^{c2c} = Q_i^c · (K_j^c)^T          # 内容Q × 内容K
A_{i,j}^{c2p} = Q_i^c · (K_{δ(i,j)}^p)^T   # 内容Q × 位置K  
A_{i,j}^{p2c} = Q_{δ(i,j)}^p · (K_j^c)^T   # 位置Q × 内容K
```

其中 `δ(i,j) = i - j` 是相对位置距离。

### 相对位置编码

DeBERTa 使用**相对位置编码**而非 BERT 的绝对位置编码：
- 关注的是 token 之间的**相对距离**（如距离1、距离2...）
- 最大相对距离通常设为 512
- 超过最大距离的 token 对共享同一个位置编码

### 直觉理解

以句子 "The cat sat on the mat" 为例：

- **c2c**: "cat" 和 "sat" 在语义上相关（主语-动词关系）→ 高注意力
- **c2p**: 动词 "sat" 倾向于关注它附近的词（主语通常在动词前1-2个位置）
- **p2c**: 位于动词后方的位置倾向于是宾语或介词短语

---

## 4. DeBERTa 核心创新 #2: Enhanced Mask Decoder (EMD)

### 问题背景

在 BERT 的 MLM（Masked Language Model）预训练任务中，需要预测被 mask 掉的词。但 DeBERTa 的解耦注意力只使用了**相对位置**，缺少**绝对位置**信息。

### 为什么绝对位置重要？

考虑句子：
```
"a new store opened beside the new mall"
```

如果要预测 "store" 和 "mall"（都被 mask 了），仅靠相对位置和上下文，模型可能无法区分这两个位置应该填什么词。**绝对位置**可以帮助消歧。

### EMD 的做法

在所有 Transformer 层之后（但在 MLM 预测头之前），将**绝对位置嵌入**加入：

```
最终表示 = Transformer输出(解耦注意力) + 绝对位置嵌入
```

这样做的好处：
1. Transformer 层中使用解耦的相对位置（更灵活）
2. 仅在最终预测时引入绝对位置（作为补充信息）
3. 两全其美

---

## 5. DeBERTa v3 的额外改进

竞赛中我们使用的是 **DeBERTa-v3**，它在 v1 基础上还有：

### REPLACED Token Detection (RTD) — 来自 ELECTRA

DeBERTa-v3 将预训练任务从 MLM 改为 **RTD**：
- 用一个小的生成器（Generator）生成替换词
- 用 DeBERTa 作为判别器（Discriminator），判断每个 token 是否被替换
- 优势：每个 token 都参与训练（MLM 只训练被 mask 的 15%）→ **训练效率更高**

### Gradient-Disentangled Embedding Sharing (GDES)

解决生成器和判别器共享 Embedding 时的梯度冲突问题。

---

## 6. DeBERTa vs BERT 核心差异对比表

| 特性 | BERT | DeBERTa |
|------|------|---------|
| **位置编码** | 绝对位置编码（与内容相加） | 相对位置编码（与内容解耦） |
| **注意力计算** | 单一混合注意力 | 3分量解耦注意力 (c2c + c2p + p2c) |
| **位置信息使用** | 输入层一次性注入 | 解耦注意力层(相对) + EMD(绝对) |
| **绝对位置** | 在输入层就加入 | 仅在解码层加入（EMD） |
| **预训练任务 (v3)** | MLM（15% tokens） | RTD（100% tokens，来自ELECTRA） |
| **参数效率** | 一般 | 更高（RTD训练更高效） |
| **典型模型大小** | base=110M | v3-base=86M (更小但更强) |

---

## 7. DeBERTa 模型规模对比

| 模型 | 参数量 | 隐层维度 | 层数 | 注意力头 |
|------|--------|----------|------|----------|
| deberta-v3-small | 44M | 768 | 6 | 12 |
| deberta-v3-base | 86M | 768 | 12 | 12 |
| deberta-v3-large | 304M | 1024 | 24 | 16 |

> **竞赛策略**: Stage 3-5 中我们需要对比这三个规模的模型效果（任务 A-W35-1）

---

## 8. 为什么 DeBERTa 适合这个竞赛？

### 竞赛任务特点
- **输入**: 学生英语作文（长文本，通常 200-1000+ tokens）
- **输出**: 6个维度的写作质量评分
- **需要**: 深入理解文本的语法、词汇、连贯性等语言质量

### DeBERTa 的优势
1. **解耦注意力**更好地捕捉词与词之间的语法和语义关系
2. **相对位置编码**更适合处理长文本（不受绝对位置限制）
3. **RTD 预训练**使模型对语法错误、用词不当等更敏感（判别器天然适合"纠错"类任务）
4. **参数效率高**：v3-base 仅 86M 参数，但性能接近更大的模型
5. **竞赛验证**：几乎所有 Feedback Prize 竞赛的 Top 方案都使用了 DeBERTa

---

## 9. 口头讲解要点提纲

> **30秒版本：**
> 
> DeBERTa 最大的改进就是把 BERT 中混在一起的"内容"和"位置"信息拆开了。BERT 在最开始就把词的含义和词的位置加在一起，之后所有计算都用这个混合向量。DeBERTa 保持它们分开，让注意力分别考虑"这两个词的意思有多相关"和"它们的距离有多近"，这样模型能更精确地理解语言结构。

> **2分钟版本（补充以下要点）：**
> 
> 1. 注意力分解成 3 个部分：内容→内容、内容→位置、位置→内容
> 2. 用相对位置（两词之间的距离）而不是绝对位置（第几个词）
> 3. 绝对位置在最后解码时才加入（EMD）
> 4. v3 版本还换了预训练方式（RTD），让每个 token 都参与训练，效率更高
> 5. 对我们竞赛来说，DeBERTa 对语法错误和用词质量更敏感，非常适合评分任务

---

## 10. 参考资源

- 📄 [DeBERTa 原始论文](https://arxiv.org/abs/2006.03654)
- 📄 [DeBERTa-v3 论文 (ELECTRA-style)](https://arxiv.org/abs/2111.09543)
- 🤗 [HuggingFace 模型页面](https://huggingface.co/microsoft/deberta-v3-base)
- 📝 [Feedback Prize 竞赛 Top 方案](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion)
