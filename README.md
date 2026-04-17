# SDSC8007 Mini Project

> **课程**：SDSC8007 Deep Learning
> **比赛**：[Kaggle - Feedback Prize: English Language Learning](https://www.kaggle.com/competitions/feedback-prize-english-language-learning)
> **任务**：多目标回归 — 对 8~12 年级英语学习者的作文在 6 个维度上评分
> （cohesion, syntax, vocabulary, phraseology, grammar, conventions）
> **评价指标**：MCRMSE (Mean Columnwise Root Mean Squared Error)
> **当前阶段**：✅ Stage 2 Baseline 已完成（DeBERTa-v3-base + 5-Fold CV）

---

## 👥 团队分工

| 成员 | 角色 | 主责模块 |
|------|------|----------|
| 郝思涵 | **A - Model Architecture** | `src/model.py` · 骨干模型 / 池化策略 / 回归头 |
| 尤良俊 | **B - Training Strategy** | `src/train.py` · 训练循环 / 优化器 / AMP / 早停 |
| 蒋羿芃 | **C - Data Engineering** | `src/dataset.py` · 分词 / K-Fold / DataLoader |

共享：`src/config.py`、`src/utils.py`、`notebooks/` 脚本、`docs/` 文档。

---

## 📂 项目结构

```
Project/
├── README.md                       ← 本文
├── requirements.txt                ← Python 依赖
│
├── data/                           ← 比赛原始数据 (需自行下载)
│   ├── train.csv                     39 K 条训练样本
│   ├── test.csv                      公开测试集
│   └── sample_submission.csv         提交格式样例
│
├── src/                            ← 核心可复用模块（三人协作）
│   ├── config.py                     Config dataclass — 集中管理超参
│   ├── dataset.py                    FeedbackDataset / create_folds / create_dataloaders
│   ├── model.py                      FeedbackModel (DeBERTa + Pooling + 回归头)
│   ├── train.py                      train_one_fold / run_kfold (5-Fold CV)
│   └── utils.py                      seed / MCRMSE / logger / device 检测
│
├── notebooks/                      ← 可独立运行的脚本/实验
│   ├── baseline.py                   本地 baseline (调用 src/)
│   ├── smoke_test.py                 冒烟测试 (1 epoch × 2 fold × 小 batch)
│   ├── debug_nan.py                  NaN 诊断脚本
│   ├── stage1_gpu_test.py            Stage 1 环境验证
│   ├── stage1_hf_deberta_basics.py   HF Transformers 学习笔记代码
│   ├── kaggle_baseline.py            ⭐ Kaggle 上一键跑的自包含脚本
│   └── kaggle_inference_only.py      ⭐ 复用已训权重只做推理
│
├── output/                         ← 本地训练产物 (fold*_best.pth, submission.csv)
│
└── docs/                           ← 文档
    ├── 项目执行计划草案.md           	 总体计划 + 任务分解
    ├── deberta_study_notes.md        DeBERTa 论文阅读笔记
    └── Kaggle操作手册.md            ⭐ Kaggle Notebook Commit / 提交流程
```

---

## 🚀 快速开始

### 方式 A：本地运行（M1/M2 Mac / CPU / CUDA GPU）

#### 1. 克隆仓库并进入目录

```bash
git clone <仓库URL>
cd Project
```

#### 2. 创建虚拟环境并安装依赖

```bash
# 建议使用 Python 3.10 或 3.11
python -m venv .venv
source .venv/bin/activate           # macOS / Linux
# .venv\Scripts\activate            # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

> 💡 **M1/M2 Mac 用户**：直接 `pip install torch` 会自动装好 MPS 支持的版本。
> 💡 **NVIDIA GPU 用户**：先装 CUDA 版 torch：
> `pip install torch --index-url https://download.pytorch.org/whl/cu121`

#### 3. 下载比赛数据

从 Kaggle 手动下载 [比赛数据](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/data) 解压后放入 `data/`：

```
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

或用 Kaggle CLI：

```bash
pip install kaggle
kaggle competitions download -c feedback-prize-english-language-learning -p data/
unzip data/feedback-prize-english-language-learning.zip -d data/
```

#### 4. 先跑冒烟测试验证环境 ✅

```bash
cd notebooks
python smoke_test.py
```

预期：1 epoch × 2 fold × 小 batch 能跑通，loss 正常下降，无 NaN。

#### 5. 正式训练 + 推理

```bash
# 在项目根目录
python -m src.train                  # 5-Fold CV 训练
python notebooks/baseline.py         # 训练 + 推理一体
```

产物会写到 `output/`：5 个 `fold*_best.pth` + `submission.csv`。

### 方式 B：Kaggle Notebook（推荐，白嫖 GPU）⭐

完整操作见 [`docs/Kaggle操作手册.md`](docs/Kaggle操作手册.md)，简要流程：

1. 比赛主页 → `Code` → `New Notebook`
2. 右侧面板：**Accelerator = GPU T4 x2**，**Environment = Pin to original**
3. 把 `notebooks/kaggle_baseline.py` 全文粘进第一个 Cell
4. 右上 **Save Version → Save & Run All (Commit)** ← 🚨 必须走 Commit，文件才会归档
7. Commit 跑完 (~2-3h) → Output 区出现 `submission.csv` → 点 `Submit to Competition`

---

## 🧠 核心技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| Backbone | **microsoft/deberta-v3-base** | ELL 比赛历年 Top 方案基石；Disentangled Attention 擅长捕捉语法/词义 |
| Pooling | **Mean Pooling (默认)** | 简单稳健；后续可切 CLS / Attention / GEM 对比 |
| Head | Dropout + Linear(768 → 6) | 多目标回归标准头 |
| Loss | **MSELoss** | 直接对齐 RMSE 评价指标 |
| Optimizer | **AdamW** (`eps=1e-6`, `foreach=False`) | `eps=1e-6` 是 NLP 回归常用值；`foreach=False` 避免 MPS 上的 NaN BUG |
| LR Schedule | Cosine with Warmup (`warmup_ratio=0.1`) | Transformer fine-tune 黄金配置 |
| CV | 5-Fold **Stratified** (多标签分层) | 6 维标签均匀分布在每个 fold |
| 训练加速 | **AMP** (CUDA 上 `autocast + GradScaler`) | 显存减半、速度提升 1.5~2× |
| 稳定性 | Gradient Clipping (`max_norm=1.0`) + Early Stopping (`patience=2`) | 防梯度爆炸、防过拟合 |

---

## ⚙️ 关键配置（`src/config.py`）

所有超参集中在 `Config` dataclass 里，改一个文件即影响全局：

```python
CFG = Config(
    model_name = "microsoft/deberta-v3-base",
    pooling    = "mean",          # mean / cls / attention / max / gem
    max_length = 512,
    n_folds    = 5,
    epochs     = 4,
    batch_size = 8,
    lr         = 2e-5,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    max_grad_norm = 1.0,
    use_amp = True,
    early_stopping_patience = 2,
)
```

要做实验就 `Config(lr=1e-5, epochs=6)` 即可，无需到处改代码。

---

## 📊 Baseline 结果（Stage 2 验收）

| 指标 | 值 |
|------|----|
| Local CV (5-Fold MCRMSE) | _待在 Kaggle 跑完后填入_ |
| Kaggle Public LB | _待提交后填入_ |
| 训练时长 | T4×2 上约 2–3 小时 |
| 显存占用 | ~10 GB / GPU (bs=8, max_len=512, AMP) |

---

## 🛠 常见问题

### ❓ 训练 loss 一直 NaN？

已知坑及修复（已内置在 `src/train.py` / `notebooks/*.py`）：

| 现象 | 原因 | 解决 |
|------|------|------|
| MPS 上训练 loss=NaN | DeBERTa disentangled attention 在 MPS 上部分算子未正确实现 | 已在脚本开头设 `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| NaN / `eps` 过小 | AdamW 默认 `eps=1e-8` 在 fp16 训练下过小 | 已统一设 `eps=1e-6, foreach=False` |
| MPS fp16 类型冲突 | DeBERTa 输出 fp16 与 Linear fp32 不匹配 | `forward` 里 MPS 设备上显式 `.float()` |

### ❓ Kaggle 上 `ValueError: Attempting to unscale FP16 gradients`？

已修复。新版 transformers 会按 checkpoint dtype 推断加载，可能把 backbone 加载成 FP16，导致 AMP 的 `scaler.unscale_` 崩。
我们在 `FeedbackModel.__init__` 里**强制** `torch_dtype=torch.float32` + `.float()`，并在 `train_one_fold` 中再加一道 `.float()` + dtype 诊断断言。

### ❓ Commit 跑完找不到 `submission.csv`？

99% 是因为你用的是 **Quick Save** 而不是 **Save & Run All**，或者还没等到 Commit 跑完就点了 Output。详见 `docs/Kaggle操作手册.md` FAQ。

### ❓ `sentencepiece` 报错？

```bash
pip install sentencepiece protobuf
```

DeBERTa-v3 tokenizer 依赖 SentencePiece；Kaggle 环境通常已预装，本地可能需要单独安装。

---

## 🗺 后续计划（Stage 3 – Stage 5）

详见 [`docs/项目执行计划草案.md`](docs/项目执行计划草案.md)。核心实验：

- [ ] **S3**: Backbone 尺寸对比（base vs large vs v3-xsmall）
- [ ] **S3**: Pooling 策略对比（mean / cls / attention / gem）
- [ ] **S4**: Layer-wise LR Decay（LLRD）
- [ ] **S4**: AWP (Adversarial Weight Perturbation)
- [ ] **S4**: 伪标签 (Pseudo Labeling) 使用 Feedback Prize 1/2 数据
- [ ] **S5**: 模型融合（不同 backbone / pooling 加权平均）
- [ ] **S5**: 最终报告 & 汇报 PPT

---

## 📖 参考文档（本项目内）

- 📘 [`docs/项目执行计划草案.md`](docs/项目执行计划草案.md) — 总计划 & 任务拆解
- 📗 [`docs/deberta_study_notes.md`](docs/deberta_study_notes.md) — DeBERTa 论文核心要点笔记
- 📙 [`docs/Kaggle操作手册.md`](docs/Kaggle操作手册.md) — Kaggle Notebook 完整操作 & FAQ

---

## 🙋 代码协作约定

1. **配置走 `src/config.py`**：不要在 notebook 里硬编码超参，改 Config 或传参。
2. **模块 import不能反向**：`config ← dataset ← model ← train`。
3. **PR 前跑 `notebooks/smoke_test.py`**：确保分支没引入 NaN / 语法错误。
4. **大产物不进 git**：`output/`、`*.pth`、`*.pkl` 应加到 `.gitignore`。
5. **实验结果集中记录**：推荐每人维护一个 `docs/experiments_<名字>.md`，记录 CV / LB / 改动点。

---

**Last update:** 2026-04-17
