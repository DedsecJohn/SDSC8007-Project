"""
=============================================================
Stage 2 Baseline — 完整 Notebook 脚本
=============================================================
Stage 2 结束检查点:
  ✅ 三人代码合并为一个完整 Notebook
  ✅ 使用 DeBERTa-v3-base + Mean Pooling + MSE Loss + 5-Fold CV
  ✅ 记录 Baseline 的 5-Fold CV MCRMSE 分数

使用方式:
  方式1 (本地): python notebooks/baseline.py
  方式2 (模块): python -m train
  方式3 (Kaggle): 将本文件内容复制到 Kaggle Notebook 中运行

如果在 Kaggle Notebook 中使用:
  1. 将 src/ 目录下的所有 .py 文件上传为 utility script
  2. 或将下方的 "Kaggle 一体化版本" 部分取消注释直接运行
=============================================================
"""

import sys
import os

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# 导入项目模块
# ============================================================
from config import Config
from utils import seed_everything, compute_mcrmse, get_logger, get_device
from dataset import create_folds, create_dataloaders, FeedbackDataset
from model import FeedbackModel
from train import train_one_fold, run_kfold

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer


def main():
    """
    Baseline 主入口。

    配置:
      - Model:    microsoft/deberta-v3-base
      - Pooling:  Mean Pooling
      - Loss:     MSE Loss
      - CV:       5-Fold Stratified
      - Epochs:   4
      - LR:       2e-5
      - Scheduler: Cosine with Warmup
      - AMP:      开启 (CUDA) / 关闭 (CPU/MPS)
    """
    # ==================== 配置 ====================
    config = Config(
        # 模型
        model_name="microsoft/deberta-v3-base",
        pooling="mean",
        hidden_dropout=0.0,
        # 数据
        max_length=512,
        n_folds=5,
        # 训练
        seed=42,
        epochs=4,
        batch_size=8,
        lr=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        use_amp=True,
        early_stopping_patience=2,
    )

    logger = get_logger("baseline")
    device = get_device()

    # ==================== 环境信息 ====================
    logger.info("=" * 60)
    logger.info("🚀 Feedback Prize Baseline")
    logger.info("=" * 60)
    logger.info(f"PyTorch 版本: {torch.__version__}")
    logger.info(f"设备: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        logger.info(f"GPU 显存: {mem:.1f} GB")
    logger.info(f"\n{config}")

    # ==================== 数据检查 ====================
    logger.info("📂 数据检查")
    if not os.path.exists(config.train_csv):
        logger.error(f"❌ 训练数据不存在: {config.train_csv}")
        logger.error("请确保 data/train.csv 在正确位置")
        return

    df = pd.read_csv(config.train_csv)
    logger.info(f"训练数据: {df.shape[0]} 条, {df.shape[1]} 列")
    logger.info(f"列名: {list(df.columns)}")
    logger.info(f"评分维度: {config.target_columns}")

    # 简要数据统计
    logger.info("\n📊 评分统计:")
    for col in config.target_columns:
        logger.info(
            f"  {col:15s}: mean={df[col].mean():.2f}, "
            f"std={df[col].std():.2f}, "
            f"min={df[col].min():.1f}, max={df[col].max():.1f}"
        )

    # 检查缺失值
    null_counts = df[config.target_columns].isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"⚠️  发现缺失值: {null_counts.to_dict()}")
    else:
        logger.info("  缺失值: 无 ✅")

    # ==================== 开始训练 ====================
    logger.info("\n" + "=" * 60)
    logger.info("🏋️ 开始 5-Fold CV Baseline 训练")
    logger.info("=" * 60)

    fold_scores = run_kfold(config)

    # ==================== 结果汇总 ====================
    logger.info("\n" + "=" * 60)
    logger.info("📋 Baseline 结果汇总")
    logger.info("=" * 60)
    logger.info(f"  模型:     {config.model_name}")
    logger.info(f"  Pooling:  {config.pooling}")
    logger.info(f"  Loss:     MSE")
    logger.info(f"  MaxLen:   {config.max_length}")
    logger.info(f"  Epochs:   {config.epochs}")
    logger.info(f"  LR:       {config.lr}")
    logger.info(f"  Batch:    {config.batch_size}")
    logger.info(f"  Seed:     {config.seed}")
    logger.info(f"  ")
    for fold, score in enumerate(fold_scores):
        logger.info(f"  Fold {fold}: MCRMSE = {score:.4f}")
    logger.info(f"  {'─'*40}")
    avg = np.mean(fold_scores)
    std = np.std(fold_scores)
    logger.info(f"  ⭐ 平均 MCRMSE = {avg:.4f} ± {std:.4f}")
    logger.info("=" * 60)

    # 写入实验记录
    record = {
        "编号": "BASE",
        "实验描述": f"{config.model_name} + {config.pooling} Pooling + MSE Loss",
        "CV Score": f"{avg:.4f}",
        "vs基线Δ": "—",
        "采纳": "✅",
    }
    logger.info(f"\n📝 实验记录表 (第一行):")
    for k, v in record.items():
        logger.info(f"  {k}: {v}")

    return fold_scores


# ============================================================
# Kaggle 一体化版本 (无需 src/ 模块)
# ============================================================
# 如果在 Kaggle 中无法导入 src/ 模块，取消注释以下代码块，
# 它包含了所有必要的代码，可以独立运行。

"""
# ===================== KAGGLE 一体化版本 =====================
# 取消注释此段代码，可在 Kaggle Notebook 中独立运行

import os, random, time, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from transformers import AutoModel, AutoConfig, AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold

# --- 配置 ---
class Config:
    model_name = "microsoft/deberta-v3-base"
    pooling = "mean"
    max_length = 512
    n_folds = 5
    target_columns = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    seed = 42
    epochs = 4
    batch_size = 8
    lr = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    max_grad_norm = 1.0
    use_amp = True
    early_stopping_patience = 2
    data_dir = "/kaggle/input/feedback-prize-english-language-learning"
    output_dir = "/kaggle/working"
    train_csv = os.path.join(data_dir, "train.csv")
    num_targets = 6
    hidden_dropout = 0.0

# (完整代码请从 src/ 目录各文件中复制粘贴)
# ===================== END KAGGLE 版本 =====================
"""


if __name__ == "__main__":
    main()
