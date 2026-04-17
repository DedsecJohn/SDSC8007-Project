"""
=============================================================
训练框架 (train.py) — 尤良俊 B-W2-2/3
=============================================================
任务:
  B-W2-2: 完整的 5-Fold CV 训练框架
          - 训练循环、验证循环
          - 学习率调度器 (Cosine with Warmup)
          - 混合精度训练 (AMP)
          - 早停机制
  B-W2-3: 训练日志记录
          - 每个 epoch 打印 train_loss 和 val_MCRMSE
          - 每个 fold 记录最佳分数
          - 最终打印 5-Fold 平均分

接口规范:
  train_one_fold(model, train_loader, val_loader, config) → float (MCRMSE)
=============================================================
"""

# ⚠️ 必须在 import torch 之前设置！
# DeBERTa 的 disentangled attention 中 XSoftmax / torch.gather / torch.einsum
# 在 MPS 后端上部分算子未正确实现，会产生 NaN。
# 设置 FALLBACK=1 后，不支持的 MPS 算子会自动回退到 CPU 执行，避免 NaN。
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
import pandas as pd

from model import FeedbackModel
from dataset import FeedbackDataset, create_folds, create_dataloaders
from utils import (
    seed_everything,
    compute_mcrmse,
    AverageMeter,
    get_logger,
    get_device,
)
from config import Config


# ============================================================
# 1. 优化器
# ============================================================

def get_optimizer(model: nn.Module, lr: float, weight_decay: float) -> AdamW:
    """
    创建 AdamW 优化器。

    对 bias 和 LayerNorm 参数不施加 weight decay（标准做法）。

    Args:
        model: 模型实例
        lr: 学习率
        weight_decay: 权重衰减系数

    Returns:
        AdamW 优化器
    """
    # 不对 bias 和 LayerNorm 做 weight decay
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_params = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]
    return AdamW(optimizer_params, eps=1e-6, foreach=False)


# ============================================================
# 2. 学习率调度器
# ============================================================

def get_scheduler(
    optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
):
    """
    创建 Cosine with Warmup 学习率调度器。

    学习率变化曲线:
      1. Warmup 阶段: 从 0 线性增长到 lr
      2. Cosine 衰减阶段: 从 lr 按余弦曲线衰减到 0

    Args:
        optimizer: 优化器
        num_training_steps: 总训练步数 (epochs * batches_per_epoch)
        warmup_ratio: warmup 步数占总步数的比例

    Returns:
        学习率调度器
    """
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return scheduler


# ============================================================
# 3. 单 Epoch 训练
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    device: torch.device,
    scaler: GradScaler = None,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
) -> float:
    """
    执行一个 epoch 的训练。

    流程: DataLoader → forward → loss → backward → optimizer.step()
    支持混合精度训练 (AMP)。

    Args:
        model: 模型
        loader: 训练 DataLoader
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 计算设备
        scaler: AMP GradScaler (仅 CUDA)
        max_grad_norm: 梯度裁剪阈值
        use_amp: 是否使用混合精度

    Returns:
        float: 平均训练 loss
    """
    model.train()
    loss_meter = AverageMeter("train_loss")
    criterion = nn.MSELoss()

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        batch_size = input_ids.size(0)

        # 根据设备选择 AMP 策略
        amp_enabled = use_amp and device.type == "cuda"

        if amp_enabled and scaler is not None:
            # CUDA 混合精度训练
            with torch.amp.autocast(device_type="cuda"):
                predictions = model(input_ids, attention_mask)
                loss = criterion(predictions, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准训练 (CPU / MPS / 不使用 AMP)
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        scheduler.step()
        loss_meter.update(loss.item(), n=batch_size)

    return loss_meter.avg


# ============================================================
# 4. 单 Epoch 验证
# ============================================================

@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple:
    """
    执行一个 epoch 的验证。

    Args:
        model: 模型
        loader: 验证 DataLoader
        device: 计算设备

    Returns:
        mcrmse: float — 验证集 MCRMSE
        per_column_rmse: list[float] — 各维度 RMSE
    """
    model.eval()
    all_predictions = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        predictions = model(input_ids, attention_mask)
        all_predictions.append(predictions.cpu().numpy())
        all_labels.append(labels.numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    mcrmse, per_column_rmse = compute_mcrmse(all_labels, all_predictions)
    return mcrmse, per_column_rmse


# ============================================================
# 5. 单 Fold 完整训练 (核心接口)
# ============================================================

def train_one_fold(
    model: nn.Module,
    train_loader,
    val_loader,
    config: Config,
    fold: int = 0,
    device: torch.device = None,
    logger=None,
) -> float:
    """
    完整的单 fold 训练流程。

    包含:
      - 优化器 + 调度器创建
      - 训练循环 + 验证循环
      - 混合精度训练 (AMP)
      - 早停机制
      - 日志记录 (B-W2-3)
      - 最佳模型保存

    接口规范:
      train_one_fold(model, train_loader, val_loader, config) → float (MCRMSE)

    Args:
        model: FeedbackModel 实例
        train_loader: 训练 DataLoader
        val_loader: 验证 DataLoader
        config: 配置对象
        fold: 当前 fold 编号
        device: 计算设备 (None 则自动检测)
        logger: Logger 实例 (None 则自动创建)

    Returns:
        float: 该 fold 的最佳验证 MCRMSE
    """
    if device is None:
        device = get_device()
    if logger is None:
        logger = get_logger("train")

    model = model.to(device)

    # ⚠️ 双保险: CUDA + AMP 下强制参数为 FP32
    # AMP 要求: 参数保持 FP32，autocast 自动处理激活的 FP16
    if device.type == "cuda":
        model = model.float()

    # 诊断: 参数 dtype 检查，防止 FP16 参数导致 scaler.unscale_ 崩溃
    dtype_counts = {}
    for p in model.parameters():
        dt = str(p.dtype)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
    logger.info(f"  参数 dtype 分布: {dtype_counts}")
    has_fp16 = any(p.dtype == torch.float16 for p in model.parameters())
    if has_fp16 and config.use_amp and device.type == "cuda":
        raise RuntimeError(
            "模型参数中存在 FP16 张量，但启用了 AMP。请检查 FeedbackModel.__init__ "
            "是否正确强制 FP32 加载 backbone (torch_dtype=torch.float32)。"
        )

    # 创建优化器和调度器
    optimizer = get_optimizer(model, config.lr, config.weight_decay)
    num_training_steps = len(train_loader) * config.epochs
    scheduler = get_scheduler(optimizer, num_training_steps, config.warmup_ratio)

    # AMP GradScaler (仅 CUDA)
    scaler = GradScaler() if (config.use_amp and device.type == "cuda") else None

    # 早停追踪
    best_mcrmse = float("inf")
    patience_counter = 0
    best_epoch = -1

    logger.info(f"{'='*60}")
    logger.info(f"📂 Fold {fold} 开始训练")
    logger.info(f"  训练样本: {len(train_loader.dataset)}, 验证样本: {len(val_loader.dataset)}")
    logger.info(f"  Epochs: {config.epochs}, LR: {config.lr}, Batch: {config.batch_size}")
    logger.info(f"  AMP: {config.use_amp and device.type == 'cuda'}, 设备: {device}")
    logger.info(f"{'='*60}")

    for epoch in range(config.epochs):
        epoch_start = time.time()

        # --- 训练 ---
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            max_grad_norm=config.max_grad_norm,
            use_amp=config.use_amp,
        )

        # --- 验证 ---
        val_mcrmse, per_col_rmse = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
        )

        elapsed = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        # --- 日志 (B-W2-3) ---
        logger.info(
            f"  Fold {fold} | Epoch {epoch+1}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | val_MCRMSE={val_mcrmse:.4f} | "
            f"lr={current_lr:.2e} | time={elapsed:.0f}s"
        )

        # --- 早停判断 ---
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            best_epoch = epoch + 1
            patience_counter = 0

            # 保存最佳模型
            save_path = os.path.join(config.output_dir, f"fold{fold}_best.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ✅ 新最佳! MCRMSE={best_mcrmse:.4f} → 模型已保存: {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info(
                    f"  ⏹ 早停触发 (patience={config.early_stopping_patience}), "
                    f"最佳 epoch={best_epoch}, MCRMSE={best_mcrmse:.4f}"
                )
                break

    logger.info(f"  Fold {fold} 完成 | 最佳 MCRMSE={best_mcrmse:.4f} (Epoch {best_epoch})")
    return best_mcrmse


# ============================================================
# 6. 5-Fold CV 主入口
# ============================================================

def run_kfold(config: Config = None):
    """
    执行完整的 5-Fold 交叉验证训练。

    流程:
      1. 读取数据 → 分层划分
      2. 对每个 fold: 创建模型 → 训练 → 记录分数
      3. 打印 5-Fold 平均 MCRMSE

    Args:
        config: Config 实例 (None 则使用默认配置)
    """
    if config is None:
        config = Config()

    logger = get_logger("train")
    device = get_device()

    logger.info("🚀 开始 K-Fold 交叉验证训练")
    logger.info(f"\n{config}")

    # 固定随机种子
    seed_everything(config.seed)

    # 读取数据
    logger.info(f"读取训练数据: {config.train_csv}")
    df = pd.read_csv(config.train_csv)
    logger.info(f"数据形状: {df.shape}")

    # 分层 K-Fold 划分
    df = create_folds(df, n_folds=config.n_folds, seed=config.seed,
                       target_columns=config.target_columns)
    logger.info(f"已完成 {config.n_folds}-Fold 分层划分")

    # 加载 tokenizer
    logger.info(f"加载 tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # K-Fold 训练
    fold_scores = []
    total_start = time.time()

    for fold in range(config.n_folds):
        fold_start = time.time()

        # 创建 DataLoader
        train_loader, val_loader = create_dataloaders(
            df=df,
            fold=fold,
            tokenizer=tokenizer,
            max_length=config.max_length,
            batch_size=config.batch_size,
            target_columns=config.target_columns,
        )

        # 创建新模型 (每个 fold 从头训练)
        model = FeedbackModel(
            model_name=config.model_name,
            pooling_type=config.pooling,
            num_targets=config.num_targets,
            hidden_dropout=config.hidden_dropout,
        )

        # 训练
        best_mcrmse = train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            fold=fold,
            device=device,
            logger=logger,
        )

        fold_scores.append(best_mcrmse)
        fold_time = time.time() - fold_start
        logger.info(f"  Fold {fold} 耗时: {fold_time/60:.1f} 分钟")

        # 释放显存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==================== 最终结果 ====================
    total_time = time.time() - total_start
    logger.info(f"\n{'='*60}")
    logger.info(f"🏆 {config.n_folds}-Fold CV 训练完成!")
    logger.info(f"{'='*60}")
    for fold, score in enumerate(fold_scores):
        logger.info(f"  Fold {fold}: MCRMSE = {score:.4f}")
    logger.info(f"  {'─'*40}")
    logger.info(f"  平均 MCRMSE = {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    logger.info(f"  总耗时: {total_time/60:.1f} 分钟")
    logger.info(f"{'='*60}")

    return fold_scores


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    """
    直接运行即可启动 5-Fold CV 训练:
      python -m train

    或自定义配置:
      from train import run_kfold
      from config import Config
      config = Config(epochs=2, batch_size=4, lr=1e-5)
      scores = run_kfold(config)
    """
    config = Config()
    scores = run_kfold(config)
