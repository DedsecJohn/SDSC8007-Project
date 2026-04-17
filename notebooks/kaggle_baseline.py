"""
=============================================================
Kaggle Baseline — Feedback Prize: English Language Learning
=============================================================
自包含 Kaggle Notebook 脚本 — 一键运行

功能:
  1. 5-Fold CV 训练 (DeBERTa-v3-base + Mean Pooling + MSE Loss)
  2. 对 test.csv 做推理 (5 模型 ensemble 平均)
  3. 生成 submission.csv

使用方式:
  - Kaggle Notebook: 直接复制粘贴全部代码到 Notebook cell 运行
  - 本地:            python notebooks/kaggle_baseline.py

竞赛: https://www.kaggle.com/competitions/feedback-prize-english-language-learning
指标: MCRMSE (Mean Column-wise Root Mean Squared Error)
=============================================================
"""

# %%  ==================== 0. 环境准备 ====================

import os
import random
import time
import logging
import warnings
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")


# %%  ==================== 1. 配置 (Config) ====================

@dataclass
class Config:
    """项目超参数配置"""

    # --- 路径 ---
    # 默认值 (会在下方被自动修正)
    data_dir: str = "./data"
    output_dir: str = "./output"

    # --- 模型 ---
    model_name: str = "microsoft/deberta-v3-base"
    pooling: str = "mean"
    hidden_dropout: float = 0.0

    # --- 数据 ---
    max_length: int = 512
    n_folds: int = 5
    target_columns: List[str] = field(default_factory=lambda: [
        "cohesion", "syntax", "vocabulary",
        "phraseology", "grammar", "conventions"
    ])

    # --- 训练 ---
    seed: int = 42
    epochs: int = 4
    batch_size: int = 8
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    early_stopping_patience: int = 2

    # --- 推理 ---
    run_inference: bool = True  # 训练完成后自动推理

    @property
    def train_csv(self):
        return os.path.join(self.data_dir, "train.csv")

    @property
    def test_csv(self):
        return os.path.join(self.data_dir, "test.csv")

    @property
    def num_targets(self):
        return len(self.target_columns)

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


CFG = Config()

# 如果在 Kaggle Notebook 中运行，手动修正路径
if os.path.exists("/kaggle/input/competitions/feedback-prize-english-language-learning"):
    CFG.data_dir = "/kaggle/input/competitions/feedback-prize-english-language-learning"
    CFG.output_dir = "/kaggle/working"
elif os.path.exists("./data/train.csv"):
    CFG.data_dir = "./data"
    CFG.output_dir = "./output"

os.makedirs(CFG.output_dir, exist_ok=True)
print(f"\n📁 Data dir:   {CFG.data_dir}")
print(f"📁 Output dir: {CFG.output_dir}")
print(f"📁 Train CSV:  {CFG.train_csv}")
print(f"📁 Test CSV:   {CFG.test_csv}")


# %%  ==================== 2. 工具函数 (Utils) ====================

def seed_everything(seed: int = 42):
    """固定所有随机种子"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mcrmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    计算 MCRMSE (Mean Column-wise Root Mean Squared Error)
    Returns: (mcrmse, per_column_rmse_list)
    """
    colwise_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    mcrmse = np.mean(colwise_rmse)
    return float(mcrmse), colwise_rmse.tolist()


class AverageMeter:
    """训练指标的滑动平均"""
    def __init__(self, name="metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def get_logger(name="SDSC8007"):
    """创建 Logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)
    return logger


def get_device():
    """自动检测: cuda > mps > cpu"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


logger = get_logger("baseline")
device = get_device()
logger.info(f"Device: {device}")
seed_everything(CFG.seed)


# %%  ==================== 3. 数据集 (Dataset) ====================

class FeedbackDataset(Dataset):
    """
    Feedback Prize 数据集
    输出: {"input_ids": Tensor, "attention_mask": Tensor, "labels": Tensor(6,)}
    """

    def __init__(self, df, tokenizer, max_length=512, target_columns=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_columns = target_columns or CFG.target_columns
        self.is_test = is_test
        self.texts = self.df["full_text"].tolist()
        if not is_test:
            self.labels = self.df[self.target_columns].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if not self.is_test:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def create_folds(df, n_folds=5, seed=42, target_columns=None):
    """分层 K-Fold (按6维评分均值分桶)"""
    target_columns = target_columns or CFG.target_columns
    df = df.copy()
    df["score_mean"] = df[target_columns].mean(axis=1)
    n_bins = min(n_folds * 3, df["score_mean"].nunique())
    df["score_bin"] = pd.cut(df["score_mean"], bins=n_bins, labels=False, duplicates="drop")
    df["score_bin"] = df["score_bin"].fillna(0).astype(int)
    df["fold"] = -1
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df["score_bin"])):
        df.loc[val_idx, "fold"] = fold_idx
    df.drop(columns=["score_mean", "score_bin"], inplace=True)
    return df


def create_dataloaders(df, fold, tokenizer, max_length=512, batch_size=8,
                       target_columns=None, num_workers=0):
    """为指定 fold 创建 train/val DataLoader"""
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)

    train_ds = FeedbackDataset(train_df, tokenizer, max_length, target_columns, is_test=False)
    val_ds = FeedbackDataset(val_df, tokenizer, max_length, target_columns, is_test=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader


# %%  ==================== 4. 模型 (Model) ====================

class MeanPooling(nn.Module):
    """Mean Pooling: attention mask 加权平均"""
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_emb = torch.sum(last_hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_emb / sum_mask


class FeedbackModel(nn.Module):
    """
    DeBERTa + Mean Pooling + Linear(hidden_size, 6)
    forward(input_ids, attention_mask) → Tensor(batch_size, 6)
    """

    def __init__(self, model_name, pooling_type="mean", num_targets=6, hidden_dropout=0.0):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        # ⚠️ 关键: 显式强制 FP32 加载 backbone
        #   在较新版本的 transformers 中，from_pretrained 可能根据 config.torch_dtype 或 checkpoint dtype 把模型加载为 FP16，导致 AMP 训练时"Attempting to unscale FP16 gradients" 错误。
        #   混合精度训练的正确姿势: 参数保持 FP32，由 autocast 自动处理激活的 FP16。
        self.backbone = AutoModel.from_pretrained(
            model_name, config=self.config, torch_dtype=torch.float32
        )
        self.backbone = self.backbone.float()  # 双保险: 即使加载错也强制转 FP32
        self.hidden_size = self.config.hidden_size
        self.pooling = MeanPooling()
        self.dropout = nn.Dropout(hidden_dropout)
        self.regression_head = nn.Linear(self.hidden_size, num_targets)
        self._init_weights(self.regression_head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        # ⚠️ 仅在 MPS 上强制转 fp32 (DeBERTa 在 MPS 上部分算子输出 fp16 会崩溃)
        # CUDA + AMP 场景下不要 .float()，否则会破坏 autocast 的 dtype 追踪，导致反向传播时 backbone 的梯度保留为 FP16，scaler.unscale_ 报错:"Attempting to unscale FP16 gradients"
        if last_hidden_state.device.type == "mps":
            last_hidden_state = last_hidden_state.float()
        pooled = self.pooling(last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        return self.regression_head(pooled)


# %%  ==================== 5. 训练框架 (Training) ====================

def get_optimizer(model, lr, weight_decay):
    """AdamW 优化器 (bias/LayerNorm 不做 weight decay)"""
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    params = [
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]
    # eps=1e-6 + foreach=False: 避免 DeBERTa 大梯度导致 AdamW 数值溢出
    return AdamW(params, eps=1e-6, foreach=False)


def get_scheduler(optimizer, num_training_steps, warmup_ratio=0.1):
    """Cosine with Warmup 学习率调度"""
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def train_one_epoch(model, loader, optimizer, scheduler, device,
                    scaler=None, max_grad_norm=1.0, use_amp=True):
    """单 epoch 训练"""
    model.train()
    loss_meter = AverageMeter("train_loss")
    criterion = nn.MSELoss()

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        bs = input_ids.size(0)

        amp_enabled = use_amp and device.type == "cuda"

        if amp_enabled and scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                preds = model(input_ids, attention_mask)
                loss = criterion(preds, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(input_ids, attention_mask)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        scheduler.step()
        loss_meter.update(loss.item(), n=bs)

    return loss_meter.avg


@torch.no_grad()
def validate_one_epoch(model, loader, device):
    """单 epoch 验证，返回 (mcrmse, per_column_rmse)"""
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        preds = model(input_ids, attention_mask)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    mcrmse, per_col = compute_mcrmse(all_labels, all_preds)
    return mcrmse, per_col


def train_one_fold(model, train_loader, val_loader, config, fold=0, device=None, logger=None):
    """
    单 fold 完整训练 (含 early stopping + 模型保存)
    返回: 最佳验证 MCRMSE
    """
    if device is None:
        device = get_device()
    if logger is None:
        logger = get_logger("train")

    model = model.to(device)
    # ⚠️ 双保险: 再次强制所有参数为 FP32
    # 混合精度训练要求: 模型参数必须是 FP32，autocast 会自动处理激活的 FP16
    # 如果参数是 FP16，其梯度也会是 FP16，导致 scaler.unscale_ 报错:"Attempting to unscale FP16 gradients"
    if device.type == "cuda":
        model = model.float()

    # 诊断: 打印参数 dtype 分布 (帮助定位 dtype 问题)
    dtype_counts = {}
    for p in model.parameters():
        dt = str(p.dtype)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
    if logger:
        logger.info(f"  参数 dtype 分布: {dtype_counts}")
    # 断言: 保证没有 FP16 参数 (否则 AMP 会崩溃)
    has_fp16 = any(p.dtype == torch.float16 for p in model.parameters())
    if has_fp16 and config.use_amp and device.type == "cuda":
        raise RuntimeError(
            "模型参数中存在 FP16 张量，但启用了 AMP。请检查 FeedbackModel.__init__ "
            "是否正确强制 FP32 加载 backbone。"
        )

    optimizer = get_optimizer(model, config.lr, config.weight_decay)
    num_steps = len(train_loader) * config.epochs
    scheduler = get_scheduler(optimizer, num_steps, config.warmup_ratio)
    scaler = GradScaler() if (config.use_amp and device.type == "cuda") else None

    best_mcrmse = float("inf")
    patience_counter = 0
    best_epoch = -1

    logger.info(f"{'='*60}")
    logger.info(f"📂 Fold {fold} | train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")
    logger.info(f"   epochs={config.epochs}, lr={config.lr}, bs={config.batch_size}, "
                f"AMP={'ON' if scaler else 'OFF'}, device={device}")
    logger.info(f"{'='*60}")

    for epoch in range(config.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            scaler, config.max_grad_norm, config.use_amp
        )
        val_mcrmse, per_col = validate_one_epoch(model, val_loader, device)

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        logger.info(
            f"  Fold {fold} | Epoch {epoch+1}/{config.epochs} | "
            f"loss={train_loss:.4f} | MCRMSE={val_mcrmse:.4f} | "
            f"lr={lr_now:.2e} | {elapsed:.0f}s"
        )

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            best_epoch = epoch + 1
            patience_counter = 0
            save_path = os.path.join(config.output_dir, f"fold{fold}_best.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ✅ Best MCRMSE={best_mcrmse:.4f} → saved {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"  ⏹ Early stop at epoch {epoch+1} (best={best_epoch})")
                break

    logger.info(f"  Fold {fold} done | Best MCRMSE={best_mcrmse:.4f} (epoch {best_epoch})")
    return best_mcrmse


# %%  ==================== 6. 推理 + 生成 Submission ====================

@torch.no_grad()
def predict_test(model, test_loader, device):
    """对测试集做推理，返回 predictions (n_samples, 6)"""
    model.eval()
    all_preds = []
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        preds = model(input_ids, attention_mask)
        all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds, axis=0)


def run_inference(config, tokenizer, device, logger):
    """
    加载 5 个 fold 的最佳模型，对 test.csv 做 ensemble 推理，生成 submission.csv

    Ensemble 策略: 5 个模型预测的简单平均
    """
    logger.info("\n" + "=" * 60)
    logger.info("🔮 开始推理 (5-Fold Ensemble)")
    logger.info("=" * 60)

    # 读取测试数据
    test_df = pd.read_csv(config.test_csv)
    logger.info(f"测试数据: {test_df.shape[0]} 条")

    # 创建测试 DataLoader
    test_ds = FeedbackDataset(
        test_df, tokenizer, config.max_length,
        config.target_columns, is_test=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size * 2,
        shuffle=False, num_workers=0, pin_memory=True
    )

    # Ensemble: 累加每个 fold 的预测
    all_predictions = np.zeros((len(test_df), config.num_targets))
    n_models = 0

    for fold in range(config.n_folds):
        model_path = os.path.join(config.output_dir, f"fold{fold}_best.pth")
        if not os.path.exists(model_path):
            logger.warning(f"  ⚠️ fold{fold}_best.pth 不存在，跳过")
            continue

        logger.info(f"  加载 Fold {fold} 模型: {model_path}")
        model = FeedbackModel(
            model_name=config.model_name,
            pooling_type=config.pooling,
            num_targets=config.num_targets,
            hidden_dropout=0.0,  # 推理时不需要 dropout
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()

        preds = predict_test(model, test_loader, device)
        all_predictions += preds
        n_models += 1
        logger.info(f"  Fold {fold} 推理完成, preds shape={preds.shape}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if n_models == 0:
        logger.error("❌ 没有找到任何训练好的模型!")
        return None

    # 平均
    all_predictions /= n_models
    logger.info(f"  Ensemble 完成: {n_models} 模型平均")

    # 生成 submission
    submission = pd.DataFrame()
    submission["text_id"] = test_df["text_id"]
    for i, col in enumerate(config.target_columns):
        submission[col] = all_predictions[:, i]

    submission_path = os.path.join(config.output_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    logger.info(f"\n📄 Submission 已保存: {submission_path}")
    logger.info(f"   shape: {submission.shape}")
    logger.info(f"\nSubmission 预览:")
    logger.info(f"\n{submission.head(10).to_string(index=False)}")

    return submission


# %%  ==================== 7. 主入口 ====================

def main():
    """
    完整流程:
      1. 读取数据 & 分层 K-Fold
      2. 5-Fold CV 训练
      3. 推理 & 生成 submission.csv
    """
    config = CFG

    logger.info("🚀 Feedback Prize — Baseline Training & Inference")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Pooling: {config.pooling}, MaxLen: {config.max_length}")
    logger.info(f"Folds: {config.n_folds}, Epochs: {config.epochs}")
    logger.info(f"LR: {config.lr}, BS: {config.batch_size}, WD: {config.weight_decay}")

    # ========== 数据 ==========
    logger.info(f"\n📂 读取训练数据: {config.train_csv}")
    df = pd.read_csv(config.train_csv)
    logger.info(f"训练数据: {df.shape[0]} 条, {df.shape[1]} 列")

    # 评分统计
    logger.info("\n📊 评分统计:")
    for col in config.target_columns:
        logger.info(f"  {col:15s}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, "
                     f"range=[{df[col].min():.1f}, {df[col].max():.1f}]")

    # 分层 K-Fold
    df = create_folds(df, config.n_folds, config.seed, config.target_columns)
    logger.info(f"\n✅ {config.n_folds}-Fold 分层划分完成")
    for fold in range(config.n_folds):
        logger.info(f"  Fold {fold}: {(df['fold']==fold).sum()} 条")

    # 加载 tokenizer
    logger.info(f"\n加载 tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # ========== 5-Fold CV 训练 ==========
    fold_scores = []
    total_start = time.time()

    for fold in range(config.n_folds):
        fold_start = time.time()

        # DataLoaders
        train_loader, val_loader = create_dataloaders(
            df, fold, tokenizer, config.max_length, config.batch_size,
            config.target_columns
        )

        # 新模型
        model = FeedbackModel(
            model_name=config.model_name,
            pooling_type=config.pooling,
            num_targets=config.num_targets,
            hidden_dropout=config.hidden_dropout,
        )

        # 训练
        best_mcrmse = train_one_fold(
            model, train_loader, val_loader, config,
            fold=fold, device=device, logger=logger
        )
        fold_scores.append(best_mcrmse)

        fold_time = time.time() - fold_start
        logger.info(f"  Fold {fold} 总耗时: {fold_time/60:.1f} min")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # ========== CV 结果汇总 ==========
    logger.info(f"\n{'='*60}")
    logger.info(f"🏆 {config.n_folds}-Fold CV 结果")
    logger.info(f"{'='*60}")
    for fold, score in enumerate(fold_scores):
        logger.info(f"  Fold {fold}: MCRMSE = {score:.4f}")
    logger.info(f"  {'─'*40}")
    avg_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    logger.info(f"  ⭐ CV MCRMSE = {avg_score:.4f} ± {std_score:.4f}")
    logger.info(f"  总训练耗时: {total_time/60:.1f} min")
    logger.info(f"{'='*60}")

    # ========== 推理 & Submission ==========
    if config.run_inference:
        submission = run_inference(config, tokenizer, device, logger)
    else:
        logger.info("\n⏭ 跳过推理 (run_inference=False)")
        submission = None

    # ========== 实验记录 ==========
    logger.info(f"\n📝 实验记录:")
    logger.info(f"  编号:     BASE")
    logger.info(f"  模型:     {config.model_name}")
    logger.info(f"  Pooling:  {config.pooling}")
    logger.info(f"  Loss:     MSE")
    logger.info(f"  MaxLen:   {config.max_length}")
    logger.info(f"  Epochs:   {config.epochs}")
    logger.info(f"  LR:       {config.lr}")
    logger.info(f"  BS:       {config.batch_size}")
    logger.info(f"  CV Score: {avg_score:.4f} ± {std_score:.4f}")

    return fold_scores, submission


# %%  ==================== 运行 ====================

if __name__ == "__main__":
    fold_scores, submission = main()
