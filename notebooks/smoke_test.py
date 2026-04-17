"""
=============================================================
Smoke Test — 快速验证训练流程是否产生 NaN
=============================================================
目的:
  用极小数据集 (16 样本) 跑 3 个 training step，
  验证 loss 和 predictions 不为 NaN。
  预期耗时: MPS ~1-2 分钟, CPU ~2-3 分钟

用法:
  cd /Users/HASEE/Desktop/MyFile/School/8007/Project
  python notebooks/smoke_test.py
=============================================================
"""

# ⚠️ 必须在 import torch 之前设置 MPS fallback
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

# 添加 src 到 path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from model import FeedbackModel
from dataset import FeedbackDataset, create_folds
from utils import seed_everything, compute_mcrmse, get_device
from config import Config
import pandas as pd


def smoke_test():
    print("=" * 60)
    print("🔥 Smoke Test — 快速 NaN 检测")
    print("=" * 60)

    seed_everything(42)
    device = get_device()
    print(f"设备: {device}")

    # -------- 1. 加载极小数据集 (16 样本) --------
    config = Config()
    df = pd.read_csv(config.train_csv)
    df = df.head(16)  # 只取 16 行
    print(f"数据: {len(df)} 样本")

    target_columns = config.target_columns

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    dataset = FeedbackDataset(
        df=df,
        tokenizer=tokenizer,
        max_length=128,  # 短序列加速
        target_columns=target_columns,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # -------- 2. 创建模型 --------
    model = FeedbackModel(
        model_name=config.model_name,
        pooling_type="mean",
        num_targets=6,
    )
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-6, foreach=False)
    criterion = nn.MSELoss()

    # -------- 3. 跑 3 个 training step --------
    print(f"\n--- 训练 3 个 step ---")
    model.train()
    nan_detected = False

    for step, batch in enumerate(loader):
        if step >= 3:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        predictions = model(input_ids, attention_mask)
        loss = criterion(predictions, labels)

        # 检查 NaN
        loss_val = loss.item()
        pred_has_nan = torch.isnan(predictions).any().item()
        loss_is_nan = math.isnan(loss_val)

        print(f"  Step {step+1}: loss={loss_val:.4f}, "
              f"pred_nan={pred_has_nan}, loss_nan={loss_is_nan}")

        if loss_is_nan or pred_has_nan:
            nan_detected = True
            print(f"  ❌ NaN 检测到!")
            break

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # -------- 4. 跑 1 个 validation step --------
    print(f"\n--- 验证 1 个 step ---")
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        predictions = model(input_ids, attention_mask)
        preds_np = predictions.cpu().numpy()
        labels_np = labels.numpy()

        pred_has_nan = np.isnan(preds_np).any()
        print(f"  预测值 (前2样本): {preds_np[:2].tolist()}")
        print(f"  真实值 (前2样本): {labels_np[:2].tolist()}")
        print(f"  pred_nan={pred_has_nan}")

        if pred_has_nan:
            nan_detected = True

        if not pred_has_nan:
            mcrmse, per_col = compute_mcrmse(labels_np, preds_np)
            print(f"  MCRMSE={mcrmse:.4f}")
            if math.isnan(mcrmse):
                nan_detected = True

    # -------- 5. 结论 --------
    print(f"\n{'='*60}")
    if nan_detected:
        print("❌ FAIL: 检测到 NaN! MPS fallback 可能未生效。")
        print("  请确认:")
        print("  1. PYTORCH_ENABLE_MPS_FALLBACK=1 在 import torch 之前设置")
        print("  2. 或尝试 device=cpu 排除 MPS 问题")
    else:
        print("✅ PASS: 训练和验证均未产生 NaN!")
        print("  MPS fallback 机制工作正常。")
        print("  可以安心运行完整的 train.py 了。")
    print(f"{'='*60}")

    return not nan_detected


if __name__ == "__main__":
    success = smoke_test()
    sys.exit(0 if success else 1)
