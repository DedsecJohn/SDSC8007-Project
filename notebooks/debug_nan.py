"""
Debug NaN issue — 逐步定位 NaN 来源
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from model import FeedbackModel
from dataset import FeedbackDataset
from utils import seed_everything
from config import Config
import pandas as pd


def check_model_weights(model, label=""):
    """检查模型参数是否有 NaN/Inf"""
    nan_params = []
    inf_params = []
    for name, p in model.named_parameters():
        if torch.isnan(p.data).any():
            nan_params.append(name)
        if torch.isinf(p.data).any():
            inf_params.append(name)
    if nan_params:
        print(f"  ⚠️ [{label}] NaN in weights: {nan_params[:5]}...")
    if inf_params:
        print(f"  ⚠️ [{label}] Inf in weights: {inf_params[:5]}...")
    if not nan_params and not inf_params:
        print(f"  ✅ [{label}] All weights finite")
    return len(nan_params) == 0 and len(inf_params) == 0


def check_grads(model, label=""):
    """检查梯度是否有 NaN/Inf"""
    nan_grads = []
    inf_grads = []
    total_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                nan_grads.append(name)
            if torch.isinf(p.grad).any():
                inf_grads.append(name)
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    print(f"  [{label}] Gradient norm: {total_norm:.4f}")
    if nan_grads:
        print(f"  ⚠️ [{label}] NaN in gradients: {nan_grads[:5]}...")
    if inf_grads:
        print(f"  ⚠️ [{label}] Inf in gradients: {inf_grads[:5]}...")
    if not nan_grads and not inf_grads:
        print(f"  ✅ [{label}] All gradients finite")
    return len(nan_grads) == 0 and len(inf_grads) == 0


def main():
    print("=" * 60)
    print("🔍 NaN Debug — 逐步排查")
    print("=" * 60)

    seed_everything(42)
    device = torch.device("cpu")  # Force CPU
    print(f"设备: {device}")

    # Load data
    config = Config()
    df = pd.read_csv(config.train_csv).head(8)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    dataset = FeedbackDataset(
        df=df, tokenizer=tokenizer, max_length=128,
        target_columns=config.target_columns,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    print(f"\n--- Step 0: 创建模型 ---")
    model = FeedbackModel(
        model_name=config.model_name, pooling_type="mean", num_targets=6,
    ).to(device)
    check_model_weights(model, "init")

    print(f"\n--- Step 1: Forward (no grad) ---")
    model.eval()
    with torch.no_grad():
        preds = model(input_ids, attention_mask)
    print(f"  predictions: {preds[0].tolist()}")
    print(f"  pred NaN: {torch.isnan(preds).any().item()}")

    print(f"\n--- Step 2: Forward (with grad) ---")
    model.train()
    preds = model(input_ids, attention_mask)
    loss = nn.MSELoss()(preds, labels)
    print(f"  predictions: {preds[0].tolist()}")
    print(f"  loss: {loss.item():.4f}")
    print(f"  pred NaN: {torch.isnan(preds).any().item()}")
    print(f"  loss NaN: {math.isnan(loss.item())}")

    print(f"\n--- Step 3: Backward ---")
    loss.backward()
    grads_ok = check_grads(model, "after backward")

    print(f"\n--- Step 4: Gradient clipping ---")
    before_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"  Grad norm before clip: {before_norm:.4f}")
    grads_ok = check_grads(model, "after clip")

    print(f"\n--- Step 5: Optimizer step ---")
    optimizer = AdamW(model.parameters(), lr=2e-5)
    optimizer.step()
    optimizer.zero_grad()
    weights_ok = check_model_weights(model, "after step")

    print(f"\n--- Step 6: Second forward ---")
    model.eval()
    with torch.no_grad():
        preds2 = model(input_ids, attention_mask)
    print(f"  predictions: {preds2[0].tolist()}")
    print(f"  pred NaN: {torch.isnan(preds2).any().item()}")

    if torch.isnan(preds2).any():
        print(f"\n--- Step 6b: 定位哪一层输出 NaN ---")
        model.eval()
        with torch.no_grad():
            # Check backbone output
            outputs = model.backbone(input_ids=input_ids, attention_mask=attention_mask)
            lhs = outputs.last_hidden_state.float()
            print(f"  backbone output NaN: {torch.isnan(lhs).any().item()}")
            print(f"  backbone output range: [{lhs.min().item():.4f}, {lhs.max().item():.4f}]")

            if torch.isnan(lhs).any():
                # Check which layer has NaN
                print("  Checking backbone layer by layer...")
                x = model.backbone.embeddings(input_ids)
                print(f"    embeddings NaN: {torch.isnan(x).any().item()}")
                for i, layer in enumerate(model.backbone.encoder.layer):
                    x = layer(x, attention_mask=attention_mask.unsqueeze(1).unsqueeze(2).float())[0]
                    has_nan = torch.isnan(x).any().item()
                    if has_nan:
                        print(f"    layer {i} NaN: True ← 首次出现!")
                        break
                    else:
                        print(f"    layer {i} NaN: False")

    # Test with zero learning rate to confirm it's the optimizer
    print(f"\n--- Step 7: 测试 lr=0 (排除优化器) ---")
    seed_everything(42)
    model2 = FeedbackModel(
        model_name=config.model_name, pooling_type="mean", num_targets=6,
    ).to(device)
    model2.train()

    optimizer2 = AdamW(model2.parameters(), lr=0.0)
    for step in range(3):
        preds = model2(input_ids, attention_mask)
        loss = nn.MSELoss()(preds, labels)
        optimizer2.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
        optimizer2.step()
        print(f"  lr=0 Step {step+1}: loss={loss.item():.4f}, NaN={torch.isnan(preds).any().item()}")

    print(f"\n{'='*60}")
    print("完成!")


if __name__ == "__main__":
    main()
