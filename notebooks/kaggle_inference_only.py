"""
=============================================================
Kaggle 推理专用脚本 — 无需重新训练
=============================================================
功能:
  加载已训练好的 5 个 fold 模型权重 (.pth 文件)，
  对 test.csv 做 ensemble 推理，生成 submission.csv

使用场景:
  ✅ 训练已经跑完一次 (有 fold0_best.pth ~ fold4_best.pth)
  ✅ 只需要重新生成 submission (比如换了 test.csv)
  ✅ 把训练好的权重下载到本地后，在 Kaggle 上只做推理

工作流程:
  第一次: 跑 kaggle_baseline.py → 训练 + 推理 + 生成 submission.csv
                                    ↓ 同时保存 5 个 .pth 模型文件
  之后:   跑本脚本 → 加载 .pth → 推理 → 生成 submission.csv (几分钟搞定)

Kaggle 使用方式:
  1. 第一次跑完 kaggle_baseline.py 后，在 Output 页面下载 fold*_best.pth
  2. 把这些 .pth 文件上传为 Kaggle Dataset (比如叫 "feedback-baseline-weights")
  3. 在新 Notebook 中挂载该 Dataset，修改下方 WEIGHTS_DIR 路径
  4. 运行本脚本 → 几分钟就能生成 submission.csv
=============================================================
"""

# %%  ==================== 0. 环境 ====================

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer

warnings.filterwarnings("ignore")

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# %%  ==================== 1. 配置 ====================

# ===== 核心：修改这里的路径 =====
# Kaggle 环境:
#   DATA_DIR    = "/kaggle/input/feedback-prize-english-language-learning"
#   WEIGHTS_DIR = "/kaggle/input/feedback-baseline-weights"   ← 你上传的权重 Dataset
#   OUTPUT_DIR  = "/kaggle/working"
#
# 本地环境:
#   DATA_DIR    = "./data"
#   WEIGHTS_DIR = "./output"   ← 训练时保存的位置
#   OUTPUT_DIR  = "./output"

if os.path.exists("/kaggle/input/feedback-prize-english-language-learning"):
    # Kaggle Notebook
    DATA_DIR = "/kaggle/input/feedback-prize-english-language-learning"
    # ⚠️ 修改为你上传权重的 Dataset 名称
    WEIGHTS_DIR = "/kaggle/input/feedback-baseline-weights"
    OUTPUT_DIR = "/kaggle/working"
elif os.path.exists("./data/test.csv"):
    # 本地 (从项目根目录运行)
    DATA_DIR = "./data"
    WEIGHTS_DIR = "./output"
    OUTPUT_DIR = "./output"
else:
    raise FileNotFoundError("找不到数据目录，请手动设置 DATA_DIR")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型配置 (必须与训练时一致!)
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 512
BATCH_SIZE = 16  # 推理时可用更大 batch
N_FOLDS = 5
TARGET_COLUMNS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]

print(f"\n📁 Data dir:    {DATA_DIR}")
print(f"📁 Weights dir: {WEIGHTS_DIR}")
print(f"📁 Output dir:  {OUTPUT_DIR}")


# %%  ==================== 2. 检查权重文件 ====================

print("\n🔍 检查模型权重文件:")
available_folds = []
for fold in range(N_FOLDS):
    path = os.path.join(WEIGHTS_DIR, f"fold{fold}_best.pth")
    exists = os.path.exists(path)
    size = f"{os.path.getsize(path)/1024/1024:.1f}MB" if exists else "NOT FOUND"
    status = "✅" if exists else "❌"
    print(f"  {status} fold{fold}_best.pth  ({size})")
    if exists:
        available_folds.append(fold)

if not available_folds:
    raise FileNotFoundError(
        f"❌ 在 {WEIGHTS_DIR} 中没有找到任何模型权重!\n"
        f"请先运行 kaggle_baseline.py 完成训练，或者上传 .pth 文件。"
    )

print(f"\n可用模型: {len(available_folds)}/{N_FOLDS} folds")


# %%  ==================== 3. 模型定义 (与训练时一致) ====================

class MeanPooling(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_emb = torch.sum(last_hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_emb / sum_mask


class FeedbackModel(nn.Module):
    def __init__(self, model_name, num_targets=6):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        # 强制 FP32 加载 (与训练时一致)
        self.backbone = AutoModel.from_pretrained(
            model_name, config=self.config, torch_dtype=torch.float32
        )
        self.backbone = self.backbone.float()
        self.pooling = MeanPooling()
        self.dropout = nn.Dropout(0.0)
        self.regression_head = nn.Linear(self.config.hidden_size, num_targets)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        # 仅 MPS 上转 fp32 (与训练时 kaggle_baseline.py 保持一致)
        if last_hidden_state.device.type == "mps":
            last_hidden_state = last_hidden_state.float()
        pooled = self.pooling(last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        return self.regression_head(pooled)


# %%  ==================== 4. 数据集 ====================

class FeedbackDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.texts = df["full_text"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


# %%  ==================== 5. 推理 ====================

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        preds = model(input_ids, attention_mask)
        all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds, axis=0)


def main():
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥 Device: {device}")

    # 读取测试数据
    test_path = os.path.join(DATA_DIR, "test.csv")
    test_df = pd.read_csv(test_path)
    print(f"📄 测试数据: {test_df.shape[0]} 条")

    # Tokenizer
    print(f"🔤 加载 tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # DataLoader
    test_ds = FeedbackDataset(test_df, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=True
    )

    # Ensemble 推理
    print(f"\n🔮 开始 Ensemble 推理 ({len(available_folds)} 模型)...")
    all_predictions = np.zeros((len(test_df), len(TARGET_COLUMNS)))

    for fold in available_folds:
        model_path = os.path.join(WEIGHTS_DIR, f"fold{fold}_best.pth")
        print(f"  Fold {fold}: 加载 {model_path}")

        model = FeedbackModel(MODEL_NAME, num_targets=len(TARGET_COLUMNS))
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model = model.to(device)

        preds = predict(model, test_loader, device)
        all_predictions += preds
        print(f"  Fold {fold}: 推理完成 ✅")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_predictions /= len(available_folds)

    # 生成 submission
    submission = pd.DataFrame()
    submission["text_id"] = test_df["text_id"]
    for i, col in enumerate(TARGET_COLUMNS):
        submission[col] = all_predictions[:, i]

    submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"\n{'='*60}")
    print(f"📄 Submission 已保存: {submission_path}")
    print(f"   shape: {submission.shape}")
    print(f"{'='*60}")
    print(f"\n预览:")
    print(submission.to_string(index=False))

    return submission


# %%  ==================== 运行 ====================

if __name__ == "__main__":
    submission = main()
