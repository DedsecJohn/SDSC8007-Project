"""
=============================================================
Kaggle 推理专用脚本 — 无需重新训练 / 无需联网
=============================================================
功能:
  加载已训练好的 5 个 fold 模型权重 (.pth 文件)，
  对 test.csv 做 ensemble 推理，生成 submission.csv

⚠️ 离线提交要求:
  Kaggle 比赛提交时 Notebook 必须关闭 Internet。
  因此脚本必须从 "挂载的 Kaggle Dataset" 加载预训练权重 + tokenizer，
  而不是从 HuggingFace Hub 下载。

需要挂载的 Kaggle 输入:
  1. 比赛数据:
     /kaggle/input/feedback-prize-english-language-learning/
  2. 预训练 DeBERTa 权重 (挂载为 Kaggle Model):
     /kaggle/input/models/petrcher/microsoftdeberta-v3-base/transformers/default/1/
       ├── config.json
       ├── pytorch_model.bin
       ├── spm.model
       └── tokenizer_config.json
  3. 自己的 fold 权重 (挂载为 Kaggle Dataset):
     /kaggle/input/feedback-baseline-weights/
       ├── fold0_best.pth
       ├── fold1_best.pth
       ├── ...
       └── fold4_best.pth

工作流程:
  第一次: 跑 kaggle_baseline.py → 训练 + 推理 + 生成 submission.csv
                                    ↓ 同时保存 5 个 .pth 模型文件
  之后:   跑本脚本 (关闭 Internet) → 加载 .pth → 离线 ensemble 推理
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

# ---------- 路径判定 ----------
# Kaggle 在比赛数据集挂载路径有两种形式，这里都兼容一下
_KAGGLE_DATA_CANDIDATES = [
    "/kaggle/input/feedback-prize-english-language-learning",
    "/kaggle/input/competitions/feedback-prize-english-language-learning",
]
_KAGGLE_DATA_DIR = next((p for p in _KAGGLE_DATA_CANDIDATES if os.path.exists(p)), None)

if _KAGGLE_DATA_DIR is not None:
    # ====== Kaggle Notebook (离线提交环境) ======
    DATA_DIR = _KAGGLE_DATA_DIR

    # 预训练骨干 (挂载的 Kaggle Model)
    # ⚠️ 如果挂载的模型名称不同，改这里
    MODEL_DIR = "/kaggle/input/models/petrcher/microsoftdeberta-v3-base/transformers/default/1"

    # 自己训练好的 fold 权重 (挂载的 Kaggle Dataset)
    # ⚠️ 如果 Dataset 名称不同，改这里
    _WEIGHTS_CANDIDATES = [
        "/kaggle/input/feedback-baseline-weights",
        "/kaggle/input/datasets/dexjohn/feedback-baseline-weights",
    ]
    WEIGHTS_DIR = next(
        (p for p in _WEIGHTS_CANDIDATES if os.path.exists(p)),
        _WEIGHTS_CANDIDATES[0],
    )

    OUTPUT_DIR = "/kaggle/working"
elif os.path.exists("./data/test.csv"):
    # ====== 本地 (从项目根目录运行) ======
    DATA_DIR = "./data"
    MODEL_DIR = "microsoft/deberta-v3-base"   # 本地允许联网，直接用 HF Hub 名
    WEIGHTS_DIR = "./output"
    OUTPUT_DIR = "./output"
else:
    raise FileNotFoundError("找不到数据目录，请手动设置 DATA_DIR")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型配置 (必须与训练时一致!)
MAX_LENGTH = 512
BATCH_SIZE = 16  # 推理时可用更大 batch
N_FOLDS = 5
TARGET_COLUMNS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]

print(f"\n📁 Data dir:    {DATA_DIR}")
print(f"📁 Model dir:   {MODEL_DIR}")
print(f"📁 Weights dir: {WEIGHTS_DIR}")
print(f"📁 Output dir:  {OUTPUT_DIR}")

# 禁用 HF 联网 (保险起见，确保离线环境不会去拉模型)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


# %%  ==================== 2. 检查文件 ====================

# --- 2.1 检查预训练骨干 ---
print("\n🔍 检查预训练骨干文件:")
if os.path.isdir(MODEL_DIR):
    for fname in ["config.json", "tokenizer_config.json", "spm.model"]:
        fpath = os.path.join(MODEL_DIR, fname)
        ok = os.path.exists(fpath)
        print(f"  {'✅' if ok else '❌'} {fname}")
    # 权重文件可能是 pytorch_model.bin 或 model.safetensors
    has_bin = os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin"))
    has_sft = os.path.exists(os.path.join(MODEL_DIR, "model.safetensors"))
    print(f"  {'✅' if (has_bin or has_sft) else '❌'} pytorch_model.bin / model.safetensors "
          f"(bin={has_bin}, safetensors={has_sft})")
else:
    print(f"  ⚠️  {MODEL_DIR} 不是目录 (可能是 HF Hub 名称，本地模式下没问题)")

# --- 2.2 检查 fold 权重 ---
print("\n🔍 检查 fold 权重文件:")
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
        f"请先运行 kaggle_baseline.py 完成训练，或者把 .pth 文件上传为 Kaggle Dataset。"
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
    def __init__(self, model_dir, num_targets=6):
        super().__init__()
        # 注意: 这里 model_dir 既可以是本地目录 (Kaggle 离线)，
        # 也可以是 HF Hub 名称 (本地调试)，AutoXxx 会自动处理。
        self.config = AutoConfig.from_pretrained(model_dir)
        # 强制 FP32 加载 (与训练时一致，避免 AMP unscale FP16 报错)
        self.backbone = AutoModel.from_pretrained(
            model_dir, config=self.config, torch_dtype=torch.float32
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

    # Tokenizer —— 从本地目录加载 (离线)
    # DeBERTa-v3 使用 SentencePiece (spm.model)，
    # 如果挂载的模型里没有 tokenizer.json，则会回退到 slow tokenizer，
    # 这要求 Kaggle 环境里已安装 sentencepiece (默认已装)。
    print(f"🔤 加载 tokenizer: {MODEL_DIR}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    except Exception as e:
        print(f"⚠️  fast tokenizer 加载失败 ({type(e).__name__}: {e})，回退 use_fast=False")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

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

        model = FeedbackModel(MODEL_DIR, num_targets=len(TARGET_COLUMNS))
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
