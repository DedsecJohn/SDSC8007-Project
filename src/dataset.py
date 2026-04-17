"""
=============================================================
数据集定义 (dataset.py) — 蒋羿芃 C-W2-1/2
=============================================================
任务:
  C-W2-1: 封装 FeedbackDataset 类，封装 tokenizer 调用、文本截断、label 读取
  C-W2-2: 实现分层 5-Fold 划分 (按6个目标分数均值分桶 + StratifiedKFold)

接口规范:
  Dataset 输出格式:
    {"input_ids": Tensor, "attention_mask": Tensor, "labels": Tensor(6,)}
=============================================================
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold


# ============================================================
# 1. FeedbackDataset (C-W2-1)
# ============================================================

class FeedbackDataset(Dataset):
    """
    Feedback Prize 竞赛数据集。

    功能:
      - 封装 tokenizer 调用（编码文本为 input_ids + attention_mask）
      - 文本截断到 max_length
      - label 读取（训练集返回 6 维评分，测试集不返回 label）

    输出格式 (约定接口):
      {"input_ids": Tensor, "attention_mask": Tensor, "labels": Tensor(6,)}
      测试集不含 "labels" 键。

    Args:
        df (pd.DataFrame):       数据 DataFrame，需含 "full_text" 列
        tokenizer:               HuggingFace tokenizer 实例
        max_length (int):        最大 token 长度
        target_columns (list):   目标列名列表 (训练时)
        is_test (bool):          是否为测试集 (不读取 label)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
        target_columns: list = None,
        is_test: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_columns = target_columns or [
            "cohesion", "syntax", "vocabulary",
            "phraseology", "grammar", "conventions"
        ]
        self.is_test = is_test
        self.texts = self.df["full_text"].tolist()

        if not is_test:
            self.labels = self.df[self.target_columns].values  # (n_samples, 6)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """
        返回单条数据。

        Returns:
            dict: {
                "input_ids":      Tensor (max_length,),
                "attention_mask": Tensor (max_length,),
                "labels":         Tensor (6,)  — 仅训练集
            }
        """
        text = self.texts[idx]

        # Tokenizer 编码
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),          # (max_length,)
            "attention_mask": encoded["attention_mask"].squeeze(0),  # (max_length,)
        }

        if not self.is_test:
            item["labels"] = torch.tensor(
                self.labels[idx], dtype=torch.float32
            )  # (6,)

        return item


# ============================================================
# 2. 分层 K-Fold 划分 (C-W2-2)
# ============================================================

def create_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    target_columns: list = None,
) -> pd.DataFrame:
    """
    按 6 个目标分数的均值进行分桶，用 StratifiedKFold 做分层 5-Fold 划分。

    原理:
      1. 计算每个样本的 6 维评分均值
      2. 用 pd.cut 将均值分为若干 bin (分桶)
      3. 以 bin 编号作为分层依据，确保每个 fold 的评分分布一致

    Args:
        df: 训练数据 DataFrame (需含 target_columns)
        n_folds: fold 数量
        seed: 随机种子
        target_columns: 目标列名列表

    Returns:
        pd.DataFrame: 新增 "fold" 列的 DataFrame
    """
    target_columns = target_columns or [
        "cohesion", "syntax", "vocabulary",
        "phraseology", "grammar", "conventions"
    ]

    df = df.copy()

    # Step 1: 计算每个样本的评分均值
    df["score_mean"] = df[target_columns].mean(axis=1)

    # Step 2: 分桶 (分为 n_folds * 3 个 bin，足够细粒度确保分层效果)
    n_bins = min(n_folds * 3, df["score_mean"].nunique())
    df["score_bin"] = pd.cut(
        df["score_mean"],
        bins=n_bins,
        labels=False,
        duplicates="drop",
    )
    # 处理可能的 NaN (边界值)
    df["score_bin"] = df["score_bin"].fillna(0).astype(int)

    # Step 3: StratifiedKFold 分层划分
    df["fold"] = -1
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df["score_bin"])):
        df.loc[val_idx, "fold"] = fold_idx

    # 清理临时列
    df.drop(columns=["score_mean", "score_bin"], inplace=True)

    return df


# ============================================================
# 3. 辅助函数：创建 DataLoader
# ============================================================

def create_dataloaders(
    df: pd.DataFrame,
    fold: int,
    tokenizer,
    max_length: int = 512,
    batch_size: int = 8,
    target_columns: list = None,
    num_workers: int = 0,
) -> tuple:
    """
    为指定 fold 创建训练和验证的 DataLoader。

    Args:
        df: 含 "fold" 列的 DataFrame
        fold: 当前 fold 编号 (0 ~ n_folds-1)
        tokenizer: HuggingFace tokenizer
        max_length: 最大 token 长度
        batch_size: batch 大小
        target_columns: 目标列名
        num_workers: DataLoader worker 数量

    Returns:
        (train_loader, val_loader)
    """
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)

    train_dataset = FeedbackDataset(
        df=train_df,
        tokenizer=tokenizer,
        max_length=max_length,
        target_columns=target_columns,
        is_test=False,
    )
    val_dataset = FeedbackDataset(
        df=val_df,
        tokenizer=tokenizer,
        max_length=max_length,
        target_columns=target_columns,
        is_test=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # 验证时可用更大 batch
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    import os
    import sys

    print("=" * 60)
    print("🧪 dataset.py 单元测试")
    print("=" * 60)

    # 加载数据
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "train.csv")
    if not os.path.exists(data_path):
        data_path = "/kaggle/input/competitions/feedback-prize-english-language-learning/train.csv"

    if not os.path.exists(data_path):
        print("⚠️  未找到 train.csv，跳过测试")
        sys.exit(0)

    df = pd.read_csv(data_path)
    print(f"数据形状: {df.shape}")

    target_columns = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]

    # 测试 1: create_folds
    print("\n[Test 1] create_folds — 分层 5-Fold 划分")
    df = create_folds(df, n_folds=5, seed=42, target_columns=target_columns)
    print(f"  fold 列分布:")
    for fold in range(5):
        fold_df = df[df["fold"] == fold]
        means = fold_df[target_columns].mean()
        print(f"    Fold {fold}: n={len(fold_df):4d}, "
              f"评分均值=[{', '.join(f'{m:.2f}' for m in means)}]")
    print("  ✅ 各 fold 样本数和评分分布基本一致")

    # 测试 2: FeedbackDataset
    print("\n[Test 2] FeedbackDataset")
    MODEL_NAME = "microsoft/deberta-v3-base"
    print(f"  加载 tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = FeedbackDataset(
        df=df[df["fold"] == 0],
        tokenizer=tokenizer,
        max_length=512,
        target_columns=target_columns,
    )
    print(f"  数据集大小: {len(dataset)}")

    sample = dataset[0]
    print(f"  sample keys: {list(sample.keys())}")
    print(f"  input_ids shape     : {sample['input_ids'].shape}")
    print(f"  attention_mask shape: {sample['attention_mask'].shape}")
    print(f"  labels shape        : {sample['labels'].shape}")
    print(f"  labels 值           : {sample['labels'].tolist()}")

    # 验证接口规范
    assert "input_ids" in sample, "缺少 input_ids"
    assert "attention_mask" in sample, "缺少 attention_mask"
    assert "labels" in sample, "缺少 labels"
    assert sample["input_ids"].shape == (512,), f"input_ids shape 错误: {sample['input_ids'].shape}"
    assert sample["labels"].shape == (6,), f"labels shape 错误: {sample['labels'].shape}"
    print("  ✅ 接口规范验证通过")

    # 测试 3: DataLoader
    print("\n[Test 3] DataLoader")
    train_loader, val_loader = create_dataloaders(
        df=df, fold=0, tokenizer=tokenizer,
        max_length=512, batch_size=4, target_columns=target_columns,
    )
    print(f"  train batches: {len(train_loader)}")
    print(f"  val batches  : {len(val_loader)}")

    batch = next(iter(train_loader))
    print(f"  batch input_ids shape     : {batch['input_ids'].shape}")
    print(f"  batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  batch labels shape        : {batch['labels'].shape}")
    assert batch["input_ids"].shape[0] == 4, "batch_size 不正确"
    assert batch["labels"].shape == (4, 6), "labels batch shape 不正确"
    print("  ✅ DataLoader 测试通过")

    print(f"\n🎉 dataset.py 全部测试通过!")
