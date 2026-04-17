"""
=============================================================
共享配置 (config.py)
=============================================================
集中管理所有超参数，便于三人代码对接。
修改配置只需改这一个文件，其他模块统一读取。
=============================================================
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """项目配置类 — 使用 dataclass 方便管理和打印"""

    # ==================== 路径配置 ====================
    # 自动检测环境：Kaggle vs 本地
    data_dir: str = (
        "/kaggle/input/competitions/feedback-prize-english-language-learning"
        if os.path.exists("/kaggle/input/competitions/feedback-prize-english-language-learning")
        else os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    )
    output_dir: str = (
        "/kaggle/working"
        if os.path.exists("/kaggle/working")
        else os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    )

    # ==================== 模型配置 ====================
    model_name: str = "microsoft/deberta-v3-base"
    pooling: str = "mean"          # 可选: "mean", "cls", "attention", "max", "gem"
    hidden_dropout: float = 0.0    # 回归头前的 dropout

    # ==================== 数据配置 ====================
    max_length: int = 512
    n_folds: int = 5
    target_columns: List[str] = field(default_factory=lambda: [
        "cohesion", "syntax", "vocabulary",
        "phraseology", "grammar", "conventions"
    ])

    # ==================== 训练配置 ====================
    seed: int = 42
    epochs: int = 4
    batch_size: int = 8
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1      # warmup 步数占总步数的比例
    max_grad_norm: float = 1.0     # 梯度裁剪
    use_amp: bool = True           # 混合精度训练
    early_stopping_patience: int = 2  # 早停耐心值

    # ==================== 派生属性 ====================
    @property
    def train_csv(self) -> str:
        return os.path.join(self.data_dir, "train.csv")

    @property
    def test_csv(self) -> str:
        return os.path.join(self.data_dir, "test.csv")

    @property
    def num_targets(self) -> int:
        return len(self.target_columns)

    def __post_init__(self):
        """创建输出目录"""
        os.makedirs(self.output_dir, exist_ok=True)

    def __str__(self) -> str:
        """格式化打印配置"""
        lines = ["=" * 60, "📋 当前配置", "=" * 60]
        for k, v in self.__dict__.items():
            lines.append(f"  {k:30s}: {v}")
        lines.append(f"  {'train_csv':30s}: {self.train_csv}")
        lines.append(f"  {'test_csv':30s}: {self.test_csv}")
        lines.append(f"  {'num_targets':30s}: {self.num_targets}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ==================== 默认配置实例 ====================
# 使用方式: from src.config import CFG
# 或: from src.config import Config; cfg = Config(lr=1e-5, epochs=5)
CFG = Config()


if __name__ == "__main__":
    print(CFG)
