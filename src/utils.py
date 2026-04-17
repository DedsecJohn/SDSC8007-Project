"""
=============================================================
工具函数 (utils.py) — 尤良俊 B-W2-1
=============================================================
任务:
  B-W2-1: 实现 MCRMSE 计算函数、随机种子固定函数

内容:
  1. seed_everything()  — 固定所有随机种子
  2. compute_mcrmse()   — 竞赛评估指标
  3. AverageMeter       — 训练过程中追踪指标
  4. get_logger()       — 简易日志工具
=============================================================
"""

import os
import random
import logging
import numpy as np
import torch


# ============================================================
# 1. 随机种子固定
# ============================================================

def seed_everything(seed: int = 42):
    """
    固定所有随机种子，确保实验可复现。

    固定范围:
      - Python random
      - NumPy
      - PyTorch CPU
      - PyTorch CUDA (如果可用)
      - cuDNN deterministic 模式

    Args:
        seed: 随机种子值

    使用方式:
        seed_everything(42)
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多 GPU 情况

    # cuDNN deterministic (牺牲一点速度换取可复现性)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. MCRMSE 计算
# ============================================================

def compute_mcrmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    计算 MCRMSE (Mean Column-wise Root Mean Squared Error)。

    公式:
      MCRMSE = (1/N_targets) * Σ sqrt( (1/N_samples) * Σ (y_true - y_pred)² )

    Args:
        y_true: np.ndarray, shape (n_samples, n_targets) — 真实值
        y_pred: np.ndarray, shape (n_samples, n_targets) — 预测值

    Returns:
        mcrmse: float — 总体 MCRMSE
        per_column_rmse: list[float] — 每个维度的 RMSE

    使用方式:
        mcrmse, per_col = compute_mcrmse(y_true, y_pred)
    """
    # 每列的 RMSE
    colwise_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    # 平均
    mcrmse = np.mean(colwise_rmse)
    return float(mcrmse), colwise_rmse.tolist()


# ============================================================
# 3. AverageMeter — 追踪训练指标
# ============================================================

class AverageMeter:
    """
    追踪指标的滑动平均值。

    用于训练过程中跟踪 loss 等指标。

    使用方式:
        meter = AverageMeter("loss")
        for batch in dataloader:
            loss = ...
            meter.update(loss.item(), n=batch_size)
        print(f"平均 loss: {meter.avg:.4f}")
    """

    def __init__(self, name: str = "metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0    # 最新值
        self.avg = 0.0    # 平均值
        self.sum = 0.0    # 累计总和
        self.count = 0    # 累计样本数

    def update(self, val: float, n: int = 1):
        """
        更新指标。

        Args:
            val: 当前 batch 的指标值
            n: 当前 batch 的样本数
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


# ============================================================
# 4. 日志工具
# ============================================================

def get_logger(name: str = "SDSC8007", log_file: str = None) -> logging.Logger:
    """
    创建一个格式统一的 Logger。

    Args:
        name: Logger 名称
        log_file: 如果指定，同时输出到文件

    Returns:
        logging.Logger 实例

    使用方式:
        logger = get_logger("train")
        logger.info("训练开始")
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出 (如果指定)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================
# 5. 设备检测
# ============================================================

def get_device() -> torch.device:
    """
    自动检测最佳计算设备。

    优先级: cuda > mps > cpu
    ⚠️ MPS 需要配合 PYTORCH_ENABLE_MPS_FALLBACK=1 环境变量，
    且 AdamW 需使用 eps=1e-6 + foreach=False 避免数值溢出。

    Returns:
        torch.device: cuda > mps > cpu
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 utils.py 单元测试")
    print("=" * 60)

    # 测试 1: seed_everything
    print("\n[Test 1] seed_everything")
    seed_everything(42)
    a1 = torch.randn(5)
    seed_everything(42)
    a2 = torch.randn(5)
    assert torch.equal(a1, a2), "种子固定失败: 两次随机结果不同"
    print(f"  第一次: {a1.tolist()}")
    print(f"  第二次: {a2.tolist()}")
    print(f"  相同: {torch.equal(a1, a2)}")
    print("  ✅ 通过")

    # 测试 2: compute_mcrmse
    print("\n[Test 2] compute_mcrmse")
    y_true = np.array([[3.5, 3.5, 3.0, 3.0, 4.0, 3.0],
                        [2.5, 2.5, 3.0, 2.0, 2.0, 2.5]])
    y_pred = np.array([[3.2, 3.3, 2.8, 2.9, 3.8, 2.8],
                        [2.8, 2.7, 2.9, 2.2, 2.3, 2.6]])
    mcrmse, per_col = compute_mcrmse(y_true, y_pred)
    target_columns = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    print(f"  各维度 RMSE:")
    for col, rmse in zip(target_columns, per_col):
        print(f"    {col:15s}: {rmse:.4f}")
    print(f"  MCRMSE = {mcrmse:.4f}")

    # 完美预测应该为 0
    mcrmse_perfect, _ = compute_mcrmse(y_true, y_true)
    assert mcrmse_perfect == 0.0, "完美预测 MCRMSE 应为 0"
    print(f"  完美预测 MCRMSE = {mcrmse_perfect}")
    print("  ✅ 通过")

    # 测试 3: AverageMeter
    print("\n[Test 3] AverageMeter")
    meter = AverageMeter("loss")
    meter.update(0.5, n=32)
    meter.update(0.3, n=32)
    meter.update(0.4, n=32)
    print(f"  三次更新后: {meter}")
    expected_avg = (0.5 * 32 + 0.3 * 32 + 0.4 * 32) / (32 * 3)
    assert abs(meter.avg - expected_avg) < 1e-6, f"平均值不正确: {meter.avg} vs {expected_avg}"
    print("  ✅ 通过")

    # 测试 4: get_logger
    print("\n[Test 4] get_logger")
    logger = get_logger("test")
    logger.info("日志功能正常")
    print("  ✅ 通过")

    # 测试 5: get_device
    print("\n[Test 5] get_device")
    device = get_device()
    print(f"  检测到设备: {device}")
    print("  ✅ 通过")

    print(f"\n🎉 utils.py 全部测试通过!")
