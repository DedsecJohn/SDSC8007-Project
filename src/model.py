"""
=============================================================
模型定义 (model.py) — 郝思涵 A-W2-1/2/3
=============================================================
任务:
  A-W2-1: 封装 FeedbackModel 类，接收 backbone 名称，
          内部加载 HuggingFace 模型 + Linear(hidden_size, 6) 回归头
  A-W2-2: 实现 Mean Pooling 作为默认 Pooling 策略
  A-W2-3: 预留 Pooling 策略切换接口 (config.pooling 参数)

接口规范:
  forward(input_ids, attention_mask) → Tensor(batch_size, 6)
=============================================================
"""

# ⚠️ MPS fallback: DeBERTa 的部分算子在 MPS 上会产生 NaN，需回退到 CPU
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


# ============================================================
# Pooling 策略实现
# ============================================================

class MeanPooling(nn.Module):
    """
    Mean Pooling (⭐ 默认策略)
    对 last_hidden_state 沿 seq_len 维度做 attention mask 加权平均。

    原理:
      1. 用 attention_mask 遮蔽 padding 位置 (乘以 0)
      2. 对 seq_len 维度求和
      3. 除以每个样本的真实 token 数

    输入: last_hidden_state (batch, seq_len, hidden_size)
          attention_mask    (batch, seq_len)
    输出: (batch, hidden_size)
    """

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len) → (batch, seq_len, 1) → broadcast to (batch, seq_len, hidden_size)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # 将 padding 位置置零后求和
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        # 真实 token 数 (clamp 防止除零)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask


class CLSPooling(nn.Module):
    """
    CLS Pooling
    直接取 [CLS] token 的输出作为句子表示。

    输入: last_hidden_state (batch, seq_len, hidden_size)
          attention_mask    (未使用)
    输出: (batch, hidden_size)
    """

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return last_hidden_state[:, 0, :]


class MaxPooling(nn.Module):
    """
    Max Pooling
    对真实 token 的每个隐层维度取最大值。

    输入: last_hidden_state (batch, seq_len, hidden_size)
          attention_mask    (batch, seq_len)
    输出: (batch, hidden_size)
    """

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # 将 padding 位置填充为极小值，使其不影响 max 操作
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        last_hidden_state[input_mask_expanded == 0] = -1e9
        max_embeddings, _ = torch.max(last_hidden_state, dim=1)
        return max_embeddings


class AttentionPooling(nn.Module):
    """
    Attention Pooling
    学习一个注意力权重向量，对所有 token 做加权平均。

    输入: last_hidden_state (batch, seq_len, hidden_size)
          attention_mask    (batch, seq_len)
    输出: (batch, hidden_size)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # 计算注意力权重: (batch, seq_len, 1)
        attn_weights = self.attention(last_hidden_state)
        # 将 padding 位置设为 -inf (softmax 后为 0)
        attn_weights[attention_mask == 0] = float("-inf")
        # softmax 归一化
        attn_weights = torch.softmax(attn_weights, dim=1)
        # 加权求和: (batch, seq_len, hidden_size) → (batch, hidden_size)
        pooled = torch.sum(last_hidden_state * attn_weights, dim=1)
        return pooled


class GEMPooling(nn.Module):
    """
    Generalized Mean (GeM) Pooling
    通过可学习的参数 p 控制 pooling 行为:
      p=1 → Mean Pooling
      p→∞ → Max Pooling

    输入: last_hidden_state (batch, seq_len, hidden_size)
          attention_mask    (batch, seq_len)
    输出: (batch, hidden_size)
    """

    def __init__(self, p: float = 3.0):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # clamp 防止负数的小数次幂出现 nan
        clamped = torch.clamp(last_hidden_state, min=1e-6)
        # x^p 后加权求和
        sum_embeddings = torch.sum(clamped.pow(self.p) * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        # 求平均后开 1/p 次方
        gem_embeddings = (sum_embeddings / sum_mask).pow(1.0 / self.p)
        return gem_embeddings


# ============================================================
# Pooling 工厂函数
# ============================================================

def get_pooling(pooling_type: str, hidden_size: int = 768) -> nn.Module:
    """
    根据名称返回对应的 Pooling 层。

    Args:
        pooling_type: "mean" | "cls" | "max" | "attention" | "gem"
        hidden_size: 模型隐层维度 (attention pooling 需要)

    Returns:
        nn.Module: Pooling 层实例
    """
    pooling_map = {
        "mean": MeanPooling,
        "cls": CLSPooling,
        "max": MaxPooling,
        "attention": lambda: AttentionPooling(hidden_size),
        "gem": GEMPooling,
    }
    if pooling_type not in pooling_map:
        raise ValueError(
            f"不支持的 Pooling 类型: '{pooling_type}'. "
            f"可选: {list(pooling_map.keys())}"
        )
    creator = pooling_map[pooling_type]
    # 如果是 class (不含参数) 直接实例化，lambda 调用返回实例
    return creator() if callable(creator) else creator


# ============================================================
# FeedbackModel 主模型
# ============================================================

class FeedbackModel(nn.Module):
    """
    Feedback Prize 竞赛模型。

    结构:
      Text → Tokenizer → Backbone (DeBERTa) → Pooling → Dropout → Linear → 6个评分

    Args:
        model_name (str):   HuggingFace 模型名称 (如 "microsoft/deberta-v3-base")
        pooling_type (str): Pooling 策略 ("mean", "cls", "attention", "max", "gem")
        num_targets (int):  回归目标数量 (默认 6)
        hidden_dropout (float): 回归头前的 dropout 比率

    接口:
        forward(input_ids, attention_mask) → Tensor(batch_size, num_targets)
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        pooling_type: str = "mean",
        num_targets: int = 6,
        hidden_dropout: float = 0.0,
    ):
        super().__init__()

        # 加载 backbone 配置和模型
        self.config = AutoConfig.from_pretrained(model_name)
        # ⚠️ 显式 FP32 加载 backbone (必须!)
        #   新版 transformers 可能根据 config.torch_dtype 把模型加载为 FP16
        #   → 在 CUDA + AMP 下会报 "Attempting to unscale FP16 gradients"
        #   正确姿势: 参数保持 FP32，由 autocast 自动处理激活的 FP16
        self.backbone = AutoModel.from_pretrained(
            model_name, config=self.config, torch_dtype=torch.float32
        )
        self.backbone = self.backbone.float()  # 双保险
        self.hidden_size = self.config.hidden_size

        # Pooling 层 (A-W2-2 + A-W2-3)
        self.pooling = get_pooling(pooling_type, self.hidden_size)
        self.pooling_type = pooling_type

        # 回归头: Linear(hidden_size, num_targets)
        self.dropout = nn.Dropout(hidden_dropout)
        self.regression_head = nn.Linear(self.hidden_size, num_targets)

        # 初始化回归头权重
        self._init_weights(self.regression_head)

    def _init_weights(self, module: nn.Module):
        """Xavier 初始化 Linear 层"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播。

        Args:
            input_ids:      (batch_size, seq_len) — token IDs
            attention_mask:  (batch_size, seq_len) — 1=真实token, 0=padding

        Returns:
            predictions: (batch_size, num_targets) — 6 个评分预测
        """
        # Step 1: Backbone forward → last_hidden_state (batch, seq_len, hidden_size)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Step 2: 仅在 MPS 上强制转 fp32
        #   - MPS: DeBERTa 某些算子输出 fp16 会引起 mps.add dtype 不匹配崩溃，必须转 fp32
        #   - CUDA + AMP: 不要 .float()! autocast 下 backbone 在 fp16 context，
        #     强制 .float() 会让后续 Pooling/Linear 的中间激活脱离 autocast 追踪，
        #     导致反向时 backbone 的梯度仍为 FP16，触发
        #     "Attempting to unscale FP16 gradients" 错误。
        #   - CPU: 一直是 fp32，无需转换
        if last_hidden_state.device.type == "mps":
            last_hidden_state = last_hidden_state.float()

        # Step 3: Pooling → (batch, hidden_size)
        pooled = self.pooling(last_hidden_state, attention_mask)

        # Step 4: Dropout + Linear → (batch, num_targets)
        pooled = self.dropout(pooled)
        predictions = self.regression_head(pooled)

        return predictions

    def freeze_backbone(self, num_layers: int = -1):
        """
        冻结 backbone 参数 (用于后续 Stage 3-5 的冻结策略实验)。

        Args:
            num_layers: 冻结前 N 层。-1 表示冻结全部 backbone。
        """
        if num_layers == -1:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            # 冻结 embeddings
            for param in self.backbone.embeddings.parameters():
                param.requires_grad = False
            # 冻结前 N 层 encoder
            if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
                for i, layer in enumerate(self.backbone.encoder.layer):
                    if i < num_layers:
                        for param in layer.parameters():
                            param.requires_grad = False

    def unfreeze_backbone(self):
        """解冻全部 backbone 参数"""
        for param in self.backbone.parameters():
            param.requires_grad = True


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("🧪 FeedbackModel 单元测试")
    print("=" * 60)

    # 检测设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"设备: {device}")

    # 测试所有 Pooling 策略
    pooling_types = ["mean", "cls", "max", "attention", "gem"]

    for pooling_type in pooling_types:
        print(f"\n--- 测试 Pooling: {pooling_type} ---")
        model = FeedbackModel(
            model_name="microsoft/deberta-v3-base",
            pooling_type=pooling_type,
            num_targets=6,
            hidden_dropout=0.1,
        )
        model = model.to(device)
        model.eval()

        # 模拟输入
        batch_size = 2
        seq_len = 128
        fake_input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        fake_attention_mask = torch.ones(batch_size, seq_len, device=device)
        # 模拟部分 padding
        fake_attention_mask[:, 100:] = 0

        with torch.no_grad():
            output = model(fake_input_ids, fake_attention_mask)

        print(f"  输入 shape  : input_ids={fake_input_ids.shape}, mask={fake_attention_mask.shape}")
        print(f"  输出 shape  : {output.shape}  ← 期望 ({batch_size}, 6)")
        assert output.shape == (batch_size, 6), f"输出 shape 不正确: {output.shape}"
        print(f"  ✅ 通过")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 测试参数统计
    print(f"\n--- 参数统计 ---")
    model = FeedbackModel("microsoft/deberta-v3-base", pooling_type="mean")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量  : {total:,} ({total/1e6:.1f}M)")
    print(f"  可训练参数: {trainable:,} ({trainable/1e6:.1f}M)")

    # 测试冻结/解冻
    print(f"\n--- 冻结/解冻测试 ---")
    model.freeze_backbone()
    trainable_frozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  冻结后可训练参数: {trainable_frozen:,}")

    model.unfreeze_backbone()
    trainable_unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  解冻后可训练参数: {trainable_unfrozen:,}")
    assert trainable_unfrozen == trainable, "解冻后参数量不一致"
    print(f"  ✅ 冻结/解冻正常")

    print(f"\n🎉 FeedbackModel 全部测试通过!")
