# Kaggle Notebook 完整操作手册

> 针对本项目（Feedback Prize - English Language Learning）的 Kaggle 实战工作流
>
> 适用代码：`notebooks/kaggle_baseline.py`（训练+推理一体）、`notebooks/kaggle_inference_only.py`（仅推理）
>
> 最后更新：2026-04-17

---

## 🎯 一张图看懂 Kaggle 的"输出保存"

```
你在 Kaggle Notebook 里运行代码
              │
              ▼
  写出到 /kaggle/working/ 的文件 ───┐
                                    │
         ┌──────────────┴──────────────┐
         │                             │
    【交互式 Run】                 【Save Version（Commit）】
  （点左上 Run All 跑）           （点右上 "Save Version" 按钮）
         │                             │
         ▼                             ▼
  session 关闭即消失 ❌         永久存入该 Version 的 Output 区 ✅
  （除非立刻手动下载）          （可挂载到别的 Notebook、可提交比赛）
```

**核心结论（务必牢记）：**

| 场景 | 操作 | 文件去向 |
|------|------|----------|
| 交互式调试 | 左上 `Run All` / 逐 Cell 运行 | 只在 `/kaggle/working/` 临时存在，**关闭 session 就丢** |
| 正式产出权重/submission | 右上 `Save Version` → **Save & Run All (Commit)** | 跑完后**自动归档**到该 Version 的 Output |
| 提交到比赛 | 在归档后的 Version 的 Output 区点 `Submit to Competition` | 进入 Leaderboard |

> ⚠️ Kaggle 有两种 Save Version 模式，**一定要选对**：
> - **Save & Run All (Commit)** ← 我们要的（从头跑全部 Cell，结果归档）
> - **Quick Save** ← 只保存代码快照，**不重新运行，不产生 Output**

---

## 📁 本项目的路径约定

代码中已做好 Kaggle 自动检测：

```python
# notebooks/kaggle_baseline.py (约 80 行处)
if os.path.exists("/kaggle/input/competitions/feedback-prize-english-language-learning"):
    CFG.data_dir   = "/kaggle/input/competitions/feedback-prize-english-language-learning"
    CFG.output_dir = "/kaggle/working"     # ← 所有产物都落在这里
```

运行完成后，`/kaggle/working/` 会有：

```
/kaggle/working/
├── fold0_best.pth         # ~700MB × 5 ≈ 3.5GB
├── fold1_best.pth
├── fold2_best.pth
├── fold3_best.pth
├── fold4_best.pth
└── submission.csv         # ← 比赛提交文件（text_id + 6 列分数）
```

> 💡 `/kaggle/working` 容量上限 **20 GB**；5 个 deberta-v3-base 权重约 3.5GB，充足。

---

## 🚀 工作流 A：首次训练并提交（kaggle_baseline.py）

### Step 1 | 创建 Notebook

1. 打开比赛主页：<https://www.kaggle.com/competitions/feedback-prize-english-language-learning>
2. 右上角 `Code` → `New Notebook`（或 `Join Competition` 后 `+ New Notebook`）
3. 右侧面板 `Add Input` 已自动挂载比赛数据，路径就是
   `/kaggle/input/competitions/feedback-prize-english-language-learning/`

### Step 2 | 配置运行环境

右侧 **Notebook Options** 面板：

| 选项 | 推荐值 | 备注 |
|------|--------|------|
| **Accelerator** | **GPU T4 x2** 或 **GPU P100** | 必须开 GPU；T4 x2 显存更大更稳 |
| **Persistence** | `Files only` 或 `No persistence` | 我们不依赖持久化 |
| **Environment** | `Pin to original environment` | 避免 transformers 版本漂移再次引发 FP16 BUG |
| **Internet** | 训练时 **On**；提交时按比赛规则 | 本比赛允许联网；若改 Code Competition 需关闭 |

### Step 3 | 粘贴/导入代码

**方案 1（推荐，简单）**：直接把本地 `notebooks/kaggle_baseline.py` 全文复制 → 粘贴进 Notebook 的第一个 Cell，点 Run。

**方案 2（进阶）**：把 `src/` 整体打包成 Kaggle **Dataset**，挂载后 `sys.path.insert(0, "/kaggle/input/xxx")`，再 `from src.train import main`。首次建议先用方案 1。

### Step 4 | 先小规模试跑（省时间）

在配置区临时把下面几项调小，跑通 5~10 分钟确认无异常：

```python
CFG.epochs  = 1
CFG.n_folds = 2        # 只跑 2 折
CFG.batch_size = 4     # 若显存紧张
```

确认 `参数 dtype 分布: {'torch.float32': XXX}` ✅、loss 正常下降 ✅ 后再撤回完整参数。

### Step 5 | 正式跑：Save Version (Commit)

**关键一步**，决定文件是否被保存：

1. 把 `epochs` / `n_folds` 改回正式值（例如 `epochs=4, n_folds=5`）
2. 右上角点 **Save Version**
3. 选 **Save & Run All (Commit)** ← ✅ 就是这个
4. 可填 Version Name，例如 `baseline-deberta-v3-base-5fold-ep4`
5. 点 **Save**，Kaggle 开始在后台从头跑一遍（你可以关掉浏览器，稍后回来看）

> 🕒 预计耗时：T4×2 上 DeBERTa-v3-base 5-Fold × 4 epoch ≈ **2–3 小时**；Kaggle 单次 Session 上限 12 小时，足够。

### Step 6 | 查看 Output 并提交比赛

Commit 跑完后：

1. Notebook 顶部切到该 **Version** → 右侧 `Output` 标签
2. 看到 `submission.csv` 和 5 个 `fold*_best.pth`
3. 在 `submission.csv` 行右侧点 **Submit to Competition**
4. 等待打分 → Leaderboard 显示 Public Score

> ⚠️ 若 `Submit to Competition` 按钮灰着，大概率是：
> - 文件不叫 `submission.csv`（查 `CFG.output_dir` 和保存文件名）
> - Submission 格式不对（应为 `text_id, cohesion, syntax, vocabulary, phraseology, grammar, conventions`）
> - 本次比赛已关闭 Late Submission（Feedback Prize 3 可能只能看分数，不计榜）

---

## ♻️ 工作流 B：复用已训好的权重（kaggle_inference_only.py）

**适用场景**：想调整推理逻辑 / 换推理超参 / 不想再等 3小时训练。

### Step 1 | 把权重打包为 Kaggle Dataset

1. 打开刚才 Commit 完成的 Notebook Version 的 Output
2. 勾选 5 个 `fold*_best.pth`
3. 右上角 `⋯` → `Create a new Dataset`（或者 `Add to Existing Dataset`）
4. 起个名字，例如 `feedback-baseline-weights`
5. 等 Dataset 上传完成（几分钟）

### Step 2 | 新建推理 Notebook

1. 新建 Notebook（同样 GPU + Pin environment）
2. `Add Input`：
   - 比赛数据：`feedback-prize-english-language-learning`
   - 刚创建的权重 Dataset：`feedback-baseline-weights`
3. 粘贴 `notebooks/kaggle_inference_only.py`
4. 确认脚本里路径：

```python
WEIGHTS_DIR = "/kaggle/input/feedback-baseline-weights"   # 你的 Dataset 名
OUTPUT_DIR  = "/kaggle/working"
```

### Step 3 | Commit & 提交

同工作流 A 的 Step 5、Step 6。推理一般几分钟跑完。

---

## 🧪 常见问题速查（FAQ）

### Q1：我 Run All 跑完关掉网页再打开，文件没了？

A：正常。没 `Save Version (Commit)`，只是 interactive session，文件会在 session timeout（~20 分钟无操作）后清空。**务必用 Commit**。

### Q2：`Save Version` 选了但找不到 Output？

A：检查：
- 选的是 **Save & Run All**（不是 Quick Save）
- 该 Version 状态为 **Complete**（Running/Failed 都没有 Output）
- 在右上 Version 下拉切到**正确的 Version**，不是最初的 Draft

### Q3：Commit 跑到一半超时？

A：对策：
- 降 `n_folds`（5 → 4）或 `epochs`（4 → 3）
- `max_len` 从 1024 降到 768 / 512
- 开 `batch_size=8, grad_accum=2`（等效 bs=16 但省显存）
- 分两次 Commit：先跑 Fold 0-2，再跑 Fold 3-4（需要改代码手动断点续跑）

### Q4：显示 "exceed 20GB quota"？

A：清理 `/kaggle/working/`：
- 只保留 `submission.csv`，训完把 `.pth` 单独打包到 Dataset 再删
- 或在训练末尾加 `os.remove(...)` 只保留最好的那个 Fold

### Q5：想让队友直接复用我的 Notebook？

A：右上 `Share` → `Public` 或 `Collaborators` → 加队友的 Kaggle 用户名，给 `Can Edit` 权限。

### Q6：怎么确认 GPU 真的启用了？

A：代码开头会打印：
```
🚀 Using device: cuda
  GPU: Tesla T4  (x2)
  参数 dtype 分布: {'torch.float32': 200}
```
若显示 `cpu` 说明 Accelerator 没设置成 GPU。

### Q7：同队 3 人怎么分工用 Kaggle？

A：推荐：
- 每人一个 Fork 的 Notebook 做实验（改配置跑不同实验）
- 所有人的权重都上传成各自的 Dataset
- 最终融合时新建一个 Notebook 挂载所有 Dataset 做 Blend

---

## ✅ 提交前 Checklist

- [ ] 代码顶部 `CFG.output_dir = "/kaggle/working"` ✔
- [ ] `Accelerator = GPU` ✔
- [ ] 试跑一次（小规模）确认 dtype 诊断全 fp32、loss 正常下降 ✔
- [ ] 恢复正式超参：`epochs=4, n_folds=5, max_len=1024` ✔
- [ ] **右上 Save Version → Save & Run All (Commit)** ✔
- [ ] Commit 完成后 Output 区出现 `submission.csv` ✔
- [ ] Output 里点 `Submit to Competition` → Leaderboard 显示分数 ✔

---

## 📚 官方参考

- Kaggle Notebook 基础：<https://www.kaggle.com/docs/notebooks>
- 比赛提交：<https://www.kaggle.com/docs/competitions#submitting-a-submission>
- Dataset 创建：<https://www.kaggle.com/docs/datasets>

---

**项目文档索引：**
- `docs/项目执行计划草案.md` — 总体计划
- `docs/Kaggle操作手册.md` — 本文
- `notebooks/kaggle_baseline.py` — 训练+推理一体脚本
- `notebooks/kaggle_inference_only.py` — 纯推理脚本（复用权重）
