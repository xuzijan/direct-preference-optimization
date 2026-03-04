# DPO 复现快速上手指南

> 按 6 步递进：Fork → 环境隔离 → 跑通示例 → 整理 I/O 与配置 → 统一评估与 AmbiCoding

---

## 论文背景

**论文**：*Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (arXiv:2305.18290)  
**机构**：Stanford University  
**核心思想**：直接从偏好数据优化策略，无需显式 reward model，用 Bradley-Terry 形式将 preference loss 转化为可微目标。

**评估场景**：偏好学习（SFT → DPO 两阶段）、人类偏好对齐  

**当前项目**：`xuzijan/direct-preference-optimization` 是 `eric-mitchell/direct-preference-optimization` 的 fork。

---

## Step 1：Fork 到自己仓库

- [x] Fork `eric-mitchell/direct-preference-optimization` → `xuzijan/direct-preference-optimization`（或你的 GitHub 用户名）
- [ ] 在 README 或 commit 中注明 fork 来源与对应 commit
- [ ] `.gitignore` 排除大文件（模型权重、checkpoint 等），只同步代码和配置

**建议 .gitignore 新增：**

```gitignore
*.safetensors
*.bin
*.pt
*.pth
.cache/
models/
checkpoints/
wandb/
```

---

## Step 2：环境隔离

每个 baseline 单独环境，避免依赖冲突。

**目录结构：**

```
/root/autodl-tmp/
├── direct-preference-optimization/   # 论文 3
│   ├── conda env dpo
│   ├── config/
│   ├── experiments/
│   └── scripts/
├── experiments/
│   ├── configs/
│   ├── data/
│   ├── eval/
│   └── outputs/
└── ...
```

**DPO 环境：**

```bash
cd /root/autodl-tmp/direct-preference-optimization
conda create -n dpo python=3.10 -y
conda activate dpo
pip install -r requirements.txt
# 若 peft/transformers 冲突，可降级: pip install "peft<0.10" "pyarrow<15"
```

**配置（可选）：**

- `wandb`：默认启用，可 `debug=true` 或 `WANDB_MODE=disabled` 禁用
- `local_dirs`：模型/数据缓存目录，默认 `[/scr-ssd, /scr, .cache]`

---

## Step 3：按原作者示例跑通一次

### 3.1 Mock 验证（完全离线，无需网络）

```bash
cd /root/autodl-tmp/direct-preference-optimization
conda activate dpo
python scripts/validate_mock.py
```

验证：mock 数据加载、batch 构建、随机初始化小模型前向、SFT 损失计算。

### 3.2 真实 API 验证（需网络 + GPU）

**SFT 阶段：**

```bash
# 小模型 gpt2（单 GPU）
python -u train.py model=gpt2 datasets=[hh] loss=sft exp_name=dpo_test \
  batch_size=4 eval_batch_size=4 trainer=BasicTrainer debug=true \
  n_examples=100 eval_every=50 sample_during_eval=false local_dirs=[.cache]

# 大模型 Pythia 2.8B（多 GPU FSDP）
python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=anthropic_dpo_pythia28 \
  gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 \
  trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16
```

**DPO 阶段：**

```bash
# 需先完成 SFT，得到 policy.pt
python -u train.py model=gpt2 datasets=[hh] loss=dpo loss.beta=0.1 \
  model.archive=/path/to/policy.pt exp_name=dpo_test \
  batch_size=4 eval_batch_size=4 trainer=BasicTrainer debug=true
```

### 3.3 论文官方数据与复现

**数据来源**：Hugging Face [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)、[stanfordnlp/SHP](https://huggingface.co/datasets/stanfordnlp/SHP)、[HuggingFaceH4/stack-exchange-preferences](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences)

**运行流程**：
```bash
# Mock 验证（离线）
python scripts/validate_mock.py

# 有网络时：SFT → DPO
python -u train.py model=pythia28 datasets=[hh] loss=sft ...
python -u train.py model=pythia28 datasets=[hh] loss=dpo loss.beta=0.1 model.archive=.../LATEST/policy.pt ...
```

**离线 mock 数据集**：`datasets=[mock]` 使用 `preference_datasets.get_mock()`，3 条内置样本，无需网络。

---

## Step 4：整理输入输出、重要参数、配置

### 4.1 输入格式（完整）

**SFT / DPO**

| 输入 | 类型 | 说明 | 来源 |
|------|------|------|------|
| 数据集 | str/list | hh、shp、se、mock | 命令行 `datasets=[hh]` |
| prompt | str | `\n\nHuman: ...\n\nAssistant:` | 数据字段 |
| chosen | str | 偏好回答 | 数据字段 |
| rejected | str | 非偏好回答（DPO 用） | 数据字段 |
| sft_target | str | SFT 目标回答 | 数据字段 |
| model | str | 模型名或路径 | 命令行 `model=gpt2` |

**偏好数据单条格式（内部）：**
```python
{
    "prompt1": {
        "responses": ["chosen", "rejected"],
        "pairs": [(0, 1)],  # (chosen_idx, rejected_idx)
        "sft_target": "chosen"
    }
}
```

### 4.2 输出格式（完整）

**SFT / DPO 训练**

| 输出 | 类型 | 说明 |
|------|------|------|
| 运行目录 | str | `{local_dir}/{user}/{exp_name}_{timestamp}/` |
| policy.pt | 文件 | 策略权重 `{run_dir}/LATEST/policy.pt` 或 `step-XXXX/policy.pt` |
| config.yaml | 文件 | 运行配置 |
| wandb 日志 | 可选 | 若 `wandb.enabled=true` |

**policy.pt 内容**：`state_dict`、`step_idx`、`metrics`

### 4.3 重要参数（完整）

| 参数 | 位置 | 默认 | 说明 |
|------|------|------|------|
| model | 命令行 | blank_model_fp32 | 模型配置，如 gpt2、pythia28 |
| datasets | 命令行 | [hh] | 数据集列表 |
| loss | 命令行 | sft | sft / dpo / ipo |
| loss.beta | 命令行 | - | DPO 温度，0.1~0.5 |
| batch_size | 命令行 | 4 | 训练 batch 大小 |
| eval_batch_size | 命令行 | 16 | 评估 batch 大小 |
| n_examples | 命令行 | null | 训练样本数（与 n_epochs 二选一） |
| n_epochs | 命令行 | 1 | 训练轮数 |
| eval_every | 命令行 | 20000 | 每 N 样本评估一次 |
| trainer | 命令行 | BasicTrainer | BasicTrainer / FSDPTrainer / TensorParallelTrainer |
| lr | config | 5e-7 | 学习率 |
| max_length | config | 512 | 最大序列长度 |
| seed | config | 0 | 随机种子 |

### 4.4 Hydra 配置文件（完整）

**路径**：`config/config.yaml`、`config/model/*.yaml`、`config/loss/*.yaml`

**主配置：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `exp_name` | str | 实验名称 |
| `seed` | int | 随机种子 |
| `batch_size` | int | 训练 batch |
| `eval_batch_size` | int | 评估 batch |
| `datasets` | list | 数据集列表 |
| `trainer` | str | 训练器 |
| `lr` | float | 学习率 |
| `max_length` | int | 最大长度 |
| `max_prompt_length` | int | 最大 prompt 长度 |
| `n_epochs` | int | 训练轮数 |
| `n_examples` | int | 训练样本数 |
| `eval_every` | int | 评估间隔 |
| `local_dirs` | list | 缓存目录 |
| `wandb.enabled` | bool | 是否启用 wandb |
| `debug` | bool | 调试模式（禁用 wandb、不保存 checkpoint） |

**模型配置：**

| 字段 | 说明 |
|------|------|
| `model.name_or_path` | 模型名或路径 |
| `model.block_name` | FSDP 块名，如 GPT2Block |
| `model.policy_dtype` | 策略 dtype |
| `model.archive` | SFT 权重路径（DPO 用） |

**损失配置（loss/dpo.yaml）：**

| 字段 | 说明 |
|------|------|
| `loss.beta` | DPO 温度 |
| `loss.reference_free` | 是否无参考模型 |
| `loss.label_smoothing` | 保守 DPO 噪声 |

### 4.5 脚本命令行参数

**validate_mock.py**

| 说明 |
|------|
| 无参数，完全离线验证 |

**train.py（Hydra 覆盖）**

| 示例 | 说明 |
|------|------|
| `model=gpt2` | 选择模型配置 |
| `datasets=[hh,mock]` | 数据集列表 |
| `loss=sft` / `loss=dpo` | 损失类型 |
| `loss.beta=0.1` | DPO beta |
| `exp_name=xxx` | 实验名 |
| `batch_size=4` | batch 大小 |
| `n_examples=100` | 训练样本数 |
| `trainer=BasicTrainer` | 训练器 |
| `debug=true` | 调试模式 |
| `local_dirs=[.cache]` | 缓存目录 |

---

## Step 5：统一评估与 AmbiCoding 适配

### 5.1 统一输入输出接口

- **统一输入**：`{id, query, context, options?, ...}` 的 dict
- **统一输出**：`{id, pred, ground_truth?, metadata?}` 的 dict
- **统一入口**：`run_baseline("dpo", config_path)` 分发到各实现

### 5.2 AmbiCoding 数据集转换

- 明确 AmbiCoding 原始格式
- 为 DPO 写转换脚本：`AmbiCoding → DPO 偏好格式`（prompt, chosen, rejected）
- 转换脚本需可复现（固定 seed、版本）

### 5.3 统一评估脚本

- 输入：各 baseline 的预测结果（统一格式）
- 输出：同一套指标（accuracy、exact match、preference 准确率等）
- 评估逻辑与 baseline 解耦

### 5.4 Prompt / 数据格式

**论文/代码中的格式**

- Prompt：`\n\nHuman: <prompt>\n\nAssistant:`
- 偏好数据：`preference_datasets.py` 中 `get_hh`、`get_shp`、`get_se`、`get_mock`
- 自定义数据：在 `get_dataset` 中新增 `get_xxx` 并注册

**建议**：在单独文件或 YAML 中保存数据格式说明，便于复现与对比。

---

## 数据流简图

```
偏好数据 (prompt, chosen, rejected) → Tokenize → Batch
        ↓
SFT: 监督 loss(chosen) → 更新 policy
        ↓
DPO: policy + reference → loss(chosen, rejected) → 更新 policy
        ↓
保存 policy.pt / step-XXXX/policy.pt
```

---

## 常见问题

| 问题 | 排查 |
|------|------|
| 网络不可用 | 使用 `datasets=[mock]` 或 `scripts/validate_mock.py` 离线验证 |
| peft Cache 导入错误 | 降级 `pip install "peft<0.10"` |
| pyarrow PyExtensionType 错误 | 降级 `pip install "pyarrow<15"` |
| 显存不足 | 减小 batch_size、增大 gradient_accumulation_steps、用 FSDPTrainer |
| 模型下载失败 | 设置 `local_dirs=[.cache]` 或使用本地模型路径 |

---

## 参考链接

- [DPO 论文](https://arxiv.org/abs/2305.18290)
- [原仓库](https://github.com/eric-mitchell/direct-preference-optimization)
