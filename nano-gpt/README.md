学习 transformer：

我的环境：WSL (Ubuntu 24.04)

```bash
cd ~/Ai-lab/nano-gpt

# 准备数据（下载一次，约 673MB；处理 2 亿字符）
uv run nano-gpt-prepare --max-chars 200000000

# 训练（默认 1000 步；这里跑 2000 步）
uv run nano-gpt-train --max-steps 2000 --out-dir checkpoints/run-2k

# 续写故事
uv run nano-gpt-sample \
  --model-dir checkpoints/run-2k/final \
  --prompt "Once upon a time" \
  --num-samples 3
```

环境检查：

```bash
uv run python scripts/check_env.py
```

## 超参配置

所有默认值（路径、数据、模型结构、训练、采样）都集中定义在
[`src/config.py`](src/config.py)；命令行参数会覆盖对应默认值。

模型结构：

| 超参数               | 值                                    |
| -------------------- | ------------------------------------- |
| 层数 `n_layer`       | 6                                     |
| 注意力头数 `n_head`  | 6                                     |
| 隐藏维度 `n_embd`    | 384（每头 64 维）                     |
| FFN 维度             | 1536（4 × n_embd）                    |
| 上下文 `n_positions` | 256                                   |
| 词表                 | 没用 BPE，每个字符是做一个token，75个 |
| Dropout              | 0.1                                   |
| 参数量               | **10,774,656**                        |

结构就是标准的 decoder-only Transformer：

token embedding + position embedding → 6 × (causal self-attention → residual/LayerNorm → MLP → residual/LayerNorm) → lm_head

随机初始化时 loss ≈ ln(75) ≈ **4.32**。

## 训练配置

训练超参的默认值同样来自 `src/config.py`：

- AdamW，`lr=6e-4`，`weight_decay=0.1`
- 200 步 warmup + cosine 衰减
- 梯度裁剪 1.0
- batch size 64，上下文 256 → 每步 16,384 个 token
- bf16 混合精度
- 每 200 步在固定验证集（2000 个窗口）上评估

在我的 RTX 5060 Laptop 8GB 上实测5分半

## 采样参数

`nano-gpt-sample` 暴露的四个旋钮（默认值同样在 `config.py` 里）：

| 参数                   | 作用                              | 建议       |
| ---------------------- | --------------------------------- | ---------- |
| `--temperature`        | 缩放 logits；越低越保守           | 0.7 ~ 0.9  |
| `--top-k`              | 只保留概率最高的 k 个字符         | 40 ~ 100   |
| `--top-p`              | 保留累计概率达到 p 的最小字符集合 | 0.9 ~ 0.95 |
| `--repetition-penalty` | 惩罚近期出现过的 token，减少复读  | 1.0 ~ 1.2  |

## 文件结构

```
src/
  __init__.py      包的版本信息
  config.py        所有默认配置：路径 / 数据 / 模型 / 训练 / 采样
  tokenizer.py     字符级 tokenizer（编码/解码/保存）
  prepare_data.py  下载 + 清洗 + 编码成 .bin
  data.py          memmap 数据集，切固定长度窗口
  model.py         按模型配置构造 GPT2LMHeadModel
  train.py         Trainer 训练入口
  sample.py        加载模型并续写
scripts/
  check_env.py     检查 GPU 与 bf16 支持
  analyze_run.py   从 run 目录打印 loss 表
data/               （git 忽略）原始 parquet + 处理后的 token
checkpoints/        （git 忽略）训练日志与权重
```
