学习 transformer：

我的环境：WSL (Ubuntu 24.04)

```bash
cd ~/Ai-lab/nano-gpt

# 准备数据（下载一次，约 673MB；处理 2 亿字符）
uv run nano-gpt-prepare --max-chars 200000000

# 训练（默认 2000 步，输出到 checkpoints/run）
uv run nano-gpt-train

# 命令行续写
uv run nano-gpt-sample \
  --model-dir checkpoints/run/final \
  --prompt "Once upon a time" \
  --num-samples 3

# 浏览器交互界面（推荐）
uv run nano-gpt-web
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
| 上下文 `n_positions` | 512                                   |
| 词表                 | 没用 BPE，每个字符是做一个token，75个 |
| Dropout              | 0.1                                   |
| 参数量               | **10,872,960**                        |

> 早期实验使用 256 上下文，参数量为 10,774,656；对应 checkpoint 仍然可以加载。

完整的模型说明（用途、数据、训练、评估与局限）见 [ModelCard.md](ModelCard.md)。

结构就是标准的 decoder-only Transformer：

token embedding + position embedding → 6 × (causal self-attention → residual/LayerNorm → MLP → residual/LayerNorm) → lm_head

随机初始化时 loss ≈ ln(75) ≈ **4.32**。

## 训练配置

训练超参的默认值同样来自 `src/config.py`：

- AdamW，`lr=6e-4`，`weight_decay=0.1`
- 200 步 warmup + cosine 衰减
- 梯度裁剪 1.0
- batch size 32，上下文 512 → 每步 16,384 个 token
- bf16 混合精度
- 每 200 步在固定验证集（2000 个窗口）上评估

## 采样参数

`nano-gpt-sample` 暴露的四个旋钮（默认值同样在 `config.py` 里）：

| 参数                   | 作用                              | 建议       |
| ---------------------- | --------------------------------- | ---------- |
| `--temperature`        | 缩放 logits；越低越保守           | 0.7 ~ 0.9  |
| `--top-k`              | 只保留概率最高的 k 个字符         | 40 ~ 100   |
| `--top-p`              | 保留累计概率达到 p 的最小字符集合 | 0.9 ~ 0.95 |
| `--repetition-penalty` | 惩罚近期出现过的 token，减少复读  | 1.0 ~ 1.2  |

## 网页界面

```bash
uv run nano-gpt-web
```

打开 <http://localhost:7860>。页面会自动扫描 `checkpoints/*/final` 和
`checkpoints/*/checkpoint-*`，可以：

- 下拉切换模型，查看上下文长度、参数量、训练步数
- 流式查看续写结果
- 勾选「对比两个模型」，用相同 prompt 和 seed 左右并排比较
- 调整 temperature / top-k / top-p / repetition penalty / max new tokens / seed
- 点击「续写输出」在现有文本后继续写

这是 TinyStories **字符级续写模型**，不是指令/聊天模型，所以它只会顺着 prompt
继续编故事。服务默认只监听 `127.0.0.1`，在 WSL 里启动后直接用 Windows 浏览器访问
`localhost` 即可。

## 文件结构

```
ModelCard.md        模型卡：用途 / 数据 / 训练 / 评估 / 局限
src/
  __init__.py      包的版本信息
  config.py        所有默认配置：路径 / 数据 / 模型 / 训练 / 采样 / 网页
  tokenizer.py     字符级 tokenizer（编码/解码/保存）
  prepare_data.py  下载 + 清洗 + 编码成 .bin
  data.py          memmap 数据集，切固定长度窗口
  model.py         按模型配置构造 GPT2LMHeadModel
  train.py         Trainer 训练入口
  generate.py      共享生成核心（逐 token 流式采样）
  sample.py        命令行续写
  registry.py      扫描、加载、缓存 checkpoints
  web.py           Gradio 网页界面
scripts/
  check_env.py     检查 GPU 与 bf16 支持
  analyze_run.py   从 run 目录打印 loss 表
data/               （git 忽略）原始 parquet + 处理后的 token
checkpoints/        （git 忽略）训练日志与权重
```
