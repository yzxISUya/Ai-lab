---
language:
  - en
library_name: transformers
pipeline_tag: text-generation
tags:
  - gpt2
  - tiny-stories
  - text-generation
  - character-level
  - from-scratch
datasets:
  - karpathy/tinystories-gpt4-clean
---

# nano-gpt（TinyStories 字符级 GPT-2）

## 模型详情

- **模型名称**：nano-gpt
- **版本**：`checkpoints/run/final`（2026-10-22）
- **类型**：decoder-only Transformer，GPT-2 架构，从零随机初始化训练
- **参数量**：10,872,960（10.87M）
- **上下文长度**：512 tokens
- **词表大小**：75
- **分词器**：自定义字符级 `CharTokenizer`，一个字符 = 一个 token，`<|endoftext|>` 的 id 为 0
- **权重**：`model.safetensors`（约 42 MB）
- **框架**：PyTorch + Hugging Face Transformers

| 超参数               | 值                        |
| -------------------- | ------------------------- |
| 层数 `n_layer`       | 6                         |
| 注意力头数 `n_head`  | 6                         |
| 隐藏维度 `n_embd`    | 384                       |
| FFN 维度 `n_inner`   | 1536                      |
| 上下文 `n_positions` | 512                       |
| Dropout              | 0.1                       |
| 激活函数             | `gelu_new`                |
| 权重共享             | 输入嵌入与 `lm_head` 共享 |

字符表（共 74 个字符，另加 `<|endoftext|>`）：

```text
\n !"$',-.0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
```

词表不包含以下字符：

```text
# % & ( ) * + / < = > @ [ \ ] ^ _ ` { | } ~
```

编码时遇到词表外字符会映射为 `<|endoftext|>`。

## 用途

### 适用场景

- TinyStories 风格的英文故事续写
- Transformer / 语言模型教学与实验
- 对比不同训练步数、上下文长度、采样参数的效果

### 不适用场景

- 指令问答、多轮对话、聊天机器人
- 事实查询、知识问答、代码生成
- 非英文任务
- 生产环境或任何高风险决策
- 需要安全对齐、内容审核的场景

本模型不是 chat / instruct 模型，没有对话模板，也不会遵循指令；它只会根据 prompt 做统计意义上的续写。

## 训练数据

- 数据集：[`karpathy/tinystories-gpt4-clean`](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean)
- 按文件顺序读取约 2 亿字符（`max_chars = 200,000,000`）
- 打乱后按 98% / 2% 划分训练集与验证集（`seed = 42`）
- 字符级编码，每个 token 占 1 字节（`uint8`）

| 划分   |  故事数 |    token 数 |
| ------ | ------: | ----------: |
| 训练集 | 244,199 | 196,235,649 |
| 验证集 |   4,983 |   4,014,682 |

数据统计同样保存在 `checkpoints/run/final/data_meta.json`。

## 训练过程

- 优化器：AdamW
- 学习率：`6e-4`，weight decay `0.1`
- 学习率调度：200 步 warmup + cosine 衰减
- 梯度裁剪：1.0
- batch size：32
- 梯度累积：1
- 精度：bf16 混合精度
- 训练步数：2,000
- 每步 token 数：32 × 512 = 16,384
- 随机种子：1337
- 评估：每 200 步，在固定的 2,000 个验证窗口上计算 loss
- checkpoint：每 500 步保存
- 硬件：NVIDIA RTX 5060 Laptop 8GB

完整训练参数见 `checkpoints/run/final/run_config.json`。

## 评估

验证集为留出的 2% 故事；评估使用固定的 2,000 个非重叠窗口。

|            步数 |  验证 loss |
| --------------: | ---------: |
| 0（随机初始化） |     4.4139 |
|             200 |     2.2411 |
|             400 |     1.4919 |
|             600 |     1.1349 |
|             800 |     0.9598 |
|            1000 |     0.8430 |
|            1200 |     0.7787 |
|            1400 |     0.7410 |
|            1600 |     0.7166 |
|            1800 |     0.7031 |
|            2000 | **0.7000** |

- 最终训练 loss：0.7468
- 最终验证 loss：0.7000
- 验证 perplexity：约 **2.01**（按字符计算）

这些指标只反映字符级下一 token 预测能力，不代表模型在语义、事实或安全性上的表现。

## 局限与风险

- **领域窄**：只见过 TinyStories，写其他主题容易跑偏
- **字符级**：词表只有 75 个符号，部分标点和符号无法表示
- **上下文短**：最多 512 个字符，长文本需要滑窗，前后可能不一致
- **没有对齐**：未做指令微调、RLHF、安全对齐或红队测试
- **可能生成错误内容**：语法错误、事实错误、重复、逻辑不连贯都可能出现
- **没有偏见评估**：训练数据可能包含社会偏见，模型可能复现
- **不是聊天模型**：不会回答问题，只会续写

## 使用方式

命令行：

```bash
uv run nano-gpt-sample \
  --model-dir checkpoints/run/final \
  --prompt "Once upon a time" \
  --num-samples 3
```

网页界面：

```bash
uv run nano-gpt-web
# 打开 http://localhost:7860
```

Python（在项目根目录执行）：

```python
from transformers import GPT2LMHeadModel

from src.generate import GenerationConfig, generate_story
from src.tokenizer import CharTokenizer

model_dir = "checkpoints/run/final"
tokenizer = CharTokenizer.load(f"{model_dir}/tokenizer.json")
model = GPT2LMHeadModel.from_pretrained(model_dir).eval()

gen = GenerationConfig(max_new_tokens=200, temperature=0.8, top_k=50, top_p=0.95)
print(generate_story(model, tokenizer, "Once upon a time", gen, "cpu"))
```

## 许可证

本地仓库没有指定模型权重或代码的许可证。数据集
[`karpathy/tinystories-gpt4-clean`](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean)
的使用请遵循其数据集卡上的条款。发布模型前建议明确选择一个许可证。

## 引用

如果使用本模型或相关数据，建议引用：

- Eldan, R., & Li, Y. (2023). _TinyStories: How Small Can Language Models Be and Still Speak Coherent English?_ arXiv:2305.07759.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). _Language Models are Unsupervised Multitask Learners._ OpenAI.
- Mitchell, M., et al. (2019). _Model Cards for Model Reporting._ arXiv:1810.03993.
