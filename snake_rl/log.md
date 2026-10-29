# Snake RL 实验日志

> 维护日期：2026-10-29  
> 项目：`snake_rl` — 用贪吃蛇做 DQN / 强化学习练习  
> 本文件是**总日志**。每次实验追加一节，记录假设、变量、配置、结果、分析和下一步。  
> 更细的原始记录在 `teach/logs/`（本地目录，未入库）；训练产物在 `runs/`（未入库）。

---

## 0. 当前结论（TL;DR）

| 问题                      | 当前答案                                                                                                  |
| ------------------------- | --------------------------------------------------------------------------------------------------------- |
| 11 维手工特征能做到多少？ | Stage 1 基线 **27.14 / 27.21**（20k / 10k 两套 100 局 eval）；3 个训练种子平均 **25.95 ± 2.38**           |
| Stage 2 最好的算法配置？  | **目标网络 + Double DQN，不开 Huber**；3 seed 平均 **26.52 ± 1.08**，最差 seed 25.41                      |
| Double 是否显著提高均值？ | **没有**。相对 baseline 平均 +0.57，95% CI [-3.30, +4.43]；主要收益是**跨 seed 方差更小、最差 seed 更高** |
| Huber 有用吗？            | **没有**。Huber − Double 平均 -0.28，CI 跨 0；跨 seed 方差反而更大（2.28 vs 1.08）                        |
| 11 维观察的上限在哪？     | 约 **27 分**；最终评估中 **95–100/100 死于 self collision**                                               |
| 全棋盘 CNN 学会了吗？     | **没有**。500 局短训练 eval 0.17；45 分钟长训练 eval 0.00（100/100 timeout，退化成绕圈）                  |
| 下一步最值得做什么？      | **10×10 小棋盘课程学习**（仍然纯 RL、全棋盘 CNN），先验证能否学会找食物，再扩到 20×20                     |

---

## 1. 评测协议与术语

### 1.1 环境（除特别说明外）

| 项             | 值                                                  |
| -------------- | --------------------------------------------------- |
| 棋盘           | 20×20                                               |
| 初始蛇长       | 3                                                   |
| timeout factor | 100（上限 = 100 × 当前蛇长）                        |
| 奖励           | food +10，death -10，step 0，timeout -10            |
| 动作           | 相对动作：直行 / 右转 / 左转                        |
| 终止           | 撞墙 / 撞自己 / 胜利 = terminated；超时 = truncated |

### 1.2 评估协议

- **greedy** 策略（epsilon=0），不渲染；
- 最终评估默认 **100 局**；
- 两组固定评估种子：
  - `20_000 + i`（`final_eval.json`）
  - `10_000 + i`（独立复核 `eval.json`）
- 训练中每 50 局做 20 局 eval，只用于看曲线，**不作为最终结论**；
- 同一训练种子的不同算法版本用**同一组评估种子**做配对比较；
- 多 seed 汇总：每个训练种子先取 20k/10k 的均值，再对 3 个训练种子求 mean ± sample SD；
- 报告指标：mean / median / max / min / std、mean steps、score/step、死因分布。

### 1.3 重要教训

1. **单训练种子不可靠。** baseline 的 3 个 seed 平均 25.95，但 SD 2.38；单 seed 的 -2.5 完全可能是种子噪声。
2. **20 局 eval 噪声很大。** 同一策略在不同 checkpoint 上可以差 ±5；结论至少用 100 局。
3. **不要用训练 score 代替 greedy eval。** 训练时带 epsilon 探索，分数不可直接比较。
4. **loss 不能跨 loss 函数直接比。** MSE 与 Huber 的尺度不同。

---

## 2. 实验总表

| ID  | 日期          | 阶段        | 优化方向                 | 观察 / 模型          | 训练种子 | 最终 eval（20k / 10k）                         | 结论                                  |
| --- | ------------- | ----------- | ------------------------ | -------------------- | -------- | ---------------------------------------------- | ------------------------------------- |
| E0  | 2026-10-25    | Stage 0     | 随机基线                 | 无                   | —        | 0.11（20×20, timeout 100）                     | 环境 sanity check，下界               |
| E1  | 2026-10-25    | Stage 1     | 复刻 DQN 基线            | 11 维 / MLP 11→256→3 | 0        | **27.14 / 27.21**                              | 可用基线，平台约 27                   |
| E2  | 2026-10-26    | Stage 2-01  | 目标网络                 | 11 维 / MLP          | 0        | 24.34 / 24.72                                  | 单 seed 看起来变差；后来证明不可靠    |
| E3  | 2026-10-26    | Stage 2-02  | + Double DQN             | 11 维 / MLP          | 0        | 28.33 / 26.82                                  | 收回目标网络的损失，约回到基线        |
| E4  | 2026-10-26    | Stage 2-02b | 多 seed 复核             | 11 维 / MLP          | 0/1/2    | baseline 25.95 ± 2.38；Double **26.52 ± 1.08** | Double 不提均值，但方差更小、下限更高 |
| E5  | 2026-10-26    | Stage 2-03  | + Huber                  | 11 维 / MLP          | 0/1/2    | 26.24 ± 2.28                                   | 不提分，方差更大；不保留              |
| E6  | 2026-10-27    | Stage 3     | 全棋盘 CNN（短）         | 5×20×20 / BoardQNet  | 0        | **0.17**                                       | 几乎没学会找食物                      |
| E7  | 2026-10-27    | Stage 3     | 全棋盘 CNN（长，45 min） | 5×20×20 / BoardQNet  | 0        | **0.00**（100/100 timeout）                    | 退化成绕圈；1 小时内失败              |
| E8  | 2026-10-27~29 | 工程        | CUDA / GPU / replay 更新 | —                    | —        | 55 tests pass                                  | 基础设施完成，非科学实验              |
| E9  | 计划          | Stage 3     | 10×10 课程学习           | 5×10×10 / BoardQNet  | 0        | —                                              | 下一步最推荐                          |

---

## 3. 公共配置

### 3.1 11 维特征（Stage 1 / Stage 2）

| 项      | 值                                                      |
| ------- | ------------------------------------------------------- |
| 观察    | 11 维：3 个危险位 + 4 个朝向 one-hot + 4 个食物相对方位 |
| 网络    | MLP 11 → 256 → 3，ReLU                                  |
| 优化    | Adam，lr=1e-3，gamma=0.9，MSE（Huber 实验除外）         |
| replay  | 容量 100k，batch 1000                                   |
| 更新    | 每步 short-memory 单样本 + 每局 replay batch            |
| epsilon | 0.4 → 0，80 局（未做 epsilon 调度实验）                 |
| 训练    | 500 局；训练 seed 0（多 seed 实验为 0/1/2）             |
| 评估    | 100 局 greedy；20_000+i 和 10_000+i 两套                |

### 3.2 全棋盘 CNN（Stage 3）

| 项     | 值                                                                                         |
| ------ | ------------------------------------------------------------------------------------------ |
| 观察   | `(5, 20, 20)`：body / head / tail / food / border                                          |
| 网络   | BoardQNet：Conv(5→32) → Conv(32→64) → Conv(64→64, stride 2) → FC 256 → FC 3；约 1.70M 参数 |
| 优化   | Adam，lr=1e-3，gamma=0.9，MSE                                                              |
| 更新   | 每 2 个环境步从 replay 采样一个 minibatch；batch 128（短）/ 256（长）                      |
| replay | 20k（短）/ 50k（长），min_replay_size=1000                                                 |
| 算法   | 目标网络 + Double DQN；Huber 关闭；`bootstrap_truncated=False`                             |
| 设备   | RTX 5060 Laptop，torch 2.14.1+cu130                                                        |
| 训练   | 500 局（短）/ 最多 5000 局、45 分钟硬超时（长）                                            |

### 3.3 复现环境

```bash
cd ~/Ai-lab/snake_rl
uv sync                     # torch 2.14.1+cu130
uv run pytest               # 55 tests
uv run python -c "import torch; print(torch.cuda.is_available())"
```

---

## 4. 实验详情

### E0 · Stage 0 随机策略基线

- **目的**：验证环境、渲染、episode 结束和评估流程；给 DQN 一个下界。
- **变量**：无（随机策略）。
- **配置**：20×20，初始蛇长 3，timeout factor 100，100 局，seed 0。
- **结果**：

| 指标       |               值 |
| ---------- | ---------------: |
| mean       |             0.11 |
| median     |              0.0 |
| max        |                2 |
| mean steps |             74.1 |
| 结束方式   | wall 99 / self 1 |

其他配置（10×10、timeout 50/10）的结果见 `teach/logs/stage0-random-baseline.md`：
随机策略在 10×10 上 mean 0.21，但平均步数只有 18.1；小棋盘更容易撞墙。

- **分析**：随机策略几乎不会 self collision（蛇太短），主要死于撞墙；平均分接近 0。
- **结论/下一步**：环境流程正确；Stage 1 应该显著超过 0.11。

---

### E1 · Stage 1：11 维 DQN 基线

- **目的**：复刻参考 DQN，建立可复现、可评估的基线。
- **变量**：无（基线）。
- **配置**：见 3.1；无目标网络、MSE、每步 short-memory + 每局 long-memory、epsilon 80 局归零。
- **命令**：

```bash
OMP_NUM_THREADS=4 .venv/bin/python scripts/train_dqn.py \
  --episodes 500 --eval-every 50 --eval-episodes 20 --final-eval-episodes 100 \
  --checkpoint-every 100 --log-every 25 --output-dir runs/stage1_seed0
```

- **结果**：

| 指标        |  seed=20k |  seed=10k |
| ----------- | --------: | --------: |
| mean        | **27.14** | **27.21** |
| median      |      26.0 |      27.5 |
| max         |        52 |        61 |
| std         |     10.13 |     11.88 |
| mean steps  |    405.44 |     410.0 |
| score/step  |    0.0669 |    0.0664 |
| self / wall |    95 / 5 |    99 / 1 |

训练：**189,708 步，922.3 秒，约 205.7 步/秒**（早期默认线程设置）。

训练中 eval 曲线：

| episode   |   50 |   100 |   150 |   200 |   250 |   300 |   350 |   400 |   450 |   500 |
| --------- | ---: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| eval mean | 0.55 | 25.20 | 26.50 | 28.85 | 30.30 | 23.85 | 25.90 | 32.60 | 25.70 | 29.60 |

- **分析**：
  - 50–100 局之间出现“开窍式”跃升；50 局时全 timeout 不能说明学不会。
  - 175 局后 mean100 平台化在 25–30。
  - 20 局 eval 波动很大（23.85–32.60），最终 100 局更可靠。
  - 主要死因从 wall 变成 **self collision**（95–99%）。
  - loss 在 3–5 左右波动，没有爆炸。
- **结论**：11 维 DQN 能学，但上限约 27；self collision 说明观察缺少身体形状/距离信息。
- **下一步**：Stage 2 算法修正；但更大的杠杆可能在 Stage 3 观察工程。

---

### E2 · Stage 2-01：目标网络

- **目的**：目标网络让 TD target 更稳定，降低 checkpoint 间波动。
- **变量**：只加 `use_target_network=True`、`target_sync_every=1000`；其余同 Stage 1（MSE、无 Double、无 truncation bootstrap）。
- **命令**：

```bash
OMP_NUM_THREADS=4 .venv/bin/python scripts/train_dqn.py \
  --episodes 500 --eval-every 50 --eval-episodes 20 --final-eval-episodes 100 \
  --use-target-network --target-sync-every 1000 \
  --output-dir runs/stage2_01_target_seed0
```

- **结果**（seed 0）：

| 指标         | Stage 1 | 目标网络 |  变化 |
| ------------ | ------: | -------: | ----: |
| mean (20k)   |   27.14 |    24.34 | -2.80 |
| mean (10k)   |   27.21 |    24.72 | -2.49 |
| median (20k) |    26.0 |     24.0 |  -2.0 |
| std (20k)    |   10.13 |     8.38 | -1.75 |
| self / wall  |  95 / 5 |   99 / 1 |     — |

配对比较：20k **-2.80**，95% CI [-5.37, -0.24]；10k **-2.49**，CI [-5.15, +0.13]。  
训练：175,523 步，93.5 秒，1,877.6 步/秒（OMP=4）；target syncs=176。

- **分析**：
  - 单 seed 看，目标网络没有提分，反而略低；loss 也没有更低。
  - 初始 Q 值没有明显下降（maxQ 5.11 → 4.93）。
  - 20 局 eval 的 checkpoint 间波动略小，但均值更低。
  - **关键修正**：后来 E4 发现 baseline 跨 seed SD 约 2.38，所以单 seed 的 -2.5 不足以证明目标网络有害。
- **结论**：保留实现（Double 依赖它），但不把这一步当作“稳定提分”的正结果；单 seed 负结论不可靠。
- **下一步**：Double DQN。

---

### E3 · Stage 2-02：Double DQN

- **目的**：降低 max 造成的 Q 值高估；验证是否能收回目标网络单独版的损失。
- **变量**：在 E2 基础上加 `double_dqn=True`；其余不变。
- **命令**：

```bash
OMP_NUM_THREADS=4 .venv/bin/python scripts/train_dqn.py \
  --episodes 500 --eval-every 50 --eval-episodes 20 --final-eval-episodes 100 \
  --use-target-network --target-sync-every 1000 --double-dqn \
  --output-dir runs/stage2_02_double_seed0
```

- **结果**（seed 0）：

| 指标         | 目标网络 |    Double |      变化 |
| ------------ | -------: | --------: | --------: |
| mean (20k)   |    24.34 | **28.33** | **+3.99** |
| mean (10k)   |    24.72 | **26.82** | **+2.10** |
| median (20k) |     24.0 |      27.0 |      +3.0 |
| max (20k)    |       46 |        64 |       +18 |
| self / wall  |   99 / 1 |    99 / 1 |         — |

与 Stage 1 基线相比：20k +1.19（CI [-1.42, +3.89]），10k -0.39（CI [-3.20, +2.27]）。  
训练：170,365 步，101.3 秒，1,681.6 步/秒；target syncs=170。

- **分析**：
  - Double 明显优于目标网络单独版：+3.99 / +2.10，20k 的 CI 不含 0。
  - 相对 Stage 1 基线没有明确优势：一个 +1.19，一个 -0.39，平均约 +0.4。
  - 仍然是 99/100 self collision；没有突破 11 维观察的上限。
- **结论**：Double 值得保留；需要 seed 1/2 确认均值差异是否真实。
- **下一步**：多 seed 复核。

---

### E4 · Stage 2-02b：Double vs Baseline 多 seed

- **目的**：补训练 seed 1/2，确认 Double 的 +0.4 是否真实。
- **配置**：baseline / Double 各跑 seed 0/1/2；其余同 3.1；评估用 20k/10k 两套 100 局。
- **逐 seed 结果**：

| 方法     | seed |   20k |   10k |  平均 | self/wall |
| -------- | ---: | ----: | ----: | ----: | --------: |
| baseline |    0 | 27.14 | 27.21 | 27.18 |      95/5 |
| baseline |    1 | 27.88 | 27.08 | 27.48 |      98/2 |
| baseline |    2 | 23.52 | 22.90 | 23.21 |      99/1 |
| Double   |    0 | 28.33 | 26.82 | 27.57 |      99/1 |
| Double   |    1 | 26.23 | 26.93 | 26.58 |     100/0 |
| Double   |    2 | 26.09 | 24.73 | 25.41 |      99/1 |

跨 seed 汇总：

| 指标      |            baseline |              Double |
| --------- | ------------------: | ------------------: |
| 平均      |    **25.95 ± 2.38** |    **26.52 ± 1.08** |
| range     | 23.21–27.48（4.27） | 25.41–27.57（2.16） |
| 最差 seed |               23.21 |           **25.41** |
| 最好 seed |               27.48 |               27.57 |

Double − Baseline：平均 **+0.57**，SD 1.56，95% CI [-3.30, +4.43]；逐 seed +0.40 / -0.90 / +2.20。

- **训练成本**：baseline seed 0/1/2 = 189,708 / 171,481 / 173,106 步（约 922 / 99 / 99 秒，seed 0 用默认线程）；Double = 170,365 / 177,963 / 179,151 步（约 101 / 105 / 103 秒，OMP=4）。

- **分析**：
  - 均值不显著；Double 的主要收益是**跨 seed 方差更小、最差 seed 更高**。
  - 这是“抬高下限”，不是“提高上限”。
  - 只有 3 个 seed，仍是提示性证据。
  - 同时修正 E2：单 seed 的目标网络负结果不可靠；baseline seed 2 本身只有 23.21。
- **结论**：当前最好的 11 维配置是 **目标网络 + Double**；保留它是为了稳定性，不是因为均值显著更高。
- **下一步**：Huber（单独开）。

---

### E5 · Stage 2-03：Huber

- **目的**：用 SmoothL1 降低大 TD error 的影响，看能否进一步稳定/提分。
- **变量**：在 Double 基础上只加 `huber=True`；梯度裁剪保持关闭。
- **配置**：3 个训练 seed；其余同 3.1。
- **逐 seed 结果**：

| 方法  | seed |   20k |   10k |      平均 | self/wall |
| ----- | ---: | ----: | ----: | --------: | --------: |
| Huber |    0 | 28.15 | 29.57 | **28.86** |      99/1 |
| Huber |    1 | 24.41 | 24.88 | **24.64** |     90/10 |
| Huber |    2 | 25.63 | 24.83 | **25.23** |      93/7 |

跨 seed：Huber **26.24 ± 2.28**；Double 26.52 ± 1.08；baseline 25.95 ± 2.38。  
Huber − Double：平均 **-0.28**，95% CI [-4.28, +3.73]；Huber − Baseline：平均 **+0.29**，CI [-6.45, +7.03]。

- **训练成本**：约 167k–176k 步；三个 seed 并行跑约 110–115 秒/个（受并行竞争影响，只作记录）。

- **分析**：
  - 均值没有提升；逐 seed 方向不一致。
  - 跨 seed 方差反而比 Double 大（2.28 vs 1.08），range 4.22 vs 2.16。
  - 死因从 self 部分转向 wall（seed 1/2 有 10/7 次 wall），但总量没有改善。
  - Huber loss 约 0.2，MSE 约 3–4，尺度不可直接比较；训练没有爆炸。
- **结论**：**Huber 不保留**。当前最好 Stage 2 = 目标网络 + Double，不开 Huber。
- **下一步**：结束算法消融/回到观察瓶颈；或补 target-only seeds、grad clip、epsilon 调度。

---

### E6 · Stage 3：全棋盘 CNN 短训练（500 局）

- **目的**：把完整 20×20 棋盘喂给 CNN，让网络自己学感知和策略。
- **变量**：观察从 11 维手工特征改为 `(5,20,20)` 棋盘；模型从 MLP 改为 BoardQNet。
- **配置**：
  - 500 局；batch 128；replay 20k；每 2 个环境步 replay 更新；
  - epsilon 0.4 → 0.02，150 局衰减；
  - 目标网络 + Double DQN；MSE；`bootstrap_truncated=False`；
  - 每 50 局 20 局 eval；最终 100 局；device=auto→CUDA。
- **命令**：

```bash
OMP_NUM_THREADS=4 uv run python scripts/train_dqn.py \
  --obs-type board --board-channels 5 \
  --episodes 500 --eval-every 50 --eval-episodes 20 --final-eval-episodes 100 \
  --batch-size 128 --max-memory 20000 \
  --replay-update-every 2 --min-replay-size 1000 \
  --use-target-network --target-sync-every 1000 --double-dqn \
  --epsilon-end 0.02 --epsilon-decay-games 150 \
  --output-dir runs/stage3_board_seed0
```

- **结果**：

| 指标               |                              值 |
| ------------------ | ------------------------------: |
| 训练步数           |                          64,367 |
| 训练时间           | 222 秒（3.7 min，约 290 步/秒） |
| replay updates     |                          31,684 |
| target syncs       |                              31 |
| 最终 eval mean     |                        **0.17** |
| median / max / std |                    0 / 2 / 0.40 |
| 结束方式           |   wall 56 / timeout 44 / self 0 |

- **分析**：
  - 几乎没学会找食物；训练 mean100 约 0.21。
  - 还没有完全退化成绕圈，但 wall/timeout 各占一半。
  - 64k 步对从零开始的 CNN 太少；11 维特征此时替网络做了大量“感知”工作。
- **结论**：500 局不够；用剩余时间跑更长训练。
- **下一步**：长训练（E7）。

---

### E7 · Stage 3：全棋盘 CNN 长训练（45 分钟，超时）

- **目的**：用剩余时间看全棋盘 CNN 是否能学会；同时测试 GPU replay minibatch。
- **变量**：5000 局上限；batch 256；replay 50k；epsilon 0.4 → 0.05，1500 局衰减；每 500 局 eval；45 分钟硬超时。
- **命令**：

```bash
OMP_NUM_THREADS=4 timeout -s INT 2700 uv run python scripts/train_dqn.py \
  --obs-type board --board-channels 5 \
  --episodes 5000 --eval-every 500 --eval-episodes 20 --final-eval-episodes 100 \
  --batch-size 256 --max-memory 50000 \
  --replay-update-every 2 --min-replay-size 1000 \
  --use-target-network --target-sync-every 1000 --double-dqn \
  --epsilon-end 0.05 --epsilon-decay-games 1500 \
  --output-dir runs/stage3_board_long_seed0
```

- **结果**（实际跑到约 2486 局，45 分钟超时）：

| 指标             |                                          值 |
| ---------------- | ------------------------------------------: |
| 训练局数         |                                       2,486 |
| 训练步数         |                                     484,183 |
| 训练时间         |            2,697 秒（45 min，约 179 步/秒） |
| 训练 score       |  mean 0.101，max 2；2,486 局中 233 局拿过分 |
| 死因（训练）     |  wall 1261 / timeout 1212 / self 13 / win 0 |
| 最近 200 局      |      mean score 0.07；timeout 187 / wall 12 |
| 最终 100 局 eval | **mean 0.00 / median 0 / max 0 / std 0.00** |
| eval 结束方式    |                       **timeout 100 / 100** |
| eval mean steps  |                                       301.0 |
| target syncs     |                                         241 |

- **分析**：
  1. **稀疏奖励 + 全棋盘**：CNN 要自己学“食物在哪、怎么过去”；48 万步不够。
  2. **绕圈是稳定局部最优**：timeout 惩罚 -10，和撞死一样；绕圈避免撞墙，惩罚延迟到 301 步。
  3. **epsilon 太低**：1500 局后到 0.05，策略已经退化，5% 随机不足以重新发现食物。
  4. **更多训练反而更差**：500 局 eval 0.17 → 2486 局 eval 0.00。
  5. **不是单纯时间问题**：需要探索/奖励/课程设计，而不是继续堆步数。
- **同时修复的 bug**：CUDA checkpoint 的 `cuda_rng_state` 在 `map_location='cuda'` 下变成 CUDA tensor，`torch.cuda.set_rng_state_all` 要求 CPU ByteTensor。已在 `DQNAgent.load` 中修复，并新增 CUDA checkpoint roundtrip 测试；**55 tests pass**。
- **结论**：**1 小时内全棋盘 CNN + 稀疏奖励没有学会 Snake**；它学到了“绕圈到超时”。下一步最值得做的是 **小棋盘课程学习**。

---

### E8 · 工程基础设施：CUDA / GPU / replay 更新

这不是科学实验，但记录影响后续训练速度/配置的工程变更。

- **依赖**：torch 从 CPU 切到 **cu130**；实测 `torch 2.14.1+cu130`、CUDA 13.0、cuDNN 92400、RTX 5060 Laptop 7.96 GB。
- **全棋盘观察**：`(5,H,W)`：body / head / tail / food / border；`--board-channels 4` 可去掉 border。
- **BoardQNet**：3 层 conv + FC；约 1.70M 参数。
- **训练循环**：新增 replay minibatch 模式：
  - `--replay-update-every N`：每 N 个环境步更新一次；
  - `--batch-size`：board 默认 128；
  - `--min-replay-size`：board 默认 1000；
  - `--replay-update-every 0` 回到 Stage 1 的 per-step short-memory + 每局 long-memory。
- **GPU 默认**：board 模式 `device=auto`；`cudnn.benchmark`、TF32、`torch.set_num_threads(1)`。
- **计时 pilot**：
  - 3 局：104 步 / 6.5 秒（包含 2 局 301 步 eval，严重低估，不能当吞吐）；
  - 20 局：1,236 步 / 8.6 秒，约 143 步/秒；
  - 500 局：64,367 步 / 222 秒，约 290 步/秒；
  - 长训练：484,183 步 / 2,697 秒，约 179 步/秒。
  - 差异主要来自 episode 长度、eval 开销、buffer/批量大小，不能直接横向比较。
- **测试**：55 个测试全部通过（含 CUDA checkpoint RNG 回归测试）。

---

### E9 · 计划中 / 未执行

按优先级：

1. **10×10 课程学习（最推荐，仍然纯 RL）**
   - 全棋盘 CNN、同样的目标网络 + Double；
   - 棋盘更小，食物更容易撞到；episode 更短；
   - 预计 10–15 分钟先验证“能否学会找食物”；
   - 学会后再扩到 20×20（FC 尺寸会变；可以重训，或改成全卷积 + global pooling）。
2. **20×20 上改探索/奖励/超时**
   - epsilon 下限 0.1–0.2，衰减更慢；
   - timeout reward 更负（如 -20）或缩短 timeout，打破“绕圈安全”；
   - 这会改变奖励/探索设计，不再是完全原始的 Stage 3 设定。
3. **奖励塑形**
   - potential-based shaping，例如 `Phi(s) = -BFS 距离`；
   - 理论上保持最优策略，但注入领域知识，与“纯 RL”目标冲突。
4. **更多算力/更长训练**
   - 全棋盘 CNN 可能需要数小时到数天；
   - 但先解决局部最优，否则加时间只会更稳定地绕圈。
5. **观察工程**
   - 距离/坐标/食物中心通道，或局部 patch + 射线；
   - 样本效率会高很多，但不是纯全棋盘。
6. **补 Stage 2 缺口**
   - target-only seed 1/2（确认目标网络单独版是否真的有害）；
   - epsilon 调度（5% 下限 / 150 局）；
   - 梯度裁剪；
   - 11 维上的标准 replay minibatch 更新。
     这些在 11 维上限下预期收益有限，优先级低于 Stage 3 课程学习。

---

## 5. 横向对比与关键结论

### 5.1 11 维阶段

| 版本                  | 训练 seeds | 20k/10k 平均 | 跨 seed SD | 最差 seed | 结论                 |
| --------------------- | ---------: | -----------: | ---------: | --------: | -------------------- |
| Stage 1 baseline      |      0/1/2 |        25.95 |       2.38 |     23.21 | 可用基线             |
| + 目标网络（单 seed） |          0 |         24.5 |          — |         — | 单 seed 不可靠       |
| + Double DQN          |      0/1/2 |    **26.52** |   **1.08** | **25.41** | 当前最好；稳定性更好 |
| + Huber               |      0/1/2 |        26.24 |       2.28 |     24.64 | 不保留               |

**结论**：

- 11 维观察的上限约 **27**；最终评估 95–100/100 死于 self collision。
- Double 的均值提升不显著（+0.57，CI 跨 0），但跨 seed 方差更小、最差 seed 更高。
- 目标网络单独版的负结果主要是单 seed 噪声；多 seed 后不要再说“目标网络有害”。
- Huber 没有帮助；不要再在 11 维上堆算法技巧。

### 5.2 全棋盘阶段

| 版本          |                    训练预算 | eval mean | 结束方式             | 结论       |
| ------------- | --------------------------: | --------: | -------------------- | ---------- |
| 全棋盘 CNN 短 |   500 局 / 64k 步 / 3.7 min |      0.17 | wall 56 / timeout 44 | 几乎没学会 |
| 全棋盘 CNN 长 | 2,486 局 / 484k 步 / 45 min |      0.00 | timeout 100/100      | 退化成绕圈 |

**结论**：

- 全棋盘 CNN 在这套“稀疏奖励 + 直接 20×20 + 1 小时”的设置下没有学会。
- 主要问题是探索/局部最优，不是单纯样本数。
- 下一步优先做小棋盘课程学习，而不是继续在 20×20 上堆时间。

### 5.3 评估方法本身的教训

- 单 seed 结论可能完全被种子噪声淹没；baseline 的 seed SD 约 2.4。
- 20 局 eval 的波动可达 ±5；最终结论至少 100 局。
- 配对比较（同一组评估种子）能降低评估噪声，但不能消除训练种子噪声。
- 看均值之外，还要看跨 seed 方差、最差 seed、死因分布。

---

## 6. 待办清单

- [ ] **P0：10×10 课程学习**（全棋盘 CNN，纯 RL，预计 10–15 分钟）。
- [ ] P1：如果 10×10 学会，扩到 20×20；考虑全卷积 + global pooling 以支持变尺寸。
- [ ] P1：20×20 全棋盘上的探索/超时/epsilon 调整（若不使用课程学习）。
- [ ] P2：补 target-only seed 1/2，完善 Stage 2 消融表。
- [ ] P2：epsilon 调度（5% 下限 / 150 局）和梯度裁剪（预期小）。
- [ ] P2：11 维上的标准 replay minibatch 更新（与 Stage 1 short/long memory 对比）。
- [ ] P3：奖励塑形 / 观察工程 / 更多算力（视“纯 RL”目标取舍）。

---

## 7. 产物与复现

### 7.1 训练产物

| 实验                 | 目录                                                           |
| -------------------- | -------------------------------------------------------------- |
| Stage 1 baseline     | `runs/stage1_seed0`（以及 `runs/stage1_seed1/2` 多 seed 复核） |
| Stage 2-01 目标网络  | `runs/stage2_01_target_seed0`                                  |
| Stage 2-02 Double    | `runs/stage2_02_double_seed0/1/2`                              |
| Stage 2-03 Huber     | `runs/stage2_03_huber_seed0/1/2`                               |
| Stage 3 全棋盘短训练 | `runs/stage3_board_seed0`                                      |
| Stage 3 全棋盘长训练 | `runs/stage3_board_long_seed0`                                 |

每个目录通常包含：`config.json`、`train.csv`、`eval.csv`、`checkpoint.pt`、
`final_eval.json`、`eval.json`、`summary.json`。

### 7.2 详细记录

- `teach/logs/stage0-random-baseline.md`
- `teach/logs/stage1-dqn-baseline.md`
- `teach/logs/stage2-01-target-network.md`
- `teach/logs/stage2-02-double-dqn.md`
- `teach/logs/stage2-02b-multiseed.md`
- `teach/logs/stage2-03-huber.md`
- `teach/logs/stage3-board-cnn.md`

> `teach/` 是本地目录（未入库）；本 `log.md` 保留关键数字和结论，作为可提交的总记录。

### 7.3 常用命令

```bash
# 测试
uv run pytest

# 11 维最佳 Stage 2 配置（目标网络 + Double，不 Huber）
OMP_NUM_THREADS=4 uv run python scripts/train_dqn.py \
  --episodes 500 --eval-every 50 --eval-episodes 20 --final-eval-episodes 100 \
  --use-target-network --target-sync-every 1000 --double-dqn \
  --output-dir runs/stage2_double_repro

# 全棋盘 CNN（GPU）
OMP_NUM_THREADS=4 uv run python scripts/train_dqn.py \
  --obs-type board --board-channels 5 \
  --episodes 500 --eval-every 50 --eval-episodes 20 --final-eval-episodes 100 \
  --batch-size 128 --max-memory 20000 \
  --replay-update-every 2 --min-replay-size 1000 \
  --use-target-network --target-sync-every 1000 --double-dqn \
  --output-dir runs/stage3_board_repro

# 评估 / 画曲线
uv run python scripts/evaluate.py --run-dir runs/stage2_double_repro --episodes 100
uv run python scripts/plot_training.py --run-dir runs/stage2_double_repro
```

---

## 8. 维护模板（以后每次实验追加）

```markdown
### E<N> · <阶段>：<优化方向>

- **目的/假设**：
- **变量**：只改了什么；对照是什么。
- **配置**：观察、网络、算法开关、batch/replay、epsilon、训练局数、seeds、device。
- **命令**：
- **结果**：mean/median/max/std、死因、steps、时间、replay updates/target syncs。
- **分析**：数据和假设是否一致；意外是什么。
- **结论/下一步**：
```

> 原则：一次只改一个变量；先 seed 0，稳定后补 seed 1/2；
> 最终结论用 100 局 greedy eval；不要用训练 score 代替评估。
