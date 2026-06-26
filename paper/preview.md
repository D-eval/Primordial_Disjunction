# 可验证符号算术轨迹的从零监督训练：一个小型 Transformer 的初步实验

## 摘要

本文研究一个可控的符号算术生成任务：给定前缀表达式，模型需要生成显式逐步计算轨迹，并输出最终十进制数值。长期动机是考察 Transformer 是否能够在符号表达式空间和数值空间之间建立可验证映射，为后续反向符号搜索、beam search verifier 和 RLHF/RLAIF 奖励优化提供基础。

当前实现训练了一个从零开始的小型 Qwen 风格因果语言模型，仅使用无限在线生成的 `forward` 数据。结果显示，在四类浅层算术模板上，训练 loss 已降至 `1e-5` 量级，训练内置周期评测在后期连续达到 `forward` exact match 和 numeric success 均为 `1.0`；可视化样例中，模型能够逐 token 复现加法、带负数减法以及进位/借位 trace。当前评测样本量较小，尚未验证深度外推、反向索引或 RLHF/RLAIF 设置，因此结论应视为一个可运行的初步正结果。

## 问题设定

原始设想是构造一个可验证的符号搜索任务：

```text
v -> e, where eval(e) ~= v
```

例如从 `3.14626` 反推出 `add(sqrt(2),sqrt(3))`。这类任务像一个从连续数值到离散语法结构的近似反向索引：若模型成功，它可能学到某种可组合的 hash inverse；若失败，则说明类似 bloom-filter 的记忆机制在符号外推上可能很弱。

当前代码库优先实现并训练了前向可验证任务：

```text
e -> trace(e), eval(e)
```

输入为：

```text
|forward| + tokens(e)
```

输出为：

```text
|beginOfThink| + trace(e) + |endOfThink| + render(eval(e))
```

## 数据生成

表达式由 `data_generate.py` 在线采样。当前激活的 `forward` 模板如下：

```text
add leaf leaf
mul leaf leaf
add neg leaf leaf
add leaf neg leaf
```

叶子常数范围为 `[1,500]`，十进制输出保留 5 位小数。对于 `add`、`mul`、`neg` 以及由负数加法诱导出的 `sub`，生成器会写出结构展开和具体竖式步骤。

加法 trace 示例：

```text
r0 = add r1 r2
r1 = 108
r2 = 107
r0 = add 108 107
    add 108 107
    = addPad 108 107
    = addPos (8 7) (0 0) (1 1)
    = addPosRes (5 1) (0 0) (2 0)
    = addRes 5 (1 0) (0 2) 0
    = addRes 5 1 2 0
    = 0215
    = 215
r0 = 215
```

这样设计的好处是，目标不是黑箱回归一个数字，而是学习一个可检查的算法轨迹。若未来加入 verifier，可以分别检查语法合法性、每一行局部计算和最终数值误差。

## 模型与训练

模型为从零训练的 Qwen 风格 decoder-only Transformer。词表由任务 token、运算符、数字、标点和 trace token 组成，而不是复用自然语言 tokenizer。

| 配置项 | 数值 |
|---|---:|
| 词表大小 | 42 |
| 参数量 | 10,450,304 |
| 层数 | 24 |
| hidden dim | 128 |
| attention heads | 8 |
| KV heads | 2 |
| head dim | 64 |
| max sequence length | 2304 |
| batch size | 4 |
| learning rate | `3e-4` |
| weight decay | `0.01` |

训练采用标准 next-token SFT。prompt 部分 label 置为 `-100`，只在输出 trace 和最终答案上计算交叉熵。训练任务当前仅启用 `forward`，因此 `inverse` 和 `simplify` 的结果不应纳入模型能力结论。

## 实验结果

当前训练日志包含 40,382 条 loss 记录，最后记录步数为 118,565，最后 loss 为 `1.91e-5`，日志中最小 loss 为 `2.95e-6`。

内置评测日志在后期每 1000 step 对 2 个 `forward` 样本评测一次；从 109,000 到 118,000 step 的最近 10 次评测中，`forward` exact match rate 和 numeric success rate 均为 `1.0`，invalid rate 为 `0.0`。

| 阶段 | exact match | numeric success | invalid |
|---|---:|---:|---:|
| step 109000--118000 的周期评测 | 1.0 | 1.0 | 0.0 |
| step 118000 可视化样例 | 1.0 | 1.0 | 0.0 |

训练 loss 图见：`../result/loss.png`

从 `result/visual.txt` 看，模型在 step 118000 对两个可视化样例均逐 token 完全匹配：

- `add(108,107)`：正确生成进位步骤并输出 `215.00000`。
- `add(384,neg(416))`：将其规约为 `sub(384,416)`，再翻转为带负号的 `sub(416,384)`，正确处理借位并输出 `-32.00000`。

## 结果怎么看

这个结果是一个清楚的工程正信号：数据格式、tokenizer、长 trace 监督、模型结构和训练循环都已经打通，小模型确实能够学会当前模板分布内的显式算术轨迹。尤其是带负数的样例说明模型不只是输出最终数字，也能复现由生成器规定的中间规约形式。

但研究结论仍要谨慎。当前 `forward` 样本空间估计为 `1,000,000`，模板只有 4 个，且评测样本数很小；模型很可能已经在这个受限分布上形成了强模式拟合。原始课题中最有趣的部分，即数值到表达式的反向搜索、深度外推、不同进制比较、beam verifier 和 RLHF/RLAIF 奖励，目前还没有形成实验证据。

## 下一步实验

1. 增加独立大样本评测脚本，至少对 `forward` 采样 1,000--10,000 个样本，报告 exact match、numeric success、invalid rate 和按模板分组的错误率。
2. 将模板从一层算术扩展到混合表达式，并做 depth split：训练 depth <= 3，测试 depth = 4/5/6。
3. 重新启用 `inverse`，先用 beam search + verifier 做 top-k success，再考虑基于奖励的强化学习或偏好优化。

奖励形式可以写成：

```text
reward = -log(abs(eval(expr) - target) + eps)
         - lambda * expr_length
         - invalid_penalty
```

## 结论

当前项目已经得到一个可信的第一阶段结果：从零训练的小型 Transformer 可以在受控模板上学习可验证的符号算术 trace，并在训练后期的小样本评测中达到完全匹配。这个结果值得作为论文的初步实验写入，但论文主张应限定为 `forward trace learning works in a controlled setting`。要支撑更强的“反向符号搜索”或“bloom-filter-like 外推”论断，还需要系统的反向任务、深度外推和大样本评测。
