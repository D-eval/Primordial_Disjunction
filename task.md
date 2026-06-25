我想到一个课题，而且我今天一天就可以做完 基于SFT和RLHF的符号搜索，目标 input: 3.14626 output: Add(sqrt(2),sqrt(3)) 训练样本允许无限生成 研究方向：bloom filter 外推是否适用

基本上就是保存一个字典，3.14626 在模型看来就是一串规律难寻的数码，他作为key，而后面的 Add(sqrt(2),sqrt(3)) 这个key对应的value，这几乎就是 https://arxiv.org/abs/2602.00906 里的实验，只不过 fact 是可验证的 fact，而不是随机取的

主要是这玩意上下文长度很短，数据无限生成，而且可以研究scale增长后对fact的储存能力，我的mac就可以跑

我还可以尝试不同进制

而且不需要微调qwen，因为词表有问题，所以从0训，我就可以给审稿人解释为啥不微调

第一步，无限生成数据：

expr -> value
Add(Sqrt(2), Sqrt(3)) -> 3.146264...
Mul(Pi, Sqrt(2)) -> 4.44288...
Div(Add(1, Sqrt(5)), 2) -> 1.61803...

表达式 grammar 限制为：

Const: 1..50, pi, e
Unary: sqrt, sin, cos, log, exp
Binary: add, sub, mul, div, pow_small

第二步，SFT：

输入是四舍五入后的数字：

<value> 3.14626

输出是表达式前缀形式：

Add(Sqrt(2),Sqrt(3))

第三步，beam search + verifier：

模型生成 32/128 个候选，全部 eval，选误差最小的。这样马上就有实验结果。

RLHF 可以换成更简单的 RLAIF/RL：

reward：

reward = -log(abs(eval(expr)-target)+eps)
         - lambda * expr_length
         - invalid_penalty



最关键的实验：

训练集：depth ≤ 3
测试集：depth = 4/5/6

看模型能不能从浅层表达式外推到深层表达式。

指标：

exact syntax match
numeric error < 1e-5
top-k success rate
invalid expression rate
average expression length

我觉得这个方向真正有趣的点不是“符号回归”，而是：

模型是否学会了表达式空间的反向索引。

也就是从数值反推出结构。

如果它成功了，就说明 SFT 模型在某种意义上学到了一个连续数值空间到离散语法空间的近似 hash inverse。

如果失败了，也很有价值：说明 bloom-filter-like 记忆结构对符号表达式外推很弱，只能插值，不能组合外推。

每个文件实现一个类，然后在 if __name__=="__main__": 里写测试内容

