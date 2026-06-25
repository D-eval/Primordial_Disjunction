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

data_generate.py ，|forward| 模式中，要生成求解步骤，比如说

mul add 12 34 56

这种应该生成求解步骤
|beginOfThink|
r0 = mul r1 r2
r1 = add r3 r4 
r3 = 12 
r4 = 34
r1 = add 12 34 
    add 12 34
    = addPad 12 34
    = addPos (2 4) (1 3)
    = addPosRes (6 0) (4 0)
    = addRes 6 (0 4) 0
    = addRes 6 4 0
    = 046
    = 46
r1 = 46
r2 = 56 
r0 = mul 46 56 
mul add 40 6 add 50 6 
= ladd (mul 40 50 mul 40 6 mul 6 50 mul 6 6)
= ladd (2000 240 300 36)
= add 2000 ladd (240 300 36)
= add 2000 add 240 ladd (300 36)
= add 2000 add 240 add 300 36
    add 300 36
    = addPad 300 036
    = addPos (0 6) (0 3) (3 0)
    = addPosRes (6 0) (3 0) (3 0)
    = addRes 6 (0 3) (0 3) 0
    = addRes 6 3 3 0
    = 0336
    = 336
= add 2000 add 240 336
    add 240 336
    = addPad 240 336
    = addPos (0 6) (4 3) (2 3)
    = addPosRes (6 0) (7 0) (5 0)
    = addRes 6 (0 7) (0 5) 0
    = addRes 6 7 5 0
    = 0576
    = 576
= add 2000 576
    add 2000 576
    = addPad 2000 0576
    = addPos (0 6) (0 7) (0 5) (2 0)
    = addPosRes (6 0) (7 0) (5 0) (2 0) 0
    = addRes 6 7 5 2 0
    = 02576
    = 2576
= 2576
|endOfThink|

加法的话，进位要考虑

哦对，还有减法

    add neg 543 357
    = add 357 neg 543
    = sub 357 543
    = neg sub 543 357
        sub 543 357
        = subPad 543 357
        = subPos (3 7) (4 5) (5 3)
        = subPos (13 7) (3 5) (5 3)
        = subPos (13 7) (13 5) (4 3)
        = subRes 5 8 1
        = 185
    = neg 185

也就是说，
需要把递归计算显式写出来
另外，每种运算也要把计算过程写出来

然后，我需要在运行 data_generate.py 的时候看到完整|forward|的SFT数据格式
在 train.py 中也要如此训练


好的，现在看看 forward_template 里的 |forward| 需要多大的上下文token





python3 data_generate.py --trace 'ladd(34,45,mul(4,5))'