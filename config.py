import argparse
import operator
import math


def _unique_keep_order(tokens):
    seen = set()
    result = []
    for token in tokens:
        if token not in seen:
            result.append(token)
            seen.add(token)
    return result


cfg = argparse.Namespace()
cfg.base_system = 10 # 数进制，一共有 这么多表示数的 token
cfg.opsToken2ops = {
    "leaf": {"arity": 0, "func": lambda x: x},
    
    "add": {"arity": 2, "func": operator.add},
    "neg": {"arity": 1, "func": operator.neg},
    "mul": {"arity": 2, "func": operator.mul},
    "inv": {"arity": 1, "func": lambda x: 1/x},
    "sqrt": {"arity": 1, "func": math.sqrt},
}

cfg.opsTokens = list(cfg.opsToken2ops.keys())
cfg.digital_token = [str(i) for i in range(cfg.base_system)]
cfg.special_token = [
    "|beginOfThink|",
    "|endOfThink|",
    "|nl|",
    "|indent|",
    "|sp|",
    "|bos|",
    "|eos|", 
]
cfg.punctuation_tokens = ["(", ")", ",", ".", "-"]
cfg.trace_tokens = [
    "=",
    "r",
    "addPad",
    "addPos",
    "addPosRes",
    "addRes",
    "ladd",
    "sub",
    "subPad",
    "subPos",
    "subRes",
]

cfg.task_describe_token = [
    # 提示任务类型
    "|forward|", # |forward|add(sqrt(2),sqrt(3)) -> 3.14626
    "|inverse|", # |inverse|3.14626 -> |bos|add(sqrt(2),sqrt(3))|eos|
    "|simplify|", # |simplify|add(2,2) -> |bos|4|eos|
]
cfg.all_tokens = _unique_keep_order(
    cfg.task_describe_token
    + cfg.opsTokens
    + cfg.digital_token
    + cfg.special_token
    + cfg.punctuation_tokens
    + cfg.trace_tokens
)



cfg.dataset = argparse.Namespace()
cfg.dataset.max_constant = 500 # leaf允许的最大常数
cfg.dataset.max_depth = 4
cfg.dataset.value_precision = 5

cfg.model = argparse.Namespace()
cfg.model.use_rope = True
cfg.model.max_seq_len = 2304
cfg.model.layer_num = 24
cfg.model.hidden_dim = 128
cfg.model.head_num = 8
cfg.model.head_dim = 64
cfg.model.kv_head_num = 2
cfg.model.intermediate_size = 704
cfg.model.rms_norm_eps = 1e-6
cfg.model.rope_theta = 1_000_000.0
cfg.model.attention_dropout = 0.0
cfg.model.hidden_act = "silu"
cfg.model.initializer_range = 0.02
cfg.model.tie_word_embeddings = True
cfg.model.vocab_size = len(cfg.all_tokens)
cfg.model.embedding_vocab_size = len(cfg.all_tokens)
cfg.model.output_vocab_size = len(cfg.all_tokens)

cfg.train = argparse.Namespace()
cfg.train.seed = 42
cfg.train.batch_size = 4
cfg.train.lr = 3e-4
cfg.train.weight_decay = 0.01
cfg.train.grad_clip = 1.0
cfg.train.log_per_step = 5
cfg.train.save_per_step = 2000
cfg.train.eval_per_step = 2000
cfg.train.eval_sample_num = 2
cfg.train.task_names = ["forward"] # ["forward", "inverse", "simplify"]
cfg.train.task_probs = [1]# [1.0, 1.0, 1.0]
cfg.train.device = "cuda"
cfg.train.param_save_dir = "./params"
cfg.train.result_save_dir = "./result"
cfg.train.visual_save_dir = "./result"
cfg.train.visual_txt_name = "visual.txt"
cfg.train.loss_png_name = "loss.png"
cfg.train.sample_space_json_name = "sample_space.json"
cfg.train.load_last_ckpt = True
cfg.train.last_ckpt_name = "last.pt"
cfg.train.best_ckpt_name = "best.pt"
cfg.train.total_step = 200000000

cfg.eval = argparse.Namespace()
cfg.eval.device = "mps"
cfg.eval.batch_size = 64
cfg.eval.sample_num = 2
cfg.eval.max_new_tokens = 2304
cfg.eval.task_names = ["forward", "inverse", "simplify"]
cfg.eval.ckpt_path = None
