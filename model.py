'''
Chaos Twins: Primordial Disjunction
目标
0.7388905057061257 --> 4 * 2**0.5/10 + 3**0.5/10
'''

import torch
from torch import nn
import torch.nn.functional as F
import math

import random
import numpy as np


# 单位加法器

# 输入:上一位输入 x_last, 本位的两个输入 x1, x2
# 输出:本位的运算结果 y_this, 下一位的输入 x_next

# x1,x2 二进制
# y_this = ((x1 != x2) != x3)
# x_next = (x1 & x2) | ((x1 | x2) & x_last)


# (x1,x2,x3):(x_next,y_this)
add_logic_dict = {
    (0,0,0):(0,0),
    (0,0,1):(0,1),
    (0,1,0):(0,1),
    (0,1,1):(1,0),
    (1,0,0):(0,1),
    (1,0,1):(1,0),
    (1,1,0):(1,0),
    (1,1,1):(1,1)
}

x1_x2_xlast_to_idx = {
    (0,0,0):0,
    (0,0,1):1,
    (0,1,0):2,
    (0,1,1):3,
    (1,0,0):4,
    (1,0,1):5,
    (1,1,0):6,
    (1,1,1):7
}

idx_to_x1_x2_xlast = {v: k for k, v in x1_x2_xlast_to_idx.items()}

x1_x2_xlast_to_label = {
    (0,0,0):None,
    (0,0,1):0,
    (0,1,0):1,
    (1,0,0):2,
    (1,1,0):0,
    (1,0,1):1,
    (0,1,1):2,
    (1,1,1):None,
}

# (x_next,y_this):(x1,x2,x_last)
add_inverse_logic_dict = {
    (0,0):[(0,0,0)],
    (0,1):[(0,0,1),(0,1,0),(1,0,0)],
    (1,0):[(1,1,0),(1,0,1),(0,1,1)],
    (1,1):[(1,1,1)]
}


# 4个embed
xN_yT_to_idx = {
    (0,0):0,
    (0,1):1,
    (1,0):2,
    (1,1):3
}

# 8个embed


def set_seed(seed: int = 42):
    """设置全套随机种子"""
    # 设置 Python 内置随机种子
    random.seed(seed)
    # 设置 NumPy 随机种子
    np.random.seed(seed)
    # 设置 PyTorch 随机种子
    torch.manual_seed(seed)
    # 如果使用 GPU，设置 CUDA 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多 GPU 情况
        torch.backends.cudnn.deterministic = True  # 确保卷积操作确定性
        torch.backends.cudnn.benchmark = False  # 关闭优化以保持确定性


set_seed(42)



class Tokenizer(nn.Module):
    def __init__(self, n_bits=32):
        super().__init__()
        self.n_bits = n_bits
    def f2b(self, x: float) -> list:
        """将小数转换为n位二进制列表（包含前导0）"""
        n_bits = self.n_bits
        binary = []
        for _ in range(n_bits):
            x *= 2
            bit = int(x)
            binary.append(bit)
            x -= bit
        return binary
    def forward(self,x:float):
        return self.f2b(x)
    def b2f(self, b: list):
        """将二进制列表转换回浮点数"""
        f = 0.0
        for i in range(len(b)):
            f += b[i] * (2 ** -(i + 1))  # 第i位对应 2^(-i-1) 的权重
        return f



class BinaryEncoder(nn.Module):
    def __init__(self, embed_dim, max_length=64, n_bits=32, num_embed=2):
        super().__init__()
        self.num_embed = nn.Embedding(num_embed, embed_dim)  # 0和1的嵌入
        self.max_length = max_length
        self.n_bits = n_bits
        self.pos_encoding = self.create_position_encoding(embed_dim, max_length)
    def create_position_encoding(self, d_model: int, max_len: int):
        """生成位置编码（类似Transformer）"""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (max_len, d_model)
    def forward(self, binary: list):
        # (b_len) , b_len > 1
        b_len = len(binary)
        # 1. 转换为二进制
        # 2. 转换为Tensor索引
        indices = torch.tensor(binary, dtype=torch.long)
        # 3. 数字嵌入
        embeddings = self.num_embed(indices)  # (b_len, embed_dim)
        # 4. 添加位置编码
        output = embeddings
        output[1:] += self.pos_encoding[:b_len-1, :] # 减去start_token
        return output  # (n_bits, embed_dim)




class BitTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=6, n_bits=32):
        super().__init__()
        # Transformer 编码器层配置
        self.pe = BinaryEncoder(embed_dim=d_model, n_bits=n_bits)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            batch_first=False  # 输入格式为 (seq_len, batch_size, d_model)
        )
        # 完整的 Transformer 编码器
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
    def forward(self, b:list):
        # b: (n_bit,)
        z = self.pe(b)
        """处理二进制位序列
        输入形状: [seq_len, d_model] (即 [32, 128])
        输出形状: [seq_len, d_model] (即 [32, 128])
        """
        # 添加 batch 维度 -> [seq_len, 1, d_model]
        z = z.unsqueeze(1)
        # 通过 Transformer 编码器
        output = self.transformer_encoder(z)
        # 移除 batch 维度 -> [32, 128]
        return output.squeeze(1)


class BitDecoder(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=6, n_bits=32):
        super().__init__()
        # Transformer 编码器层配置
        self.n_bits = n_bits
        self.pe = BinaryEncoder(num_embed=8+1,embed_dim=d_model, n_bits=n_bits)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8)
        # 完整的 Transformer 编码器
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    def forward(self, x, z, causal_mask, padding_mask):
        # b: (seq_len,)
        # z: (src_seq_len, d_model)
        tgt = self.pe(x)
        """处理二进制位序列
        输入形状: [seq_len, d_model] (即 [32, 128])
        输出形状: [seq_len, d_model] (即 [32, 128])
        """
        # 添加 batch 维度 -> [seq_len, 1, d_model]
        tgt = tgt.unsqueeze(1)
        # 通过 Transformer 编码器
        output = self.transformer_decoder(
            tgt=tgt,  # 目标序列 (seq_len, batch_size)
            memory=z.unsqueeze(1),  # 编码器输出 (src_seq_len, batch_size, d_model)
            tgt_mask=causal_mask,  # 自回归掩码 (seq_len, seq_len)
            tgt_key_padding_mask=padding_mask  # 填充掩码 (batch_size, seq_len)
        )
        # 移除 batch 维度 -> [32, 128]
        return output


class SingleAdder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x_last):
        # x1,x2,x_last: {0,1}, 二进制
        # y_this = ((x1 != x2) != x_last)
        # x_next = bool((x1 & x2) | ((x1 | x2) & x_last))
        return add_logic_dict[(x1,x2,x_last)] # (x_next,y_this)
        # 十进制
        #res = x1 + x2 + x_last
        #y_this = res % 10
        #x_next = int(res // 10)
        #return x_next, y_this
        
'''
    def inverse(self,z,x_next,y_this):
'''     

# z: (B,128) -> (B,num_class)
class ClassifierHead(nn.Module):
    def __init__(self, input_dim=128, mlp_layers=[128, 256, 256], num_class=3):
        super().__init__()
        # 构建 MLP 层
        layers = []
        prev_dim = input_dim
        for hidden_dim in mlp_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())  # 激活函数
            prev_dim = hidden_dim
        # 最后一层输出分类结果
        layers.append(nn.Linear(prev_dim, num_class))
        # 将层组合为 Sequential
        self.mlp = nn.Sequential(*layers)
    def forward(self, z):
        """前向传播
        输入: z (B, 128)
        输出: logits (B, num_class)
        """
        return self.mlp(z)



def generate_idx_to_mlp(latent_dim):
    idx_to_mlp = {
        0:None,
        1:ClassifierHead(num_class=3, input_dim=latent_dim),
        2:ClassifierHead(num_class=3, input_dim=latent_dim),
        3:None
    }
    return idx_to_mlp


class Adder(nn.Module):
    def __init__(self):
        super().__init__()
        self.adder = SingleAdder()
    def forward(self,t1,t2):
        # t1,t2: {0,1} list
        t1 = t1[::-1]
        t2 = t2[::-1]
        x_next_all = []
        y_this_all = []
        for i in range(len(t1)):
            if i==0:
                x_next, y_this = self.adder(t1[i],t2[i],0)
            else:
                x_next, y_this = self.adder(t1[i],t2[i],x_next_all[-1])
            x_next_all.append(x_next)
            y_this_all.append(y_this)
        #del x_next_all[-1]
        return x_next_all[::-1], y_this_all[::-1]


class AddiDecomposer(nn.Module):
    def __init__(self,latent_dim = 128,n_bits = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_bits = n_bits
        # 特殊token
        self.start_token = 8
        # self.mask_token = 9
        # 解析器们
        self.tokenizer = Tokenizer(n_bits=n_bits)
        self.adder = Adder()
        # 模型们
        self.idx_to_classifier = generate_idx_to_mlp(latent_dim)
        self.encoder = BitTransformer(d_model=latent_dim,nhead=8,num_layers=6,n_bits=n_bits)
        self.decoder = BitDecoder(d_model=latent_dim,n_bits=n_bits)
    def get_loss(self,x1,x2):
        '''get_loss
        x1,x2: float < 0.5
        t1,t2: 两个真实的加数的二进制编码 List[{0,1}]*n_bits
        '''
        # 加载参数
        tokenizer = self.tokenizer
        encoder = self.encoder
        decoder = self.decoder
        adder = self.adder
        idx_to_classifier = self.idx_to_classifier
        n_bits = self.n_bits
        start_token = self.start_token
        # 转换为2进制list [{0,1}] * n_bits
        t1 = tokenizer(x1)
        t2 = tokenizer(x2)
        t = adder(t1,t2)
        x_next_all, y_this_all = t
        z = encoder(y_this_all) # 编码
        xN_yT_to_classifier_idx = xN_yT_to_idx
        input_len = n_bits
        # [(x_next,y_this),...]
        xN_yT_idx = list(zip(
            x_next_all,
            y_this_all
        ))
        # [(x1,x2,x_last)]
        decoder_output_expected_b = list(zip(
        t1[:input_len],
        t2[:input_len],
        (x_next_all+[0])[1:input_len+1]))
        # 对齐: xN_yT_idx -> decoder_output_expected_b
        decoder_temp_input = [xN_yT_to_idx[i] for i in xN_yT_idx]
        decoder_label = [x1_x2_xlast_to_idx[i] for i in decoder_output_expected_b]
        #
        decoder_input = torch.tensor([start_token] + decoder_label[:-1],dtype=torch.long)  # (seq_len+1)
        decoder_label = torch.tensor(decoder_label,dtype=torch.long)
        #
        padding_mask = None #(tgt == mask_idx).unsqueeze(0)  # (batch_size, seq_len)
        # 自回归掩码
        causal_mask = torch.triu(torch.ones(input_len, input_len), diagonal=1).bool()  # (seq_len, seq_len)
        #
        output = decoder(decoder_input, z, causal_mask, padding_mask)
        #
        logits = [0] * output.shape[0]
        #
        for i in range(output.shape[0]):
            xN_yT_idx_temp = decoder_temp_input[i]
            classifier = idx_to_classifier[xN_yT_idx_temp]
            if classifier is None:
                logits[i] = None
            else:
                logits_temp = classifier(output[i])
                logits[i] = logits_temp
        #
        labels = [x1_x2_xlast_to_label[i] for i in decoder_output_expected_b]
        #
        valid_labels = []
        valid_logits = []
        for label, logit in zip(labels, logits):
            if label is not None and logit is not None:
                valid_labels.append(label)
                valid_logits.append(logit)
        # 将 valid_labels 和 valid_logits 转换为 Tensor
        valid_labels = torch.tensor(valid_labels, dtype=torch.long)  # (N,)
        valid_logits = torch.cat(valid_logits, dim=0)  # (N, C)
        valid_probs = F.softmax(valid_logits,dim=1)
        # 计算交叉熵损失
        loss = F.cross_entropy(valid_probs, valid_labels)
        return loss
    def infer(self,x):
        b = self.tokenizer(x)
        # infer
        '''infer
        确保模型处于eval()模式下
        b: 待分解的二进制数 List[{0,1}]*n_bits
        '''
        # 加载参数
        tokenizer = self.tokenizer
        encoder = self.encoder
        decoder = self.decoder
        adder = self.adder
        idx_to_classifier = self.idx_to_classifier
        n_bits = self.n_bits
        start_token = self.start_token
        #
        z = encoder(b)
        x_next_all = [0]
        decoder_input = [start_token]
        for i in range(n_bits):
            xN_yT_temp = (x_next_all[-1],b[i])
            decoder_temp_input = xN_yT_to_idx[xN_yT_temp]
            classifier = idx_to_classifier[decoder_temp_input]
            if classifier is None:
                output_temp = add_inverse_logic_dict[xN_yT_temp][0]
            else:
                temp_len = len(decoder_input)
                # 推理的时候也要加causal_mask,否则会使得前面时刻的中间值和训练不符
                causal_mask = torch.triu(torch.ones(temp_len, temp_len), diagonal=1).bool()  # (seq_len, seq_len)
                padding_mask = torch.zeros((1,temp_len),dtype=bool)
                output = decoder(decoder_input, z, causal_mask, padding_mask)
                output = output[-1]
                logits = classifier(output)
                pred = torch.argmax(logits,dim=1)
                output_temp = add_inverse_logic_dict[xN_yT_temp][pred]
            x_next_all += [output_temp[2]]
            decoder_input += [x1_x2_xlast_to_idx[output_temp]]
        x1_x2_xlast_pred = [idx_to_x1_x2_xlast[i] for i in decoder_input[1:]]
        b1_pred = [i[0] for i in x1_x2_xlast_pred]
        b2_pred = [i[1] for i in x1_x2_xlast_pred]
        x1_pred = tokenizer.b2f(b1_pred)
        x2_pred = tokenizer.b2f(b2_pred)
        return x1_pred, x2_pred



''' 结果等价验证
decoder.eval()
output = decoder(decoder_input, z, causal_mask, padding_mask)
output = output[-1]

output1 = decoder(decoder_input, z, causal_mask, None)
output1 = output1[-1]

causal_mask2 = torch.triu(torch.ones(temp_len+1, temp_len+1), diagonal=1).bool()  # (seq_len, seq_len)
padding_mask2 = torch.zeros((1,temp_len+1),dtype=bool)
output2 = decoder(decoder_input+[mask_token], z, causal_mask2, padding_mask2)
output2 = output2[-2]
'''



'''
tgt = decoder.pe(tgt)
tgt = tgt.unsqueeze(1)
# 通过 Transformer 编码器
output = decoder.transformer_decoder(
    tgt=tgt,  # 目标序列 (seq_len, batch_size, d_model)
    memory=z.unsqueeze(1),  # 编码器输出 (src_seq_len, batch_size, d_model)
    tgt_mask=causal_mask,  # 自回归掩码 (seq_len, seq_len)
    tgt_key_padding_mask=None  # 填充掩码 (batch_size, seq_len)
)
xN_yT = (
    y_this_all[input_len],
    x_next_all[input_len]
)
xN_yT_idx = xN_yT_to_idx[xN_yT]
classifier = idx_to_classifier[xN_yT_idx]
model = SingleAdder()
'''
