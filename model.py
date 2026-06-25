from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from config import cfg


class SymbolicVocabulary:
    def __init__(self):
        self.tokens = list(cfg.all_tokens)
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def bos_id(self) -> int:
        return self.token_to_id["|bos|"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["|eos|"]

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id[token] for token in tokens]

    def decode(self, token_ids: List[int]) -> List[str]:
        return [self.id_to_token[token_id] for token_id in token_ids]


@dataclass
class TinyQwenConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    attention_dropout: float
    hidden_act: str
    initializer_range: float
    tie_word_embeddings: bool
    use_rope: bool

    @classmethod
    def from_runtime_cfg(cls) -> "TinyQwenConfig":
        return cls(
            vocab_size=cfg.model.vocab_size,
            hidden_size=cfg.model.hidden_dim,
            intermediate_size=cfg.model.intermediate_size,
            num_hidden_layers=cfg.model.layer_num,
            num_attention_heads=cfg.model.head_num,
            num_key_value_heads=cfg.model.kv_head_num,
            head_dim=cfg.model.head_dim,
            max_position_embeddings=cfg.model.max_seq_len,
            rms_norm_eps=cfg.model.rms_norm_eps,
            rope_theta=cfg.model.rope_theta,
            attention_dropout=cfg.model.attention_dropout,
            hidden_act=cfg.model.hidden_act,
            initializer_range=cfg.model.initializer_range,
            tie_word_embeddings=cfg.model.tie_word_embeddings,
            use_rope=cfg.model.use_rope,
        )


def build_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_name in {"cuda", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class QwenRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class QwenRotaryEmbedding(nn.Module):
    def __init__(self, config: TinyQwenConfig):
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1).to(hidden_states.device)
        position_ids = position_ids[:, None, :].float()
        freqs = (inv_freq.float() * position_ids).transpose(1, 2)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype=hidden_states.dtype), emb.sin().to(dtype=hidden_states.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        n_rep,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


class QwenMLP(nn.Module):
    def __init__(self, config: TinyQwenConfig):
        super().__init__()
        self.hidden_act = config.hidden_act
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.hidden_act != "silu":
            raise ValueError(f"Unsupported hidden activation: {self.hidden_act}")
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class QwenAttention(nn.Module):
    def __init__(self, config: TinyQwenConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout

        q_out_dim = self.num_attention_heads * self.head_dim
        kv_out_dim = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_out_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, kv_out_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, kv_out_dim, bias=True)
        self.o_proj = nn.Linear(q_out_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.head_dim,
        ).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class QwenDecoderLayer(nn.Module):
    def __init__(self, config: TinyQwenConfig):
        super().__init__()
        self.self_attn = QwenAttention(config)
        self.mlp = QwenMLP(config)
        self.input_layernorm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QwenModel(nn.Module):
    def __init__(self, config: TinyQwenConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([QwenDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = QwenRotaryEmbedding(config) if config.use_rope else None
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def _build_attention_mask(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        dtype = self.embed_tokens.weight.dtype
        device = input_ids.device

        causal_mask = torch.full(
            (seq_len, seq_len),
            fill_value=torch.finfo(dtype).min,
            device=device,
            dtype=dtype,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)

        if attention_mask is None:
            return causal_mask

        padding_mask = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
        return causal_mask + padding_mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        batch_size, seq_len, _ = hidden_states.shape
        full_attention_mask = self._build_attention_mask(input_ids=input_ids, attention_mask=attention_mask)

        position_embeddings = None
        if self.rotary_emb is not None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, full_attention_mask, position_embeddings)

        return self.norm(hidden_states)


class QwenForCausalLM(nn.Module):
    def __init__(self, config: TinyQwenConfig):
        super().__init__()
        self.config = config
        self.model = QwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        output = {"logits": logits}

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            output["loss"] = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        generated = input_ids
        generated_attention_mask = attention_mask
        if generated_attention_mask is None:
            generated_attention_mask = torch.ones_like(generated, dtype=torch.long)

        for _ in range(max_new_tokens):
            outputs = self(generated, attention_mask=generated_attention_mask)
            next_token = outputs["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            generated_attention_mask = torch.cat(
                [generated_attention_mask, torch.ones_like(next_token, dtype=generated_attention_mask.dtype)],
                dim=-1,
            )
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
        return generated

    def count_parameters(self, trainable_only: bool = False) -> int:
        parameters = self.parameters() if not trainable_only else (
            parameter for parameter in self.parameters() if parameter.requires_grad
        )
        return sum(parameter.numel() for parameter in parameters)


def build_model() -> QwenForCausalLM:
    return QwenForCausalLM(TinyQwenConfig.from_runtime_cfg())


if __name__ == "__main__":
    from data_generate import SymbolicDatasetGenerator

    torch.manual_seed(cfg.train.seed)

    vocab = SymbolicVocabulary()
    model = build_model()
    dataset = SymbolicDatasetGenerator(seed=cfg.train.seed, value_precision=cfg.dataset.value_precision)
    sample = dataset.sample_inverse_example()
    input_ids = torch.tensor([vocab.encode(sample["full_tokens"])], dtype=torch.long)
    labels = input_ids.clone()
    labels[:, : len(sample["input_tokens"])] = -100

    outputs = model(input_ids=input_ids, labels=labels)
    generated = model.generate(
        input_ids=torch.tensor([vocab.encode(sample["input_tokens"])], dtype=torch.long),
        max_new_tokens=cfg.eval.max_new_tokens,
        eos_token_id=vocab.eos_id,
    )

    print("=== model summary ===")
    print(f"vocab size            : {vocab.vocab_size}")
    print(f"hidden size           : {model.config.hidden_size}")
    print(f"intermediate size     : {model.config.intermediate_size}")
    print(f"layer num             : {model.config.num_hidden_layers}")
    print(f"attention heads       : {model.config.num_attention_heads}")
    print(f"kv heads              : {model.config.num_key_value_heads}")
    print(f"head dim              : {model.config.head_dim}")
    print(f"max seq len           : {model.config.max_position_embeddings}")
    print(f"parameter count       : {model.count_parameters():,}")
    print(f"trainable parameters  : {model.count_parameters(trainable_only=True):,}")
    print("\n=== forward test ===")
    print(f"logits shape          : {tuple(outputs['logits'].shape)}")
    print(f"loss                  : {outputs['loss'].item():.6f}")
    print("\n=== greedy decode preview ===")
    print(vocab.decode(generated[0].tolist()))
