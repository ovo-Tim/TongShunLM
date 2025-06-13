from dataclasses import dataclass
from typing import Optional

@dataclass
class config:
    n_layer: int
    n_embd: int
    D_DECAY_LORA: int
    D_AAA_LORA: int
    D_MV_LORA: int
    D_GATE_LORA: int
    vocab_size: int
    head_size: int

    final_ffn: Optional[int] = None  # 默认等于 dim_ffn

    dim_att: Optional[int] = None  # 默认等于 n_embd
    dim_ffn: Optional[int] = None  # 默认等于 n_embd * 4
    # head_size: Optional[int] = None  # 默认等于 head_size_a
    n_head: Optional[int] = None  # 默认等于 dim_att // head_size

    def __post_init__(self):
        self.dim_att = self.n_embd if self.dim_att is None else self.dim_att
        self.dim_ffn = (self.n_embd * 4) if self.dim_ffn is None else self.dim_ffn
        assert self.dim_att % self.head_size == 0, "dim_att must be divisible by head_size"
        self.n_head = (self.dim_att // self.head_size) if self.n_head is None else self.n_head

        self.final_ffn = self.dim_ffn if self.final_ffn is None else self.final_ffn