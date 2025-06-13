from .template import config

n_embd = 768

args = config(
    n_layer=12,
    n_embd=n_embd,
    D_DECAY_LORA=64,
    D_AAA_LORA=64,
    D_MV_LORA=32,
    D_GATE_LORA=128,
    vocab_size=65536,
    head_size=64,
    dim_ffn=n_embd * 4
)