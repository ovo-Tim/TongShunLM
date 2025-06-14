from .template import config

n_embd = 512

args = config(
    n_layer=3,
    n_embd=n_embd,
    D_DECAY_LORA=64,
    D_AAA_LORA=64,
    D_MV_LORA=32,
    D_GATE_LORA=64,
    vocab_size=3654,
    head_size=64,
    dim_ffn=n_embd * 2,
    final_ffn=n_embd * 4,
)