from .template import config
# This is ridiculous! The performance is still as well as previous ones.
n_embd = 32

args = config(
    n_layer=5,
    n_embd=n_embd,
    D_DECAY_LORA=64,
    D_AAA_LORA=64,
    D_MV_LORA=32,
    D_GATE_LORA=64,
    vocab_size=3654,
    head_size=32,
    dim_ffn=n_embd * 5,
    final_ffn=n_embd * 4,
)