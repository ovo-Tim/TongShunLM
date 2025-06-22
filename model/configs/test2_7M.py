from .template import config
# In this test, we decrease the size of embd and increase the size of attn.
# The result is shockingly well
n_embd = 256

args = config(
    n_layer=4,
    n_embd=n_embd,
    D_DECAY_LORA=64,
    D_AAA_LORA=64,
    D_MV_LORA=32,
    D_GATE_LORA=64,
    vocab_size=3654,
    head_size=128,
    dim_ffn=n_embd * 4,
    final_ffn=n_embd * 4,
)