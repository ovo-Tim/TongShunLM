########################################################################################################
# TongShunLM, based on RWKV-7.
# Modified from https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_demo.py
########################################################################################################

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)

'''
This will load RWKV-7 "Goose" x070 and inference in GPT-mode (slower than RNN-mode for autoregressive generation)
'''

DTYPE = torch.bfloat16
# DTYPE = torch.half # better

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyStatic = torch.jit.script
# MyModule = nn.Module
# MyFunction = lambda fn: fn

########################################################################################################
# RWKV TimeMix
########################################################################################################

class RWKV_Tmix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.n_head = args.n_head
        self.head_size: int = args.head_size
        self.DTYPE = DTYPE

        H = args.n_head
        N = args.head_size
        C = args.n_embd

        self.x_r = nn.Parameter(torch.rand(1,1,C))
        self.x_w = nn.Parameter(torch.rand(1,1,C))
        self.x_k = nn.Parameter(torch.rand(1,1,C))
        self.x_v = nn.Parameter(torch.rand(1,1,C))
        self.x_a = nn.Parameter(torch.rand(1,1,C))
        self.x_g = nn.Parameter(torch.rand(1,1,C))

        self.w0 = nn.Parameter(torch.rand(1,1,C))
        self.w1 = nn.Parameter(torch.rand(C, args.D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.rand(args.D_DECAY_LORA, C))

        self.a0 = nn.Parameter(torch.rand(1,1,C))
        self.a1 = nn.Parameter(torch.rand(C, args.D_AAA_LORA))
        self.a2 = nn.Parameter(torch.rand(args.D_AAA_LORA, C))

        self.v0 = nn.Parameter(torch.rand(1,1,C))
        self.v1 = nn.Parameter(torch.rand(C, args.D_MV_LORA))
        self.v2 = nn.Parameter(torch.rand(args.D_MV_LORA, C))

        self.g1 = nn.Parameter(torch.rand(C, args.D_GATE_LORA))
        self.g2 = nn.Parameter(torch.rand(args.D_GATE_LORA, C))

        self.k_k = nn.Parameter(torch.rand(1,1,C))
        self.k_a = nn.Parameter(torch.rand(1,1,C))
        self.r_k = nn.Parameter(torch.rand(H,N))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5, dtype=self.DTYPE) # !!! notice eps value !!!

    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        # print(f"Input: min={x.min()}, max={x.max()}, mean={x.mean()}")
        x = self.RWKV7_OP(r, w, k, v, -kk, kk*a)
        # print(f"Input: min={x.min()}, max={x.max()}, mean={x.mean()}")
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first

    @MyFunction
    def RWKV7_OP(self, r, w, k, v, a, b):
        B, T, C = r.size()
        H = C // self.head_size
        N = self.head_size
        r = r.view(B, T, H, N).float()
        k = k.view(B, T, H, N).float()
        v = v.view(B, T, H, N).float()
        a = a.view(B, T, H, N).float()
        b = b.view(B, T, H, N).float()
        w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
        out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
        state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)

        for t in range(T):
            kk = k[:, t, :].view(B, H, 1, N)
            rr = r[:, t, :].view(B, H, N, 1)
            vv = v[:, t, :].view(B, H, N, 1)
            aa = a[:, t, :].view(B, H, N, 1)
            bb = b[:, t, :].view(B, H, 1, N)
            state = state * w[: , t, :, None, :] + state @ aa @ bb + vv @ kk
            out[:, t, :] = (state @ rr).view(B, H, N)

        return out.view(B, T, C).to(dtype=self.DTYPE)

########################################################################################################
# RWKV ChannelMix
########################################################################################################

class RWKV_CMix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            self.x_k = nn.Parameter(torch.rand(1, 1, args.n_embd))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x

        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

########################################################################################################
# RWKV Block
########################################################################################################

class Block(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln0 = nn.LayerNorm(args.n_embd) # only used in block 0, should be fused with emb
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

    @MyFunction
    def forward(self, x, v_first):

        if self.layer_id == 0:
            x = self.ln0(x)

        xx, v_first = self.att(self.ln1(x), v_first)
        x = x + xx
        x = x + self.ffn(self.ln2(x))

        return x, v_first

class RWKV(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.emb = nn.Embedding(model_args.vocab_size, model_args.n_embd)

        self.blocks = nn.ModuleList([Block(model_args, i) for i in range(model_args.n_layer)])

    def forward(self, idx):

        x = self.emb(idx)

        v_first = torch.rand_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)

        return x

class tongshun(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.rwkv = RWKV(args)
        self.ffn = nn.Sequential(
            nn.LayerNorm(args.n_embd*2),
            nn.Linear(args.n_embd * 2, args.final_ffn),
            nn.SiLU(),
            nn.Linear(args.final_ffn, args.final_ffn),
            nn.SiLU(),
            nn.Linear(args.final_ffn, args.final_ffn),
            nn.SiLU(),
            nn.Linear(args.final_ffn, 1)
        )

    def forward(self, x1, x2):
        x1 = self.rwkv(x1)[:, -1, :]
        x2 = self.rwkv(x2)[:, -1, :]
        # Shape: (batch_size, 2*args.n_embd)
        x = torch.cat([x1, x2], dim=1)
        x = self.ffn(x)
        x = x.squeeze(-1)
        return x

if __name__ == '__main__':
    from configs.test1_20M import args
    model = tongshun(args)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6}M")

    print(model(torch.tensor([[   1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
           1,    1,    1, 1261, 2427, 3445, 2815, 2368, 1451, 1010, 3017, 2368,
         883, 2088, 2115,   35, 2858,  424]], dtype=torch.int32), torch.tensor([[   1,    1,    1,    1,    1,    0,    0,  441, 3322, 2885]],dtype=torch.int32)))