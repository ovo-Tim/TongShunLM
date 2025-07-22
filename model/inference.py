from tokenizer.tokrnizer import Tokenizer
import torch
from torch import tensor
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from model.tongshun_model_rwkv import tongshun
    model_type = tongshun
else:
    model_type = Any

# Settings
model_conf = "configs.test7_124k"
model_def = "tongshun_model_rwkv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_args = getattr(__import__(model_conf, fromlist=["args"]), "args")
model: model_type = getattr(__import__(model_def, fromlist=["tongshun"]), "tongshun")(model_args)

# Dirty transform
_state_dict: dict = torch.load('pretrained.pt', map_location=device)['state_dict']
state_dict = {}
for k, v in _state_dict.items():
    state_dict[k[6:]] = v

model.load_state_dict(state_dict)
model.eval()
tokrnizer = Tokenizer("model/tokenizer/tokenizer.json")

while True:
    x1 = input("context:")
    x2 = input("Test words(split by space):").split(' ')
    x1 = tensor([tokrnizer.encode(x1)])
    res = {}
    for i in x2:
        _i = tensor([tokrnizer.encode(i)])
        res[i] = model(x1, _i)
    print(dict(sorted(res.items(), key=lambda item: item[1])))