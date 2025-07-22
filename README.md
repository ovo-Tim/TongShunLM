# TongShunLM
基于 RWKV-7 架构的用词通顺度打分模型，输入两个句子序列，输出通顺程度分数。模型参数量仅为 124k (占用空间 1.6MiB).


## 安装依赖：
1. 前往 [Pytorch 官网](https://pytorch.org/get-started/locally/) 安装 pytorch
2. `pip -r ./model/requirements.txt `

## 推理测试
``` bash
> python model/inference.py
context:不同模型在相同温度下的表现也存在
Test words(split by space):采用 参与 朝右 刺眼 差异
{'朝右': tensor([5.1472], grad_fn=<SqueezeBackward1>), '刺眼': tensor([5.4400], grad_fn=<SqueezeBackward1>), '参与': tensor([10.5972], grad_fn=<SqueezeBackward1>), '差异': tensor([12.1750], grad_fn=<SqueezeBackward1>), '采用': tensor([13.9028], grad_fn=<SqueezeBackward1>)}
context:但也并非是把温度设的越大就
Test words(split by space):优化 用户 一会 越好
{'用户': tensor([8.9341], grad_fn=<SqueezeBackward1>), '越好': tensor([10.5526], grad_fn=<SqueezeBackward1>), '优化': tensor([12.0162], grad_fn=<SqueezeBackward1>), '一会': tensor([13.4863], grad_fn=<SqueezeBackward1>)}
context:持续高血糖是引起糖尿病的主要原因,同时也会引起多种
Test words(split by space):并发症 煲仔饭 百分之 不放在 不负责
{'煲仔饭': tensor([-2.3387], grad_fn=<SqueezeBackward1>), '不放在': tensor([11.8902], grad_fn=<SqueezeBackward1>), '并发症': tensor([13.0803], grad_fn=<SqueezeBackward1>), '不负责': tensor([13.7572], grad_fn=<SqueezeBackward1>), '百分之': tensor([15.5749], grad_fn=<SqueezeBackward1>)}
```