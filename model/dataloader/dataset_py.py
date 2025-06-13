'''
纯 python 实现，不支持多进程(IterableDataset)，但看起来性能够用了
返回性状: ((上下文， 输入), 标签(0/1))
测试:
数据集大小: 1.1G
字典大小: 34M
平均耗时(ms): 0.002
峰值内存使用: 387.61 MB
'''
from torch.utils.data import IterableDataset, DataLoader
import random
from pathlib import Path
from marisa_trie import Trie
from itertools import repeat

def is_chinese(char):
    return '\u4e00' <= char <= '\u9fff'

_end_chars = {'。', '！', '？', '!', '?'}

class TongShunDataset(IterableDataset):
    def __init__(self, file_paths: list[Path], voca: list, chinese_only=True, negative_sample_rate:int=3):
        '''
        negative_sample_rate: 负样本采样率，但由于每个字符会生成三条数据，所以实际负样本采样率为 negative_sample_rate/3
        voca: 输入法可用词库
        '''
        self.file_paths = file_paths
        self.chinese_only = chinese_only
        self.negative_sample_rate = negative_sample_rate
        self.voca = voca

        # Build trie tree
        self.voca_tree = Trie([i[::-1] for i in voca])

    def data_generator(self, char, context, sentence):
        sentence += char
        if not (self.chinese_only and not is_chinese(char)):
            for i in self.get_y(context):
                l = len(i)
                # Full context
                yield (context[:-l], i), 1
                # Random context
                if (len(context) - l) > 15:
                    yield (context[random.randint(0, len(context) - l - 12):-l], i), 1
                # Full sentence(even if sentence is empty)
                if sentence != context:
                    yield (sentence[:-l], i), 1

            # Random negative sample
            _contexts = [context, sentence]
            for _ in repeat(None, self.negative_sample_rate):
                yield (random.choice(_contexts), random.choice(self.voca)), 0

    def get_y(self, string):
        y = self.voca_tree.prefixes(string[::-1])
        y = [i[::-1] for i in y]
        y.append(string[-1])
        # The users may input someting crazy, so let's add some randomness
        if len(string) >= 15:
            y.append(string[random.randint(0, len(string) - 12):])
        return y

    def __iter__(self):
        return self._generator()

    def _generator(self):
        for file_path in self.file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                context = ""
                sentence = ""
                while char := file.read(1):
                    if char == '\n':
                        context = ""
                        sentence = ""
                        continue
                    context += char
                    if char in _end_chars:
                        sentence = ""
                        continue
                    yield from self.data_generator(char, context, sentence)

# Test
if __name__ == "__main__":
    import tracemalloc
    import timeit
    tracemalloc.start()

    with open("./model/dict.txt", "r") as f:
        voca = f.read().splitlines()
    # data = TextFilesDataset([Path("test_data.txt")], voca)
    data = TongShunDataset([Path("/tmp/chinese_output.txt")], voca)
    t = iter(data)
    n = 1500

    # for _ in repeat(None, 100):
    #     print(next(t))

    print("平均耗时(ms):", timeit.timeit(lambda: next(t), number=n)*1000/n)

    current, peak = tracemalloc.get_traced_memory()
    print(f"峰值内存使用: {peak / 10**6:.2f} MB")
    tracemalloc.stop()


