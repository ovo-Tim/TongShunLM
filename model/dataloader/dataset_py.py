'''
纯 python 实现，不支持多进程(IterableDataset)，但看起来性能够用了
返回性状: ((上下文， 输入), 标签(0/1))
测试:
数据集大小: 1.1G
字典大小: 34M
平均每个数据耗时(ms): 0.01 (Tokenized)
峰值内存使用: 387.61 MB
'''
from torch.utils.data import IterableDataset
from torch import tensor
import random
from pathlib import Path
from marisa_trie import Trie
from itertools import repeat
import torch

def pad_list(input_list, pad_len, pad_value):
    padding_size = max(0, pad_len - len(input_list))
    return [pad_value] * padding_size + input_list[-pad_len:]

def is_chinese(char):
    return '\u4e00' <= char <= '\u9fff'

_end_chars = {'。', '！', '？', '!', '?'}

class RandomQueue:
    def __init__(self):
        self._data = []

    def add(self, item):
        """Add an item to the queue."""
        self._data.append(item)

    def pop(self):
        """Remove and return a random item from the queue. O(1) time complexity."""
        i = random.randint(0, len(self._data) - 1)
        # Swap with the last item
        self._data[i], self._data[-1] = self._data[-1], self._data[i]
        return self._data.pop()  # Pop the last item

    def __len__(self):
        return len(self._data)

class TongShunDataset(IterableDataset):
    def __init__(self, file_paths: list[Path], voca: list, chinese_only=True, negative_sample_rate:int=3, tokenizer = lambda a:a, pad_len:tuple[int, int]=(30, 10), val_mode=False):
        '''
        negative_sample_rate: 负样本采样率，但由于每个字符会生成三条数据，所以实际负样本采样率为 negative_sample_rate/3
        voca: 输入法可用词库
        '''
        self.file_paths = file_paths
        self.chinese_only = chinese_only
        self.negative_sample_rate = negative_sample_rate
        self.tokenizer = tokenizer
        self.val_mode = val_mode
        self.voca = [self.tokenizer(i) for i in voca]

        self.pad_context_len, self.pad_y_len = pad_len

        # Build trie tree
        self.voca_tree = Trie([i[::-1] for i in voca])

        self.random_queue = RandomQueue()

        self._generator = self.generator()
        for _ in repeat(None, 30):
            self.random_queue.add(next(self._generator))

    def data_generator(self, char, context, sentence):
        sentence += char
        if not (self.chinese_only and not is_chinese(char)):
            for i in self.get_x(context):
                l = len(i)

                # Tokenize
                _context = self.tokenizer(context)
                _sentence = self.tokenizer(sentence)
                _i = self.tokenizer(i)

                # Full context
                yield (_context[:-l], _i), 1
                # Random context
                if (len(context) - l) > 15:
                    yield (_context[random.randint(0, len(context) - l - 12):-l], _i), 1
                # Full sentence(even if sentence is empty)
                if sentence != context:
                    yield (_sentence[:-l], _i), 1

            # Random negative sample
            _contexts = [_context, _sentence]
            for _ in repeat(None, self.negative_sample_rate):
                yield (random.choice(_contexts), random.choice(self.voca)), 0

    def pad_context(self, string):
        return pad_list(string, self.pad_context_len, 1)

    def pad_x(self, string):
        return pad_list(string, self.pad_y_len, 1)

    def get_x(self, string):
        y = self.voca_tree.prefixes(string[::-1])
        y = [i[::-1] for i in y]
        y.append(string[-1])
        # The users may input someting crazy, so let's add some randomness
        if len(string) >= 15:
            y.append(string[random.randint(0, len(string) - 12):])
        return y

    def __iter__(self):
        return self.shuffled()

    def generator(self):
        for file_path in self.file_paths:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                context = ""
                sentence = ""
                if self.val_mode:
                    file.seek(random.randint(0,10000))
                while char := file.read(1):
                    if char == '\n':
                        context = ""
                        sentence = ""
                        continue
                    context += char
                    if char in _end_chars:
                        sentence = ""
                        continue
                    # yield from self.data_generator(char, context, sentence)
                    for i in self.data_generator(char, context, sentence):
                        (con, x), y = i
                        yield (tensor(self.pad_context(con), dtype=torch.int32), tensor(self.pad_x(x), dtype=torch.int32)), tensor(y, dtype=torch.half)

    def shuffled(self):
        for i in self._generator:
            self.random_queue.add(i)
            yield self.random_queue.pop()

# Test
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))

    from tokenizer.tokrnizer import Tokenizer
    import tracemalloc
    import timeit
    tracemalloc.start()

    with open("./model/dict.txt", "r") as f:
        voca = f.read().splitlines()
    tokenizer = Tokenizer("./model/tokenizer/tokenizer.json")
    data = TongShunDataset([Path("./val_datas/test_data.txt")], voca, tokenizer=tokenizer.encode, val_mode=True)
    # data = TongShunDataset([Path("/tmp/chinese_output.txt")], voca, tokenizer=tokenizer.encode)
    t = iter(data)
    n = 1500

    for _ in repeat(None, 10):
        a = next(t)
        print(a[0][0], a[0][1])

    print("平均耗时(ms):", timeit.timeit(lambda: next(t), number=n)*1000/n)

    current, peak = tracemalloc.get_traced_memory()
    print(f"峰值内存使用: {peak / 10**6:.2f} MB")
    tracemalloc.stop()


