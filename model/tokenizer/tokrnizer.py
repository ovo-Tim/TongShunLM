import json

class Tokenizer:
    def __init__(self, tokenizer_path):
        with open(tokenizer_path, "r") as f:
            self.tokenizer = json.load(f)

    def encode(self, text):
        return [self.tokenizer.get(token, 0) for token in text]

if  __name__ == "__main__":
    tokenizer = Tokenizer("model/tokenizer/tokenizer.json")
    print(tokenizer.encode("你好 123 abc / .  , ！"))

    import timeit
    print(timeit.timeit(lambda: tokenizer.encode("你好 123 abc / .  , ！"), number=100000))