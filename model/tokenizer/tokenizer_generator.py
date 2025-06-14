import json

'''
0: Unknown char
1: pad
'''

t = {}
with open("./model/tokenizer/vocab.txt") as f:
    for i in f.read():
        t[i]=len(t)+2

with open("./model/tokenizer/tokenizer.json", 'w') as f:
    json.dump(t, f, ensure_ascii=False)