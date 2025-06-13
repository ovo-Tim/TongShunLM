'''
将 CCI4.0 数据集的 jsonl 文件转换为 txt 文件，只保留中文占比 ≥70% 的文本。
'''
import os
import ujson as json
import re

CHINESE_CHAR_PATTERN = re.compile(r'[\u4e00-\u9fa5]')

def chinese_ratio(text):
    """计算文本中中文字符所占比例"""

    s = int(len(text)/3)
    e = int(len(text)*2/3)
    chinese_count = sum(1 for _ in CHINESE_CHAR_PATTERN.finditer(text[s:e]))

    return chinese_count / (len(text) / 3)

def is_chinese_majority(text, threshold=0.7):
    """判断文本是否以中文为主（中文字符占比 >= threshold）"""
    return chinese_ratio(text) >= threshold

def extract_text_from_jsonl(jsonl_file_path):
    texts = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i > 8000:
                break
            try:
                data = json.loads(line.strip())
                if 'text' in data:
                    text = data['text']
                    if is_chinese_majority(text):  # 判断中文占比是否超 70%
                        texts.append(text.strip())
            except json.JSONDecodeError:
                print(f"警告：无法解析文件 {jsonl_file_path} 中的一行")
    return texts

def main(input_folder, output_file):
    all_texts = []

    # 遍历文件夹中的所有 .jsonl 文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(input_folder, filename)
            print(f"正在处理文件: {file_path}")
            texts = extract_text_from_jsonl(file_path)
            all_texts.extend(texts)

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for text in all_texts:
            f_out.write(text + '\n \n')

    print(f"完成！共提取并写入 {len(all_texts)} 条中文占比 ≥70% 的文本到 {output_file}")

# 示例用法：
if __name__ == '__main__':
    input_folder = '/tmp/'   # 替换为你的 jsonl 文件所在目录
    output_file = '/tmp/chinese_output.txt'   # 输出文本文件路径
    main(input_folder, output_file)