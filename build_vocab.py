import os
from collections import Counter
from tqdm import tqdm

def extract_one_file(filename):
    with open(filename, mode="r") as file:
        for line_no, line in enumerate(file.readlines()):
            if line_no == 1:
                line = line.strip()
                items = line.split(" ")
                items = [item for item in items if item != ""]
                return items
    return []


def process_pinyin(pinyin):
    """

    :param pinyin:
    :return: 拼音和声调
    """
    tone = 0
    try:
        tone = int(pinyin[-1])
        pinyin = pinyin[:-1]
    except Exception as e:
        pass

    return pinyin, tone


def build_vocab(file_paths):
    counter = Counter()
    for file_path in tqdm(file_paths, desc="files"):
        items = extract_one_file(file_path)
        for item in items:
            pinyin, tone = process_pinyin(item)
            counter.update([pinyin, tone])
    return counter


def write_vocab(vocab_dict, o_file_path):
    with open(o_file_path, mode='w') as file:
        items = list(vocab_dict.items())
        items.sort(key=lambda x: x[1], reverse=True)
        for item in tqdm(items, "write vocab"):
            file.write("{}\n".format(item[0]))


if __name__ == '__main__':
    root = "/Users/ky/datasets/data_thchs30/data/"
    file_names = os.listdir(root)
    file_paths = [os.path.join(root, file_name) for file_name in file_names if file_name.endswith("trn")]
    vocab = build_vocab(file_paths)
    print(vocab)
    write_vocab(vocab, "vocab.txt")
