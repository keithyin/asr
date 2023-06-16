import torch
from torch.utils import data as torch_data
from tqdm import tqdm
import torchaudio
import os


def collate_fn(batch):
    feature_max_len = 0
    label_max_len = 0
    for instance in batch:
        feature_len = instance[2].numpy()
        label_len = instance[3].numpy()
        feature_max_len = max([feature_len, feature_max_len])
        label_max_len = max([label_len, label_max_len])

    padded_instance = []
    for idx in range(len(batch)):
        instance = batch[idx]
        feature, label, feature_len_ori, label_len_ori = instance
        feature_len = feature_len_ori.numpy()
        label_len = label_len_ori.numpy()
        feature_pad_len = feature_max_len - feature_len
        label_pad_len = label_max_len - label_len
        feature = THCHSDatasets.feature_pad(feature, feature_pad_len)
        label = THCHSDatasets.label_pad(label, label_pad_len)
        padded_instance.append([feature, label, feature_len_ori, label_len_ori])

    batch_features = torch.stack([instance[0] for instance in padded_instance], dim=0)
    batch_labels = torch.stack([instance[1] for instance in padded_instance], dim=0)
    batch_feature_lens = torch.stack([instance[2] for instance in padded_instance], dim=0)
    batch_label_lens = torch.stack([instance[3] for instance in padded_instance], dim=0)

    return batch_features, batch_labels, batch_feature_lens, batch_label_lens


class THCHSDatasets(torch_data.Dataset):
    EPSILON_IDX = 0
    UNK_IDX = 1
    PAD_IDX = 2
    SPACE_IDX = 3

    def __init__(self, subtitle_file_paths, vocab_path, sample_ratio=1.):
        self.vocab = {
            "<EPSILON>": THCHSDatasets.EPSILON_IDX,
            "<UNK>": THCHSDatasets.UNK_IDX,
            "<PAD>": THCHSDatasets.PAD_IDX,
            "<SPACE>": THCHSDatasets.SPACE_IDX
        }

        self.sample_ratio = sample_ratio

        self.id2token = self.build_vocab_dict(vocab_path)
        self.subtitle_file_paths = subtitle_file_paths
        pass

    def __getitem__(self, item):
        subtitle_path = self.subtitle_file_paths[item]
        wav_path = THCHSDatasets.subtitle_path_2_wav_path(subtitle_path)
        waveform, sample_rate = torchaudio.load(wav_path)
        sample_transform = torchaudio.transforms.Resample(sample_rate, new_freq=int(sample_rate/10))
        waveform = sample_transform(waveform)
        waveform = waveform[0]

        # print("waveform:{}".format(wavform), wavform.shape)
        # exit(0)
        label = self.build_label(subtitle_path)
        return (waveform, label,
                torch.tensor(waveform.shape[0], dtype=torch.int64),
                torch.tensor(label.shape[0], dtype=torch.int64))

    def __len__(self):
        return len(self.subtitle_file_paths)

    def build_vocab_dict(self, vocab_path):
        with open(vocab_path, mode="r") as file:
            for line in tqdm(file.readlines(), desc="vocab"):
                line = line.strip()
                self.vocab[line] = len(self.vocab)
        id2token = {v: k for k, v in self.vocab.items()}
        return id2token

    def build_label(self, subtitle_path):
        with open(subtitle_path, mode="r") as file:
            label_line = file.readlines()[1].strip()
            tokens = THCHSDatasets.tokenization(label_line, insert_space=True)
            idx = self.tokens2ids(tokens)
            return torch.tensor(idx, dtype=torch.int64)

    def tokens2ids(self, tokens):
        return [self.vocab[token] for token in tokens]

    def ids2tokens(self, ids):
        tokens = [self.id2token[id] for id in ids]
        return tokens

    def num_classes(self):
        return len(self.vocab)

    @staticmethod
    def subtitle_path_2_wav_path(subtitle_path):
        return subtitle_path[:-4]

    @staticmethod
    def tokenization(line, insert_space=False, insert_blank=False):
        line = line.strip()
        items = line.split(" ")
        items = [item for item in items if item != ""]
        tokens = []
        for pinyin in items:
            tokens.extend(THCHSDatasets.process_pinyin(pinyin))
            # 每个拼音后加一个space！用来区分不同拼音
            if insert_space:
                tokens.append("<SPACE>")
        if insert_space:
            tokens.pop()
        tokens = list(map(str, tokens))

        return tokens

    @staticmethod
    def process_pinyin(pinyin):
        tone = 0
        try:
            tone = int(pinyin[-1])
            pinyin = pinyin[:-1]
        except Exception as e:
            pass

        return pinyin, tone

    @staticmethod
    def feature_pad(ori, pad_len):
        if pad_len == 0:
            return ori
        pad_t = torch.zeros(size=[pad_len], dtype=torch.float32)
        return torch.concat([ori, pad_t], dim=0)

    @staticmethod
    def label_pad(ori, pad_len):
        if pad_len == 0:
            return ori
        pad_t = torch.ones(size=[pad_len], dtype=torch.int64) * THCHSDatasets.PAD_IDX
        return torch.concat([ori, pad_t], dim=0)


if __name__ == '__main__':
    root = "/Users/ky/datasets/data_thchs30/data/"
    file_names = os.listdir(root)
    file_paths = [os.path.join(root, file_name) for file_name in file_names if file_name.endswith("trn")]
    dataset = THCHSDatasets(file_paths, "vocab.txt")
    loader = torch_data.DataLoader(dataset, collate_fn=collate_fn, batch_size=2)
    for data in loader:
        feature, label, feature_lens, label_lens = data
        print(label, label_lens)
        break
