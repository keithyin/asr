import torch
from torch import nn
from data import THCHSDatasets, collate_fn
import torch.utils.data as torch_data
import os
from tqdm import tqdm


class SpeechModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=4, num_layers=1,
                            batch_first=True)
        self.fc = nn.Linear(in_features=4, out_features=num_classes)

    def forward(self, x):
        """
        :param x: [b, T]
        :return: x:[b, T, num_classes] 包含blank
        """
        x = torch.unsqueeze(x, -1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    root = "/Users/ky/datasets/data_thchs30/data/"
    file_names = os.listdir(root)
    file_paths = [os.path.join(root, file_name) for file_name in file_names if file_name.endswith("trn")]
    dataset = THCHSDatasets(file_paths, "vocab.txt")
    model = SpeechModel(num_classes=dataset.num_classes())
    loss_fn = nn.CTCLoss()
    model.load_state_dict(torch.load("ckpt-10600.pth"))
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)
    step = 0
    for epoch in range(100):
        loader = torch_data.DataLoader(dataset, collate_fn=collate_fn, batch_size=8, shuffle=True)

        for data in tqdm(loader):
            step += 1
            feature, label, feature_lens, label_lens = data
            # print(label, label_lens)
            x = model(feature)

            # [b, t, num_classes] -> [t, b, num_classes]
            x = torch.transpose(torch.log_softmax(x, dim=2), dim0=0, dim1=1)
            loss = loss_fn(x, label, feature_lens, label_lens)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if step % 50 == 0:
                print("epoch:{}, step:{}, loss:{}".format(epoch, step, loss))
                if step % 200 == 0:
                    torch.save(model.state_dict(), f"ckpt-{step}.pth")
