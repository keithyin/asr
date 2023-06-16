import torch
import torchaudio
import torchaudio.functional as audio_f
import torchaudio.transforms as audio_t
import matplotlib.pyplot as plt


def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show(block=True)


def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.show(block=True)


if __name__ == '__main__':

    # print(torch.__version__)
    # print(torchaudio.__version__)
    # waveform, sample_rate = torchaudio.load("/Users/ky/datasets/data_thchs30/train/C8_749.wav")
    # plot_waveform(waveform, sample_rate)
    # plot_specgram(waveform, sample_rate)
    # print("sample_rate:{}".format(sample_rate))
    # print("waveform:{}".format(waveform))
    a = torch.tensor(list(range(100)), dtype=torch.int64)
    print(int(a.shape[0] * 0.8))
