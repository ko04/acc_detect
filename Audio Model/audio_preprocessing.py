import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import librosa
from IPython.display import Audio, display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_RATE = 16000     
FFT_SIZE = 626                  # chosen so that resulting size is 256x256
MEL_BINS = 256          
MEL_SPECTROGRAM = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,    # frequency in Hz
    n_fft=FFT_SIZE,             # # of samples used for one STFT
    win_length=FFT_SIZE,        # same as n_fft
    hop_length=int(FFT_SIZE/2), # # of samples by which to slide STFT by every time - default FFT_SIZE/2
    n_mels=MEL_BINS             # # of frequency bands in STFT (height of y-axis)
)


def load_wav(wav):
    # load takes path and returns tensor representation and sample rate
    # waveform tensor is default shape (channels, )
    waveform, _ = torchaudio.load(wav)
    # downsample to standard sample rate
    waveform = torchaudio.functional.resample(waveform, _, SAMPLE_RATE)
    # convert to mono if audio is stereo
    return torch.mean(waveform, dim=0), SAMPLE_RATE


def play_audio(waveform, sample_rate):
    waveform = waveform.numpy()
    # sometimes doesn't work in vscode
    display(Audio(waveform, rate=sample_rate))


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_frames = waveform.shape[0]
    time_axis = torch.arange(0, num_frames) / sample_rate

    fig, axes = plt.subplots(1, 1)
    axes.plot(time_axis, waveform, linewidth=1)
    axes.grid(True)
    if xlim:
        axes.set_xlim(xlim)
    if ylim:
        axes.set_ylim(ylim)
    axes.set_title(title)
    plt.show(block=False)


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)
