import matplotlib.pyplot as plt
import torch
import json
import hifigan

def plot_mel(data, titles, n_attn=0, save_dir=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    if n_attn > 0:
        # Plot Mel Spectrogram
        plot_mel_(fig, axes, data[:-n_attn], titles)

        # Plot Alignment
        xlim = data[0].shape[1]
        for i in range(-n_attn, 0):
            im = axes[i][0].imshow(data[i], origin='lower', aspect='auto')
            axes[i][0].set_xlabel('Decoder timestep')
            axes[i][0].set_ylabel('Encoder timestep')
            axes[i][0].set_xlim(0, xlim)
            axes[i][0].set_title(titles[i], fontsize="medium")
            axes[i][0].tick_params(labelsize="x-small")
            axes[i][0].set_anchor("W")
            fig.colorbar(im, ax=axes[i][0])
    else:
        # Plot Mel Spectrogram
        plot_mel_(fig, axes, data, titles)

    fig.canvas.draw()
    # data = save_figure_to_numpy(fig)
    if save_dir is not None:
        plt.savefig(save_dir)
    # plt.close()
    return fig #, data

def plot_mel_(fig, axes, data, titles, tight_layout=True):
    if tight_layout:
        fig.tight_layout()

    for i in range(len(data)):
        mel = data[i]
        if isinstance(mel, torch.Tensor):
            mel = mel.detach().cpu().numpy()
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

path = "./LJSpeech-mel-LJ007-0055.npy"
import numpy as np
mel = np.load(path).T
mels = mel.reshape(1,80,388)
mels = torch.from_numpy(mels)
fig = plot_mel(
    [
        mel.T,
    ],["Synthetized Spectrogram"],save_dir="./test.png"
)
plt.close()

def get_vocoder(device):
    name = "HiFi-GAN"
    speaker = "LJSpeech"

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("./hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("./hifigan/generator_LJSpeech.pth.tar", map_location=device)
        elif speaker == "universal":
            ckpt = torch.load("./hifigan/generator_universal.pth.tar.zip", map_location=device)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder

def vocoder_infer(mels, vocoder, lengths=None):
    name = "HiFi-GAN"
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * 32768.0
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocoder = get_vocoder(device)

wav_predictions = vocoder_infer(
    mels, vocoder
)
print(len(wav_predictions))
from scipy.io import wavfile
wavfile.write("./test.wav",22050,wav_predictions[0])
