import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def calc_fft(sig_in, rate_in):
    n = len(sig_in) # signal length
    freq = np.fft.rfftfreq(n, d=1/rate_in) # amount of time between each sample, map to freq
    mag_sig = abs(np.fft.rfft(sig_in)/n) # normalize for length of signal, scaled by length of signal balanced for long signals
    return mag_sig, freq

def envelope(sig_in, rate_in, threshold):
    sig = pd.Series(sig_in).apply(np.abs)
    mask = []
    sig_mean = sig.rolling(window=int(rate_in/10), min_periods=1, center=True).mean()
    for mean in sig_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def main():
    df = pd.read_csv('UrbanSound8K.csv')
    df.set_index('slice_file_name', inplace=True)
    for f in df.index:
        df.at[f, 'length'] = df.at[f, 'end'] - df.at[f, 'start']
    
    df.rename(columns={'class' : 'label'}, inplace=True)
    classes = list(np.unique(df.label))
    class_dict = df.groupby(['label'])['length'].mean()

    fig, ax = plt.subplots()
    ax.set_title('Class Distribution', y=1.08)
    ax.pie(class_dict, labels=class_dict.index, autopct='%1.1f%%', shadow=False, startangle=90)
    ax.axis('equal')
    plt.savefig("distribution")
    plt.show(block=False)
    df.reset_index(inplace=True)

    signals = {}
    fft = {}
    fbank = {}
    mfccs = {}

    for c in classes:
        wav_file = df[df.label == c].iloc[1][0] # df filename
        fold = df[df.label == c].iloc[1][5]
        signal, rate = librosa.load('audio/fold'+str(fold)+'/'+wav_file, sr=44100)
        mask = envelope(signal, rate, 0.000005)
        signal = signal[mask]
        signals[c] = signal
        fft[c] = calc_fft(signal, rate) 

        bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103)
        fbank[c] = bank
        mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
        mfccs[c] = mel
    
    plot_signals(signals)
    plt.savefig("time_series")
    plt.show()
    
    plot_fft(fft)
    plt.savefig("fft")
    plt.show()

    plot_fbank(fbank)
    plt.savefig("filterbank")
    plt.show()

    plot_mfccs(mfccs)
    plt.savefig("mfcc")
    plt.show()
    
    df.set_index('slice_file_name', inplace=True)
    
    if len(os.listdir('clean_data')) == 0:
        for f in tqdm(df.index):
            signal, rate = librosa.load('audio/fold'+str(df.at[f, 'fold'])+'/'+f, sr=16000)
            mask = envelope(signal, rate, 0.000005)
            wavfile.write(filename='clean_data/' + f, rate=rate, data=signal[mask])



if __name__ == "__main__":
    main()