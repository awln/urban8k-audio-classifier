# urban8k-audio-classifier
<p> Urban8k sound classification based on an AlexNet convolutional neural network architecture </p>
<p> An important aspect is the preprocessing of the audio for defining characteristics. This is done by taking the short fourier transform of the signal with a small time frame, then applying the mel filter bank on the periodogram. We then calculate the mel cepstral coefficient of the filter bank energies then use this processed audio for machine learning.</p>
<p> Dataset found at https://urbansounddataset.weebly.com/urbansound8k.html </p>

### Dependencies
Developed with Python 3.7 with the following libraries:
* Tensorflow
* keras
* Librosa
* tqdm
* pandas
* matplotlib
* sklearn
* numpy
* pickle

## Exploratory Data Analysis
Sample audio distribution

<img src="https://raw.githubusercontent.com/awln/urban8k-audio-classifier/master/distribution.png"/>

Time Series on classes
<img src="https://raw.githubusercontent.com/awln/urban8k-audio-classifier/master/time_series.png"/>

short-time fast fourier transform on samples of different classes
<img src="https://raw.githubusercontent.com/awln/urban8k-audio-classifier/master/fft.png"/>

Filter Bank TFD on different classes
<img src="https://raw.githubusercontent.com/awln/urban8k-audio-classifier/master/filterbank.png"/>

Mel frequency cepstral coefficients
<img src="https://raw.githubusercontent.com/awln/urban8k-audio-classifier/master/mfcc.png"/>
    
## Testing and Training Analysis
Training on the audio was based off k-fold cross validation.

## Predictions

## Citations
http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
