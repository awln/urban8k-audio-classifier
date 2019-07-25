import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
#from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Activation, Dropout, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
import pickle
from Config import Config
import librosa
import random
    
def build_predictions(audio_dir):
    y_true = []
    y_pred = []
    fn_prob = {}
    
    print('Extracting features from audio')
    for fn in tqdm(os.listdir(audio_dir)):
        wav, rate = librosa.load(os.path.join(audio_dir, fn))
        

def build_rand_feat():
    #if os.path.isfile(os.path.join('pickles', 'conv.p')):
    #    print("Loading pickle model")
    #    with open(os.path.join('pickles', 'conv.p'), 'rb') as handle:
    #        tmp = pickle.load(handle)
            
    x = []
    y = []
    _min, _max = float('inf'), -float('inf') # to update
    for it in tqdm(range(len(df))): # for num of samples, select random to add to the list x and y
        #rand_class = np.random.choice(class_dist.index, p=prob_dist) # select a random class bassed on the distribution
        
        #file = np.random.choice(df[df.label==rand_class].index) # select a random file based on the class
        # print(df.at[file, 'fold'])
        #while df.at[file, 'fold'] == 10:
        #    file = np.random.choice(df[df.label==rand_class].index)
        fold = df.iloc[it]["fold"]
        file = df.iloc[it]["slice_file_name"]
        label = df.iloc[it]["label"]
        wav, rate = librosa.load('audio/fold' + str(fold) + '/' + file) # read rate and wave
        # wav, rate = librosa.load('data/clean/' + file)
        label = df.iloc[it]['label'] # set label var
        mfccs = np.mean(librosa.feature.mfcc(wav, rate, n_mfcc=40).T,axis=0)
        melspec = np.mean(librosa.feature.melspectrogram(y=wav, sr=rate, n_mels=40,fmax=8000).T,axis=0)
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=wav, sr=rate,n_chroma=40).T,axis=0)
        chroma_cq = np.mean(librosa.feature.chroma_cqt(y=wav, sr=rate,n_chroma=40).T,axis=0)
        chroma_cens = np.mean(librosa.feature.chroma_cens(y=wav, sr=rate,n_chroma=40).T,axis=0)
        featureset = np.reshape(np.vstack((mfccs,melspec,chroma_stft,chroma_cq,chroma_cens)),(40,5))
        if(fold != 10):
          x_train.append(featureset)
          y_train.append(classes.index(label))
        else:
          x_test.append(featureset)
          y_test.append(classes.index(label))
          
    #with open(os.path.join('pickles', 'conv.p'), 'rb') as handle:
    #    pickle.dump(config, handle, protocol=2)
          
# =============================================================================
#         while wav.shape[0]-config.step <= 0:
#             file = np.random.choice(df[df.label==rand_class].index)
#             #while df.at[file, 'fold'] == 10:
#             #    file = np.random.choice(df[df.label==rand_class].index)
#             wav, rate = librosa.load('data/clean/' + file) # read rate and wave
#             label = df.at[file, 'label'] # set label var
#             
#         rand_index = np.random.randint(0, wav.shape[0]-config.step) # rand index into wav file
#         sample = wav[rand_index:rand_index+config.step] # sample of wav file from (rand_index, config.step)
#         x_sample = mfcc(sample, rate,
#                         numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft) # mfcc on the sample
#         _min = min(np.amin(x_sample), _min)
#         _max = max(np.amin(x_sample), _max)
# =============================================================================
# =============================================================================
#         x.append(x_sample)
#         y.append(classes.index(label))
# =============================================================================
# =============================================================================
#     config.min = _min
#     config.max = _max
#     x, y = np.array(x), np.array(y)
#     x = (x - _min) / (_max - _min)
#     x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
#     y = to_categorical(y, num_classes=10)
#     print(x.shape)
#     print(y.shape)
#     folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=1).split(x, y))
#     return folds, x, y
# =============================================================================

def build_test_feat():
    pass

def get_alexnet_model():
    model = Sequential()
    # layer 1
    model.add(Conv2D(16, (3, 3), activation='relu', strides=(1,1),
                     padding='same', input_shape=input_shape))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
    
    # layer 2
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1,1), padding='same'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
    
    # layer 3
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1,1), padding='same'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
    
    # layer 4
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1,1),
                     padding='same'))
    # model.add(BatchNormalization())
        
    # layer 5
    #model.add(Conv2D(256, (3, 3), activation='relu', strides=(1,1),
    #                 padding='same'))
    # model.add(BatchNormalization())
    
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    # layer 6
    model.add(Dense(128, activation='relu'))
    # model.add(BatchNormalization())
    
    # layer 7
    model.add(Dense(64, activation='relu'))
    # model.add(BatchNormalization())
    
    # layer 8
    model.add(Dense(10, activation='softmax'))
    # model.add(BatchNormalization())
    
    model.summary()
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['acc'])
    return model

config = Config()

df = pd.read_csv('UrbanSound8k.csv')
#df.set_index('slice_file_name', inplace=True)
print(df.shape)

#for f in df.index:    
#    df.at[f, 'length'] = df.at[f, 'end'] - df.at[f, 'start']
    
df.rename(columns={'class' : 'label'}, inplace=True)
#df.reset_index()
classes = list(np.unique(df.label))
#class_dist = df.groupby(['label'])['length'].mean()

#num_samples = 2 * int(df['length'].sum()/0.1)
num_samples = int(30000)
#prob_dist = class_dist / class_dist.sum()

#choices = np.random.choice(class_dist.index, p=prob_dist)

#folds, x, y = build_rand_feat()
x_train = []
y_train = []
x_test = []
y_test = []

build_rand_feat()

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


#x_train_2d = np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
#x_test_2d = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
#x_train_2d.shape,x_test_2d.shape



#y_flat = np.argmax(y, axis=1)

x_train = np.reshape(x_train, (x_train.shape[0], 40, 5, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 40, 5, 1))
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
input_shape = (40, 5, 1)
model = get_alexnet_model()

#class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)

checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=True, mode='max')

model.fit(x_train,y_train,batch_size=50,epochs=30,validation_data=(x_test,y_test))

train_loss_score=model.evaluate(x_train,y_train)
test_loss_score=model.evaluate(x_test,y_test)
print(train_loss_score)
print(test_loss_score)
    
model.save(config.model_path)


