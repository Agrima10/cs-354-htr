import numpy as np
import cv2
import os
import pandas as pd
import string
import matplotlib.pyplot as plt

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import tensorflow.keras.backend as K

from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.saving import register_keras_serializable


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from PIL import Image

import tensorflow as tf
import matplotlib.pyplot as plt

import tensorflow.keras.backend as K
import Levenshtein as lv

char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

valid_images_path = 'valid_images.npy'
valid_images = np.load(valid_images_path)

valid_original_text_path = 'valid_original_text.npy'
valid_original_text = np.load(valid_original_text_path)

def Model1():
    # input with shape of height=32 and width=128 
    inputs = Input(shape=(32,128,1))

    # convolution layer with kernel size (3,3)
    conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
    # poolig layer with kernel size (2,2)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

    conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

    conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)

    conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)
    # poolig layer with kernel size (2,1)
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

    conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(conv_5)

    conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

    conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)

    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(blstm_1)

    outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)

    # model to be used at test time
    act_model = Model(inputs, outputs)
    
    return act_model,outputs,inputs
    
act_model,outputs,inputs=Model1()
act_model.summary()

filepath = '.\sgdo-25000r-25e-18074t-10b-2007v.hdf5'

# Load the saved best model weights
act_model.load_weights(filepath)

# Predict outputs on validation images
prediction = act_model.predict(valid_images)

# Use CTC decoder
decoded = K.ctc_decode(prediction, 
                        input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                        greedy=True)[0][0]
out = K.get_value(decoded)

total_jaro = 0

# See the results
for i, x in enumerate(out):
    letters = ''
    for p in x:
        if int(p) != -1:
            letters += char_list[int(p)]
    total_jaro += lv.jaro(letters, valid_original_text[i])

print('Jaro Score:', total_jaro / len(out))

# Predict outputs on a subset of validation images
i = 2000
j = 2005
prediction = act_model.predict(valid_images[i:j])

# Use CTC decoder
decoded = K.ctc_decode(prediction,   
                        input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                        greedy=True)[0][0]

out = K.get_value(decoded)

# See the results
for _, x in enumerate(out):
    print("Original Text: ", valid_original_text[i])
    print("Predicted Text: ", end='')
    for p in x:
        if int(p) != -1:
            print(char_list[int(p)], end='')
    plt.imshow(valid_images[i].reshape(32, 128), cmap=plt.cm.gray)
    plt.show()
    i += 1
    print('\n')

