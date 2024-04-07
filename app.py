from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf

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
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import backend as tf_keras_backend
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten

tf_keras_backend.set_image_data_format('channels_last')
tf_keras_backend.image_data_format()


app = Flask(__name__)

filepath='sgdo-25000r-25e-18074t-10b-2007v.hdf5'

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def process_image(img):
    """
    Converts image to shape (32, 128, 1) & normalize
    """
    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aspect Ratio Calculation
    w1, h1 = gray_img.shape
    new_w = 32
    new_h = int(h1 * (new_w / w1))
    img_resized = cv2.resize(gray_img, (new_h, new_w))

    w, h = img_resized.shape

    img_resized = img_resized.astype('float32')

    # Converts each to (32, 128, 1)
    if w < 32:
        add_zeros = np.full((32-w, h), 255)
        img_resized = np.concatenate((img_resized, add_zeros))
        w, h = img_resized.shape

    if h < 128:
        add_zeros = np.full((w, 128-h), 255)
        img_resized = np.concatenate((img_resized, add_zeros), axis=1)
        w, h = img_resized.shape

    if h > 128 or w > 32:
        dim = (128, 32)
        img_resized = cv2.resize(img_resized, dim)

    img_normalized = cv2.subtract(255, img_resized)
    img_normalized = np.expand_dims(img_normalized, axis=2)

    # Normalize
    img_normalized = img_normalized / 255

    return img_normalized


char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def squeeze_layer(x):
    return K.squeeze(x, 1)

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

    # import tensorflow as tf

    squeezed = layers.Lambda(squeeze_layer)(conv_7)


    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(blstm_1)

    outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)

    # model to be used at test time
    act_model = Model(inputs, outputs)

    return act_model,outputs,inputs

# Predict text from an image
def load_model_weights(model,filepath):
    try:
        model.load_weights(filepath)
    except Exception as e:
        print("Error loading model weights:", e)

def predict(img):
    r=""
    try:
        prediction = act_model.predict(np.array([img]))  # Use act_model instead of model

        # use CTC decoder
        decoded = K.ctc_decode(prediction,
                            input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                            greedy=True)[0][0]
        out = K.get_value(decoded)
        
        for i, x in enumerate(out):
            for p in x:
                if int(p) != -1:
                    r+=char_list[int(p)]
    except:
        print("hi2")
    return r

act_model,outputs,inputs=Model1()
load_model_weights(act_model,filepath)

# Define route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define route for the prediction
@app.route('/predict', methods=['POST'])
def predict_image():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['file']
        # Read the image file
        img_array = np.frombuffer(file.read(), np.uint8)
        # Decode the image
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # Process the image
        processed_img = process_image(image)
        print("ImageProcessed")
        # Predict text
        predicted_text = predict(processed_img)
        return render_template('result.html', predicted_text=predicted_text)

# def predict(img):
#     try:
#         test_img = process_image(img)
#     except:
#         print('Error processing image')
#         return 'Error processing image'

#     try:
#         prediction = act_model.predict(np.array([test_img]))
#         # Use CTC decoder
#         decoded = K.ctc_decode(prediction,
#                             input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
#                             greedy=True)[0][0]
#         out = K.get_value(decoded)

#         predicted_text = ""
#         for i, x in enumerate(out):
#             for p in x:
#                 if int(p) != -1:
#                     s = char_list[int(p)]
#                     predicted_text += s
#         return predicted_text
#     except Exception as e:
#         print('Error predicting text:', e)
#         return 'Error predicting text'

if __name__ == '__main__':
    app.run(debug=True)