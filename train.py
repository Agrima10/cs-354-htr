# Import necessary libraries
import numpy as np
import cv2
import os
import pandas as pd
import string
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import tensorflow as tf
from models import Model1

# Ignore warnings in the output
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Define data paths and constants
RECORDS_COUNT = 25000
char_list = string.ascii_letters + string.digits + string.punctuation
max_label_len = 0

# Load and preprocess personalized data
# Assuming you have engl dataframe loaded here

train_images = []
train_labels = []
train_input_length = []
train_label_length = []
train_original_text = []

valid_images = []
valid_labels = []
valid_input_length = []
valid_label_length = []
valid_original_text = []

''' for i in range(len(engl)):
    splits_id = engl["image"][i].split('/')
    filepath = '../input/english-handwritten-characters-dataset/{}/{}'.format(splits_id[0], splits_id[1])

    # Process image
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    try:
        img = process_image(img)
    except:
        continue

    # Process label
    try:
        word = engl['label'][i]
        label = encode_to_labels(word)
    except:
        continue

    if i % 10 == 0:
        valid_images.append(img)
        valid_labels.append(label)
        valid_input_length.append(31)
        valid_label_length.append(len(word))
        valid_original_text.append(word)
    else:
        train_images.append(img)
        train_labels.append(label)
        train_input_length.append(31)
        train_label_length.append(len(word))
        train_original_text.append(word)

    if len(word) > max_label_len:
        max_label_len = len(word)

    if i >= RECORDS_COUNT:
        break
    '''

train_padded_label = pad_sequences(train_labels, maxlen=max_label_len, padding='post', value=len(char_list))
valid_padded_label = pad_sequences(valid_labels, maxlen=max_label_len, padding='post', value=len(char_list))

# Load the model
act_model, outputs, inputs = Model1()

the_labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, the_labels, input_length, label_length])

# Create the model for training
model = Model(inputs=[inputs, the_labels, input_length, label_length], outputs=loss_out)

# Define training parameters
batch_size = 5
epochs = 20
optimizer_name = 'sgd'

# Compile the model
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer_name, metrics=['accuracy'])

# Define the filepath for saving the best model
filepath="{}o-{}r-{}e-{}t-{}v.hdf5".format(optimizer_name,
                                             str(RECORDS_COUNT),
                                             str(epochs),
                                             str(len(train_images)),
                                             str(len(valid_images)))

# Set up ModelCheckpoint callback to save the best model
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

# Train the model
history = model.fit(x=[train_images, train_padded_label, train_input_length, train_label_length],
                    y=np.zeros(len(train_images)),
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=([valid_images, valid_padded_label, valid_input_length, valid_label_length],
                                     [np.zeros(len(valid_images))]),
                    verbose=1,
                    callbacks=callbacks_list)
