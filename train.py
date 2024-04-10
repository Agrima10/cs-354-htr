import numpy as np
import string
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Lambda
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import tensorflow.keras.backend as K
from models import Model1
from preprocessing import preprocess_data

def train_model():
    train_images, train_labels, train_input_length, train_label_length, valid_images, valid_labels, valid_input_length, valid_label_length = preprocess_data()

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Define data paths and constants
    RECORDS_COUNT = 25000
    char_list = string.ascii_letters + string.digits + string.punctuation
    max_label_len = max(max([len(label) for label in train_labels]), max([len(label) for label in valid_labels]))

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
    batch_size = 10
    epochs = 25
    optimizer_name = 'sgd'

    # Compile the model
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer_name, metrics=['accuracy'])

    # Define the filepath for saving the best model
    filepath = "{}o-{}r-{}e-{}t-{}v.hdf5".format(optimizer_name,
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

    return history

# Call the train_model function to train the model
history = train_model()