import numpy as np
import matplotlib.pyplot as plt

import tensorflow.keras.backend as K
import Levenshtein as lv

def test_model(act_model, valid_images, valid_original_text, char_list):
    filepath = './sgdo-25000r-25e-21143t-2348v.hdf5'
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

    # Plot accuracy and loss
    def plotgraph(epochs, acc, val_acc):
        # Plot training & validation accuracy values
        plt.plot(epochs, acc, 'b')
        plt.plot(epochs, val_acc, 'r')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plotgraph(epochs, loss, val_loss)
    plotgraph(epochs, acc, val_acc)

    # Get the index of the best model
    minimum_val_loss = np.min(history.history['val_loss'])
    best_model_index = np.where(history.history['val_loss'] == minimum_val_loss)[0][0]

    best_loss = str(history.history['loss'][best_model_index])
    best_acc = str(history.history['accuracy'][best_model_index])
    best_val_loss = str(history.history['val_loss'][best_model_index])
    best_val_acc = str(history.history['val_accuracy'][best_model_index])

    print("Best Loss:", best_loss)
    print("Best Accuracy:", best_acc)
    print("Best Validation Loss:", best_val_loss)
    print("Best Validation Accuracy:", best_val_acc)