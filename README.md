# Handwriting Recognition 

## Overview
Handwritten Text Recognition (HTR) is a challenging domain within computer vision and image processing, with significant implications for various applications such as reading bank checks and prescriptions. Despite advancements in Optical Character Recognition (OCR) technology, recognizing handwritten text remains challenging due to factors like varied handwriting styles and limited datasets.

This project proposes a hybrid approach to address these challenges, aiming to enhance the accuracy of handwritten text recognition from images. By integrating Convolutional Neural Networks (CNN) and Bidirectional Long Short-Term Memory (BiLSTM) with a Connectionist Temporal Classification (CTC) decoder, the proposed model achieves substantial improvement in accuracy.

##Installation

1. Clone the repository:

```git clone https://github.com/Agrima10/cs-354-htr.git```

2. Navigate to project directory:

```cd CS-354-HTR```

3. Install dependencies:

```pip install -r requirements.txt```

## IAM dataset:

```The dataset used for this project can be downloaded from here: https://www.kaggle.com/datasets/bigkizd/iam-dataset```

## Preprocessing Data:

After downloading the dataset, preprocess it by:

```python preprocess.py```

This is going to save preprocessed np arrays in the form of .npy files which can be used later for training and testing.

## Training the model:

After the data is processed, the model can be trained using:

```python main.py```

The number of epochs and batch size can be customized. This also generates a training loss vs testing loss and a training accuracy vs testing accuracy graph. In the end the model is saved as a .hdf5 file which can be used later.

## Testing the data:

Jaro has been used to calculate the word error rate. Update the path for the saved model in test.py and run it using:

```python test.py```

This will generate jaro test scores and also demonstrate how the model works on randomly selected images from the testinng dataset.

## Running the application

Run the flask application with:

```python app.py```

Currently it uses the already saved model, but the path can be customized to use any desired model.

## Contributors:

This was a project made for the CS 354 Lab in the year 2024 under the guidance of Dr. Aruna Tiwari. The contributors are:

1. Sana Presswala (210001062)
2. Agrima Bundela (210002009)
3. Kirtan Kushwaha (210001030)

