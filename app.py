from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model
from keras import backend as K 

app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')



# Define image dimensions expected by the model
# image_width = 28
# image_height = 28

# Preprocess the input image
def process_image(img):
    """
    Converts image to shape (32, 128, 1) & normalize
    """
    w, h = img.shape

#     _, img = cv2.threshold(img,
#                            128,
#                            255,
#                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Aspect Ratio Calculation
    new_w = 32
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape

    img = img.astype('float32')

    # Converts each to (32, 128, 1)
    if w < 32:
        add_zeros = np.full((32-w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape

    if h < 128:
        add_zeros = np.full((w, 128-h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape

    if h > 128 or w > 32:
        dim = (128,32)
        img = cv2.resize(img, dim)

    img = cv2.subtract(255, img)

    img = np.expand_dims(img, axis=2)

    # Normalize
    img = img / 255

    return img

char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Predict text from an image
def predict(img):
    try:
        test_img = process_image(img)
    except:
        print('hi')

    try:
        prediction = model.predict(np.array([test_img]))

        # use CTC decoder
        decoded = K.ctc_decode(prediction,
                            input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                            greedy=True)[0][0]
        out = K.get_value(decoded)

        for i, x in enumerate(out):
            print("predicted text = ", end = '')
            for p in x:
                if int(p) != -1:
                    s=char_list[int(p)]
                    print(s)
    except:
        print("hi2")



predict('../input/my-own/alphabet/char')

# Define route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['file']
        # Read the image file
        img_array = np.frombuffer(file.read(), np.uint8)
        # Decode the image
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # Predict text
        predicted_text = predict(image)
        return render_template('result.html', predicted_text=predicted_text)

if __name__ == '__main__':
    app.run(debug=True)
