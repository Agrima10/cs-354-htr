import app
from train import train_model
from preprocessing import preprocess_data
from test import test_model

def main():
    # Preprocessing the data
    preprocessed_data = preprocess_data()

    # Training the model
    trained_model = train_model(preprocessed_data)

    # Testing the model
    test_model(trained_model, preprocessed_data)

    # Running app
    app.run()

if __name__ == "__main__":
    main()