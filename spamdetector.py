import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def load_data(file_path):
    """
    Load dataset and clean columns
    """
    try:
        data = pd.read_csv(file_path, encoding="latin-1")
    except FileNotFoundError:
        print("Dataset file not found.")
        return None

    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']

    return data


def preprocess_text(text):
    """
    Basic text cleaning
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


def train_model(messages, labels):
    """
    Train spam detection model
    """
    vectorizer = CountVectorizer(stop_words='english')

    processed_messages = [preprocess_text(msg) for msg in messages]

    X = vectorizer.fit_transform(processed_messages)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels,
        test_size=0.2,
        random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Model Accuracy:", round(accuracy * 100, 2), "%")

    return model, vectorizer


def save_model(model, vectorizer):
    """
    Save trained model to file
    """
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)


def load_model():
    """
    Load saved model
    """
    try:
        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        return model, vectorizer
    except:
        return None, None


def predict_message(model, vectorizer, message):
    """
    Predict spam or not spam
    """
    message = preprocess_text(message)
    message_vector = vectorizer.transform([message])

    prediction = model.predict(message_vector)

    if prediction[0] == "spam":
        return "Spam"
    else:
        return "Not Spam"


def main():

    data = load_data("spam.csv")

    if data is None:
        return

    model, vectorizer = load_model()

    if model is None:

        print("Training model...")

        model, vectorizer = train_model(
            data['message'],
            data['label']
        )

        save_model(model, vectorizer)

        print("Model saved successfully.")

    print("\nSpam Email Detector is ready.\n")

    while True:

        message = input(
            "Enter a message (type 'exit' to quit): "
        )

        if message.lower() == "exit":
            print("Exiting program.")
            break

        result = predict_message(
            model,
            vectorizer,
            message
        )

        print("Prediction:", result)


if __name__ == "__main__":
    main()
