import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


def load_data():
    data = pd.read_csv("spam.csv", encoding="latin-1")

    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']

    return data


def train_model(data):

    messages = data['message'].apply(preprocess_text)
    labels = data['label']

    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(messages)

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

    print("Model Accuracy:", accuracy)

    return model, vectorizer


def predict_message(model, vectorizer):

    while True:

        message = input(
            "Enter message (type 'exit' to quit): "
        )

        if message.lower() == "exit":
            break

        message = preprocess_text(message)

        message_vector = vectorizer.transform([message])

        prediction = model.predict(message_vector)

        if prediction[0] == "spam":
            print("Spam")
        else:
            print("Not Spam")


def main():

    print("Loading data...")

    data = load_data()

    print("Training model...")

    model, vectorizer = train_model(data)

    print("Model is ready.")

    predict_message(model, vectorizer)


if __name__ == "__main__":
    main()
