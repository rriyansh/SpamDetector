import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def load_data(file_path):
    """
    Load and clean dataset
    """
    data = pd.read_csv(file_path, encoding="latin-1")
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']

    data['label'] = data['label'].map({
        'ham': 0,
        'spam': 1
    })

    return data


def train_model(messages, labels):
    """
    Train the ML model
    """
    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(messages)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    return model, vectorizer


def predict_email(model, vectorizer, email_text):
    """
    Predict if email is spam or not
    """
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)

    return "Spam 🚨" if prediction[0] == 1 else "Not Spam ✅"


def main():
    data = load_data("spam.csv")

    model, vectorizer = train_model(
        data['message'],
        data['label']
    )

    email = input("Enter an email message: ")

    result = predict_email(model, vectorizer, email)

    print("\nPrediction:", result)


if __name__ == "__main__":
    main()
