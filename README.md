# Spam Email Detector

This project is a simple machine learning application that detects whether a message is spam or not spam.
I originally built this project as a basic implementation to understand how text classification works using Python and scikit-learn.<br><br>

Over time, I decided to improve the project by adding better structure, preprocessing, model saving, and evaluation features so that it feels closer to a real-world machine learning workflow.

---

## What This Project Was Initially

In the beginning, this project was a basic spam detection script that:

Loaded a dataset<br>
Converted text into vectors<br>
Trained a simple Naive Bayes model<br>
Predicted whether a message was spam or not

It worked correctly, but the structure was very simple and lacked features that are commonly used in practical machine learning projects.

---

## What Has Been Added and Improved

The project has now been updated with several improvements to make it more realistic and maintainable.

Added proper function-based code structure<br>
Implemented text preprocessing for cleaner input<br>
Added model accuracy evaluation<br>
Added model saving and loading using pickle<br>
Improved user interaction with a continuous input loop<br>
Added basic error handling for missing files

These changes make the project easier to extend and closer to how real machine learning systems are built.

---

## Features

Detects spam and non-spam messages<br>
Uses Natural Language Processing techniques<br>
Automatically trains the model if it does not exist<br>
Saves the trained model for future use<br>
Displays model accuracy after training<br>
Clean and modular function-based design

---

## Concepts Used

Text preprocessing<br>
Bag of Words model<br>
Naive Bayes classification<br>
Train and test data splitting<br>
Model persistence using pickle<br>
Accuracy evaluation

---

## Tech Stack

Python 3<br>
Pandas<br>
Scikit-learn

---

## Project Structure

SpamDetector<br>
│<br>
├── spam_detector.py<br>
├── spam.csv<br>
├── model.pkl<br>
├── vectorizer.pkl<br>
├── requirements.txt<br>
└── README.md

---

## Functions Explained

load_data(file_path)<br>
This function loads the dataset from a CSV file and prepares the columns for training.

preprocess_text(text)<br>
This function cleans the input text by converting it to lowercase and removing unnecessary characters.

train_model(messages, labels)<br>
This function trains the machine learning model using the training dataset and calculates accuracy.

save_model(model, vectorizer)<br>
This function saves the trained model and vectorizer so they can be reused later without retraining.

load_model()<br>
This function loads the saved model from disk if it already exists.

predict_message(model, vectorizer, message)<br>
This function predicts whether a message is spam or not.

main()<br>
This is the main program loop that handles training, loading, and user interaction.

---

## Example Usage

Enter a message:

Congratulations! You have won a free gift.

Prediction:

Spam

---

## Why I Updated This Project

I wanted to move beyond a simple beginner script and make the project more structured and practical.
The goal was to learn how machine learning models are trained, evaluated, saved, and reused in real applications.

This update also helped me practice writing cleaner code and organizing logic into reusable functions.

---

## Possible Future Improvements

Add TF-IDF vectorization<br>
Add support for email datasets from different sources<br>
Build a web interface using Flask or Streamlit<br>
Deploy the model as an API<br>
Improve accuracy using advanced NLP techniques

---

## Recent Updates

Added text preprocessing for better message cleaning
Added model accuracy display after training
Improved code structure using functions
Added requirements.txt for dependency management.
