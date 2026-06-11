import re
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Dataset
data = {
    "review": [
        "I loved this movie, it was fantastic!",
        "Worst movie ever, totally boring",
        "Amazing acting and great storyline",
        "I did not like the film at all",
        "Best movie of the year",
        "Terrible plot and bad acting",
        "Absolutely wonderful experience",
        "Waste of time and money",
        "Brilliant direction and screenplay",
        "Horrible movie"
    ],
    "sentiment": [1,0,1,0,1,0,1,0,1,0]
}

df = pd.DataFrame(data)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['review'] = df['review'].apply(clean_text)

# Logistic Regression Pipeline
lr_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", LogisticRegression())
])

# Naive Bayes Pipeline
nb_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])

lr_pipeline.fit(df['review'], df['sentiment'])
nb_pipeline.fit(df['review'], df['sentiment'])

def predict_sentiment(text, model_choice):
    text = clean_text(text)

    model = lr_pipeline if model_choice == "Logistic Regression" else nb_pipeline

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    confidence = round(np.max(probabilities) * 100, 2)

    fig, ax = plt.subplots()
    ax.bar(["Negative", "Positive"], probabilities)
    ax.set_ylim(0, 1)
    ax.set_title("Sentiment Probability Distribution")

    return sentiment, f"{confidence}%", fig

interface = gr.Interface(
    fn=predict_sentiment,
    inputs=[
        gr.Textbox(
            lines=4,
            placeholder="Type a movie review, tweet, or feedback..."
        ),
        gr.Radio(
            ["Logistic Regression", "Naive Bayes"],
            label="Choose Model",
            value="Logistic Regression"
        )
    ],
    outputs=[
        gr.Textbox(label="Predicted Sentiment"),
        gr.Textbox(label="Confidence Score"),
        gr.Plot(label="Sentiment Probability")
    ],
    title="Advanced Sentiment Analysis NLP App",
    description="""
    This application performs real-time sentiment analysis using NLP.

    • TF-IDF Vectorization
    • Machine Learning Models
    • Confidence & Probability Visualization
    """
)

interface.launch()

