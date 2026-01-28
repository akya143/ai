import nltk
import string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import movie_reviews
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
nltk.download('movie_reviews')
nltk.download('stopwords')
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
texts = [' '.join(doc) for doc, _ in documents]
labels = [category for _, category in documents]
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
classifier = LogisticRegression(max_iter=1000)
model = make_pipeline(vectorizer, classifier)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
example_text = ["I love this movie, it's amazing!", "I hate this movie, it's terrible."]
predictions = model.predict(example_text)
print("\nPredictions for Example Texts:")
for text, pred in zip(example_text, predictions):
    print(f"Text: {text} -> Predicted Sentiment: {pred}")
