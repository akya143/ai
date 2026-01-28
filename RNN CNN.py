import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data (For a full implementation, replace this with a larger dataset)
texts = [
    "I love this movie",
    "This movie is terrible",
    "Amazing plot and great acting",
    "I hate this movie",
    "The movie was okay, not great",
    "Fantastic experience, very enjoyable",
    "Not worth watching",
    "I would watch it again, highly recommend",
    "I dislike the plot but the acting was good",
    "A masterpiece in cinema"
]
labels = [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]  # 1: Positive, 0: Negative

# Tokenize the text data
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)

# Pad sequences to ensure they are of the same length
X = pad_sequences(X, maxlen=10)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Ensure data is in the right format
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Define the LSTM-based RNN model
model_rnn = models.Sequential([
    layers.Embedding(input_dim=10000, output_dim=64, input_length=10),
    layers.LSTM(64, return_sequences=False),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification (positive/negative)
])

model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the RNN model (LSTM)
model_rnn.fit(X_train, y_train, epochs=5, batch_size=2, validation_data=(X_test, y_test))

# Evaluate the model
y_pred_rnn = (model_rnn.predict(X_test) > 0.5).astype(int)

print("RNN (LSTM) Accuracy:", accuracy_score(y_test, y_pred_rnn))

