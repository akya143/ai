import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Flatten, Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'item_id': ['A', 'B', 'C', 'A', 'C', 'A', 'B', 'C', 'B', 'C'],
    'rating': [5, 4, 3, 4, 5, 3, 2, 5, 5, 3]
}
df = pd.DataFrame(data)
user_map = {user: idx for idx, user in enumerate(df['user_id'].unique())}
item_map = {item: idx for idx, item in enumerate(df['item_id'].unique())}
df['user_id'] = df['user_id'].map(user_map)
df['item_id'] = df['item_id'].map(item_map)
X = df[['user_id', 'item_id']].values
y = df['rating'].values
user_input = Input(shape=(1,))
item_input = Input(shape=(1,))
user_embedding = Embedding(input_dim=len(user_map), output_dim=10)(user_input)
item_embedding = Embedding(input_dim=len(item_map), output_dim=10)(item_input)
user_embedding = Flatten()(user_embedding)
item_embedding = Flatten()(item_embedding)
merged = Concatenate()([user_embedding, item_embedding])
dense = Dense(128, activation='relu')(merged)
dense = Dense(64, activation='relu')(dense)
output = Dense(1)(dense)
model = Model(inputs=[user_input, item_input], outputs=output)
model.compile(optimizer=Adam(), loss='mean_squared_error')
model.fit([X[:, 0], X[:, 1]], y, epochs=10, batch_size=2, verbose=1)
def recommend(user_id, top_n=3):
    # Get all items
    all_items = list(item_map.values())
    user_input_data = np.array([user_map[user_id]] * len(all_items))
    item_input_data = np.array(all_items)
    predictions = model.predict([user_input_data, item_input_data])
    item_ids_sorted = np.argsort(predictions.flatten())[::-1]
    recommended_item_ids = item_ids_sorted[:top_n]
    recommended_items = [list(item_map.keys())[list(item_map.values()).index(idx)] for idx in recommended_item_ids]
    
    return recommended_items
user_id = 1
recommended_items = recommend(user_id)
print(f"Top 3 recommendations for user {user_id}: {recommended_items}")

