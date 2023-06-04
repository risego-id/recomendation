import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the food data
food_data = pd.read_csv('./data/Food and Calories.csv')

# Load the user data
user_data = pd.read_csv('./data/User data.csv')

user_data['User_ID'] = range(1, len(user_data) + 1)
food_data['Food_ID'] = range(1, len(food_data) + 1)

# Create a rating column (calories needed) in the user data
def calculate_calories(row):
    if row['Gender'] == 'M':
        bmr = 10 * row['Weight'] + 6.25 * row['Height'] - 5 * row['Age'] + 5
    else:
        bmr = 10 * row['Weight'] + 6.25 * row['Height'] - 5 * row['Age'] - 161
    activity_level_mapping = {
        'sedentary': 1.2,
        'lightly active': 1.375,
        'moderately active': 1.55,
        'very active': 1.725,
        'extra active': 1.9
    }
    calories_needed = bmr * activity_level_mapping[row['Activity Level']]
    return calories_needed


user_data['Calories'] = user_data.apply(calculate_calories, axis=1)

# Merge user data and food data
merged_data = pd.merge(user_data, food_data, how='cross')

# Split the data into train and test sets
train_data, test_data = train_test_split(merged_data, test_size=0.2, random_state=42)

# Merge the 'Calories' column from user_data into train_data
train_data = pd.merge(train_data, user_data[['User_ID', 'Calories']], on='User_ID', how='left')

# Create user and item input tensors
user_ids = train_data['User_ID'].values
food_ids = train_data['Food_ID'].values

# Normalize the ratings (calories)
min_rating = train_data['Calories'].min()
max_rating = train_data['Calories'].max()
ratings = (train_data['Calories'].values - min_rating) / (max_rating - min_rating)

# Define the neural network model
embedding_dim = 8  # Number of latent dimensions for user and item embeddings
num_users = len(user_data)
num_food = len(food_data)

# Determine the maximum user and food IDs
max_user_id = merged_data['User_ID'].max()
max_food_id = merged_data['Food_ID'].max()

# Create user and item input tensors
user_input = tf.keras.layers.Input(shape=(1,))
food_input = tf.keras.layers.Input(shape=(1,))

user_embedding = tf.keras.layers.Embedding(max_user_id + 1, embedding_dim, input_length=1)(user_input)
food_embedding = tf.keras.layers.Embedding(max_food_id + 1, embedding_dim, input_length=1)(food_input)

# Adjust the learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

concatenated = tf.keras.layers.Concatenate()([user_embedding, food_embedding])
dense1 = tf.keras.layers.Dense(32, activation='relu')(concatenated)
output = tf.keras.layers.Dense(1, activation='linear')(dense1)

model = tf.keras.Model(inputs=[user_input, food_input], outputs=output)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Increase the number of epochs and batch size
history = model.fit([user_ids, food_ids], ratings, epochs=10, batch_size=128)

def get_food_recommendations(model, user_data, food_data, num_food, calorie_input):
    # Find the closest calorie value in the user data
    closest_calories = user_data['Calories'].values
    closest_index = np.argmin(np.abs(closest_calories - calorie_input))
    user_id = user_data.loc[closest_index, 'User_ID']

    # Predict ratings (calories) for all food items for the given user
    predicted_ratings = model.predict([np.array([user_id] * num_food), np.arange(1, num_food + 1)])

    # Sort the predicted ratings in descending order
    sorted_indices = np.argsort(predicted_ratings, axis=0)[::-1]

    # Get the top 10 food recommendations based on calories needed
    top_food_indices = sorted_indices[0:10, 0] + 1  # Add 1 to adjust for 0-based indexing

    # Retrieve the food names for the recommended food indices
    recommended_food = food_data.loc[top_food_indices.ravel(), 'Food'].tolist()

    return recommended_food

# Define the API endpoint
@app.route('/food_recommendations', methods=['POST'])
def food_recommendations():
    data = request.json  # Get the JSON data from the request
    data["Activity Level"] = "lightly active"
    calorie_input = calculate_calories(data)  # Extract the desired calorie value from the JSON data

    # Get the food recommendations
    recommendations = get_food_recommendations(model, user_data, food_data, num_food, calorie_input)

    # Return the food recommendations as a JSON response
    response = {
        'recommendations': recommendations[0:5],
        'recommendations_2': recommendations[5:10]
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
