#%%
from flask import Flask, render_template, request, jsonify
import dill
import pandas as pd
from flask import Flask
from flask_cors import CORS
import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from flask_socketio import SocketIO
#%%

## Fucntions for the recommend songs

## Numeric column features
number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']



## Function to get the name of the song from the data

def get_song_data(song, spotify_data):
    
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    
    except IndexError:
        return None

## Finding a mean of the feature set selected previously   
def get_mean_vector(song_list, spotify_data):
    
    song_vectors = []

    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in the database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)

    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict
def adjust_weights_based_on_feedback(index, user_feedback):
    liked_songs = user_feedback['liked_songs']
    disliked_songs = user_feedback['disliked_songs']

    weights = np.ones(len(index))  # Initialize weights to 1

    for liked_song in liked_songs:
        if liked_song in index:
            weights[index.index(liked_song)] *= 15  # Increase the weight for liked songs

    for disliked_song in disliked_songs:
        if disliked_song in index:
            weights[index.index(disliked_song)] *= -0.1  # Decrease the weight for disliked songs

    # Normalize weights to sum to 1
    weights /= np.sum(weights)

    return weights

#%%
# Initializes the flask app
app = Flask(__name__)
socketio = SocketIO(app)


user_feedback = {'liked_songs': [], 'disliked_songs': []}
songs=[]

CORS(app, resources={r"/": {"origins": "*"}})
# Load the recommendation function
with open('song_model.pkl', 'rb') as file:
    recommend_songs = dill.load(file)

# Load your data
data = pd.read_csv("/Users/richikghosh/Documents/GitHub/music-recommendation-system/data/data.csv")


#Defines a route for the root URL
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    global current_songs
    if request.method == 'POST':
        # Get data from the request
        input_data = request.get_json()
        current_songs = input_data.get('songs', [])
        # Call the recommendation function
        recommendations = recommend_songs(current_songs, data,user_feedback)

        # Return recommendations as JSON
        return jsonify(recommendations)
    

@app.route('/send_feedback', methods=['POST'])
def send_feedback():
    global user_feedback
    global current_songs
    feedback_data = request.get_json()

    print("Received Feedback Data:", feedback_data)
    
    # Store feedback in the dictionary
    user_feedback['liked_songs'].extend(feedback_data.get('liked_songs', []))
    user_feedback['disliked_songs'].extend(feedback_data.get('disliked_songs', []))


    print("Updated user feedback:", user_feedback)
    print("Calling recommend_songs with songs:", current_songs)
    recommendations = recommend_songs(current_songs, data, user_feedback)
    print("Generated recommendations:", recommendations)


    # Return a confirmation response along with recommendations
    return jsonify({'status': 'Feedback received successfully', 'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)

# %%

# %%
