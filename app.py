#%%
from flask import Flask, render_template, request, jsonify
import dill
import pandas as pd
from flask import Flask
from flask_cors import CORS
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
#%%
def flatten_dict_list(dict_list):
    
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict
#%%
def get_song_data(song, spotify_data):
    
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    
    except IndexError:
        return None
#%%
number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

#%%
def get_mean_vector(song_list, spotify_data):
    
    song_vectors = []
    
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)
#%%

app = Flask(__name__)

user_feedback = {'liked_songs': [], 'disliked_songs': []}


CORS(app, resources={r"/": {"origins": "*"}})
# Load the recommendation function
with open('song_model.pkl', 'rb') as file:
    recommend_songs = dill.load(file)

# Load your data
data = pd.read_csv("/Users/shreya/Documents/GitHub/DATS-6103-FA-23-SEC-11/music-recommendation-system/data/data.csv")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    if request.method == 'POST':
        # Get data from the request
        input_data = request.get_json()
        songs = input_data.get('songs', [])

        # Call the recommendation function
        recommendations = recommend_songs(songs, data)

        # Return recommendations as JSON
        return jsonify(recommendations)
    

@app.route('/send_feedback', methods=['POST'])
def send_feedback():
    global user_feedback
    feedback_data = request.get_json()

    print("Received Feedback Data:", feedback_data)
    
    # Store feedback in the dictionary
    user_feedback['liked_songs'].extend(feedback_data.get('liked_songs', []))
    user_feedback['disliked_songs'].extend(feedback_data.get('disliked_songs', []))

    print("Updated User Feedback:", user_feedback)
    
    # Return a confirmation response
    return jsonify({'status': 'Feedback received successfully'})


if __name__ == '__main__':
    app.run(debug=True, port=5001)

# %%
user_feedback
# %%
