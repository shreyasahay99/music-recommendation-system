#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import dill
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle

#%% [markdown]
## Data Import
#%%
data = pd.read_csv("/Users/shreya/Documents/GitHub/DATS-6103-FA-23-SEC-11/music-recommendation-system/data/data.csv")
data_by_artist = pd.read_csv("/Users/shreya/Documents/GitHub/DATS-6103-FA-23-SEC-11/music-recommendation-system/data/data_by_artist.csv")
data_by_year=pd.read_csv("/Users/shreya/Documents/GitHub/DATS-6103-FA-23-SEC-11/music-recommendation-system/data/data_by_year.csv")
data_by_genre=pd.read_csv("/Users/shreya/Documents/GitHub/DATS-6103-FA-23-SEC-11/music-recommendation-system/data/data_by_genres.csv")
#data_wt_genre=pd.read_csv("/Users/richikghosh/Documents/GitHub/music-recommendation-system/data/data_w_genres.csv")

#%%


#%%
####################################################
############ Recommendation System ################
###################################################

from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib

## Numeric column features
number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


#%%

## Function to get the name of the song from the data

def get_song_data(song, spotify_data):
    
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    
    except IndexError:
        return None

#%%

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

#%%

def flatten_dict_list(dict_list):
    
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict

# %%
def recommend_songs( song_list, spotify_data, n_songs=10):

    # metadata_cols = ['name', 'year', 'artists']
    # song_dict = flatten_dict_list(song_list)
    # song_center = get_mean_vector(song_list, spotify_data)
    # scaled_data = StandardScaler().fit_transform(spotify_data[number_cols])
    # scaled_song_center = StandardScaler().fit_transform(song_center.reshape(1, -1))

    # # Reshape scaled_song_center to be 2D
    # scaled_song_center = scaled_song_center.reshape(-1, len(number_cols))

    # distances = cdist(scaled_song_center, scaled_data, 'euclidean').flatten()
    # sorted_indices = np.argsort(distances)
    # index = sorted_indices[:n_songs]
    # rec_songs = spotify_data.iloc[index]
    # rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    # return rec_songs[metadata_cols].to_dict(orient='records')

    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    song_center = get_mean_vector(song_list, spotify_data)

    # Calculate correlation between input features and song features
    correlations = np.corrcoef(song_center, rowvar=False)[0]
    recommendation_scores = np.sum(np.multiply(correlations, scaled_data), axis=1)


    # Sort distances in ascending order
    sorted_indices = np.argsort(recommendation_scores)[:n_songs]

    # Selecting the n_songs by index
    rec_songs = spotify_data.iloc[sorted_indices]

    # Removing the names of the songs which are present in the given list
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]

    return rec_songs[metadata_cols].to_dict(orient='records')

# %%
recommend_songs([{'name': 'Needed Me', 'year':2016},
                {'name': 'Neighbors', 'year': 2016},
                {'name': 'Feel No Ways', 'year': 2016},
                {'name': 'Middle', 'year': 2016},
                {'name': 'Try Everything', 'year': 2016}],  data)






#########################################################################################################################################################

# %%
# with open('song_model.pkl', 'wb') as file:
#     dill.dump(recommend_songs, file)

# # Load the function
# with open('song_model.pkl', 'rb') as file:
#     loaded_model = dill.load(file)

# # Use the loaded function
# result = loaded_model([{'name': 'you broke me first', 'year':2020},
#                        {'name': 'Dynamite', 'year': 2000},
#                        {'name': 'Therefore I Am', 'year': 2020},
#                        {'name': 'WAP (feat. Megan Thee Stallion)', 'year': 2020},
#                        {'name': 'Levitating (feat. DaBaby)', 'year': 2000}],  data)
# %%

# %%
