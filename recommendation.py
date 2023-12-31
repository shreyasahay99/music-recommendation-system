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
data = pd.read_csv("/Users/richikghosh/Documents/GitHub/music-recommendation-system/data/data.csv")
data_by_artist = pd.read_csv("/Users/richikghosh/Documents/GitHub/music-recommendation-system/data/data_by_artist.csv")
data_by_year=pd.read_csv("/Users/richikghosh/Documents/GitHub/music-recommendation-system/data/data_by_year.csv")
data_by_genre=pd.read_csv("/Users/richikghosh/Documents/GitHub/music-recommendation-system/data/data_by_genres.csv")
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

# def recommend_songs( song_list, spotify_data, n_songs=10):
#     scaler = StandardScaler()
#     metadata_cols = ['name', 'year', 'artists']
#     song_dict = flatten_dict_list(song_list)
#     song_center = get_mean_vector(song_list, spotify_data)
#     scaled_data = scaler.fit_transform(spotify_data[number_cols])
#     scaled_song_center = scaler.fit_transform(song_center.reshape(1, -1))
#     # This finds the euclidean distance
#     distances = cdist(scaled_song_center, scaled_data, 'cosine')
#     # Sorted in ascending order then choose the first n_songs
#     index = list(np.argsort(distances)[:, :n_songs][0])
#     # selecting the n_songs by index
#     rec_songs = spotify_data.iloc[index]
#     # removing the names of the songs which are present in the given list
#     rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
#     return rec_songs[metadata_cols].to_dict(orient='records')
#%%
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

def recommend_songs(song_list, spotify_data, n_songs=10, sort_by='release_date'):
    np.random.seed(42)  # Set the random seed for reproducibility
    scaler = StandardScaler()
    metadata_cols = ['name', 'year', 'artists']
    
    song_dict = flatten_dict_list(song_list)
    song_center = get_mean_vector(song_list, spotify_data)
    
    scaled_data = scaler.fit_transform(spotify_data[number_cols])
    scaled_song_center = scaler.fit_transform(song_center.reshape(1, -1))
    
    # Using k-means clustering with k = n_songs
    kmeans = KMeans(n_clusters=2*n_songs, random_state=42)
    kmeans.fit(scaled_data)
    
    # Find the cluster to which the song center belongs
    cluster_label = kmeans.predict(scaled_song_center)[0]
    
    # Get the indices of songs in the same cluster
    index = list(np.where(kmeans.labels_ == cluster_label)[0])
    
    # Selecting the n_songs by index
    rec_songs = spotify_data.iloc[index]
    
    # Removing the names of the songs which are present in the given list
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    
    # Sort the recommended songs by the specified criteria (e.g., popularity)
    rec_songs = rec_songs.sort_values(by=sort_by, ascending=False)
    
    # Return the top 10 songs
    rec_songs_top10 = rec_songs.head(n_songs)
    
    return rec_songs_top10[metadata_cols].to_dict(orient='records')

# %%
recommend_songs([{'name': 'Needed Me', 'year':2016},
                {'name': 'Neighbors', 'year': 2016},
                {'name': 'Feel No Ways', 'year': 2016},
                {'name': 'Middle', 'year': 2016},
                {'name': 'Try Everything', 'year': 2016}],  data)



#%%
recommend_songs([{'name': 'Come As You Are', 'year':1991},
                {'name': 'Smells Like Teen Spirit', 'year': 1991},
                {'name': 'Lithium', 'year': 1992},
                {'name': 'All Apologies', 'year': 1993},
                {'name': 'Stay Away', 'year': 1993}],  data)



#########################################################################################################################################################

# %%
# with open('song_model.pkl', 'wb') as file:
#     dill.dump(recommend_songs, file)

# # Load the function
# with open('song_model.pkl', 'rb') as file:
#     loaded_model = dill.load(file)

# # Use the loaded function
recommend_songs([{'name': 'Gati Bali', 'year':1921},
                 {'name': 'Danny Boy', 'year':1921}],  data)

# %%

# %%
data.head()
# %%
