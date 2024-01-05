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



#### K Means Clustering ####


#### K Means Clustering ####


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


def recommend_songs(song_list, spotify_data, n_songs=10, sort_by='popularity'):
    np.random.seed(42)  # Set the random seed for reproducibility
    metadata_cols = ['name', 'year', 'artists']
    
    song_dict = flatten_dict_list(song_list)
    song_center = get_mean_vector(song_list, spotify_data)
    
    # Use StandardScaler to scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    
    # Use PCA for dimensionality reduction
    pca = PCA(n_components=len(spotify_data[number_cols].columns))
    scaled_data_pca = pca.fit_transform(scaled_data)
    scaled_song_center_pca = pca.transform(scaled_song_center)
    
    # Using k-means clustering with k = n_songs
    kmeans = KMeans(n_clusters=2 * n_songs, random_state=42)
    kmeans.fit(scaled_data_pca)
    
    # Find the cluster to which the song center belongs
    cluster_label = kmeans.predict(scaled_song_center_pca)[0]
    
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
#%%

########
#### Self-Organizing Maps (SOM) ####

#######
from minisom import MiniSom
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# def recommend_songs(song_list, spotify_data, n_songs=10, sort_by='year', som_seed=None):
#     np.random.seed(42)  # Set the random seed for reproducibility
#     metadata_cols = ['name', 'year', 'artists']

#     song_dict = flatten_dict_list(song_list)
#     song_center = get_mean_vector(song_list, spotify_data)

#     # Use StandardScaler to scale the data
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(spotify_data[number_cols])
#     scaled_song_center = scaler.transform(song_center.reshape(1, -1))

#     # Use PCA for dimensionality reduction
#     pca = PCA(n_components=len(spotify_data[number_cols].columns))
#     scaled_data_pca = pca.fit_transform(scaled_data)
#     scaled_song_center_pca = pca.transform(scaled_song_center)

#     # Using Self-Organizing Maps (SOM)
#     if som_seed is not None:
#         np.random.seed(som_seed)
#     som_size = (2 * n_songs, 2)
    
#     som = MiniSom(som_size[0], som_size[1], scaled_data_pca.shape[1], sigma=0.3, learning_rate=0.09)
#     som.train_random(scaled_data_pca, 1000)  # You may need to adjust the number of training iterations

#     # Find the best-matching unit (BMU) for the song center
#     bmu_coords = som.winner(scaled_song_center_pca)

#     # Get the indices of songs in the same cluster (considering 1D SOM)
#     index = [i for i, coords in enumerate(som.win_map(scaled_data_pca).get(bmu_coords, []))]

#     # Selecting the n_songs by index
#     rec_songs = spotify_data.iloc[index]

#     # Removing the names of the songs which are present in the given list
#     rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]

#     # Sort the recommended songs by the specified criteria (e.g., popularity)
#     rec_songs = rec_songs.sort_values(by=sort_by, ascending=False)

#     # Return the top 10 songs
#     rec_songs_top10 = rec_songs.head(n_songs)

#     return rec_songs_top10[metadata_cols].to_dict(orient='records')


#%%
# from sklearn.mixture import GaussianMixture
# from sklearn.preprocessing import MinMaxScaler  # Use MinMaxScaler instead of StandardScaler
# import numpy as np

# def recommend_songs(song_list, spotify_data, n_songs=10, sort_by='release_date'):
#     # np.random.seed(42)  # Set the random seed for reproducibility
#     scaler = MinMaxScaler()  # Use MinMaxScaler instead of StandardScaler
#     metadata_cols = ['name', 'year', 'artists']
    
#     song_dict = flatten_dict_list(song_list)
#     song_center = get_mean_vector(song_list, spotify_data)
    
#     scaled_data = scaler.fit_transform(spotify_data[number_cols])
#     scaled_song_center = scaler.fit_transform(song_center.reshape(1, -1))
    
#     # Using Gaussian Mixture Models with n_components = n_songs
#     gmm = GaussianMixture(n_components=n_songs, random_state=42)
#     gmm.fit(scaled_data)
    
#     # Find the cluster probabilities for the song center
#     cluster_probs = gmm.predict_proba(scaled_song_center)[0]
    
#     # Choose the cluster with the highest probability
#     cluster_label = np.argmax(cluster_probs)
   
#     # Get the indices of songs in the same cluster
#     index = list(np.where(gmm.predict(scaled_data) == cluster_label)[0])
    
#     # Selecting the n_songs by index
#     rec_songs = spotify_data.iloc[index]
    
#     # Removing the names of the songs which are present in the given list
#     rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    
#     # Sort the recommended songs by the specified criteria (e.g., popularity)
#     rec_songs = rec_songs.sort_values(by=sort_by, ascending=False)
    
#     # Return the top 10 songs
#     rec_songs_top10 = rec_songs.head(n_songs)
    
#     return rec_songs_top10[metadata_cols].to_dict(orient='records')

# %%
recommend_songs([{'name': 'Needed Me', 'year':2016},
                {'name': 'Neighbors', 'year': 2016},
                {'name': 'Feel No Ways', 'year': 2016},
                {'name': 'Middle', 'year': 2016},
                {'name': 'Try Everything', 'year': 2016}],  data)



#%%

recommend_songs([{'name': 'Smells Like Teen Spirit', 'year': 1991}],  data)

#########################################################################################################################################################

# %%
with open('song_model.pkl', 'wb') as file:
    dill.dump(recommend_songs, file)

# # Load the function
# with open('song_model.pkl', 'rb') as file:
#     loaded_model = dill.load(file)

# # Use the loaded function
recommend_songs([{'name': 'Come Back To Erin', 'year':1921},
                 {'name': 'When We Die', 'year':1921}],  data)

# %%
recommend_songs([{'name': 'Sweater Weather', 'year':2013},
                {'name': 'Pompeii', 'year': 2013},
                {'name': 'You Know You Like It', 'year': 2014},
                {'name': 'Gucci Gang', 'year': 2017},
                {'name': 'I <3 My Choppa', 'year': 2017}],  data)

# %%
data.columns
# %%

#%%
# merge_columns=['mode', 'key','energy']
# data_y=data_by_genre[['mode', 'key','genres','energy']]
# merged_data = pd.merge(data,data_y , how='inner', on=merge_columns)




# %%


# %%
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# def collect_feedback():
#     positive_feedback = input("Enter positive feedback (comma-separated): ").split(',')
#     negative_feedback = input("Enter negative feedback (comma-separated): ").split(',')
#     return [feedback.strip() for feedback in positive_feedback], [feedback.strip() for feedback in negative_feedback]


# def get_mean_vector_with_feedback(song_dict, spotify_data):
#     """
#     Adjust the mean vector based on user feedback.
#     """
#     # Extract features and feedback from the song_dict
#     features = spotify_data[number_cols].values
#     positive_feedback = song_dict.get('positive_feedback', [])
#     negative_feedback = song_dict.get('negative_feedback', [])

#     # Calculate the mean vector based on feedback
#     if positive_feedback:
#         positive_vectors = features[spotify_data['name'].isin(positive_feedback)]
#         mean_positive_vector = np.mean(positive_vectors, axis=0)
#     else:
#         mean_positive_vector = np.zeros(features.shape[1])

#     if negative_feedback:
#         negative_vectors = features[spotify_data['name'].isin(negative_feedback)]
#         mean_negative_vector = np.mean(negative_vectors, axis=0)
#     else:
#         mean_negative_vector = np.zeros(features.shape[1])

#     # Combine positive and negative feedback to adjust the mean vector
#     adjusted_mean_vector = features.mean(axis=0) + mean_positive_vector - mean_negative_vector

#     return adjusted_mean_vector


# def recommend_songs(song_list, spotify_data, n_songs=10, sort_by='popularity'):
#     np.random.seed(42)  # Set the random seed for reproducibility
#     metadata_cols = ['name', 'year', 'artists']
    
#     # Collect feedback from the user
#     positive_feedback, negative_feedback = collect_feedback()
    
#     # Update song_dict with feedback
#     song_dict = flatten_dict_list(song_list)
#     song_dict['positive_feedback'] = positive_feedback
#     song_dict['negative_feedback'] = negative_feedback
    
#     # Apply feedback to adjust the mean vector
#     song_center = get_mean_vector_with_feedback(song_dict, spotify_data)
    
#     # Use StandardScaler to scale the data
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(spotify_data[number_cols])
#     scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    
#     # Use PCA for dimensionality reduction
#     pca = PCA(n_components=len(spotify_data[number_cols].columns))
#     scaled_data_pca = pca.fit_transform(scaled_data)
#     scaled_song_center_pca = pca.transform(scaled_song_center)
    
#     # Using k-means clustering with k = n_songs
#     kmeans = KMeans(n_clusters=2 * n_songs, random_state=42)
#     kmeans.fit(scaled_data_pca)
    
#     # Find the cluster to which the adjusted song center belongs
#     cluster_label = kmeans.predict(scaled_song_center_pca)[0]
    
#     # Get the indices of songs in the same cluster
#     index = list(np.where(kmeans.labels_ == cluster_label)[0])
    
#     # Selecting the n_songs by index
#     rec_songs = spotify_data.iloc[index]
    
#     # Removing the names of the songs which are present in the given list
#     rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    
#     # Sort the recommended songs by the specified criteria (e.g., popularity)
#     rec_songs = rec_songs.sort_values(by=sort_by, ascending=False)
    
#     # Return the top 10 songs
#     rec_songs_top10 = rec_songs.head(n_songs)
    
#     return rec_songs_top10[metadata_cols].to_dict(orient='records')



# %%
# def retrain_kmeans_model(data, user_feedback):
#     # Extract numerical features for clustering
#     number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
#                    'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']
#     features = data[number_cols]

#     # Include user feedback in training data
#     user_liked_data = data[data['name'].isin(user_feedback['liked_songs'])][number_cols]
#     user_disliked_data = data[data['name'].isin(user_feedback['disliked_songs'])][number_cols]
#     training_data = pd.concat([features, user_liked_data, user_disliked_data], axis=0)

#     # Standardize features
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(training_data)

#     # Apply PCA for dimensionality reduction
#     pca = PCA(n_components=len(number_cols))
#     scaled_data_pca = pca.fit_transform(scaled_data)

#     # Retrain k-means model
#     kmeans = KMeans(n_clusters=2 * 10, random_state=42)
#     kmeans.fit(scaled_data_pca)

#     return kmeans, scaler, pca
# %%
