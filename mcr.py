#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


#%% [markdown]
## Data Import
#%%
data = pd.read_csv("/Users/richikghosh/Documents/GitHub/music-recommendation-system/data/data.csv")
data_by_artist = pd.read_csv("/Users/richikghosh/Documents/GitHub/music-recommendation-system/data/data_by_artist.csv")
data_by_year=pd.read_csv("/Users/richikghosh/Documents/GitHub/music-recommendation-system/data/data_by_year.csv")
data_by_genre=pd.read_csv("/Users/richikghosh/Documents/GitHub/music-recommendation-system/data/data_by_genres.csv")
#data_wt_genre=pd.read_csv("/Users/richikghosh/Documents/GitHub/music-recommendation-system/data/data_w_genres.csv")

#%%
data.artists.value_counts()
#%%




# %%

### Making the pipeline with Scaling the data and then using  Clustering

# Load your data
# Assuming your data is stored in a DataFrame named 'data'
# Drop non-numeric columns for simplicity in this example
numeric_columns = data.select_dtypes(include=['float64', 'int64']).drop(['year'], axis=1)

# Define features for clustering
features = numeric_columns.columns

# Create a pipeline
pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('clustering', KMeans(n_clusters=9,random_state=42))
])
#%%


                                ################
                                # Find best k #
                                ################



# Range of clusters to try
k_values = range(2, 50)  # You can adjust this range based on your analysis

# List to store distortions (inertia)
distortions = []

# Evaluate different values of k
for k in k_values:
    pipeline.set_params(clustering__n_clusters=k)
    
    # Fit the pipeline to your data
    data['cluster'] = pipeline.fit_predict(numeric_columns[features])
    
    # Calculate distortion (inertia)
    distortions.append(pipeline.named_steps['clustering'].inertia_)

# Plot Elbow Method (Distortion/Inertia)
plt.figure(figsize=(8, 5))
plt.plot(k_values, distortions, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion (Inertia)')
plt.show()
# %%

# Range of clusters to try
k_values = range(2, 30)  # You can adjust this range based on your analysis

# List to store silhouette scores
silhouette_scores = []

# Evaluate different values of k
for k in k_values:
    pipeline.set_params(clustering__n_clusters=k)
    
    # Fit the pipeline to your data
    data['cluster'] = pipeline.fit_predict(numeric_columns[features])
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(numeric_columns, data['cluster'])
    silhouette_scores.append(silhouette_avg)

# Plot Silhouette Score
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score for Different Values of k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.show()




# %%
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load your data
# Assuming your data is stored in a DataFrame named 'data'
# Drop non-numeric columns for simplicity in this example
numeric_columns = data.select_dtypes(include=['float64', 'int64']).drop(['year'], axis=1)

# Define features for clustering
features = numeric_columns.columns

# Create a pipeline with PCA
pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('clustering', KMeans(n_clusters=9, random_state=42))
])

# Fit the pipeline to your data
data['cluster'] = pipeline.fit_predict(numeric_columns[features])

# Apply PCA on the transformed data from the clustering step
pca = PCA(n_components=2)
pca_result = pca.fit_transform(numeric_columns[features])

# Plot PCA results
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=data['cluster'], cmap='viridis')
plt.title('PCA Clustering Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# %%
pca_result
# %%
## Visualizations

import plotly.express as px 
song_cluster_pipeline = Pipeline([('scalar',StandardScaler()), 
                                  ('kmeans', KMeans(n_clusters=20, 
                                   verbose=False))
                                 ], verbose=False)

X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels

from sklearn.decomposition import PCA

pca_pipeline = Pipeline([('scalar',StandardScaler()),('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['artists']
projection['cluster'] = data['cluster_label']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
fig.show()







# %%



#################

## Content-Based Filtering:

#################
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load your dataset
df = data

# Select relevant features for content-based filtering
features = ['valence', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness', 'tempo']

# Create a feature matrix (use a subset of data for testing)
subset_size = 6000  # Adjust the subset size based on your system's capacity
df_subset = df.head(subset_size)
X = df_subset[features]

# Normalize numerical features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Calculate cosine similarity matrix
similarity_matrix = cosine_similarity(X_normalized)

# Choose a target item (replace with an actual item name or ID)
target_item_index = 0

# Retrieve similarity scores for the target item
item_similarity_scores = similarity_matrix[target_item_index]

# Rank items based on similarity scores
recommendations = df_subset[['name', 'artists', 'popularity']].copy()
recommendations['similarity_score'] = item_similarity_scores
recommendations = recommendations.sort_values(by='similarity_score', ascending=False)
#%%
N=5
# Display top N recommendations
top_recommendations = recommendations.head(N)
print(top_recommendations[['name', 'artists', 'popularity','similarity_score']])

# %%
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load your dataset
df = data

# Select relevant features for clustering
features = ['valence', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness', 'tempo']

# Create a feature matrix
X = df[features]

# Standardize numerical features
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Apply dimensionality reduction for visualization (e.g., using PCA)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_standardized)

# Choose the number of clusters (you may need to tune this parameter)
n_clusters = 7
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X_standardized)

# Print the coordinates of the cluster centers in the PCA-transformed space
cluster_centers_pca = pca.transform(kmeans.cluster_centers_)
print("Cluster Centers (PCA coordinates):\n", cluster_centers_pca)


# Plot the clusters
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d') 
for cluster_label in range(n_clusters):
    cluster_data = X_pca[df['cluster'] == cluster_label]
    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], label=f'Cluster {cluster_label}')

# Plot the cluster centers
ax.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], cluster_centers_pca[:, 2], marker='X', s=200, color='black', label='Cluster Centers')
ax.set_title('Clustering of Music Data')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.legend()
plt.show()


# %%
## FInding the best cluster
max_clusters=30
distortions=[]
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_normalized)
    distortions.append(kmeans.inertia_)

# Plot the Elbow curve
plt.plot(range(1, max_clusters + 1), distortions, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion (Inertia)')
plt.show()
# %%
import plotly.express as px
import plotly.graph_objects as go
# Add PCA components to DataFrame for plotting
df['pca_1'] = X_pca[:, 0]
df['pca_2'] = X_pca[:, 1]
df['pca_3'] = X_pca[:, 2]

# Get cluster centers in PCA space
cluster_centers_pca = pca.transform(kmeans.cluster_centers_)

# Create a 3D scatter plot using Plotly.graph_objects
fig = go.Figure()

# Plot the clusters
for cluster_label in range(n_clusters):
    cluster_data = df[df['cluster'] == cluster_label]
    fig.add_trace(go.Scatter3d(
        x=cluster_data['pca_1'],
        y=cluster_data['pca_2'],
        z=cluster_data['pca_3'],
        mode='markers',
        marker=dict(size=5, opacity=0.7),
        text=cluster_data['artists'],
        name=f'Cluster {cluster_label}'
    ))

# Plot the cluster centers
fig.add_trace(go.Scatter3d(
    x=cluster_centers_pca[:, 0],
    y=cluster_centers_pca[:, 1],
    z=cluster_centers_pca[:, 2],
    mode='markers',
    marker=dict(size=10, color='black', symbol='cross'),
    name='Cluster Centers'
))

# Customize layout
fig.update_layout(
    scene=dict(
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        zaxis_title='Principal Component 3'
    ),
    title='Clustering of Music Data'
)

fig.show()
# %%

from sklearn.neighbors import NearestNeighbors

# Load your dataset
df = data

# Select relevant features for recommendation
features = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

# Create a feature matrix
X = df[features]

# Standardize numerical features
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Aggregate features for each artist (mean or sum)
artist_features = df.groupby('artists')[features].mean()  # You can use other aggregation functions like sum, max, etc.

# Fit a k-nearest neighbors model on artist features
knn_model = NearestNeighbors(n_neighbors=8, metric='cosine', algorithm='brute')
knn_model.fit(artist_features)
# Function to get similar artists for a given artist
def get_similar_artists(artist_name):
    if artist_name in artist_features.index:
        artist_features_standardized = scaler.transform([artist_features.loc[artist_name]])
        _, indices = knn_model.kneighbors(artist_features_standardized)
        similar_artists = artist_features.iloc[indices[0]].index.tolist()
        return similar_artists
    else:
        return []
#%%
    
# Example: Get similar artists for a specific artist
artist_to_find_similar = "['Eminem']"
similar_artists = get_similar_artists(artist_to_find_similar)
print(f"Artists similar to {artist_to_find_similar}:")
print(similar_artists)

# %%

# %%
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Load your dataset
df = data

# Select relevant features for clustering
features = ['valence', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness', 'tempo']

# Create a feature matrix
X = df[features]

# Standardize numerical features
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Apply dimensionality reduction for visualization (e.g., using t-SNE)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_standardized)

# Choose the number of clusters (you may need to tune this parameter)
n_clusters = 7
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X_standardized)

# Plot the clusters
fig, ax = plt.subplots(figsize=(12, 8))
for cluster_label in range(n_clusters):
    cluster_data = X_tsne[df['cluster'] == cluster_label]
    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_label}')

# Annotate with artist names
for i, row in df.iterrows():
    ax.annotate(row['artists'], (X_tsne[i, 0], X_tsne[i, 1]))

ax.set_title('Clustering of Music Data')
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.legend()
plt.show()

# %%

## Function to find the song in spotify and fetch its features
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
import os
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="SPOTIFY_CLIENT_ID",client_secret="SPOTIFY_CLIENT_SECRET"))
def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)

#%%[markdown]

## Recommendation System


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
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
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
    
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    song_center = get_mean_vector(song_list, spotify_data)
    scaled_data = scaler.fit_transform(spotify_data[number_cols])
    scaled_song_center = scaler.fit_transform(song_center.reshape(1, -1))
    # This finds the euclidean distance
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    # Sorted in ascending order then choose the first n_songs
    index = list(np.argsort(distances)[:, :n_songs][0])
    # selecting the n_songs by index
    rec_songs = spotify_data.iloc[index]
    # removing the names of the songs which are present in the given list
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')
# %%
recommend_songs([{'name': 'Come As You Are', 'year':1991},
                {'name': 'Smells Like Teen Spirit', 'year': 1991},
                {'name': 'Lithium', 'year': 1992},
                {'name': 'All Apologies', 'year': 1993},
                {'name': 'Stay Away', 'year': 1993}],  data)




#########################################################################################################################################################

# %%
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def recommend_songs(song_list, spotify_data, n_songs=10, n_neighbors=10):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    # Extract numeric features
    X = spotify_data[number_cols].values

    # Create and fit StandardScaler and KNeighborsRegressor in a pipeline
    knn_pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=n_neighbors))
    
    # Fit the pipeline to the scaled data
    knn_pipeline.fit(X, X)  # Using X as both input and target for unsupervised learning
    
    # Transform the mean vector and find neighbors
    song_center = get_mean_vector(song_list, spotify_data)
    scaled_song_center = knn_pipeline.named_steps['standardscaler'].transform(song_center.reshape(1, -1))
    neighbors = knn_pipeline.predict(scaled_song_center)

    # Add clustering labels to the original data
    cluster_labels = knn_pipeline.predict(scaled_song_center)
    # Add clustering labels to the original data
    spotify_data['cluster_label'] = cluster_labels[0] 
    # Extract indices of the neighbors
    indices = np.argsort(neighbors.flatten())[:n_songs]
    # Select recommended songs by index
    rec_songs = spotify_data.iloc[indices]

    # Remove songs already in the given list
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]

    return rec_songs[metadata_cols].to_dict(orient='records'), knn_pipeline

def get_cluster_labels(songs, spotify_data, knn_pipeline):
    X = spotify_data[number_cols].values
    song_vectors = [get_song_data(song, spotify_data) for song in songs]

    # Filter out None values
    song_vectors = [song[number_cols].values for song in song_vectors if song is not None]

    if not song_vectors:
        # Handle the case where all songs are None
        print('Warning: None of the provided songs exist in Spotify or in the database')
        return []

    # Ensure that the input to knn_pipeline.predict is a 2D array
    scaled_song_vectors = knn_pipeline.named_steps['standardscaler'].transform(song_vectors)
    cluster_labels = knn_pipeline.predict(scaled_song_vectors.reshape(1, -1))

    return cluster_labels


# Example usage
song_list = [
    {'name': 'Come As You Are', 'year': 1991},
    {'name': 'Smells Like Teen Spirit', 'year': 1991},
    {'name': 'Lithium', 'year': 1992},
    {'name': 'All Apologies', 'year': 1993},
    {'name': 'Stay Away', 'year': 1993}
]

# Get recommendations and the fitted pipeline
recommendations, knn_pipeline = recommend_songs(song_list, data)

# Get cluster labels for the provided songs
cluster_labels = get_cluster_labels(song_list, data, knn_pipeline)

# Find songs in the same cluster
songs_in_same_cluster = data[data['cluster_label'].isin(cluster_labels)]

print(songs_in_same_cluster[metadata_cols].to_dict(orient='records'))



# %%
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def recommend_songs(song_list, spotify_data, n_songs=10, eps=0.5, min_samples=5):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    # Extract numeric features
    X = spotify_data[number_cols].values

    # Create StandardScaler and DBSCAN in a pipeline
    dbscan_pipeline = make_pipeline(StandardScaler(), DBSCAN(eps=eps, min_samples=min_samples))
    
    # Fit the pipeline to the scaled data
    cluster_labels = dbscan_pipeline.fit_predict(X)
    
    # Transform the mean vector and find cluster label
    song_center = get_mean_vector(song_list, spotify_data)
    scaled_song_center = dbscan_pipeline.named_steps['standardscaler'].transform(song_center.reshape(1, -1))
    cluster_label = dbscan_pipeline.predict(scaled_song_center)[0]
    
    # Select indices of songs in the same cluster
    indices = np.where(cluster_labels == cluster_label)[0][:n_songs]

    # Select recommended songs by index
    rec_songs = spotify_data.iloc[indices]

    # Remove songs already in the given list
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]

    return rec_songs[metadata_cols].to_dict(orient='records')

# Example usage
recommendations = recommend_songs([
    {'name': 'Come As You Are', 'year': 1991},
    {'name': 'Smells Like Teen Spirit', 'year': 1991},
    {'name': 'Lithium', 'year': 1992},
    {'name': 'All Apologies', 'year': 1993},
    {'name': 'Stay Away', 'year': 1993}], data)

print(recommendations)


# %%

def func(n,k):
    if (k==n+1):
        return
    print(k,end=' ')
    func(n,k+1)
k=1  
func(5,1)
# %%
def func(n,k):
    if (n==0 or n==1):
        return k
    k=k*n 
    return func(n-1,k)
print(func(6,1))
# %%
def func(k):
    if(k<=0):
        return k
    if k==1:
        return 0
    if k==2:
        return 1
    return func(k-2)+func(k-1)
print(func(0))
# %%
def func(n):
    if n<=1:
        return n 
    return n+func(n-1)
print(func(3))
# %%
def func(n,k=1):
    if n==0:
        return n
    if n<=1:
        return k 
    return func(n-1,n+k)
print(func(3))
# %%


def func(use_str, last, first=0):
    if last<=first:
        return True
    if(use_str[last]==use_str[first]):
        print(f"{use_str[last],use_str[first]}")
        return func(use_str,last-1,first+1)
    else:
        print(f"Not palindrome {use_str[last],use_str[first]}")
        return False
    
def palichcekcall(arr):
        if func(arr,len(arr)-1)==True:
            print("Yes it is palindrome")
        else:
            print("Its not pali")


palichcekcall(input(""))



# %%
def func(num, sum=0):
    if num<=0:
        return sum
    return func(num//10,num%10+sum)
func(253)
# %%

# %%
