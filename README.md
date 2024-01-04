# Music Recommendation System
<p align="center">
<img src="data/Music.png" width="400" height="350" title="StockSense" alt="StockSense Logo">
</p>

## Description

This web application suggests 10 similar songs based on a user-provided list of songs using unsupervised learning. Additionally, it features a feedback system to gather user input for continuous improvement of the application.

## Data

The data has been collected from Kaggle and it has a record of 1.2 million songs.


## Model Building
In this project we have scaled the data using scaling techniques and removed the dimentionality using PCA built after this we have built two models:
- K Means Clustering
- Self organizing Maps(SOMs)

After this the closest songs to the list provided are selected and presented to the end user. This is constantly being changed based on the feedback provided by the end user.

## Deployment

We have used Flask to built the webpage of the recommendation system.







**Data Source**: https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset
