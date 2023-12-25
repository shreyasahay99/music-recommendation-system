#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#%% [markdown]
## Data Import
#%%
data = pd.read_csv("/Users/shreya/Documents/GitHub/DATS-6103-FA-23-SEC-11/music-recommendation-system/data/data.csv")

#%%
data.head()
#%%
data.info()
# %%
data.isna().sum()
# %%
