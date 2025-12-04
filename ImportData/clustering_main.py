import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from icecream import ic

df = pd.read_csv(r'Dataset\Winequality\winequality-white.csv', header = 0, sep = ';')

column_names = df.columns.tolist()

X = df[column_names[:-1]]

Y = df['quality']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Run K-Means on the SCALED data
km_scaled = KMeans(n_clusters=2,
                   init = 'random',
                   n_init = 10,
                   max_iter = 300,
                   tol = 1e-04,
                   random_state = 0)
y_km_scaled = km_scaled.fit_predict(X_scaled)



