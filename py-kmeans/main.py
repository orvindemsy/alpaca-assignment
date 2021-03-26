# -*- coding: utf-8 -*-
"""
Author: Orvin Demsy
Date  : 2021-03-26
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas  as pd
import seaborn as sns
from kmeans import KMeans

# === Simulated dataset of two features, three different clusters === #
np.random.seed(1234)
a = np.random.randn(50, 2)
b = np.random.randn(50, 2)+3
c = np.random.randn(50, 2)+np.array([1, 7])

print('=== Simulated Data of Three Class ===')
for name, x in zip(['a', 'b', 'c'], [a, b, c]):
    print(f'cluster {name} has mean of {np.round(x.mean(0), 2)}')
    
# === Visualizing data, creating true label === #
y_true = np.concatenate([np.zeros(50), np.ones(50)*1, np.ones(50)*2])
X = np.vstack([a, b, c])  
df = pd.DataFrame(X, columns=['feat1', 'feat2'])
df = pd.concat([df, pd.Series(y_true, name='cluster')], axis=1)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Visualizing Dataset', fontsize=15)
ax[0].set_title('Unclustered')
sns.scatterplot(x='feat1', y='feat2', data=df, ax=ax[0], palette='muted')
ax[1].set_title('Clustered')
sns.scatterplot(x='feat1', y='feat2', hue='cluster', data=df, ax=ax[1], palette='muted')
plt.show()

# === Code testing === #
kmeans = KMeans(n_clusters=3, n_iter=10)
cent   = kmeans.fit(X)
y_pred = kmeans.predict(X) 

print('Coordinates of centroids at the end of iteration:\n', np.round(cent, 3))
print('')
print('A set of cluster id that each point belongs to:\n', y_pred)
print('')

# === Plot fitted X ==== #
kmeans.plot(X)

# == Comparing cluster visualization using `y_pred` and `y_true` === #
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Cluster Comparison Using y-pred and y-true Labels', fontsize=15)
for ax, y, title in zip(axes.ravel(), (y_pred, y_true), ('y_pred', 'y_true')):
    ax.set_title(f'Using {title}')
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, ax=ax, palette='muted')
    ax.set(xlabel='feat1', ylabel='feat2')
plt.show()
