import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.DataFrame({
    'x': [37, 251, 336, 359, 360, 378, 381],
    'y': [129, 156, 352, 345, 364, 343, 363]
})

np.random.seed(200)
k = 2
# centroids[i] = [x, y]
centroids = {
    i + 1: [np.random.randint(0, 400), np.random.randint(0, 400)]
    for i in range(k)
}

fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color='k')
colmap = {1: 'r', 2: 'g'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 400)
plt.ylim(0, 400)
plt.show()