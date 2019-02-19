from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


IR_data = np.array([ [226,155], [247,188], [360,346], [362,366], [380,344], [383,365] ])


kmeans = KMeans(n_clusters=2, random_state=0).fit(IR_data)

plt.scatter(IR_data[:,0],IR_data[:,1],color='black')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], color='green')

 # creates an array of the labels, look like [1, 1, 0, 0, 0] where 1 means
 # coordinate[x] in IR_data array belongs to cluster 1
labels = kmeans.labels_

print("Points belonging to cluster 0:")
for idx, clusterNum in enumerate(labels):
    if (clusterNum==0):
        print(IR_data[idx])
    
print("\nPoints belonging to cluster 1:")
for idx, clusterNum in enumerate(labels):
    if (clusterNum==1):
        print(IR_data[idx])
    
plt.show()
