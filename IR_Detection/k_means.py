from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import os

# checks the path 'outputcoordinates' for how many files exist so it knows how many times to loop
list = os.listdir("outputCoordinates")
number_files = len(list)
number_coordinates = number_files/2 # since theres 2 coordinate files (one x one y) for each pic


# loops for as many coordinate files there are in the outputCoordinates folder
for coordNum in range(int(number_coordinates)):

        # indicate which pic it is scanning
        print("\n\nPic Num: "+str(coordNum))

        # read file with name "outputCoordinates/XCoords-00001.txt" for example, number will increase as loop increments
        fileX = open("outputCoordinates/XCoords-"+str(coordNum).zfill(5)+".txt", "r")
        fileY = open("outputCoordinates/YCoords-"+str(coordNum).zfill(5)+".txt", "r")

        # read in coordinates as a numpy array, then convert into a list of lists 
        xData = np.array([int(line.rstrip()) for line in fileX])
        xData = [[i] for i in xData]

        yData = np.array([int(line.rstrip()) for line in fileY])
        yData = [[i] for i in yData]

        # close files
        fileX.close()
        fileY.close()

        # combine both arrays to create a list of pairs(nested list) of x,y coordinates
        IR_data = np.concatenate((xData, yData), 1)

        # perform kmeans algorithm on the data with certain clusters  
        kmeans = KMeans(n_clusters=2, random_state=0).fit(IR_data)

        # plot all the points as well as the kmeans clusters
        plt.scatter(IR_data[:, 0], IR_data[:, 1], color='black')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], color='red')

        # create an array of the labels, looks like [1, 1, 0, 0, 0] where 1 means
        # coordinate[x] in IR_data array belongs to cluster 1
        labels = kmeans.labels_

        print("Points belonging to cluster 0:")
        for idx, clusterNum in enumerate(labels):
                if clusterNum == 0:
                        print(IR_data[idx])
        
        print("\nPoints belonging to cluster 1:")
        for idx, clusterNum in enumerate(labels):
                if clusterNum == 1:
                        print(IR_data[idx])


        # shows the plot, will pop up for each image, comment out to avoid annoyance
        plt.ylim(0, 480)
        plt.xlim(0, 640)
        plt.gca().invert_yaxis()
        plt.show()
