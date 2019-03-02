# import numpy as np

# for i in range (0, 10):
#     print(i)
#     fileName = "test/file{0}".format(i)
#     file = open('{0}.txt'.format(fileName), 'w')
#     file.close


import os

list = os.listdir("outputCoordinates") # dir is your directory path
number_files = len(list)
print (number_files/2)

for i in range(int(number_files/2)):
    print(i)