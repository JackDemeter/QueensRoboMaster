# FIND SMALL BRIGHT SPOTS IN AN IMAGE
# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
import csv

# CONSTS
XTHRESHOLD = 5
YTHRESHOLD = 5
MULT = 0
XSHIFT = 0
YSHIFT = 0
XSTRETCH = 0
YSTRETCH = 0




def make_square(pair1, pair2, mult = 0):
    coords1, coords2 = pair1
    coords3, coords4 = pair2
    list_x = [coords1[0],coords2[0],coords3[0],coords4[0]]
    list_y = [coords1[1],coords2[1],coords3[1],coords4[1]]
    # returns the x,y,w,h of the bounding box
    xmin = min(list_x)
    ymin = min(list_y)
    xmax = max(list_x)
    ymax = max(list_y)

    xcenter = (xmin + xmax)/2
    ycenter = (ymin + ymax)/2

    xmin = int(xmin - abs(xmin - xcenter) * mult)
    ymin = int(ymin - abs(ymin - ycenter) * mult)
    xmax = int(xmax + abs(xmin - xcenter) * mult)
    ymax = int(ymax + abs(ymin - ycenter) * mult)

    return [int(xmin + XSHIFT + (xcenter - 320) * XSTRETCH) ,int(ymin + YSHIFT + (ycenter - 400) * YSTRETCH), xmax -xmin+ 4, ymax-ymin+4]

def square_coords(pair1, pair2, mult = 0):
    coords1, coords2 = pair1
    coords3, coords4 = pair2
    list_x = [coords1[0],coords2[0],coords3[0],coords4[0]]
    list_y = [coords1[1],coords2[1],coords3[1],coords4[1]]
    # returns the x,y,w,h of the bounding box
    xmin = min(list_x)
    ymin = min(list_y)
    xmax = max(list_x)
    ymax = max(list_y)


    return [xmin, ymin, xmax, ymax]

# open a text file called dotCoords.txt, make it if it doesnt exist. This file will store the coordinates of the emitters
with open("batch1.csv", mode='a', newline='') as batch:
    batch_writer = csv.writer(batch, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    batch_writer.writerow(['filename', 'width', 'height', 'class', 'xmin','ymin', 'xmax', 'ymax'])
    # for loop that runs the whole program, set the range(#) to whatever number of pictures you have so the program runs through each one
    for picNum in range(1,100):
    
        #fileX = open("outputCoordinates/XCoords-"+str(picNum).zfill(5)+".txt", "w+")
        #fileY = open("outputCoordinates/YCoords-"+str(picNum).zfill(5)+".txt", "w+")

        # set the image path, this will increment by one each loop. looks like "IR-#####.jpg" with incrementing numbers (and leading zeroes)
        imagePath = "IR1-"+str(picNum).zfill(5)+".jpg"
        imagePath2 = "WB1-"+str(picNum+4).zfill(5)+".jpg"

        # print the picture we are on
        print("picNum = " + str(picNum))

        # load the image, convert it to grayscale, and blur it
        image = cv2.imread("images/"+imagePath)
        image2 = cv2.imread("images/"+imagePath2)

        # grayscale the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # blur the image to reduce high frequency noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image to reveal light regions in the blurred image
        thresh = cv2.threshold(blurred, 125, 255, cv2.THRESH_BINARY)[1]

        # OLD, FROM EXAMPLE CODE
        # perform a series of erosions and dilations to remove
        # any small blobs of noise from the thresholded image
        ##thresh = cv2.erode(thresh, None, iterations=2)
        ##thresh = cv2.dilate(thresh, None, iterations=4)

        # perform a connected component analysis on the thresholded image, then
        # initialize a mask to store only the "large" components
        labels = measure.label(thresh, neighbors=8, background=0)
        mask = np.zeros(thresh.shape, dtype="uint8")

        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue

            # otherwise, construct the label mask and count the number of pixels
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)

            # if the number of pixels in the blob is inbetween these two parameters, then add it as a blob
            if numPixels > 1 and numPixels < 100:
                mask = cv2.add(mask, labelMask)

        # find the contours in the mask, then sort them from left to right
        cnts = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # if no bright spots are found, continue
        if not cnts:
            print("Picture #" + str(picNum) +
                " did not have any identifiable dots, moving onto next picture...\n\n")
            continue

        # sort the contours
        cnts = contours.sort_contours(cnts)[0]
        # loop over the contours
        z = 0
        square_detected = False

        x_array = []
        y_array = []
        for (i, c) in enumerate(cnts):
            # draw the bright spot on the image
            (x, y, w, h) = cv2.boundingRect(c)
            if (y > 250):
                ((cX, cY), radius) = cv2.minEnclosingCircle(c)
                # cv2.circle(image, (int(cX), int(cY)), int(radius),
                #       (0, 0, 255), 2)
                
                # this code was to put a number label on the processed images
                # cv2.putText(image, "#{}".format(i + 1), (x, y - 15),
                # cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                # Print Location of Dot
                print("Location of dot " + str(i+1) + "\nx: " +
                    str(x) + "\ny: " + str(y) + "\n\n")

                # add value to xy arrays to determine if they are close enough to form a square
                x_array.append(x)
                y_array.append(y)
                z += 1
                # write coords to file
                # fileX.write("%d\n" % (x))
                # fileY.write("%d\n" % (y))

            # throw away any image without atleast 4 points

        


        # show the output image
        ##cv2.imshow("Image", image)
        x_pair = []
        avg_y_list = []
        avg_x_list = []
        # thresholding
        for n in range(z):
            for j in range(n+1,z):
                if abs(x_array[n] - x_array[j]) < XTHRESHOLD:
                    avg_y = abs((y_array[n]+y_array[j])/2)
                    avg_y_list.append(avg_y)
                    avg_x = abs((x_array[n]+x_array[j])/2)
                    avg_x_list.append(avg_x)
                    x_pair.append(([x_array[n], y_array[n]],[x_array[j], y_array[j]] ))

     
        # compare all pairs and make a bounding box
        for n in range(len(avg_y_list) - 1):
            for j in range(n+1, len(avg_y_list)):
                if abs(avg_y_list[n]-avg_y_list[j]) < YTHRESHOLD:
                    square_detected = True
                    x,y,w,h = make_square(x_pair[n], x_pair[j], MULT)
                    square_coords_list = square_coords(x_pair[n], x_pair[j], MULT)
                    batch_writer.writerow([imagePath, 640, 480, 'plate', square_coords_list[0], square_coords_list[1], square_coords_list[2], square_coords_list[3]])
                    cv2.rectangle(image, (x,y) ,(x+w,y+h), (0,255,0), 2)
                    cv2.rectangle(image2, (x,y) ,(x+w,y+h), (0,255,0), 2)


        # save the image to the outputImages folder
        if square_detected:
            cv2.imwrite("outputImages/"+imagePath, image)
            cv2.imwrite("outputImages/"+imagePath2, image2)
            


        # close the coordinate text files
        #fileX.close()
        #fileY.close()



