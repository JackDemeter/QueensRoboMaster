# FIND SMALL BRIGHT SPOTS IN AN IMAGE

# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2

# open a text file called dotCoords.txt, make it if it doesnt exist. This file will store the coordinates of the emitters
file = open("emitterCoords.txt", "w+")

# for loop that runs the whole program, set the range(#) to whatever number of pictures you have so the program runs through each one
for picNum in range(20):
    # set the image path, this will increment by one each loop. looks like "IR-#####.jpg" with incrementing numbers (and leading zeroes)
    imagePath = "IR-"+str(picNum).zfill(5)+".jpg"

    # print the picture we are on
    print("picNum = " + str(picNum))

    # load the image, convert it to grayscale, and blur it
    image = cv2.imread("images/"+imagePath)

    # grayscale the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blur the image to reduce high frequency noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # threshold the image to reveal light regions in the blurred image
    thresh = cv2.threshold(blurred, 105, 255, cv2.THRESH_BINARY)[1]

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
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(cX), int(cY)), int(radius),
                   (0, 0, 255), 2)

        # this code was to put a number label on the processed images
        # cv2.putText(image, "#{}".format(i + 1), (x, y - 15),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # Print Location of Dot
        print("Location of dot " + str(i+1) + "\nx: " +
              str(x) + "\ny: " + str(y) + "\n\n")

        # write coords to file
        file.write("%d %d\n" % (x, y))

    # write a new line for each photo
    file.write("\n")

    # show the output image
    ##cv2.imshow("Image", image)

    # save the image to the outputImages folder
    cv2.imwrite("outputImages/"+imagePath, image)


# close the coordinate text file
file.close()
