# features.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numpy as np
import util
import samples

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def basicFeatureExtractor(datum):
    """
    Returns a binarized and flattened version of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features indicating whether each pixel
            in the provided datum is white (0) or gray/black (1).
    """
    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1
    return features.flatten()

def enhancedFeatureExtractor(datum):
    """
    Returns a feature vector of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features designed by you. The features
            can have any length.

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    ##
    """
    """ This one only got 82% accuracy
    prev = datum[0][0]
    for i in range(n):
        for j in range(m):
            pixel = datum[i][j]
            if prev == 0 and pixel > 0: number_contwhite+=1
            prev = datum[i][j]
    """

    "*** YOUR CODE HERE ***"
    """ This only got 87% accuracy"""
    # # cont spaces based on regions and neighbors
    # region = set()
    # conti = 0
    # for x in range(DIGIT_DATUM_WIDTH):
    #     for y in range(DIGIT_DATUM_HEIGHT):
    #         if (x, y) not in region and datum[x][y] < 2:
    #             conti += 1
    #             stack = [(x, y)]
    #             while stack:
    #                 pixel = stack.pop()
    #                 region.add(pixel)
    #                 for npixel in getNeighborpixels(pixel[0], pixel[1]):
    #                     if datum[npixel[0]][npixel[1]] < 2 and npixel not in region:
    #                         stack.append(npixel)
    #
    # white = np.zeros(3, dtype=int)
    # white[0] = conti % 2 # only one of these will be true at a time. Separate the cont data into 3 as said in project
    # white[1] = (conti >> 1) %2
    # white[2] = (conti >> 2) %2
    # whitespace = np.concatenate((features,white)) # concatenate these two
    """This one worked, used count of cont spaces using single pass"""
    from collections import deque
    pixellist = deque() #deq of pixels to scan

    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1 #use the binary image, but still in 2 dimensional format,
    number_regions = 0 #the numebr of regions found
    visited = set()     # a set of visited pixels
    def getPoint():     # gets a new point to search, the next black pixel not visited in image, returns negatives when scanned all image
        for i in range(DIGIT_DATUM_WIDTH):
            for j in range(DIGIT_DATUM_HEIGHT):
                if features[i][j] == 0 and (i,j) not in visited:
                    visited.add((i,j))
                    return i,j
        return -1,-1
    i,j = getPoint()
    while i >= 0 and j>= 0: # while theres still pixels to scan
        pixellist.append((i,j))    #append pixel to scan list
        number_regions = number_regions+1 # every time we find a new pixel we add to number of regions
        while pixellist:
            x,y = pixellist.popleft() #scan all pixels in list
            neighbors = getNeighborpixels(x,y) # get the neihbors of pixel
            for neighbor in neighbors:
                z,w = neighbor
                if features[z][w] == 0 and (z,w) not in visited:   #if its black, then add to visited and to scan list
                    visited.add((z,w))
                    pixellist.append((z,w))

        i,j = getPoint()
    myfeature = np.zeros(3)
    if number_regions == 1: myfeature = [1,0,0]
    if number_regions == 2: myfeature = [0,1,0]
    if number_regions > 2:  myfeature = [0,0,1]

    original = basicFeatureExtractor(datum)
    final = np.concatenate((original, myfeature))
    return final


def getNeighborpixels(x, y):
    neighbors = []
    if x > 0:
        neighbors.append((x - 1, y))
    if x < DIGIT_DATUM_WIDTH - 1:
        neighbors.append((x + 1, y))
    if y > 0:
        neighbors.append((x, y - 1))
    if y < DIGIT_DATUM_HEIGHT - 1:
        neighbors.append((x, y + 1))
    return neighbors

def analysis(model, trainData, trainLabels, trainPredictions, valData, valLabels, validationPredictions):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the print_digit(numpy array representing a training example) function
    to the digit

    An example of use has been given to you.

    - model is the trained model
    - trainData is a numpy array where each row is a training example
    - trainLabel is a list of training labels
    - trainPredictions is a list of training predictions
    - valData is a numpy array where each row is a validation example
    - valLabels is the list of validation labels
    - valPredictions is a list of validation predictions

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
   # for i in range(len(trainPredictions)):
    #    prediction = trainPredictions[i]
     #   truth = trainLabels[i]
      #  if (prediction != truth):
       #      print "==================================="
        #     print "Mistake on example %d" % i
         #    print "Predicted %d; truth is %d" % (prediction, truth)
           #  print "Image: "
          #   print_digit(trainData[i,:])


## =====================
## You don't have to modify any code below.
## =====================

def print_features(features):
    str = ''
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    for i in range(width):
        for j in range(height):
            feature = i*height + j
            if feature in features:
                str += '#'
            else:
                str += ' '
        str += '\n'
    print(str)

def print_digit(pixels):
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    pixels = pixels[:width*height]
    image = pixels.reshape((width, height))
    datum = samples.Datum(samples.convertToTrinary(image),width,height)
    print(datum)

def _test():
    import datasets
    train_data = datasets.tinyMnistDataset()[0]
    for i, datum in enumerate(train_data):
        print_digit(datum)

if __name__ == "__main__":
    _test()
