import cv2
import numpy as np
import math
import helperfuncs

def getMatches(f1,d1,f2,d2):
    print("|--- Getting matches...")
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    try:
        matches = flann.match(d1,d2)
        matches = sorted(matches, key = lambda x:x.distance)
    except:
        print("Error in getMatches().")
        print(d1.dtype)
        print(d2.dtype)

    return matches

def computeTranslation(images):
    print("Creating translation matricies...")
    # parameters #
    # threshold =     10
    # confidence =    0.99
    # inlierRatio =   0.3
    # epsilon =       1.5
    
    sift = cv2.xfeatures2d.SIFT_create()

    xwidth,ywidth = images.shape
    translations = np.empty((xwidth,ywidth),dtype=object)
    for x in range(0,xwidth):
        for y in range(0,ywidth):
            translations[x,y] = np.identity(3)
    
    for x in range(0,xwidth):
        for y in range(0,ywidth):
            # Assuming we are using OpenCV 3.x #

            # Get SIFT Features
            img = images[x,y].astype(np.uint8)
            if x==0 and y==0:
                features2,descriptors2 = sift.detectAndCompute(img,None)
            else:
                features1 = features2
                descriptors1 = descriptors2
                features2,descriptors2 = sift.detectAndCompute(img,None)
                matches = getMatches(features1,descriptors1,features2,descriptors2)
                
    return translations

def warp(image,f):
    print("Performing an image warp...")

    sizey, sizex, numChannels = image.shape
    output = np.zeros((sizey,sizex,numChannels))

    xcenter = int(sizex/2)
    ycenter = int(sizey/2)

    # This creates a reference matrix
    # that contains our warp transform.
    # We'll map image intensities from
    # img(x,y) => newimg(xx[x,y],yy[x,y]).

    # The amount of spherical warping is
    # determined by the factor 'f.'

    x = np.arange(0, sizex) - xcenter
    y = np.arange(0, sizey) - ycenter
    xx,yy = np.meshgrid(x,y)
    yy = (np.divide((f*yy),(np.sqrt(np.power(xx,2)+np.power(f,2)))))+ycenter
    xx = (f * np.arctan(xx/f)) + xcenter
    xx = np.floor(xx+0.5)
    yy = np.floor(yy+0.5)

    yy = yy.astype(int)
    xx = xx.astype(int)


    cylinder = np.zeros((sizey,sizex,numChannels),np.uint8)

    for i in range(0,numChannels):
        for n in range(0,sizey):
            for m in range(0,sizex):
                try:
                    cylinder[yy[n,m],xx[n,m],i] = image[n,m,i]
                except:
                    print("Error in warp. Indicies m=",str(m)," n=",str(n)," i=",str(i))
               
    cv2.imshow('okay',image)
    cv2.imshow('warped',cylinder)

    return cylinder


def create(images,f):
    print("Creating panorama...")
    cylindricalImages = []
    panorama = []

    for image in images:
        # warp images cylindrically
        cylindricalImages.append(warp(image,f))
    
    # translations = computeTranslations(cylindricalImages,f)


    return panorama

def getImages():
    print("Getting images...")
    scalefactor = 0.25
    xwidth = 17
    ywidth = 3
    dataset = np.empty((xwidth,ywidth),dtype=object)
    for x in range(0,xwidth):
        for y in range (0,ywidth):
            filename = helperfuncs.getFilename(x,y)
            img = cv2.imread(filename)
            img = cv2.resize(img, None, fx = scalefactor, fy = scalefactor, interpolation = cv2.INTER_CUBIC)
            # plt.subplot(3,11,i)
            # plt.imshow(img)
            # i = i+1
            dataset[x,y] = img
    return dataset

def main():
    # get images #
    dataset = getImages()


    # parameters #
    f = 1000  # [UNKNOWN] Samsung Galaxy S7 #

    # execution #
    panorama = create(dataset,f)


# main()