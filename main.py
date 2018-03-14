import cv2
import numpy as np
import math
import helperfuncs
from scipy import ndimage
import pdb

def merge(imgs,transforms,newHeight,newWidth,f):
    panorama = []

    height, width, numChannels = imgs[0,0].shape
    panowidth,panoheight = imgs.shape

    # Set up the mask #
    mask = warp(np.ones((height,width,numChannels)),f)
    mask = cv2.bitwise_not(mask)    # get image compliment
    mask = ndimage.distance_transform_edt(mask)   # bwdist eqivalent from matlab
    mask = np.divide(mask,mask.max(0))

    # Finally, let's perform merge. #
    max_h = 0
    min_h = 0
    max_w = 0
    min_w = 0
    for x in range(0,panowidth):
        for y in range(0,panoheight):
            p_prime = np.multiply(transforms[x,y],np.transpose([1,1,1]))
            try:
                p_prime = np.divide(p.prime,p.prime[2,0])
            except:
                print("Array division by zero detected: merge(), (",str(x),",",str(y),")")
                p_prime = p_prime
            base_h = floor(p_prime[0,0])
            base_w = floor(p_prime[1,0])
            if base_h > max_h
                max_h = base_h
            if base_h < min_h
                min_h = base_h
            if base_w > max_w
                max_w = base_w
            if base_h < min_w
                max_w = base_w
    panorama = zeros((newHeight+20,newWidth+20,numChannels))
    denominator = zeros((newHeight+20,newWidth+20,numChannels))

    for x in range(0,panowidth):
        for y in range(0,panoheight):
            p_prime = transforms[x,y]


    return panorama

def getMatches(f1,d1,f2,d2):
    print("|--- Getting matches...")
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(d1,d2,k=2)

    # print("     matches",type(matches)," size:",len(matches))
    
    return matches

def computeTranslation(images):
    print("Creating translation matricies...")
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
                # Use Lowe's Ratio Test to determine good matches.
                # See: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
                matchesMask = [[0,0] for i in range(len(matches))]
                good = []
                for m,n in matches:
                    if m.distance < 0.7*n.distance:
                        good.append(m)
                if len(good) > 10:  # we want at least 10 matches to proceed
                    src_pts = np.float32([ features1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                    dst_pts = np.float32([ features2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                    M,mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                    translations[x,y] = M
                else:
                    print("**** NOT ENOUGH MATCHES ****")
    return translations

def warp(image,f):
    print("|--- Performing an image warp...")

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

    return cylinder

def create(images,f):
    print("Creating panorama...")

    full360 = False     # Disable end to end stitching for now

    cylindricalImages = images
    panorama = []
    panowidth, panoheight = images.shape
    for x in range(0,panowidth):
        for y in range(0,panoheight):
            # warp images cylindrically
            cylindricalImages[x,y] = (warp(images[x,y],f))
    translations = computeTranslation(cylindricalImages)

    print("|--- Computing absolute translations.")    
    absoluteTrans = np.zeros(translations.shape,translations.dtype)
    for x in range(0,panowidth):
        for y in range(0,panoheight):
            if x==0 and y==0:
                absoluteTrans[x,y] = translations[x,y]
                prev = absoluteTrans[x,y]
            else:
                absoluteTrans[x,y] = prev * translations[x,y]
                prev = absoluteTrans[x,y]
    
    imgheight0, imgwidth0, ch0 = cylindricalImages[0,0].shape
    # imgheight1, imgwidth1, ch1 = cylindricalImages[0,1].shape
    # imgheight2, imgwidth2, ch2 = cylindricalImages[0,2].shape
    if full360:
        # For a partial panorama, full360 will be false.
        # This will be for testing.
        # Otherwise, we need to fix endings.
        helperfuncs.todo()
    else:
        helperfuncs.todo()
        maxY = imgheight0
        minY = 1
        maxX = imgwidth0
        minX = 1
        for n in range(0,panowidth):
            for m in range(0,panoheight):
                curr = absoluteTrans[n,m]
                maxY = max(maxY,curr[0,2] + imgheight0)
                maxX = max(maxX,curr[1,2] + imgwidth0)
                minY = min(minY,curr[0,2])
                minX = min(minX,curr[1,2])
                curr[1,2] = curr[1,2] - math.floor(minX)
                absoluteTrans[n,m] = curr
        panorama_h = math.ceil(maxY) - math.floor(minY) + 1
        panorama_w = math.ceil(maxX) - math.floor(minX) + 1
        for n in range(0,panowidth):
            for m in range(0,panoheight):
                curr = absoluteTrans[n,m]
                curr[1,2] = curr[1,2] - math.floor(minX)
                curr[0,2] = curr[0,2] - math.floor(minY)
                absoluteTrans[n,m] = curr

    panorama = merge(cylindricalImages,absoluteTrans,panorama_h,panorama_w,f)

    return panorama

def getImages():
    print("Getting images...")
    scalefactor = 0.25
    xwidth = 4      # 3 for testing, use 17 in actual
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