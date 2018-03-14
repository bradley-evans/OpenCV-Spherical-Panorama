import main
import cv2
import helperfuncs

## TESTS FOR IMAGE WARPING ##

# scalefactor = 0.25
# testimage = cv2.imread('spherical_testimages2/img_00_0.jpg')
# testimage = cv2.resize(testimage, None, fx = scalefactor, fy = scalefactor, interpolation = cv2.INTER_CUBIC)
# cv2.imshow('original',testimage)
# warped = main.warp(testimage,2000)
# cv2.imshow('warped',warped)
# while(True):
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()

## TESTS FOR DATA RETRIEVAL ##

dataset = main.getImages()
# print(dataset)
# print(str(dataset.shape))

## TESTS FOR TRANSLATION COMPUTATION ##

# translations = main.computeTranslation(dataset)

## TESTS FOR PANO CREATE FUNCTION ##
main.create(dataset,2000)

print("I survived.")