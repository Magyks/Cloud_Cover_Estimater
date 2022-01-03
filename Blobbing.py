from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy 
import cv2 as cv2

file_name = ".\\Fit_Images\\Partially.fit"
hdul = fits.open(file_name)
hdu_data = hdul[0].data
hdul.close()

class blobbing:
    def __init__(self,hdu_data,params_args=[],name = 1):
        self.name = name
        img8 = (hdu_data/256).astype('uint8')
        params = cv2.SimpleBlobDetector_Params()
        if len(params_args) == 0:
            params.filterByColor        = True
            params.filterByArea         = True
            params.filterByCircularity  = False
            params.filterByInertia      = False
            params.filterByConvexity    = False
        else:
            params.filterByColor        = params_args[0]
            params.filterByArea         = params_args[1]
            params.filterByCircularity  = params_args[2]
            params.filterByInertia      = params_args[3]
            params.filterByConvexity    = params_args[4]

        """ print(params.filterByColor )
        print(params.filterByArea )
        print(params.filterByCircularity )
        print(params.filterByInertia )
        print(params.filterByConvexity ) """

        params.blobColor = 255
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(img8)
        self.img_with_keypoints = cv2.drawKeypoints(img8,
                                        keypoints,
                                        numpy.array([]),
                                        (0,200,200),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



print("OpenCv")


params_list = [True,False,False,False,False]  ## Colour filter
blob_1 = blobbing(hdu_data,params_list,1)

params_list = [False,True,False,False,False]  ## Area filter
blob_2 = blobbing(hdu_data,params_list,2)

params_list = [False,False,True,False,False]  ## Circularity filter
blob_3 = blobbing(hdu_data,params_list,3)

params_list = [False,False,False,True,False]  ## Inertia filter
blob_4 = blobbing(hdu_data,params_list,4)

params_list = [False,False,False,False,True]  ## Convexity filter
blob_5 = blobbing(hdu_data,params_list,5)

blob_list = [blob_1,blob_2,blob_3,blob_4,blob_5]

fig, ax = plt.subplots(1,5)
for i in range(len(blob_list)):
    img = blob_list[i].img_with_keypoints
    ax[i].imshow(img)
plt.show()
"""
params_list = [False,False,False,False,True]  ## Colour filter
best_solution = blobbing(hdu_data,params_list)
plt.imshow(best_solution.img_with_keypoints)
plt.show()
"""

