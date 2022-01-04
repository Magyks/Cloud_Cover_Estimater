from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy
import cv2 as cv
import os
import copy
import time


class image_process:
    def __init__(self,file_names,file_location):
        self.file_names = file_names
        self.file_location = file_location

        self.hdu_list = []
        self.hdu_data = []
        for i in range(len(file_names)):
            hdul = fits.open(file_location+file_names[i])
            self.hdu_list.append(hdul[0])
            self.hdu_data.append(hdul[0].data)
            hdul.close()

    def show_img(self,time=3,CMAP = "gray",idx=0):
        plt.clf()
        plt.imshow(self.hdu_data[idx],cmap=CMAP)
        plt.pause(time)
        plt.close()

    def create_masks_old(self,threshold = 3600,overwrite=False,save=True):
        "Creates a mask of all the given array using a standard deviation threshold test" 
        x = os.path.exists(self.file_names[0][:len(self.file_names[0])-4]+"_mask.txt")
        print("Overwrite is set to ",overwrite,".")
        print("Mask for image 1 already exists.")
        if overwrite == False and x:
            print("Mask(s) already exist, loading.")
            self.loadmask()
        else:
            print("Creating new mask(s).")
            self.mask_data = []
            for k in range(len(self.hdu_data)):
                array = copy.copy(self.hdu_data[k])
                x_max = len(array)
                y_max = len(array[0])
                for i in range(x_max):
                    for j in range(y_max):
                        self.numstd(array,(i,j)) 
                        if self.std > threshold :
                            array[i][j] = 0
                print("mask ",k+1,"out of ",len(self.hdu_data)," created.")
                self.mask_data.append(array)
            if save==True:
                self.save_mask()

    def create_masks(self,threshold = 3600,new_mask = False,save = True):
        "Create mask method re-written to allow for individual file evaluation and mask creation"
        "This method creates or loades the masks of the provided files, later turned into PKL files"
        print("New mask? :",new_mask,", Save? :",save)
        self.mask_data = []
        for i in range(len(self.file_names)):
            curr_name = self.file_names[i]                                ## Current file name
            x = os.path.exists(self.file_location+curr_name[:len(curr_name)-4]+"_mask.txt")  ## Check the file exists
            if x == False or new_mask == True:                            ## if the file dosen's exist or needs to be re-written
                ## Create a new mask
                print("Creating a new mask for file ",i)
                array = copy.copy(self.hdu_data[i])
                x_max = len(array)
                y_max = len(array[0])
                for j in range(x_max):
                    for k in range(y_max):
                        self.numstd(array,(j,k)) 
                        if self.std > threshold :
                            array[j][k] = 0
                if save:
                    ## save the new mask
                    print("Mask saved")
                    numpy.savetxt(self.file_location+curr_name[:len(curr_name)-4]+"_mask.txt",array)
                else:
                    print("Single run mask only, not saved.")
            else:
                ## Load a previous mask
                print("Loading mask ",i)
                array = numpy.loadtxt(self.file_location+curr_name[:len(curr_name)-4]+"_mask.txt","uint8")
            
            ## Add it to the mask array
            self.mask_data.append(array)
            print("Mask ",i+1,"out of ",len(self.hdu_data)," loaded/created.")

    def loadmask(self):
        try:
            x = self.mask_data[0]
        except  AttributeError:
            self.mask_data = []
            for i in range(len(self.file_names)):
                mask = numpy.loadtxt(self.file_names[i][:len(self.file_names[i])-4]+"_mask.txt","uint8")
                self.mask_data.append(mask)
            print("Mask(s) loaded")
        else:
            print("Mask(s) are already loaded.")

    def numstd(self,array,point,side_length=2):
        "Returns the standard deviation of a given array using numpy"
        self.values(array,point,size=side_length)   ## Gives valyes in a dimond shape around the point
        self.std = numpy.std(self.value_list)           ## returns the given std for the selected array

    def values(self,array,point,size=2):

        "Gives the values around a point in an array of dimond of side length 2*size"
        #array is a np array, point is a tuple (x,y)
        value_list = []
        x0 = point[0]
        y0 = point[1]
        for i in range(-size,size):
            for j in range(-size,size):
                try:
                    value_list.append(array[x0+i][y0+j])
                except IndexError:
                    continue

        self.value_list = value_list

    def save_mask(self):
        print("Saving mask data.")
        for i in range(len(self.mask_data)):
            numpy.savetxt(self.file_names[i][:len(self.file_names[i])-4]+"_mask.txt",self.mask_data[i])

    def data_w_mask(self):
        print("Creating images with masks applied.")
        self.trimmed_data = []
        for k in range(len(self.hdu_data)):
            array = copy.copy(self.hdu_data[k])
            mask_data = self.mask_data[k]
            for i in range(len(array)):
                for j in range(len(array[0])):
                    if mask_data[i][j] != 0:
                        array[i][j] = 0
            self.trimmed_data.append(array)
        print("Mask(s) applied.")

    def circlesum(self,array,c_point=(464,470),size=464):
        sum = []
        x = numpy.arange(c_point[0]-size,size+1+c_point[0],1)
        #print(x)
        y = numpy.sqrt((size)**2-(x-c_point[0])**2) + c_point[1]
        #print(y)
        y = y.astype(int)
        for i in range(len(x)):
            for j in range(int(y[i]-size),int(y[i]+size)): 
                try:
                    value = array[int(x[i])][j]
                except IndexError:
                    continue
                else:
                    #print("before",value,array[int(x[i])][j])
                    if value != 0:
                        #print("added",value,array[int(x[i])][j])
                        sum.append(array[int(x[i])][j])
        #print(sum)
        #time.sleep(3)
        self.circle_sum = sum

    def centrelum(self,array,c_point=(464,470),size=30):
        self.circlesum(array,c_point,size) 
        np_sum = numpy.array(self.circle_sum)
        np_sum = np_sum / len(np_sum)
        self.c_sum = numpy.sum(np_sum)

    def edgelum(self,array,c_point=(464,470)):
        self.circlesum(array,c_point=(464,470),size = 460)
        max = self.circle_sum
        self.circlesum(array,c_point=(464,470),size = 200)
        min = self.circle_sum
        max_sum = numpy.sum(max)
        min_sum = numpy.sum(min)
        average = (max_sum - min_sum) / (len(max) - len(min))
        self.e_avg = average

    def median_edgelum(self,array,c_point=(464,470)):
        val_list = []
        r_c = 130
        c_l = numpy.pi*2*330 #circumference 
        circle_num = numpy.floor(c_l/r_c)
        for i in range(int(circle_num)):
            x = 464 + 330*numpy.cos(2*numpy.pi*i/circle_num)
            y = 464 + 330*numpy.sin(2*numpy.pi*i/circle_num)
            self.circlesum(array,c_point=(x,y),size = r_c)
            val = self.circle_sum
            z = numpy.sum(val)/len(val)
            #print(z)
            val_list.append(z)
        self.e_avg = numpy.median(val_list)

    def wholelum(self,array,c_point=(464,470)):
        self.circlesum(array,c_point=(464,470),size=464)
        np_sum = numpy.array(self.circle_sum)
        average = numpy.sum(np_sum / len(np_sum))
        self.w_avg = average 

    def countour_meth(self,idx=0):
        if type(idx) == int:
            data = self.hdu_data[idx]
        elif type(idx) == tuple:
            for i in range(len(idx)-1):
                data = self.hdu_data[idx[i]]
                img = (data/256).astype('uint8')
                mask = cv.threshold(img, 100, 255, cv.THRESH_BINARY)[1]
                result = cv.inpaint(img, mask, 21, cv.INPAINT_TELEA) 
                ret, thresh = cv.threshold(result, 100, 150, 3)
                contours, hierarchy = cv.findContours(thresh,
                                                    cv.RETR_TREE,
                                                    cv.CHAIN_APPROX_NONE  #CHAIN_APPROX_NONE or CHAIN_APPROX_SIMPLE
                                                    )
                                                        # source image, retrival mode , proximation method
                                                        
                cv.drawContour(img, contours, -1, (0,255,0), 3)
                plt.figure(1,figsize=(5,3),dpi=300)
                plt.imshow(img)
                plt.show()
            idx = idx[i+1]
        
        data = self.hdu_data[idx]
        img = (data/256).astype('uint8')
        mask = cv.threshold(img, 100, 255, cv.THRESH_BINARY)[1]
        result = cv.inpaint(img, mask, 21, cv.INPAINT_TELEA) 
        ret, thresh = cv.threshold(result, 100, 150, 3)
        contours, hierarchy = cv.findContours(thresh,
                                            cv.RETR_TREE,
                                            cv.CHAIN_APPROX_NONE  #CHAIN_APPROX_NONE or CHAIN_APPROX_SIMPLE
                                            )
                                                # source image, retrival mode , proximation method
                                                
        cv.drawContour(img, contours, -1, (0,255,0), 3)
        plt.figure(1,figsize=(5,3),dpi=300)
        plt.imshow(img)
        plt.show()

    def blobbing_meth(self,idx=0,params_args=[]):
        params = cv.SimpleBlobDetector_Params()
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

        if type(idx) == int:
            data = self.hdu_data[idx]
        elif type(idx) == tuple:
            for i in range (len(idx)-1):
                img8 = (self.hdu_data[idx[i]]/256).astype('uint8')
                params = cv.SimpleBlobDetector_Params()
                params.blobColor = 255
                detector = cv.SimpleBlobDetector_create(params)
                keypoints = detector.detect(img8)
                self.img_with_keypoints = cv.drawKeypoints(img8,
                                                keypoints,
                                                numpy.array([]),
                                                (0,200,200),
                                                cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                plt.imshow(img8)
                plt.show()
            idx = idx[i+1]


        img8 = (self.hdu_data[idx[i]]/256).astype('uint8')
        params.blobColor = 255
        detector = cv.SimpleBlobDetector_create(params)
        keypoints = detector.detect(img8)
        self.img_with_keypoints = cv.drawKeypoints(img8,
                                        keypoints,
                                        numpy.array([]),
                                        (0,200,200),
                                        cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(img8)
        plt.show()

    def cloud_percentage_original(self,array,type="median"):
            self.centrelum(array,size = 30)
            Yc = self.c_sum
            if type == "median":
                self.median_edgelum(array)
            else:
                self.edgelum(array)
            Ye = self.e_avg 
            self.circlesum(array)
            Ya = numpy.sum(self.circle_sum) / len(self.circle_sum)
            max = numpy.max(array)
            #print("Max             :",max)
            #print("Yc              :",Yc)
            #print("Ye              :",Ye)
            #print("Yc-Ye           :",Yc-Ye)
            print("(Yc-Ye)/Average :",(Yc-Ye)/Ya)
            print("(Yc-Ye) / Max   :",(Yc-Ye)/max)
            if numpy.isnan(float((Yc-Ye)/Ya)) or (Yc-Ye)/Ya < 0:
                a = 0
            else:
                a = (Yc-Ye)/Ya

            if numpy.isnan(float((Yc-Ye)/max)) or Yc-Ye/max <0 :
                b = 0
            else:
                b = (Yc-Ye)/max
            return a,b

    def perfect_minus(self,array,perfect):
        self.circlesum(array)
        ag_1 = numpy.sum(self.circle_sum) /len(self.circle_sum)
        self.circlesum(perfect)
        ag_2 = numpy.sum(self.circle_sum)/len(self.circle_sum)
        x = len(array)
        y = len(array[0])
        if x != len(perfect) or y != len(perfect[0]):
            print(x,y,len(perfect),len(perfect[0]))
        for i in range(x):
            for j in range(y):
                array[i][j] -= perfect[i][j] *ag_1/ag_2
        return array