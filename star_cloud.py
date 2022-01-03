import Star_measure as sm
import matplotlib.pyplot as plt
import copy,pickle,nn ,os
import numpy as np
import tensorflow as tf
import os

file_names =   ["Full_cloud.fit","Partially.fit","Starry.fit","Sky_1.fit","Sky_2.fit","Sky_3.fit","Sky_4.fit","Sky_5.fit","Sky_6.fit","Sky_7.fit","Sky_8.fit","Sky_9.fit","Sky_10.fit"]
sorted_names = ["Full_cloud.fit","Sky_1.fit","Sky_3.fit","Sky_10.fit","Sky_8.fit","Partially.fit","Sky_5.fit","Sky_6.fit","Sky_2.fit","Sky_7.fit","Starry.fit","Sky_9.fit","Sky_4.fit"]
estimated_cloud = [100,40,0,100,20,100,0,30,20,0,100,0,100]
sorted_cloud =    [100,100,100,100,100,40,30,20,20,0,0,0,0]
file_location = ".\\Fit_Images\\"

training_location = ".\\Fit_Images\\Training\\"
training_files = os.listdir(training_location)


def create_values(overwrite = False):
    if os.path.exists(".//Results.pkl") and overwrite == False:
        loaded_values = load_values()
    else:
        file_names =   ["Full_cloud.fit","Partially.fit","Starry.fit","Sky_1.fit","Sky_2.fit","Sky_3.fit","Sky_4.fit","Sky_5.fit","Sky_6.fit","Sky_7.fit","Sky_8.fit","Sky_9.fit","Sky_10.fit"]
        sorted_names = ["Full_cloud.fit","Sky_1.fit","Sky_3.fit","Sky_10.fit","Sky_8.fit","Partially.fit","Sky_5.fit","Sky_6.fit","Sky_2.fit","Sky_7.fit","Starry.fit","Sky_9.fit","Sky_4.fit"]
        estimated_cloud = [100,40,0,100,20,100,0,30,20,0,100,0,100]
        sorted_cloud =    [100,100,100,100,100,40,30,20,20,0,0,0,0]
        file_location = ".\\Fit_Images\\"

        perfect = sm.image_process(["Starry.fit"],".\\Fit_Images\\")
        trial = sm.image_process(file_names,file_location)
        trial.create_masks()
        trial.save_mask()
        trial.data_w_mask()
        list1 = []
        list2 = []
        for i in range(len(file_names)):
            a1,a2 = trial.cloud_percentage_original(trial.trimmed_data[i])            
            list1.append([a1,a2])
            adj_img = trial.perfect_minus(trial.hdu_data[i],perfect.hdu_data[0])
            b1,b2 = trial.cloud_percentage_original(adj_img)
            list2.append([b1,b2])
            print(i+1,"/",len(file_names))
            #print("Image shown")
            #fig, (ax0,ax1) = plt.subplots(nrows=1,ncols=2,sharex=True)
            #ax0.imshow(img,cmap = "gray")
            #ax1.imshow(adj_img,cmap = "gray")
            #plt.show()
        
        myFile = open("Results.pkl","wb")
        pickle.dump([list1,list2],myFile)
        myFile.close()
        loaded_values = [list1,list2]
    return loaded_values

def load_values():
    myFile = open("Results.pkl","rb")
    loaded_list = pickle.load(myFile)
    myFile.close()
    return loaded_list

def flatten(array1,array2):
    output = []
    for i in range(len(array1)):
        idx_1 = array1[i][0]
        idx_2 = array1[i][1]
        idx_3 = array2[i][0]
        idx_4 = array2[i][1]    

        a = normalise(idx_1)
        b = normalise(idx_2)
        c = normalise(idx_3)
        d = normalise(idx_4)

        output.append([a,b,c,d])

    return output

def normalise(value):
    return 0.5(value/(1+value)+1)

load_list = create_values()
l1 = load_list[0]
l2 = load_list[1]
#print(l1,l2)

train = flatten(l1,l2)
training_values = copy(estimated_cloud)
print(training_values)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(4)),  ## Flattens the 28 by 28 training image to a 784 1D array. Defines input nodes
    tf.keras.layers.Dense(15, activation='relu'),  ## Defines the first layer of the nn with a ReLu activation function
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)])  

model.compile(optimizer='adam',                     ## Algorithem for updating the model using the loss function
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), ## Specify where the loss function is calculated from
              metrics=['accuracy'])    

model.fit(train, training_values, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

""" net = nn.net((4,10,5,10,1))
for k in range(100):
    for i in range(len(file_names)):
        x = l1[i]
        y = l2[i]
        z = [x[0],x[1],y[0],y[1]]
        net.network[0][0] = np.array(z)
        net.for_pass()
        model = net.f_pass_out[len(net.f_pass_out)-1][0]   ## only one value
        print("predicted :",model,"estimated :",estimated_cloud[i])
        net.error_calc_norm([estimated_cloud[i]],[model])
        net.back_pass(mu=0.01)

net.save() """


""" perfect = sm.image_process(["Starry.fit"],
                            ".\\Fit_Images\\")
perfect.median_edgelum(perfect.hdu_data[0])
x = perfect.e_avg
print(x) """