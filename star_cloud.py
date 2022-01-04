import Star_measure as sm
import matplotlib.pyplot as plt
import copy,pickle,nn ,os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

## Main code

## Test Files, Main ones are the first 3
file_names =   ["Full_cloud.fit","Partially.fit","Starry.fit","Sky_1.fit","Sky_2.fit","Sky_3.fit","Sky_4.fit","Sky_5.fit","Sky_6.fit","Sky_7.fit","Sky_8.fit","Sky_9.fit","Sky_10.fit"]
estimated_cloud = [100,40,0,100,20,100,0,30,20,0,100,0,100]
estimated_cloud = np.array(estimated_cloud)/100
file_location = ".\\Fit_Images\\"

## Training files, size should be 999
training_location = ".\\Fit_Images\\Training\\"
directory_files = os.listdir(training_location)

training_files = []
for i in range(len(directory_files)):
    if "___" in directory_files[i] and "mask" not in directory_files[i] and "pkl" not in directory_files[i]:
        training_files.append(directory_files[i])

training_values = []
for i in range(len(training_files)):
    end = training_files[i].split("___")
    val = int(end[1][:len(end[1])-4])
    if val < 0 or val > 100:
        print("Value invalid for file :",training_files[i])
    else:
        training_values.append(val/100)
print("Value list created.")


def create_values(file_names,file_location,overwrite = False):
    perfect = sm.image_process(["Starry.fit"],".\\Fit_Images\\")
    
    problem_idxs = []
    loaded_values = []
    for i in range(len(file_names)):
        x=file_names[i]
        if os.path.exists(file_location+"\\Results_"+str(x[:len(x)-4])+".pkl"):
            myFile = open(file_location+"\\Results_"+str(x[:len(x)-4])+".pkl","rb")
            loaded_list = pickle.load(myFile)
            loaded_values.append(loaded_list)
            myFile.close()
        else:
            trial = sm.image_process([file_names[i]],file_location)
            trial.create_masks()
            trial.data_w_mask()
            a1,a2 = trial.cloud_percentage_original(trial.trimmed_data[0])   
            try:         
                adj_img = trial.perfect_minus(trial.hdu_data[0],perfect.hdu_data[0])
            except IndexError:
                print("Bad image, dimensions are not 928 by 928")
                problem_idxs.append(i)
            else:
                b1,b2 = trial.cloud_percentage_original(adj_img)
                c1 = normalise(a1)
                c2 = normalise(a2)
                d1 = normalise(b1)
                d2 = normalise(b2)
                loaded_values.append([c1,c2,d1,d2])
                myFile = open(file_location+"\\Results_"+str(x[:len(x)-4])+".pkl","wb")
                pickle.dump([c1,c2,d1,d2],myFile)
                myFile.close()

        print(i+1,"/",len(file_names))

    return loaded_values,problem_idxs


def normalise(value):
    return 0.5*(value/(1+value)+1)

if os.path.exists(".\\Cloud_cover_model.h5"):
    model = tf.keras.models.load_model('.\\Cloud_cover_model.h5')
else:
    load_list,problem_idxs = create_values(training_files,training_location)
    for i in range(len(problem_idxs)):
        training_values.pop(problem_idxs[len(problem_idxs)-1-i])

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(4)),  ## Flattens the 28 by 28 training image to a 784 1D array. Defines input nodes
        tf.keras.layers.Dense(15, activation='relu'),  ## Defines the first layer of the nn with a ReLu activation function
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)])  

    model.compile(optimizer='adam',                     ## Algorithem for updating the model using the loss function
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), ## Specify where the loss function is calculated from
                metrics=['accuracy'])    

    model.fit(load_list, training_values, epochs=10)
    model.save("Cloud_cover_model.h5")


## Create test data
load_list = create_values(file_names,file_location)


## Evaluate the test data against the model for an accuracy test
test_loss, test_acc = model.evaluate(load_list,  estimated_cloud, verbose=2)
print('\nTest accuracy:', test_acc)

## Return the actual value obtained from the model
probability_model = tf.keras.Sequential([model,                         
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(load_list) 
print(predictions)    

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