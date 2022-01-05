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
estimated_cloud = [99,40,0,99,20,99,0,30,20,0,99,0,99]
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
        raise IndexError
    elif val == 100:
        val = 99
        training_values.append(val/100)
    else:
        training_values.append(val/100)


actual_images = []
actual_values = []
trial = sm.image_process(training_files,training_location)
for i in range(len(training_files)):
    hdu_dims = np.array(trial.hdu_data[i]).shape
    print(hdu_dims)
    if hdu_dims[0] == 928 and hdu_dims[1] == 928 and os.path.exists(training_location+training_files[i]):
        actual_images.append(training_files[i])
        actual_values.append(training_values[i])

image_data = sm.image_process(actual_images,training_location)
image_data.create_masks()
image_data.data_w_mask()
load_list = image_data.trimmed_data

if os.path.exists(".\\Cloud_cover_model.h5"):
    model = tf.keras.models.load_model('.\\Cloud_cover_model.h5')
else:
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(928, 928)),  ## Flattens the 28 by 28 training image to a 784 1D array. Defines input nodes
        tf.keras.layers.Dense(200, activation='relu'),  ## Defines the first layer of the nn with a ReLu activation function
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(1)])  

    model.compile(optimizer='adam',                     ## Algorithem for updating the model using the loss function
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), ## Specify where the loss function is calculated from
                metrics=['accuracy'])    

    print(training_values)
    model.fit(load_list, training_values, epochs=100)
    model.save("Cloud_cover_model.h5")


## Create test data
image_data = sm.image_process(file_names,file_location)
image_data.create_masks()
image_data.data_w_mask()
load_list = image_data.trimmed_data


## Evaluate the test data against the model for an accuracy test
test_loss, test_acc = model.evaluate(load_list,  list(estimated_cloud), verbose=2)
print('\nTest accuracy:', test_acc)

## Return the actual value obtained from the model
probability_model = tf.keras.Sequential([model,                         
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(load_list) 
print(predictions)    

