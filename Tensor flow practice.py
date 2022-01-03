# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist  ## import the training data and training labels including the test data/labels

## Define the training stuff and the test stuff
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

## Define the names of the labels like a dictionary
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

## Show an example of the training image 

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

## Scale the training data from 0 to 1
train_images = train_images / 255.0

test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  ## Flattens the 28 by 28 training image to a 784 1D array. Defines input nodes
    tf.keras.layers.Dense(128, activation='relu'),  ## Defines the first layer of the nn with a ReLu activation function
    tf.keras.layers.Dense(10)])                     ## Specifies the last layer (output layer) size

model.compile(optimizer='adam',                     ## Algorithem for updating the model using the loss function
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), ## Specify where the loss function is calculated from
              metrics=['accuracy'])                 ## Gives the accuracy of the model based on the current state of the model and the test data

model.fit(train_images, train_labels, epochs=10)    ## Used to start training the model

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2) ## Compare the model against the test data to find the accuracy and loss value

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model,                         ## Create a probability model based of the previous model
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)                    ## Calculate the predicted label for each test image with its probability

np.argmax(predictions[0])                           ## Find the largest value in a 1D array, illistrating the largest probability index

img = test_images[1]                                ## Take a test image
img = (np.expand_dims(img,0))                       ## add it to a array of only itself
predictions_single = probability_model.predict(img) ## Pass it into the probability model to calculate the label

print(predictions_single)



