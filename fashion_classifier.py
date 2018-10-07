#!/usr/local/bin/python

# To avoid AVX2 CPU warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#tensorflow, Keras, numpy, plot package 
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#test "Hello World"
hello = tf.constant('Hello, Tensorflow!')
sess = tf.Session()
sess.run(hello)

#test evaluate 
a = tf.constant(10)
b = tf.constant(32)
sess.run(a+b)

#print tensorflow version
#

####################################
#load fashion_mnist data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# import pdb; pdb.set_trace()

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()


#####################################
#
# Show original pics
#
#####################################
#scale values to a range of 0 to 1 (!!! This step is very important, which impacts accuracy very significantly)
train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize=(10,10))
# for i in range(25):
# 	plt.subplot(5,5,i+1)
# 	plt.xticks([])
# 	plt.yticks([])
# 	plt.grid(False)
# 	plt.imshow(train_images[i], cmap=plt.cm.binary)
# 	plt.xlabel(class_names[train_labels[i]])
# plt.show()

#####################################
#
# Setup the layers (identify the network structure)
# AKA setup layers and nodes per layer
#
#####################################

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28))) #input layer, only reformat the data, no parameters
model.add(keras.layers.Dense(128,activation=tf.nn.relu)) #hidden layer
model.add(keras.layers.Dense(10,activation=tf.nn.softmax)) #output layer

#####################################
#
# Compile the model
# (Loss Function, Optimizer and Metrics are identified here)
# Metrics = Used to monitor the training and testing steps
#
#####################################

model.compile(optimizer=tf.train.AdamOptimizer(),
			  loss='sparse_categorical_crossentropy',
			  metrics=['accuracy'])

#####################################
#
# Train the model
# Steps:
# 1. Feed the training data to the model (training data AND training labels)
# 2. Model in training (auto)
# 3. We ask the model to make predictions about a test set, feed test data and match with the test_labels array
# * identify the number of epochs
#
#####################################

# Steps 1 & 2
model.fit(train_images, train_labels,epochs=5)

# Step 3
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy: ', test_acc)

#####################################
#
# Make Predictions
# 
#####################################

predictions = model.predict(test_images)




