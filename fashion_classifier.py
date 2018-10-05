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
#flatten_train_images = train_images.reshape(60000,784)

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(train_images[i], cmap=plt.cm.binary)
	plt.xlabel(class_names[train_labels[i]])
plt.show()

