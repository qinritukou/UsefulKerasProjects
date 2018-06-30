from __future__ import print_function 
import random 
from os import listdir 
import glob 
import time 

import numpy as np 
from scipy import misc 
import tensorflow as tf 
import h5py 

from keras.datasets import mnist 
from keras.utils import np_utils 

import matplotlib.pyplot as plt 

# Setting the random seed so that the results are reproducible.
random.seed(101)
train_datasize = 60000
test_datasize = 10000

# setting variables for MNIST image dimensions 
mnist_image_height = 28 
mnist_image_width = 28 

# Import MNIST data from keras 
(X_train, y_train), (X_test, y_test) = mnist.load_data() 

# Checking the downloaded data 
print("Shape of training dataset: {}".format(np.shape(X_train)))
print("Shape of test dataset: {}".format(np.shape(X_test)))

# plt.figure() 
# plt.imshow(X_train[0], cmap='gray')
# plt.show()

print("Label for image: {}".format(y_train[0]))

def build_synth_data(data, labels, dataset_size):
    # Define synthetic image dimensions 
    synth_img_height = 64 
    synth_img_width = 64 

    # Define synthetic data 
    synth_data = np.ndarray(shape=(dataset_size, synth_img_height, synth_img_width), dtype=np.float32)

    # Define synthetic labels 
    synth_labels = []

    # For a loop till the size of the synthetic dataset 
    for i in range(0, dataset_size):
        print("{} created.".format(i))
        # Pick a random number of digits to be in the dataset 
        num_digits = random.randint(1, 5)

        # Randomly sampling indices to extract digits + labels afterwards 
        s_indices = [random.randint(0, len(data) - 1) for p in range(0, num_digits)]

        # stitch images together 
        new_image = np.hstack([X_train[index] for index in s_indices])
        # stitch the labels together 
        new_label = [y_train[index] for index in s_indices]

        # Loop till number of digits - 5, to concatenate blanks images, and blank label together 
        for j in range(0, 5 - num_digits):
            new_image = np.hstack([new_image, np.zeros(shape=(mnist_image_height, mnist_image_width))])
            new_label.append(10) # Might need to remove this step 
        # Resize image 
        new_image = misc.imresize(new_image, (64, 64))
        # Assign the image to synth_data 
        synth_data[i,:,:] = new_image
        # Assign the label to synth_data 
        synth_labels.append(tuple(new_label))

    # Return the synthetic dataset 
    return synth_data, synth_labels

# Building the training dataset 
X_synth_train, y_synth_train = build_synth_data(X_train, y_train, train_datasize)

# Building the test dataset 
X_synth_test, y_synth_test = build_synth_data(X_test, y_test, test_datasize)

# Checking a sample 
# plt.figure() 
# plt.imshow(X_synth_test[232], cmap='gray')
# plt.show()
# print(y_synth_train[232])


"""
   Preparatory Preprocessing
""" 
# Converting labels to One-hot represents of shape (set_size, digits, classes)
possible_classes = 11 

def convert_labels(labels):
    #As per Keras conventions, the multiple labels need to be of the form [array_digit1,...5]
    #Each digit array will be of shape (60000,11)
    #Declare output ndarrays
    dig0_arr = np.ndarray(shape=(len(labels),possible_classes))
    dig1_arr = np.ndarray(shape=(len(labels),possible_classes))
    dig2_arr = np.ndarray(shape=(len(labels),possible_classes))
    dig3_arr = np.ndarray(shape=(len(labels),possible_classes)) #5 for digits, 11 for possible classes  
    dig4_arr = np.ndarray(shape=(len(labels),possible_classes))

    for index,label in enumerate(labels):
        
        #Using np_utils from keras to OHE the labels in the image
        dig0_arr[index,:] = np_utils.to_categorical(label[0],possible_classes)
        dig1_arr[index,:] = np_utils.to_categorical(label[1],possible_classes)
        dig2_arr[index,:] = np_utils.to_categorical(label[2],possible_classes)
        dig3_arr[index,:] = np_utils.to_categorical(label[3],possible_classes)
        dig4_arr[index,:] = np_utils.to_categorical(label[4],possible_classes)

    return [dig0_arr,dig1_arr,dig2_arr,dig3_arr,dig4_arr]

train_labels = convert_labels(y_synth_train)
test_labels = convert_labels(y_synth_test)

#Checking the shape of the OHE array for the first digit position
print(np.shape(train_labels[0]))

print(np_utils.to_categorical(y_synth_train[234][0],11))

"""
    Preprocessing Images for model
"""
def prep_data_keras(img_data):
    
    #Reshaping data for keras, with tensorflow as backend
    img_data = img_data.reshape(len(img_data),64,64,1)
    
    #Converting everything to floats
    img_data = img_data.astype('float32')
    
    #Normalizing values between 0 and 1
    img_data /= 255
    
    return img_data

train_images = prep_data_keras(X_synth_train)
test_images = prep_data_keras(X_synth_test)

print(np.shape(train_images))
print(np.shape(test_images))

"""
    Model Building
"""
#Importing relevant keras modules
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D

#Building the model

batch_size = 128
nb_classes = 11
nb_epoch = 12

#image input dimensions
img_rows = 64
img_cols = 64
img_channels = 1

#number of convulation filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

#defining the input
inputs = Input(shape=(img_rows,img_cols,img_channels))

#Model taken from keras example. Worked well for a digit, dunno for multiple
cov = Convolution2D(nb_filters,kernel_size[0],kernel_size[1],border_mode='same')(inputs)
cov = Activation('relu')(cov)
cov = Convolution2D(nb_filters,kernel_size[0],kernel_size[1])(cov)
cov = Activation('relu')(cov)
cov = MaxPooling2D(pool_size=pool_size)(cov)
cov = Dropout(0.25)(cov)
cov_out = Flatten()(cov)


#Dense Layers
cov2 = Dense(128, activation='relu')(cov_out)
cov2 = Dropout(0.5)(cov2)



#Prediction layers
c0 = Dense(nb_classes, activation='softmax')(cov2)
c1 = Dense(nb_classes, activation='softmax')(cov2)
c2 = Dense(nb_classes, activation='softmax')(cov2)
c3 = Dense(nb_classes, activation='softmax')(cov2)
c4 = Dense(nb_classes, activation='softmax')(cov2)

#Defining the model
model = Model(input=inputs,output=[c0,c1,c2,c3,c4])

#Compiling the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#Fitting the model
model.fit(train_images,train_labels,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,
          validation_data=(test_images, test_labels))

model.save('DigitSequenceRecognition.h5');

# Test 
predictions = model.predict(test_images)
print(np.shape(predictions))
print(len(predictions[0]))
print(np.shape(test_labels))


def calculate_acc(predictions,real_labels):
    
    individual_counter = 0
    global_sequence_counter = 0
    for i in range(0,len(predictions[0])):
        #Reset sequence counter at the start of each image
        sequence_counter = 0 
        
        for j in range(0,5):
            if np.argmax(predictions[j][i]) == np.argmax(real_labels[j][i]):
                individual_counter += 1
                sequence_counter +=1
        
        if sequence_counter == 5:
            global_sequence_counter += 1
         
    ind_accuracy = individual_counter/(50000.0)
    global_accuracy = global_sequence_counter/(10000.0)
    
    return ind_accuracy,global_accuracy


ind_acc,glob_acc = calculate_acc(predictions,test_labels)
print("The individual accuracy is {} %".format(ind_acc*100))
print("The sequence prediction accuracy is {} %".format(glob_acc*100))

#Printing some examples of real and predicted labels
for i in random.sample(range(0,test_datasize),5):
    
    actual_labels = []
    predicted_labels = []
    
    for j in range(0,5):
        actual_labels.append(np.argmax(test_labels[j][i]))
        predicted_labels.append(np.argmax(predictions[j][i]))
        
    print("Actual labels: {}".format(actual_labels))
    print("Predicted labels: {}\n".format(predicted_labels))