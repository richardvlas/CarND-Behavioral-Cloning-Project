import os
import csv
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D 
from keras.layers import Conv2D, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint


def nvidia_nn(dropout=0.0):
    """
    Nvidia convolutional neural network
    Ref: http://images.nvidia.com/content/tegra/automotive/images/2016/
    solutions/pdf/end-to-end-dl-using-px.pdf
    :dropout: float between 0 and 1. Fraction of the input units to drop.
    """
    # Sequential model - linear stack of layers
    model = Sequential()
    
    # Lambda layer: Preprocessing images - 1. Normalizing, 2. Mean centering
    # Input image shape: 160x320px, 3 color channels (RGB) 
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    
    # Cropping layer: Crops images (top, bottom) = (70, 25)
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    
    # 1. Convolutional Layer
    # number of filters (out. space dim.) = 24, filter window size = 5x5
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
    
    # 2. Convolutional layer
    # number of filters (out. space dim.) = 36, filter window size = 5x5
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
    
    # 3. Convolutional layer
    # number of filters (out. space dim.) = 48, filter window size = 5x5
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
    
    # 4. Convolutional layer
    # number of filters (out. space dim.) = 64, filter window size = 3x3
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))

    # 5. Convolutional layer
    # number of filters (out. space dim.) = 64, filter window size = 3x3
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))

    # Flatten layer: Flattens the input
    model.add(Flatten())
    
    # Dense (fully connected) Layers, output = 1 (steering angle)
    model.add(Dense(100))
    # Dropout layer to prevent overfitting
    model.add(Dropout(rate=dropout))
    model.add(Dense(50))
    # Dropout layer to prevent overfitting
    model.add(Dropout(rate=dropout))
    model.add(Dense(10))
    model.add(Dense(1))

    # Configure the learning process
    # Optimizer = adam, loss function = mse (mean square error)
    model.compile(optimizer='adam', loss='mse')

    return model


# Load and prepare samples for image generator
def load_img_paths(log_file):
    samples = []
    with open(log_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples

img_path = './driving_data/driving_log.csv'
samples = load_img_paths(img_path)

# Split images into random train and test subsets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Number of images: {}'.format(len(samples)))
print('Number of training samples: {}'.format(len(train_samples)))
print('Number of test samples: {}'.format(len(validation_samples)))


def generator(samples, fliped=False, all_cameras=False, correction=0.1, batch_size=32):
    """ 
    Define generator to load image data
    :batch_size: Number of input images to yield at once
                 Note that # output images will depend on fliped and
                 all_cameras flags
    """
    num_samples = len(samples)
    # Loop forever so the generator never terminates
    while 1:
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images          = []
            steering_angles = []
            
            for batch_sample in batch_samples:
            
                steering_center = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                
                ## read in and preprocess images from center camera
                img_center = cv2.imread(batch_sample[0])                
                img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB)
                images.append(img_center)
                steering_angles.append(steering_center)
                #horizontal flip
                if fliped:
                    images.append(cv2.flip(img_center,1))
                    steering_angles.append(-steering_center)

                if all_cameras:
                    ## read in and preprocess images from left camera
                    img_left = cv2.imread(batch_sample[1])
                    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
                    images.append(img_left)
                    steering_angles.append(steering_left)
                    ## read in and preprocess images from right camera
                    img_right = cv2.imread(batch_sample[2])
                    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
                    images.append(img_right)
                    steering_angles.append(steering_right)

                    if fliped:
                        #horizontal flip - left image
                        images.append(cv2.flip(img_left,1))
                        steering_angles.append(-steering_left)
                        #horizontal flip - right image
                        images.append(cv2.flip(img_right,1))
                        steering_angles.append(-steering_right)    
                    
            X_features = np.array(images)
            y_labels = np.array(steering_angles)

            yield sklearn.utils.shuffle(X_features, y_labels)


# Set batch size
batch_size = 64

# Save training and validation generators
train_generator      = generator(train_samples, fliped=True, all_cameras=True, 
                                 correction=0.1, batch_size=batch_size)
validation_generator = generator(validation_samples, fliped=True, all_cameras=True, 
                                 correction=0.1, batch_size=batch_size)

# Save Nvidia neural network model
model = nvidia_nn(dropout=0.35)

# Prints a string summary of the network 
print(model.summary())

# Callbacks function EarlyStopping stops training when a monitored quantity has stopped improving.
# This function will be passed to the .fit_generator() method of the Sequential class.  
early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=3)
# Save the best model after every epoch 
model_ckp = ModelCheckpoint('./models/best_model.h5', monitor='val_loss', 
                            verbose=1, save_best_only=True)

# Train the model for a fixed number of epochs
history_object = model.fit_generator(train_generator, 
                                     steps_per_epoch=np.ceil(len(train_samples)/batch_size), 
                                     validation_data=validation_generator, 
                                     validation_steps=np.ceil(len(validation_samples)/batch_size), 
                                     epochs=30,
                                     callbacks=[early_stop, model_ckp],
                                     verbose=1)

# Save the model to a single HDF5 file
model.save('./models/model.h5')

print('Model Saved!')