# Cloning Driving Behavioral using Deep Learning
## Overview
This project uses deep neural networks and convolutional neural networks to train a vehicle drive in the middle of the road. Specifically, a vehicle simulator is used to collect images of manual driving and a deep learning model then predicts steering angle in autonomous driving mode.

<img src="img/driving_simulator.png" width="75%" height="75%">

### The Goals of this Project :
* Use vehicle simulator to collect training images
* Load, preprocess and explore the training data set
* Design a deep neural network architecture using Keras API
* Train the deep neural network model and validate it
* Test the model to ensure it predicts accurate steering commands and drives in the middle of the road


### Submitted Files
The project includes the following files:
* `README.md` - A markdown file explaining the project structure and training approach
* `model.py` - The script used to create and train the model
* `drive.py` - The script to drive the car
* `model.h5` - The saved deep learning model
* `best_model.h5` - The saved deep learning model due to 'early stopping' function
* `video.mp4` - A video recording of the vehicle driving autonomously one lap around the track

### Running the Model
You can test your model by launching the simulator and entering autonomous mode. The car will just sit there until your Python server connects to it and provides it steering angles. Here’s how you start your Python server:

```python 
python drive.py model.h5
```
Once the model is up and running in drive.py, you should see the car move around


Below I will consider the [rubric points](https://review.udacity.com/#!/rubrics/1968/view) of this project individually and describe how I addressed each point in my implementation. Here is the link to my [project exploratory code](Behavioral_Cloning.ipynb).

## Training Data Collection

The goal of this project is to drive down the center of the the road. To achieve this, it's necessary to capture a good driving behavior which means driving the vehicle in the center of the road while capturing the training images. 

**My strategy for collecting training data**:
* Drive 2 laps counter-clockwise on the 1st track while staying in the center of the road as much as possible
* Drive 1 lap and record recovery from side of the road - this helps the model learn coming back to the center of the road  
* Drive 1 lap counter-clockwise while staying in the center of the road as much as possible - the model learns to generalize
* Record driving through sharp curves (each curve 2x times) - sharp curves are difficult task for the model to learn, so increasing the amount of data for the model to learn from
* Drive 1 lap on the 2nd track to help generalize the model

The vehicle simulator collects images from 3 cameras which can be used for the model training to improve the accuracy especially for recovery driving from road sides. More details in next section. 

Total number of collected images (per camera): **11555**

Here is a visualization of steering angle command versus image #:

<img src="img/steer_command.png" width="40%" height="40%">


Note that the first part image 0 to image 7000 represents driving on the 1st track, while the last part represents driving on the 2nd track. One can notice that the steering angle command values are much larger for the 2nd track as the track consists of sharp left and right turns. Such data can very well generalize the model as we will see later on. 

In order to explain why I decided to follow my strategy for collecting the training data, let's first look on the steering angle command distribution for data collected on the first track driving it counter-clockwise for approximately 2 laps:

<img src="img/steer_command_hist_2_laps.png" width="75%" height="75%">

It's clear from the histogram that the distribution is left skewed representing a bias in steering angle command as we drove the vehicle in counter-clock wise direction.


The next figure shows all data collected on the first track (including clock-wise, recovery and smooth driving through curves)

<img src="img/steer_command_hist_track_1.png" width="75%" height="75%">

Although the distribution slightly improves it's still left skewed.


The last portion of data collected represents driving 1 lap on second track (jungle) 

<img src="img/steer_command_hist_track_2.png" width="75%" height="75%">

The steering command data is distributed much more symetric than when driving on the 1st track alone. This proves that the data is indeed suitable for the training to generalize the model.


All images combined create dataset with the folowing distribution

<img src="img/steer_command_hist.png" width="75%" height="75%">

A nice symmetric distribution that can help the model to learn predict the steering angle command easier.

Using these data we have a baseline dataset that will be expanded as describe in the next section in order to create even more image data for the training.

## Loading Data, Preprocessing and Data Augmentation

This section explain how the collected images are loaded and futher processed to make the most use of them.

A function called `load_images()` to implement all these steps was defined as

```Python
load_images(img_path, fliped=False, all_cameras=False, correction=0.1)
```

* `img_path`    - path to image log file

* `fliped`      - data augmentation by flipping images and steering measurements around vertical axis

Flipping images and steering measurements is an effective technique to reduce turn bias whenever the data skewed distributed. An example of image and its flipped version is shown below 

|steering angle = -0.103|steering angle = 0.103|
|---|---|
|<img src="img/center_2020_04_05_12_43_58_782.jpg">|<img src="img/center_2020_04_05_12_43_58_782_flipped.jpg">|

* `all_cameras` - if set to `True` all 3 camera images will be loaded and used
* `correction`  - correction factor to create adjusted steering measurements for the side camera images 

This factor helps correct the steering angle command value due to the shifted location of the side cameras. This ensures that the model doen't learn to steer either too soft or too strong when learning from side cameras.

Here is an example of an image captured with from all cameras at a certain time showing the steering angle command adjusted with correction factor `correction=0.1`   

|st. angle = -0.003|st. angle = -0.103|st. angle = -0.203|
|---|---|---|
| <img src="img/left_2020_04_05_12_43_58_782.jpg"> | <img src="img/center_2020_04_05_12_43_58_782.jpg"> | <img src="img/right_2020_04_05_12_43_58_782.jpg"> |

If call the function with the parameters set to `True`

```Python
load_images(img_path, fliped=True, all_cameras=True, correction=0.1)
```

then the total number of images loaded and created by data augmentation is equal to: **69330**

So we have 6x more images (2 more camera images at each timestep and 2x more data due to augmentation) to be used for training and validation respectively.

All 69330 images combined create dataset with the folowing distribution

<img src="img/steer_command_hist_data_proc_aug.png" width="45%" height="45%">

The disadvantage of the `load_images()` function is that it loads all images in memory at once and so for large datasets would allocate too much memory making this function memory inefficient!

A better approach would be to define a Generator function which is an efficient way of building iterators in Python. Iterators are memory-efficient in the sense that they can process a desired portion of data at a given time instead of loading all data at once.

Here we define a generator that will be used later on in the training of the model:
```Python
generator(samples, fliped=False, all_cameras=False, correction=0.1, batch_size=32)
```

It takes one additional parameter `batch_size` which specifies number of input images to yield at once

The generator function also shuffles the sample data to make sure the learning process is not dependent on the order of data collected

```Python
samples = sklearn.utils.shuffle(samples)
```

## Model Architecture and Training Strategy
This section describes the deep neural network model deployed and the training strategy used.

### Model Architecture
I first tried to use LeNet neural network to see how well the model would predict steering angle command. I mainly used it to get started with the project and develop the training pipeline since I was already familiar with the model. Nevertheless, the model seemed not to predict the steering command accurately in all cases. Especially in sharp curves without lane markings the model went off the road. 

So, I decided to use a more powerfull network. One of the more advanced model architectures is a convolutional neural network published by vehicle team at NVIDIA that maps raw pixels from a camera directly to steering commands. 
Here is a [link](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) that describes the model in detail.

The network consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers. It has about 27 million connections and 250 thousand parameters as shown in the figure below:


<img src="img/NVIDIA_CNN.png" width="45%" height="45%">


I took this architecture as a baseline and performed few modifications. The input to the network is different from the original model, since the training images come in shape of 160x320px with 3 color channels (RGB) that are passed to the network. The first layer performs image preprocessing: Normalization and mean centering. This was implemented using `Lambda` layer:

```python
Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3))
```

The next layer is used to crops the images on top 70px and on bottom 25px. This step is done to make sure that the images capture mainly the road and nothing else that would complicate the training from these images. It also reduces training time as the image size gets smaller. This was implemented using `Cropping2D` layer:

```python
Cropping2D(cropping=((70,25), (0,0)))
```

Next, series of 5 convolutional layers with striding an relu activation function was added. The first three convolutional layers use a 2×2 stride and a 5×5 kernel and the last two convolutional layers use a 1x1 stride convolution with a 3×3 kernel size.

```python
Conv2D(24, (5, 5), strides=(2, 2), activation="relu")
Conv2D(36, (5, 5), strides=(2, 2), activation="relu")
Conv2D(48, (5, 5), strides=(2, 2), activation="relu")
Conv2D(64, (3, 3), strides=(1, 1), activation="relu")
Conv2D(64, (3, 3), strides=(1, 1), activation="relu")
```

The output from the last convolutional layer is first flattened into shape of 2112 inputs and then feeded into a series of 3 fully connected layers. The first fully connected layer consists of 100 output units. To avoid overfitting a dropout layer is added with 35% dropout probability. A next fully connected layer with 50 output units is again used and followed by a dropout layer. The third fully connected layer has 10 outputs and no dropout layer is used anymore.

```python
Flatten()
Dense(100)
Dropout(rate=0.35)
Dense(50)
Dropout(rate=0.35)
Dense(10)
```

Since the output from the neural network model is predicting one value (steering angle command), a fully connected layer with one output unit is used here.

```python
Dense(1)
```

**Final model architecture**

The following table shows the final model architecture:

|Layer|Description|Param #|
|---|---|---|
|Input | 160x320x3 image|0|
|Lambda| Normalization & mean centering, output 160x320x3| 0|
|Cropping2D | Crop input images, output 65x320x3 | 0|
|Convolution 5x5| 2x2 stride, activation:relu, output 31x158x24| 1824|
|Convolution 5x5| 2x2 stride, activation:relu, output 14x77x36| 21636|
|Convolution 5x5| 2x2 stride, activation:relu, output 5x37x48| 43248|
|Convolution 3x3| 1x1 stride, activation:relu, output 3x35x64| 27712|
|Convolution 3x3| 1x1 stride, activation:relu, output 1x33x64| 36928|
|Flatten| output 2112| 0|
|Fully connected| output 100| 211300|
|Dropout|drop probability 35%| 0|
|Fully connected| output 50| 5050|
|Dropout|drop probability 35%| 0| 
|Fully connected| output 10| 510|
|Fully connected| output 1| 11|

* Total params: 348,219
* Trainable params: 348,219
* Non-trainable params: 0

This model achitecture is defined in `nvidia_nn()` function. 

As a next step I configured the model for training. Specificaly, the command below (part of `nvidia_nn()` function)

```python
model.compile(optimizer='adam', loss='mse')
```
sets the type of optimizer to be used and the objective function to minimized. I selected `'adam'` algorithm which is an extension to stochastic gradient descent that updates network weights iteratively for each NN parameter instead of maintaining a single learning rate for all updates during training. This makes `'adam'` a great choice as it requires little tuning effort. 

It's well worth to have a look on the training performance for different optimization algoritms to see how well Adam outperforms some of the other types of optimizators. Here is provided [here](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/). 

The steering angle command prediction we are trying to solve in this project represents a classical regression problem. Regression problem optimization can be mathematically achieved by minimizing an objective function. A suitable objective function is for example **mean squared error** function. Therefore, the next paramter to be set in `model.compile()` is `loss='mse'`  

### Model Training
Having the model architecture defined, the training step can be conducted. As mentioned in previous above a Python generator will be used to load and preprocess image training data in batches which is much more memory-efficient than loading all image data in memory at once.


In the first step the image dataset `samples` is splitted by sklearn module's function `train_test_split` into training `train_samples` and validation set `validation_samples`. 20% of the data set will be used for validation 
```python
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
```

Then two generators (one for training and one for validation) are saved as
```python
train_generator      = generator(train_samples, fliped=True, all_cameras=True, 
                                 correction=0.1, batch_size=batch_size)
validation_generator = generator(validation_samples, fliped=True, all_cameras=True, 
                                 correction=0.1, batch_size=batch_size)
```

Next, the neural network model with dropout set to 35% is saved as
```python
model = nvidia_nn(dropout=0.35)
```

One of the challenges when training neural networks is to set appropriate number of epochs before the training is terminated.
Keras offers a great solution to this problem by using Callbacks which is a set of functions to be applied at given stages of the training procedure. 

Two callback functions can be used to ensure the model stops at appropriate time, doesn't overfit and the best model from the training phase is saved: `EarlyStopping` and `ModelCheckpoint`:

`EarlyStopping` called as: 
```python
early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=3)
```
stops the training when a monitored quantity has stopped improving. Here the quantity to be monitored is the validation loss.
The `patience` parameter specifies number of epochs that produced the monitored quantity with no improvement after which training will be stopped.

`ModelCheckpoint` called as:
```python
model_ckp = ModelCheckpoint('./models/best_model.h5', monitor='val_loss', 
                            verbose=1, save_best_only=True)
```
saves the model after every epoch. But since `save_best_only=True` only the the latest best model with respect to validation loss will be saved.


Last step is to train the model for a fixed number of epochs by calling `model.fit_generator()`:
```python
history_object = model.fit_generator(train_generator, 
                                     steps_per_epoch=np.ceil(len(train_samples)/batch_size), 
                                     validation_data=validation_generator, 
                                     validation_steps=np.ceil(len(validation_samples)/batch_size), 
                                     epochs=30,
                                     callbacks=[early_stop, model_ckp],
                                     verbose=1)
```

This will train the model on data generated batch-by-batch by the Python generator for 30 epochs unless early stopping is activated.

To monitor training and validation loss metrics a model history object which is returned from `model.fit_generator()` is saved under `history_object` variable that contains both loss functions for each epoch. 

Here is the visualization of training/validation loss vs epoch #:

<img src="img/Nvidia_3_cams_flip_generator.png" width="45%" height="45%">

The `best_model.h5` file contains model parameters from epoch=27. The training was terminated due to 'early stopping' at this epoch. The file `model.h5` contains parameters from the last epoch and it is not necessarily the best model from the training as can be seen from the validation loss figure, with the loss being slightly increased.


## Simulation and Model Testing

Once the model is trained making good predictions on the training and validation sets, you can test the model by launching the simulator and entering autonomous mode.

```python
python drive.py best_model.h5
```

Once the model is up and running in drive.py, you will see the car move around. 

To meet the project specifications: 
* The car must successfully drive around track one
* No tire may leave the drivable portion of the track surface
* The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle)

The video below shows the final model driving the vehicle in autonomous mode:

![](img/autonomous_driving.gif)

