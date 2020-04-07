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
* `video.mp4` - A video recording of the vehicle driving autonomously one lap around the track

### Running the Model
You can test your model by launching the simulator and entering autonomous mode. The car will just sit there until your Python server connects to it and provides it steering angles. Here’s how you start your Python server:

```python 
python drive.py model.h5
```
Once the model is up and running in drive.py, you should see the car move around



Below I will consider the [rubric points](https://review.udacity.com/#!/rubrics/1968/view) of this project individually and describe how I addressed each point in my implementation. Here is the link to my [project exploratory code](Behavioral_Cloning.ipynb).


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
 

### Data Preprocessing and Data Augmentation 

### Model Training


## Simulation and Model Testing



