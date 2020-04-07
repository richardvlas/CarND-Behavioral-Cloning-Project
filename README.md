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
* `readme.md` - A markdown file explaining the project structure and training approach
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
This section describes the deep neural network model deployed and the training strategy used

### Model Architecture
I first tried to use LeNet neural network to see how well the model would predict steering angle command. I mainly used it to get started with the project and develop the training pipeline since I was already familiar with the model. Nevertheless, the model seemed not to predict the steering command accurately in all cases. Especially in sharp curves without lane markings the model went off the road. 

So, I decided to use a more powerfull network. One of the more advanced model architectures is a convolutional neural network published by vehicle team at NVIDIA that maps raw pixels from a camera directly to steering commands. 
Here is a [link](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) that describes the model in detail.

The network has about 27 million connections and 250 thousand parameters as shown in the following figure

<img src="img/NVIDIA_CNN.png" width="75%" height="75%">




started by using 

The selection of the neural network model I used 

### Data Preprocessing and Data Augmentation 

### Model Training


## Simulation and Model Testing



