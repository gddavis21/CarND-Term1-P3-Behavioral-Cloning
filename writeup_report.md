# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolutional neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image_training_progress]: ./examples/training_progress.png
[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model consists of a convolution neural network with 5x5 and 3x3 filter sizes, and depths between 24 and 64 (model.py lines 146-173). 

The model includes RELU activation and Max-Pooling layers to introduce nonlinearity, and the input image data is cropped and normalized in the model using Keras cropping & lambda layers (model.py line 132-137). 

#### 2. Attempts to reduce overfitting in the model

Training my final model architecture with the final selection of training data, model parameters, and image augmentation/pre-processing displayed no apparent evidence of overfitting (or underfitting). I experimented with adding dropout layers, varying the number, placement and dropout probabilities. In all cases adding dropout appeared to result in underfitting the model and worse steering performance on the track. My conclusion was that for this model, the utilized image augmentation methods provided sufficient robustness to overfitting, precluding the need for dropout or any other explicit overfitting reduction methods (like L1/L2 regularization).

The following plot illustrates progress of training & validation loss during a training cycle on my model (trained for 30 epochs). I used MSE for the training loss metric, but plotted RMSE as it seems to enhance the visualization without distorting its meaning.

![Model training progress][image_training_progress]

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 298-323). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

I used mini-batch gradient descent (via Keras Model.fit_generator) to train the model.
* The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 314).
* Training for 30 epochs with batch size of 128 proved to be an effective combination.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to build a CNN model similar to LeNet, a relatively simple and well-known CNN architecture. While I suspected LeNet would ultimately prove too simple for an effective steering angle model, it had the merits of being easy to implement and fast to train. 

I used my LeNet-based model as a test-bed for developing other project building blocks: collecting training data, randomized image augmentation, running the training/validation pipeline, and testing the trained model in the simulator. I did eventually decide to implement a more capable model, but the LeNet-based model was very useful in proving out the functionality of these other components.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I used MSE for the training loss metric, and monitored model training for evidence of underfitting (training loss stays too high) and overfitting (validation loss levels off and/or increases while training loss decreases).

My next step was to build a substantially more complex & capable CNN model based on the architecture described in the article [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf), published by NVIDIA in 2016. Since this architecture was designed for the steering angle regression problem, I thought it would be an excellent solution for this project.

While I was able to train the NVIDIA-based model and get it to successfully drive around Track #1, model training was slow and seemed prone to overfitting (validation loss leveled out and even increased after a small number of training epochs). I experimented with adding Dropout between various layers in the model to reduce overfitting. This was somewhat successful, albeit time-consuming.

In the end I decided to "split the difference" and implement a model architecture more complex than LeNet but less complex than the NVIDIA model. The details of my final architecture are found in the next section. With a minimal amount of trial-and-error experimentation, I arrived at an architecture that seemed to strike a nice balance between training time and performance on the track, with the unexpected bonus of neither underfitting nor overfitting (as far as I could tell).

For the sake of being thorough, I did experiment with adding Dropout to this model. However, even a single Dropout layer with a low dropout rate (0.2) resulted in noticeable underfitting. I concluded that overfitting reduction wasn't necessary at that point.

While testing the steering performance on Track #1, I did notice that at times the car got close to the left or right lane lines, and stayed there instead of moving back to the center. I decided to collect several more batches of training data where I specifically started the car on either side of the road, and steered back to the center. I noticed an immediate performance improvement--when the car veered to the edge of the lane, it would steer smoothly back to the center.

At the end of the process, the vehicle is able to drive autonomously around both Track #1 and Track #2 at the default speed of 9 MPH without leaving the road. I have also experimented with increasing the speed. The vehicle is able to successfully navigate Track #1 at speeds up to 30 MPH, and Track #2 at speeds up to 15 MPH. Sadly, at 20 MPH on Track #2 it steers off the road on a sharp down-hill turn just short of the end of the lap. Room for improvement!

#### 2. Final Model Architecture

The final model architecture (model.py lines 121-173) consists of a convolutional neural network with the following layers and layer sizes:

* Image data is pre-processed in the model (model.py function create_initial_model):
    * model assumes input RGB image data of size 320x160
    * top 60 and bottom 20 pixels are cropped via Keras Cropping2D layer
    * image data converted to floating-point and normalized via Keras Lambda layer  

* Neural network layers added to the model (model.py function add_my_CNN_steering_layer):
    1. Convolutional layer: 5x5 filter, depth 24, RELU activation
    2. Convolutional layer: 5x5 filter, depth 48, RELU activation 
    3. Max-Pooling layer
    4. Convolutional layer: 3x3 filter, depth 64, RELU activation
    5. Convolutional layer: 3x3 filter, depth 64, RELU activation 
    6. Max-Pooling layer
    7. Dense layer, width 128, RELU activation
    8. Dense layer, width 64, RELU activation
    9. Steering angle readout layer (1 output)

* The model includes RELU layers and Max-Pooling layers to introduce nonlinearity. 
* All weights are initialized using [He initialization](https://arxiv.org/abs/1502.01852), which has been shown to be effective in combination with RELU activation.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
