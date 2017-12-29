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

[image_steering_angle_histogram]: ./examples/steering_angle_hist.png
[image_training_progress]: ./examples/training_progress.png
[image_model_architecture]: ./examples/model.png
[image_track1_center_lane_1]: ./examples/track1_center_lane_1.png
[image_track1_center_lane_2]: ./examples/track1_center_lane_2.png
[image_track1_center_lane_3]: ./examples/track1_center_lane_3.png
[image_track2_center_lane_1]: ./examples/track2_center_lane_1.png
[image_track2_center_lane_2]: ./examples/track2_center_lane_2.png
[image_track2_center_lane_3]: ./examples/track2_center_lane_3.png
[image_recover_from_left_1]: ./examples/recover_from_left_1.png
[image_recover_from_left_2]: ./examples/recover_from_left_2.png
[image_recover_from_left_3]: ./examples/recover_from_left_3.png
[image_recover_from_right_1]: ./examples/recover_from_right_1.png
[image_recover_from_right_2]: ./examples/recover_from_right_2.png
[image_recover_from_right_3]: ./examples/recover_from_right_3.png
[image_left_camera]: ./examples/track1_left_camera.png
[image_center_camera]: ./examples/track1_center_camera.png
[image_right_camera]: ./examples/track1_right_camera.png
[image_track1_normal]: ./examples/track1_normal.jpg
[image_track1_flipped]: ./examples/track1_flipped.jpg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 video of autonomous driving on Track #1 at default 9 mph
* Track1-30mph.mp4 video of autonomous driving on Track #1 at 30 mph
* Track2-18mph.mp4 video of autonomous driving on Track #2 at 18 mph
* writeup_report.md which describes the project solution and summarizes results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture & Training Strategy: Overview

#### 1. An appropriate model architecture has been employed

The model consists of a convolution neural network with 5x5 and 3x3 filter sizes, and depths between 24 and 64 (model.py lines 146-173). 

The model includes RELU activation and Max-Pooling layers to introduce nonlinearity, and the input image data is cropped and normalized in the model using Keras cropping & lambda layers (model.py line 132-137). 

#### 2. Attempts to reduce overfitting in the model

Training my final model architecture with the final selection of training data, model parameters, and image augmentation/pre-processing displayed no apparent evidence of overfitting (or underfitting). I experimented with adding dropout layers, varying the number, placement and dropout probabilities. In all cases adding dropout appeared to result in underfitting the model and worse steering performance on the track. My conclusion was that for this model, the utilized image augmentation methods provided sufficient robustness to overfitting, precluding the need for dropout or any other explicit overfitting reduction methods (like L1/L2 regularization).

The following illustrates progress of training & validation loss while training my model (for 30 epochs). I used MSE for the loss metric, but plotted RMSE as it seems to enhance the visualization without distorting its meaning.

![Model training progress][image_training_progress]

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 298-323). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

I used mini-batch gradient descent (via Keras Model.fit_generator) to train the model.
* The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 314).
* Training for 30 epochs with batch size of 128 proved to be an effective combination.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of:
1. 1 lap of center-lane driving on Track #1
2. 1 lap of center-lane driving on Track #1, driving in opposite direction
3. 1 lap of center-lane driving on Track #2
4. 1 lap of center-lane driving on Track #2, driving in opposite direction
5. multiple recoveries from the left and right sides of the road (on Track #1)

For details about how I created the training data, see the next section. 

### Model Architecture & Training Strategy: In-Depth

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a proven model (or 2) and iterate toward a model tuned for performing well on this problem & data set.

My first step was to build a CNN model similar to [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf), a relatively simple and well-known CNN architecture. While I suspected LeNet would ultimately prove too simple for an effective steering angle model, it had the merits of being easy to implement and fast to train. 

I used my LeNet-based model as a test-bed for developing other project building blocks: data collection, image augmentation, model training & validation, and model testing in the simulator. I did eventually decide to implement a more capable steering model, but the LeNet-based network was very useful in proving out the functionality of these other components.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I used MSE for the training loss metric, and monitored model training for evidence of underfitting (training loss stays too high) and overfitting (validation loss levels off and/or increases while training loss continues to decrease).

My next step was to build a substantially more complex & capable CNN model based on the architecture described in the article [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf), published by NVIDIA in 2016. Since this architecture was designed for the steering angle regression problem, I thought it would be an excellent solution for this project.

While I was able to train the NVIDIA-based model and get it to successfully navigate Track #1, model training was slow and seemed prone to overfitting. I experimented with adding Dropout between various layers in the model to reduce overfitting. This was somewhat successful, albeit time-consuming.

In the end I decided to "split the difference" and implement a model architecture more complex than LeNet but less complex than the NVIDIA network. The details of my final architecture are found in the next section. With a minimal amount of trial-and-error experimentation, I arrived at an architecture that struck a nice balance between training time and performance on the track, with the unexpected bonus of neither underfitting nor overfitting the training data (as far as I could tell).

For the sake of being thorough, I did experiment with adding Dropout to this model. However, even a single Dropout layer with a low dropout rate (0.2) resulted in noticeable underfitting. I concluded that overfitting reduction wasn't necessary.

While testing the steering performance on Track #1, I did notice that at times the car got close to the left or right lane lines, and "stuck" there instead of moving back to the center. I decided to collect another batches of training data with several instances of starting the car on either side of the road, and steered back to the center. I noticed an immediate performance improvement--when the car veered to the edge of the lane, it would then steer smoothly back to the center.

At the end of the process, the vehicle is able to drive autonomously around both Track #1 and Track #2 at the default speed of 9 MPH without leaving the road. I also experimented with increasing the speed (by editing the drive.py script). The vehicle can successfully navigate Track #1 at speeds up to 30 MPH, and Track #2 at speeds up to 18 MPH. At 20 MPH on Track #2 it steers off the road on a sharp down-hill turn just short of the end of the lap. Room for improvement!

Note that in addition to the required project submission of the Track #1 video driving at 9 MPH (video.mp4), I also recorded videos of driving Track #1 at 30 MPH and Track #2 at 18 MPH (track1-30mph.mp4 & track2-18mph.mp4).

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

Here is a visualization of the architecture:

![Model architecture][image_model_architecture]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 2 laps of center-lane driving on Track #1 -- 1 lap in the default/forward direction, and another in the opposite direction (by turning the car around). Track #1 includes a lot more left turning than right turning in the forward direction, so recording a lap in the opposite direction balances the data set. Here are example images of center-lane driving on Track #1:

![Track #1][image_track1_center_lane_1]
![Track #1][image_track1_center_lane_2]
![Track #1][image_track1_center_lane_3]

To reduce overfitting the model to Track #1, I then recorded 2 more laps of center-lane driving, but on Track #2 instead. As before, I recorded 1 lap in the forward direction and 1 lap in the opposite direction.

![Track #2][image_track2_center_lane_1]
![Track #2][image_track2_center_lane_2]
![Track #2][image_track2_center_lane_3]

I then recorded the vehicle recovering from the left and right sides of the road back to center so that the vehicle would learn to steer back toward the center whenever it veers off to the left or right side of the lane. These images show what a recovery looks like starting from the left side of the road:

![Left side][image_recover_from_left_1]
![Correcting][image_recover_from_left_2]
![Back in center][image_recover_from_left_3]

Similarly, these images show what a recovery looks like starting from the right side of the road:

![Right side][image_recover_from_right_1]
![Correcting][image_recover_from_right_2]
![Back in center][image_recover_from_right_3]

##### Training Data Augmentation

The augmentation of training data is applied dynamically during training, resulting in a continuously varied stream of training samples. I found in practice this was an effective means of preventing the model from overfitting to the training data.

1. Each training sample includes images captured with center, left and right cameras. To increase variation in the training set, I randomly selected one of these images for each training sample:
    * Center-camera image for 50% of samples
    * Left-camera image for 25% of samples
    * Right-camera image for 25% of samples
    * When using left/right image, the steering angle was adjusted by a (hyper-parameter) offset. The steering angle data ranges from -1 to 1. With some experiementation I settled on a steering adjustment of 0.15.

![Left-camera][image_left_camera]
![Center-camera][image_center_camera]
![Right-camera][image_right_camera]

2. To completely remove left/right steering bias, I randomly flipped the training images (and corresponding steering angles) on 50% of training samples. For example, here is an image that has then been flipped:

![Original image][image_track1_normal]
![Flipped image][image_track1_flipped]

3. While collecting training images, I generated a histogram of the trained steering angles. Not surprisingly, the number of zero-angle samples are clearly over-represented in the data. To balance the distribution of training data, I made the training data generator randomly ignore 50% of zero-angle samples.

![Trained steering angles][image_steering_angle_histogram]

##### Model Training & Validation

Overall I collected 10540 training data samples. The training data is randomly shuffled, and 20% is separated into a validation data set. 

Images are pre-processed in the model as follows:
1. Top 60 and bottom 20 pixels are cropped out
2. Image is converted to floating-point
3. Image is normalized to range [-1.0, 1.0]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
