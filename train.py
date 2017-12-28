import os
import csv
# import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
# from keras.utils import np_utils

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('epochs', 30, 'Training epochs')
flags.DEFINE_integer('batch_size', 128, 'Training batch size')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_float('steering_correction', 0.15, 'Left/right camera steering angle correction')
flags.DEFINE_float('keep_zeros', 0.5, 'Zero-angle sample retention rate')

def load_training_samples(log_path):
    '''
    Read CSV data from driving log file created by simulator, and
    return list of CSV lines.
    '''
    samples = []
    with open(log_path) as csv_file:
        reader = csv.reader(csv_file)
        return [line for line in reader]
    

def training_data_generator(
    samples, 
    batch_size=32, 
    keep_zeros=0.5, 
    steering_correction=0.15):
    '''
    training data generator for Keras Model.fit_generator
      - use user-define percentage of zero-angle samples (to balance training set)
      - use center camera image 50%, left camera 25%, right camera 25%
      - apply user-defined steering adjustment for left/right camera images
      - flip image vertically 50%
      - generates RGB images
    '''
    sample_count = len(samples)
    while 1:
        for offset in range(0, sample_count, batch_size):
            images = []
            angles = []
            while len(images) < min(batch_size, sample_count - offset):
                
                # randomly select sample from training set
                sample = random.choice(samples)

                # read image path & steering angle from sample data
                image_path = sample[0]
                steering_angle = float(sample[3])

                # skip zero-angle samples at user-defined rate
                if steering_angle == 0.0 and random.random() > keep_zeros:
                    continue
                    
                # randomly use left/right images 25% each
                # apply user-defined steering correction to left/right images
                if random.random() < 0.5:
                    if random.random() < 0.5:
                        image_path = sample[1]
                        steering_angle += steering_correction
                    else:
                        image_path = sample[2]
                        steering_angle -= steering_correction

                # images are stored BGR, model requires RGB
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # randomly flip images 50%
                if random.random() < 0.5:
                    img = cv2.flip(img, 1)
                    steering_angle = -steering_angle
                    
                images.append(img)
                angles.append(steering_angle)
            images_array = np.array(images)
            angles_array = np.array(angles)
            yield (images_array, angles_array)

def validation_data_generator(samples, batch_size=32):
    '''
    validation data generator for Keras Model.fit_generator
        - select samples in order
        - only use center camera images
        - generates RGB images
    '''
    sample_count = len(samples)
    while 1:
        for offset in range(0, sample_count, batch_size):
            end = offset + batch_size
            batch_samples = samples[offset:end]
            images = []
            angles = []
            for sample in batch_samples:
                img = cv2.imread(sample[0])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                steering_angle = float(sample[3])
                images.append(img)
                angles.append(steering_angle)
            images_array = np.array(images)
            angles_array = np.array(angles)
            yield (images_array, angles_array)

            
def create_initial_model(model):
    '''
    create Keras sequential model with image pre-processing layer
      - crop top & bottom margins to focus on relevant image region
      - normalize images to floating-point in range [-1.0, 1.0]
    '''

    # start with Keras sequential model
    model = Sequential()

    # crop top 60 & bottom 20 pixels (eliminate irrelevant data)
    model.add(Cropping2D(
        cropping=((60,20),(0,0)), 
        input_shape=(160, 320, 3)))

    # convert to floating-point & normalize to [-1.0, 1.0] range
    model.add(Lambda(lambda x: (x / 127.5) - 1.0))
    return model

    
def add_my_CNN_steering_layer(model):
    '''
    Add CNN steering regression layer to Keras model.
    This is the model used in final project submission.
    '''
    model.add(Convolution2D(
        24, 5, 5, subsample=(2,2), border_mode='valid', 
        init='he_normal'))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(
        48, 5, 5, subsample=(2,2), border_mode='valid', 
        init='he_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Convolution2D(
        64, 3, 3, subsample=(1,1), border_mode='valid', 
        init='he_normal'))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(
        64, 3, 3, subsample=(1,1), border_mode='valid', 
        init='he_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(128, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(64, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(1, init='he_normal'))  # regression


def add_LeNet_steering_layer(model):
    '''
    add LeNet CNN steering regression layer to Keras model
    (for comparison purposes, not used in final project submission)
    '''
    model.add(Convolution2D(
        6, 5, 5, subsample=(1,1), border_mode='valid', 
        init='he_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Convolution2D(
        16, 5, 5, subsample=(1,1), border_mode='valid', 
        init='he_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(120, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(84, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(1, init='he_normal'))

    
def add_Nvidia_CNN_steering_layer(model):
    '''
    add CNN steering regression layer (published by Nvidia) to Keras model
    (for comparison purposes, not used in final project submission)
    '''
    # input: 80x320, output: 24@38x158
    model.add(Convolution2D(
        24, 5,5, 
        subsample=(2,2), 
        border_mode='valid',
        init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # output: 36@19x80
    model.add(Convolution2D(
        36, 5,5, 
        subsample=(2,2), 
        border_mode='same',
        init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # output: 48@10x40
    model.add(Convolution2D(
        48, 5,5, 
        subsample=(2,2), 
        border_mode='same',
        init='he_normal'))
    model.add(Activation('relu'))

    # output: 64@8x38
    model.add(Convolution2D(
        64, 3,3, 
        subsample=(1,1), 
        border_mode='valid',
        init='he_normal'))
    model.add(Activation('relu'))

    # output: 64@6x36
    model.add(Convolution2D(
        64, 3,3, 
        subsample=(1,1), 
        border_mode='valid',
        init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(100, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(50, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, init='he_normal'))
    model.add(Activation('relu'))

    model.add(Dense(1, init='he_normal'))  # regression
    
    
def load_training_data():
    '''
    Load data from all training sets, return list of CSV sample lines
    '''
    samples = []
    samples.extend(load_training_samples('../sim-data/Track1-F/driving_log.csv'))
    samples.extend(load_training_samples('../sim-data/Track1-R/driving_log.csv'))
    samples.extend(load_training_samples('../sim-data/Track2-F/driving_log.csv'))
    samples.extend(load_training_samples('../sim-data/Track2-R/driving_log.csv'))
    samples.extend(load_training_samples('../sim-data/Recovery/driving_log.csv'))
    return samples


def plot_training_progress(fit_history):
    plt.plot([x**0.5 for x in fit_history.history['loss']][1:])
    plt.plot([x**0.5 for x in fit_history.history['val_loss']][1:])
    plt.title('Model MSE Loss')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper right')
    plt.show()    


def main(_):
    print('Number of epochs: %d' % FLAGS.epochs)
    print('Batch size: %d' % FLAGS.batch_size)
    print('Learning rate: %f' % FLAGS.learning_rate)
    print('Steering correction: %.2f' % FLAGS.steering_correction)
    
    # load training data, split into training/validation sets
    samples = load_training_data()
    random.shuffle(samples)
    train_samples, valid_samples = train_test_split(samples, test_size=0.2)

    print('Training samples: %d' % len(train_samples))
    print('Validation samples: %d' % len(valid_samples))

    train_gen = training_data_generator(
        train_samples, 
        FLAGS.batch_size,
        FLAGS.keep_zeros,
        FLAGS.steering_correction)
        
    valid_gen = validation_data_generator(valid_samples, FLAGS.batch_size)
    
    # setup model
    model = create_initial_model()
    add_my_CNN_steering_layer(model)
    model.compile(loss = 'mse', optimizer = 'adam', metrics=['mae'])

    # train & save model, plot training progress
    fit_history = model.fit_generator(
        train_gen,
        samples_per_epoch=len(train_samples),
        validation_data=valid_gen,
        nb_val_samples=len(valid_samples),
        nb_epoch=FLAGS.epochs,
        verbose=2)
        
    model.save('model.h5')
    plot_training_progress(fit_history)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
