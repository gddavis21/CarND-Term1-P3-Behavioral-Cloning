import os
import csv
import pickle
import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.utils import np_utils

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
# flags.DEFINE_string('training_file', '', "Training data directory file")
flags.DEFINE_integer('epochs', 20, 'Training epochs')
flags.DEFINE_integer('batch_size', 128, 'Training batch size')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_float('steering_correction', 0.15, 'Left/right camera steering angle correction')

def load_training_samples(log_path):
    '''
    '''
    samples = []
    with open(log_path) as csv_file:
        reader = csv.reader(csv_file)
        return [line for line in reader]
    
def training_generator(samples, batch_size=32, steering_correction=0.2):
    sample_count = len(samples)
    while 1:
        random.shuffle(samples)
        zero_count = 0
        for offset in range(0, sample_count, batch_size):
            images = []
            angles = []
            while len(images) < min(batch_size, sample_count - offset):
                sample = random.choice(samples)
                image_path = sample[0]
                steering_angle = float(sample[3])
                if steering_angle == 0.0:
                    zero_count += 1
                    if zero_count % 2 == 0:
                        continue
                k = random.randint(0,3)
                if k == 2:
                    # use left image 25%
                    image_path = sample[1]
                    steering_angle += steering_correction
                elif k == 3:
                    # use right image 25%
                    image_path = sample[2]
                    steering_angle -= steering_correction
                # if random.randint(0,1) == 1:
                    # image = np.fliplr(image)
                    # steering_angle = -steering_angle
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                angles.append(steering_angle)
            images_array = np.array(images)
            angles_array = np.array(angles)
            yield (images_array, angles_array)

def testing_generator(samples, batch_size=32):
    sample_count = len(samples)
    while 1:
        # random.shuffle(samples)
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
            
def add_preprocessing_layer(model):
    '''
    '''
    model.add(Cropping2D(
        cropping=((60,20),(0,0)), 
        # data_format='channels_last',
        # dim_ordering='tf',
        input_shape=(160, 320, 3)))

    model.add(Lambda(lambda x: (x / 127.5) - 1.0))
    # model.add(Lambda(lambda x: x - 127.5))
    
def add_LeNet_steering_layer(model):
    '''
    '''
    model.add(Convolution2D(6, 5, 5, subsample=(1,1), border_mode='valid', init='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(16, 5, 5, subsample=(1,1), border_mode='valid', init='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(120, init='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Dense(84, init='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, init='he_normal'))
    
def add_my_CNN_steering_layer(model):
    '''
    '''
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='valid', init='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='valid', init='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='valid', init='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='valid', init='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, init='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(64, init='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, init='he_normal'))  # regression

    
def add_Nvidia_CNN_steering_layer(model):
    '''
    '''
    # input: 80x320, output: 24@38x158
    model.add(Convolution2D(
        24, 5,5, 
        subsample=(2,2), 
        border_mode='valid',
        init='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # output: 36@19x80
    model.add(Convolution2D(
        36, 5,5, 
        subsample=(2,2), 
        border_mode='same',
        init='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # output: 48@10x40
    model.add(Convolution2D(
        48, 5,5, 
        subsample=(2,2), 
        border_mode='same',
        init='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))

    # output: 64@8x38
    model.add(Convolution2D(
        64, 3,3, 
        subsample=(1,1), 
        border_mode='valid',
        init='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))

    # output: 64@6x36
    model.add(Convolution2D(
        64, 3,3, 
        subsample=(1,1), 
        border_mode='valid',
        init='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(100, init='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(50, init='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, init='he_normal'))
    # model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(1, init='he_normal'))  # regression
    
def main(_):
    # load training data
    samples = []
    samples.extend(load_training_samples('../sim-data/Track1-F/driving_log.csv'))
    samples.extend(load_training_samples('../sim-data/Track1-R/driving_log.csv'))
    samples.extend(load_training_samples('../sim-data/Track2-F/driving_log.csv'))
    samples.extend(load_training_samples('../sim-data/Track2-R/driving_log.csv'))
    samples.extend(load_training_samples('../sim-data/Recovery/driving_log.csv'))
    
    random.shuffle(samples)
    train_samples, valid_samples = train_test_split(samples, test_size=0.2)
    
    print('Training samples: %d' % len(train_samples))
    print('Validation samples: %d' % len(valid_samples))
    
    train_gen = training_generator(train_samples, FLAGS.batch_size, FLAGS.steering_correction)
    valid_gen = testing_generator(valid_samples, FLAGS.batch_size)
    
    # setup model
    model = Sequential()
    add_preprocessing_layer(model)
    add_my_CNN_steering_layer(model)

    model.compile(loss = 'mse', optimizer = 'adam', metrics=['mae'])

    # train & save model
    fit_history = model.fit_generator(
        train_gen,
        samples_per_epoch=len(train_samples),
        validation_data=valid_gen,
        nb_val_samples=len(valid_samples),
        nb_epoch=FLAGS.epochs,
        verbose=2)
        
    model.save('model.h5')
    
    # visualize training results
    # plt.plot(fit_history.history['mean_absolute_error'])
    # plt.plot(fit_history.history['val_mean_absolute_error'])
    plt.plot([x**0.5 for x in fit_history.history['loss']])
    plt.plot([x**0.5 for x in fit_history.history['val_loss']])
    plt.title('Model MSE Loss')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper right')
    plt.show()    

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
