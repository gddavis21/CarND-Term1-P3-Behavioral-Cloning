import os
import csv
import pickle
import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def load_steering_angles(log_path):
    '''
    '''
    samples = []
    with open(log_path) as csv_file:
        reader = csv.reader(csv_file)
        samples = [line for line in reader]
    return [float(s[3]) for s in samples]
    
angles = []
angles.extend(load_steering_angles('../sim-data/Track1-F/driving_log.csv'))
angles.extend(load_steering_angles('../sim-data/Track1-R/driving_log.csv'))
angles.extend(load_steering_angles('../sim-data/Track2-F/driving_log.csv'))
angles.extend(load_steering_angles('../sim-data/Track2-R/driving_log.csv'))
angles.extend(load_steering_angles('../sim-data/Recovery/driving_log.csv'))

print(angles.count(0.0))
print(len(angles))

n, bins, patches = plt.hist(angles, 51, normed=1, histtype='stepfilled')
plt.show()