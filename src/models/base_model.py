"""
    This file defines an interface for a base prediction model. 
    This is the parent class for all machine learning models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import wavfile

class BaseModel():

    def __init__(self):
        self.labels = []
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    # This function is not implemented here and is a virtual
    # function and every child class must implement this function
    def train(self):
        raise NotImplementedError()

    # This function is not implemented here and is a virtual
    # function and every child class must implement this function
    def predict(self):
        raise NotImplementedError()

    # This function is not implemented here and is a virtual
    # function and every child class must implement this function
    def save_model(self):
        raise NotImplementedError()

    # This function will get the data that is needed to train
    # the model
    def get_data(self, test_size=0.2):
        all_data_labels = []
        X = []
        # Get the metadata from the audio files
        data_labels = np.genfromtxt('data/gmu-audio.txt',skip_header=1,  delimiter='\n', dtype=None, encoding=None)
        for data in range(len(data_labels)):
            all_data_labels.append(data_labels[data].split(', '))
        # Filter out any files that are from outside the USA
        # Extracts the features from the audio file and stores them
        # Gets the state the soeaker is from for the label
        for label in range(len(all_data_labels)):
            data = all_data_labels[label]
            if data[len(data)-1] == 'usa,':
                file_path = 'data/gmu/' + data[0] + '.wav'
                samp_rate, audio_data = wavfile.read(file_path)
                X.append(audio_data.tolist())
                self.labels.append(data[len(data)-2])
        # Splits the data up for training and testing the machine learning models
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, self.labels, test_size=test_size)

    # This function will return the labels that are in the
    # training data
    def get_labels(self):
        # Checks that there are labels
        # If not tells the user that there is currently no data
        if len(self.labels) == 0:
            print("There are currently no labels. Read in the data to get labels for the data.")
            return None
        else:
            return self.labels
