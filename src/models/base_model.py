"""
    This file defines an interface for a base prediction model. 
    This is the parent class for all machine learning models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
        # Find the best way to get X and Y from the data
        labels = []
        data_labels = np.genfromtxt('data/gmu-audio.txt',skip_header=1,  delimiter='\n', dtype=None, encoding=None)
        for label in range(len(data_labels)):
            labels.append(data_labels[label].split(', '))
        print(labels)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size)

    # This function will return the labels that are in the
    # training data
    def get_labels(self):
        if len(self.labels) == 0:
            print("There are currently no labels. Read in the data to get labels for the data.")
            return None
        else:
            return self.labels
