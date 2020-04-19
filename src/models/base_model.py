"""
    This file defines an interface for a base prediction model. 
    This is the parent class for all machine learning models
"""


class BaseModel():

    def __init__(self):
        self.labels = []

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
    def get_data(self):
        pass

    # This function will return the labels that are in the
    # training data
    def get_labels(self):
        if len(self.labels) == 0:
            print("There are currently no labels. Read in the data to get labels for the data.")
            return None
        else:
            return self.labels
