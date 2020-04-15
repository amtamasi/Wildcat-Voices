"""
    This file defines an interface for a base prediction model. 
    This is the parent class for all machine learning models
"""


class BaseModel():

    def __init__(self):
        pass

    # This function is not implemented here and is a virtual
    # function and every child class must implement this function
    def train(self):
        raise NotImplementedError()

    # This function is not implemented here and is a virtual
    # function and every child class must implement this function
    @abc.abstractmethod
    def predict(self):
        raise NotImplementedError()

    # This function is not implemented here and is a virtual
    # function and every child class must implement this function
    @abc.abstractmethod
    def save_model(self):
        raise NotImplementedError()

    # This function will get the data that is needed to train
    # the model
    def get_data(self):
        pass

    # This function will process the audio file to minimize noise
    # and have some consistentcy with the audio files
    def preprocess_data(self):
        pass
