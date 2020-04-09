"""
    This file will contain the class for the Support
    Vector Machine machine learning model.
"""

import base_model
from sklearn import svm
# This can be used to read in the audio file
# possible that this gets used in the BaseModel class
from scipy.io import wavfile

class SVM(BaseModel):
    def __init__(self):
        self.max_iter = -1
        self.kernel_type = 'rbf'
        self.reg_param = 1.0
        self.model = svm.SVC()

    # Trains the model on the data
    # This function should be in the base model
    def train(self):
        pass

    # This function should probably be in the base model
    # and have it be the same for all models
    def predict(self, filename):
        # This function will get a prediction given
        # an audio file 
        # Get the file from the database

        # Feed in the preprocessed data from the file
        # to get a prediction

        # Return the prediction
        pass

    def set_kernel(self, kernel):
        self.kernel_type = kernel
        
    def set_max_iterations(self, interations):
        self.max_iter = interations

