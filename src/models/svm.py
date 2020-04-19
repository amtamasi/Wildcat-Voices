"""
    This file will contain the class for the Support
    Vector Machine machine learning model.
"""

from .base_model import BaseModel
from sklearn import svm
# This can be used to read in the audio file
# possible that this gets used in the BaseModel class
from scipy.io import wavfile

class SVM(BaseModel):
    def __init__(self):
        self.max_iter = -1
        self.kernel_type = 'rbf'
        self.reg_param = 1.0
        self.model = svm.SVC(C=self.reg_param, kernel=self.kernel_type, max_iter=self.max_iter)

    # Trains the model on the data
    # This function implements a virtual function in the base model
    # This function will get the
    def train(self):
        X_data, labels = self.get_data()
        self.model.fit(X_data, labels)

    # This function should probably be in the base model
    # and have it be the same for all models
    def predict(self, filename):
        # This function will get a prediction given
        # an audio file 
        # Get the file from the database

        # Return the prediction
        pass

    # This function will recreate the SVM model with any new parameters
    # The model will need to be trained again after this function is called
    def save_model(self):
        self.model = svm.SVC(C=self.reg_param, kernel=self.kernel_type, max_iter=self.max_iter)

    def set_kernel(self, kernel):
        self.kernel_type = kernel
        
    def set_max_iterations(self, iterations):
        self.max_iter = iterations

    def set_reg_param(self, reg_param):
        self.reg_param = reg_param
