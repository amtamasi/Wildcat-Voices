"""
    This file will contain the class for the Support
    Vector Machine machine learning model.
"""

from .base_model import BaseModel
from sklearn import svm

from scipy.io import wavfile

class SVM(BaseModel):
    def __init__(self):
        super().__init__()
        self.max_iter = -1
        self.kernel_type = 'rbf'
        self.reg_param = 1.0
        self.model = svm.SVC(C=self.reg_param, kernel=self.kernel_type, max_iter=self.max_iter)

    # Trains the model on the data
    def train(self):
        # Gets the data from the audio files
        self.get_data()
        # Fits the model to the training data
        self.model.fit(self.x_train, self.y_train)
        # Returns the score the model had on the testing data
        return self.model.score(self.x_test, self.y_test)

    # This function should probably be in the base model
    # and have it be the same for all models
    def predict(self, filename):
        # This function will get a prediction given
        # an audio file 
        # Get the file from the database
        samp_rate, audio_data = wavfile.read(filename)
        # Return the prediction
        return self.model.predict([audio_data])

    # This function will recreate the SVM model with any new parameters
    # The model will need to be trained again after this function is called
    def save_model(self):
        self.model = svm.SVC(C=self.reg_param, kernel=self.kernel_type, max_iter=self.max_iter)

    # Sets the kernel type to the given kernel
    def set_kernel(self, kernel):
        self.kernel_type = kernel

    # Sets the maximum iterations to the given number
    def set_max_iterations(self, iterations):
        self.max_iter = iterations

    # Sets the regularization parameter to the given number
    def set_reg_param(self, reg_param):
        self.reg_param = reg_param
