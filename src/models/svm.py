"""
    This file will contain the class for the Support
    Vector Machine machine learning model.
"""

from .base_model import BaseModel
from sklearn import svm

import pickle

class SVM(BaseModel):
    def __init__(self):
        super().__init__()
        self.max_iter = -1
        self.kernel_type = 'rbf'
        self.reg_param = 1.0
        self.model = svm.SVC(C=self.reg_param, kernel=self.kernel_type, max_iter=self.max_iter)

    # Trains the model on the data
    def train(self):
        self.model = svm.SVC(C=self.reg_param, kernel=self.kernel_type, max_iter=self.max_iter)
        # Fits the model to the training data
        self.model.fit(self.x_train, self.y_train)
        # Returns the score the model had on the testing data
        return self.model.score(self.x_test, self.y_test)

    # This function takes in a features array that was
    # already computed and returns the prediction
    def predict(self, file_data):
        # Return the prediction
        return self.model.predict([file_data])[0]

    # This function will recreate the SVM model with any new parameters
    # The model will need to be trained again after this function is called
    def save_model(self, filename='svm_model.pkl'):
        with open(filename, 'wb') as pickle_file:
            pickle.dump(self.model, pickle_file)

    def load_model(self, filename='svm_model.pkl'):
        with open(filename, 'rb') as pickle_file:
            self.model = pickle.load(pickle_file)

    # Sets the kernel type to the given kernel
    def set_kernel(self, kernel):
        self.kernel_type = kernel

    # Sets the maximum iterations to the given number
    def set_max_iterations(self, iterations):
        self.max_iter = iterations

    # Sets the regularization parameter to the given number
    def set_reg_param(self, reg_param):
        self.reg_param = reg_param
