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
        self.max_iter = 0
        self.kernel_type = None
