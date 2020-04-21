"""
    This file defines an interface for a random prediction model, used for a baseline.
"""
import random
import pickle

from .base_model import BaseModel

class RandomModel(BaseModel):
    def __init__(self):
        super().__init__()

    def train(self):
        return "No training needed for this model"

    def predict(self):
        return random.choice(self.labels)

    def save_model(self, filename='random_model.pkl'):
        with open(filename, 'wb') as pickle_file:
            pickle.dump(self.labels, pickle_file)

    def load_model(self, filename='random_model.pkl'):
        with open(filename, 'rb') as pickle_file:
            self.labels = pickle.load(pickle_file)
