"""
    This file defines an interface for a neural network prediction model, used for a baseline.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf 
from tensorflow import keras

from .base_model import BaseModel

class NeuralNetModel(BaseModel):
    def __init__(self):
        self.model = None

        #Data
        self.num_classes = len(self.classes)
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        #Training parameters
        self.num_epochs = 10

    def build_model(self):
        #Define the model architecture
        model = keras.Sequential([
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(self.num_classes)
        ])

        #"Compile" the model so that it can train on data
        model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        return model

    def save_model(self, model_file_name=None, weights_file_name=None):
        #Serialize model architecture to JSON
        model_json = self.model.to_json()
        with open(model_file_name, 'w') as json_file:
            json_file.write(model_json)

        #Serialize model weights to HDF5 format
        self.model.save_weights(weights_file_name)
        

    def load_model(self, model_file_name=None, weights_file_name=None):
        #Load model architecture 
        json_file = open(model_file_name)
        loaded_json_model = json_file.read()
        json_file.close()

        #Load model weights
        loaded_model = keras.models.model_from_json(loaded_json_model)
        loaded_model.load_weights(weights_file_name)

        return loaded_model


    def train(self):
        history = self.model.fit(self.x_train, self.y_train, epochs=self.num_epochs, validation_data=(self.x_test, self.y_test))
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)


    def predict(self, single_sample_x):
        '''
        Parameter: single_sample_x - One preprocessed audio example to make a prediction on.
        '''
        prediction = None
        prediction = self.model.predict( np.array( [single_sample_x] ) )[0]

        return prediction


'''Sample code for how to use
if __name__ == "__main__":
    #Make a new model, then train and save it to a file 
    nn_model = NeuralNetModel()
    nn_model.build_model()
    nn_model.get_data("database_file")
    nn_model.train()
    nn_model.save_model("model_filename")

    #Later on, when you want to predict
    NN_model = NeuralNetModel()
    NN_model.load_model("model_filename")
    prediction = NN_model.predict(sample audio)
    print("Neural Net model's prediction:", prediction)
'''