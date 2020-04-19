"""
    This file contains the tests for the base model class
"""

import unittest

from src.models.base_model import BaseModel

class BaseModelTester(unittest.TestCase):

    def setUp(self):
        self.test_model = BaseModel()

    def test_train(self):
        exception = NotImplementedError
        with self.assertRaises(exception):
            self.test_model.train()

    def predict(self):
        exception = NotImplementedError
        with self.assertRaises(exception):
            self.test_model.predict()       

    def test_save_model(self):
        exception = NotImplementedError
        with self.assertRaises(exception):
            self.test_model.save_model()

    def test_get_data(self):
        self.test_model.get_data()

    def test_get_labels_empty(self):
        expected_return_labels = None
        labels = []
        self.test_model.labels = labels
        return_labels = self.test_model.get_labels()
        self.assertEqual(return_labels, expected_return_labels)

    def test_get_labels(self):
        expected_return_labels = ['label1', 'label2', 'label3']
        self.test_model.labels = expected_return_labels
        return_labels = self.test_model.get_labels()
        self.assertEqual(return_labels, expected_return_labels)

if __name__ == "__main__":
    unittest.main()

