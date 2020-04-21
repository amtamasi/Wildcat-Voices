"""
    This file contains the tests for the base model class
"""

import mock
import unittest
import numpy as np

from src.models.base_model import BaseModel

class BaseModelTester(unittest.TestCase):

    def setUp(self):
        self.test_model = BaseModel()

    # Tests that the not implemented error will be raised
    def test_train(self):
        exception = NotImplementedError
        with self.assertRaises(exception):
            self.test_model.train()

    # Tests that the not implemented error will be raised
    def predict(self):
        exception = NotImplementedError
        with self.assertRaises(exception):
            self.test_model.predict()

    # Tests that the not implemented error will be raised
    def test_save_model(self):
        exception = NotImplementedError
        with self.assertRaises(exception):
            self.test_model.save_model()

    # Mocks out functions that call external libraries
    @mock.patch('src.models.base_model.np')
    @mock.patch('src.models.base_model.BaseModel.process_data')
    @mock.patch('src.models.base_model.train_test_split')
    def test_get_data(self, mock_split, mock_data, mock_np):
        # Sets return values for the mocked out functions
        mock_np.genfromtxt.return_value = np.array(["english1, male, pittsburgh, pennsylvania, usa,",
                                                    "english2, male, lexington, kentucky, usa,",
                                                    "english3, male, lexington, kentucky, usa,"])
        mock_data.return_value = np.array([0,0,0,1,1,1,0])
        mock_split.return_value = ([[0,0,0,1,1,1,0],[0,0,0,1,1,1,0]], ['kentucky','kentucky'], [[0,0,0,1,1,1,0]], ['kentucky'])
        # Calls the function that is being tested
        self.test_model.get_data()
        self.assertTrue(len(self.test_model.x_train) > 0)
        self.assertTrue(len(self.test_model.y_train) > 0)
        self.assertTrue(len(self.test_model.x_test) > 0)
        self.assertTrue(len(self.test_model.y_test) > 0)

    # Tests that the return_labels function will return None
    # when there are no labels
    def test_get_labels_empty(self):
        expected_return_labels = None
        labels = []
        self.test_model.labels = labels
        return_labels = self.test_model.get_labels()
        self.assertEqual(return_labels, expected_return_labels)

    # Tests that the return_labels function will return the 
    # labels stored
    def test_get_labels(self):
        expected_return_labels = ['label1', 'label2', 'label3']
        self.test_model.labels = expected_return_labels
        return_labels = self.test_model.get_labels()
        self.assertEqual(return_labels, expected_return_labels)

if __name__ == "__main__":
    unittest.main()

