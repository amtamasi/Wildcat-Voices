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

    @mock.patch('src.models.base_model.np')
    @mock.patch('src.models.base_model.wavfile')
    @mock.patch('src.models.base_model.train_test_split')
    def test_get_data(self, mock_split, mock_wav, mock_np):
        mock_np.genfromtxt.return_value = np.array(["english1, male, pittsburgh, pennsylvania, usa,",
                                                    "english2, male, lexington, kentucky, usa,",
                                                    "english3, male, lexington, kentucky, usa,"])
        mock_wav.read.return_value = (12, np.array([0,0,0,1,1,1,0]))
        mock_split.return_value = ([[0,0,0,1,1,1,0],[0,0,0,1,1,1,0]], ['kentucky','kentucky'], [[0,0,0,1,1,1,0]], ['kentucky'])
        self.test_model.get_data()
        mock_np.genfromtxt.assert_called_with('data/gmu-audio.txt',skip_header=1,  delimiter='\n', dtype=None, encoding=None)
        mock_split.assert_called_with([[0,0,0,1,1,1,0],[0,0,0,1,1,1,0],[0,0,0,1,1,1,0]], ['pennsylvania', 'kentucky', 'kentucky'], test_size=0.2)

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

