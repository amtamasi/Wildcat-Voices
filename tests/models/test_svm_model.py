"""
    This file contains the tests for the SVM model
"""

import unittest
import mock
import numpy as np

from sklearn import svm
from src.models.svm import SVM

class SVMModelTester(unittest.TestCase):
    
    def setUp(self):
        self.test_model = SVM()

    @mock.patch('src.models.svm.svm.SVC.fit')
    @mock.patch('src.models.svm.svm.SVC.score')
    def test_train(self, mock_score, mock_fit):
        # Store the expected return value
        expected_output = 0.8
        # Set up the return values for the mocked out functions
        mock_fit.return_value = None
        mock_score.return_value = 0.8
        # Set up the training and testing data
        self.test_model.x_train = [[0,0,0,1,1,1,0],[0,0,0,1,1,1,0]]
        self.test_model.y_train = ['ohio','kentucky']
        self.test_model.x_test = [[0,0,0,1,1,1,0]]
        self.test_model.y_test = ['kentucky']
        # Call the train function
        actual_output = self.test_model.train()
        # Check that the actual output is the expected output
        self.assertEqual(actual_output, expected_output)
        # Check that the functions were called with the correct parameters
        mock_fit.assert_called_with([[0,0,0,1,1,1,0],[0,0,0,1,1,1,0]], ['ohio', 'kentucky'])
        mock_score.assert_called_with([[0,0,0,1,1,1,0]], ['kentucky'])

    @mock.patch('src.models.svm.svm.SVC.fit')
    @mock.patch('src.models.svm.svm.SVC.predict')
    def test_predict(self, mock_predict, mock_fit):
        # Store the expected value
        expected_output = 'kentucky'
        # Set up the return values for the mocked functions
        mock_fit.return_value = None
        mock_predict.return_value = ['kentucky']
        # Fit the model and call the predict function
        self.test_model.model.fit([[0,0,0,1,1,1,0],[0,0,0,1,1,1,0],[0,0,0,1,1,1,0]], ['kentucky','kentucky','kentucky]'])
        actual_output = self.test_model.predict(np.array([0.5, 0.4, 1.2, 2.3, 0.9, 9.3]))
        self.assertEqual(actual_output, expected_output)
        # Check that the predict function was called with the expected parameters
        #mock_predict.assert_called_with(np.array([0.5, 0.4, 1.2, 2.3, 0.9, 9.3]))

    def test_set_kernel(self):
        kernel = 'sigmoid'
        # Change the kernel type
        self.test_model.set_kernel(kernel)
        # Check that the kernel changed
        self.assertEqual(self.test_model.kernel_type, kernel)

    def test_set_max_iterations(self):
        max_iter = 100
        # Change the maximum number of iterations
        self.test_model.set_max_iterations(max_iter)
        # Checked that the maximum number of iterations changed
        self.assertEqual(self.test_model.max_iter, max_iter)

    def test_set_reg_param(self):
        reg_param = 2.0
        # Change the regularization parameter
        self.test_model.set_reg_param(reg_param)
        # Check that the regularization parameter changed
        self.assertEqual(self.test_model.reg_param, reg_param)

if __name__ == "__main":
    unittest.main()
