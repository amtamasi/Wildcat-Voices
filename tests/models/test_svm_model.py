"""
    This file contains the tests for the SVM model
"""

import unittest
from sklearn import svm
from src.models.svm import SVM

class SVMModelTester(unittest.TestCase):
    
    def setUp(self):
        self.test_model = SVM()

    def test_train(self):
        # Check that self.get_data was called
        # Check that self.model.fit was called with given parameters
        pass

    def test_predict(self):
        # Check that reading the file was called
        # Check that self.model.predict was called with given parameters
        pass

    def test_save_model(self):
        kernel = 'sigmoid'
        max_iter = 100
        reg_param = 2.0
        svm_model = svm.SVC(C=reg_param, kernel=kernel, max_iter=max_iter)
        self.test_model.kernel_type = kernel
        self.test_model.max_iter = max_iter
        self.test_model.reg_param = reg_param
        self.test_model.save_model()
        self.assertEqual(self.test_model.model.get_params(), svm_model.get_params())

    def test_set_kernel(self):
        kernel = 'sigmoid'
        self.test_model.set_kernel(kernel)
        self.assertEqual(self.test_model.kernel_type, kernel)

    def test_set_max_iterations(self):
        max_iter = 100
        self.test_model.set_max_iterations(max_iter)
        self.assertEqual(self.test_model.max_iter, max_iter)

    def test_set_reg_param(self):
        reg_param = 2.0
        self.test_model.set_reg_param(reg_param)
        self.assertEqual(self.test_model.reg_param, reg_param)

if __name__ == "__main":
    unittest.main()
