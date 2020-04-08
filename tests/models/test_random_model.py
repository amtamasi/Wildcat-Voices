"""
    This file defines the tests for the random prediction model.
"""
import unittest

from src.models.random_model import RandomModel

class RandomModelTester(unittest.TestCase):
    def test_classes_not_empty(self):
        model = RandomModel()
        num_classes = len(model.classes)
        self.assertTrue(num_classes > 0)

    def test_train(self):
        model = RandomModel()
        output = model.train()
        expected_output = "No training needed for this model"
        self.assertEqual(output, expected_output)

    def test_predict(self):
        pass

        #self.assertIsInstance(, ) 


if __name__ == "__main__":
    unittest.main()