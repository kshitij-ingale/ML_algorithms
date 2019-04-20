from logistic import Logistic_regression
import unittest
import numpy as np

class test_class(unittest.TestCase):
	def test_check_parameters_shape(self):
		obj = Logistic_regression()
		X = np.array([[1,2,3],[4,5,6],[7,8,9],[17,18,19]])
		y = np.array([0,1,1,0])
		obj.train(X,y)
		self.assertEqual(obj.parameters.shape, (3,1))




if __name__ == '__main__':
	unittest.main()