"""
This module implements the logistic regression from scratch
"""
import numpy as np
from utils import *

class Logistic_regression:
	def __init__(self):
		self.parameters = None

	def train(self, X, y, num_iterations = 50, learning_rate = 0.1):
		self.parameters = np.zeros([X.shape[1],1])
		
		for iteration in range(num_iterations):
			self.parameters += learning_rate * (np.dot(X, (sigmoid(np.dot(X,self.parameters)) - y)))




if __name__ == '__main__':
	obj = Logistic_regression()
	read_data()