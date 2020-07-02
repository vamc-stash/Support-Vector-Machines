import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import math


def accuracy_score(y_true, y_pred):

	"""	score = (y_true - y_pred) / len(y_true) """

	return round(float(sum(y_pred == y_true))/float(len(y_true)) * 100 ,2)

def pre_processing(df):

	""" partioning data into features and target """

	X = df.drop([df.columns[-1]], axis = 1)
	y = df[df.columns[-1]]

	return X, y

def train_test_split(x, y, test_size = 0.25, random_state = None):

	""" partioning the data into train and test sets """

	x_test = x.sample(frac = test_size, random_state = random_state)
	y_test = y[x_test.index]

	x_train = x.drop(x_test.index)
	y_train = y.drop(y_test.index)

	return x_train, x_test, y_train, y_test


class SVM:

	def __init__(self, kernel = "linear", learning_rate = 1e-4, regularization_strength = 1.0, max_iter = 2000):
 
		self.num_feats = int
		self.train_size = int
		self.weights = np.array 
		self.y_train = np.array 
		self.input_matrix = np.array

		self.kernel = kernel
		self.kernel_matrix = np.array
		self.support_vectors = np.array
		self.learning_rate = learning_rate 	#Learning rate for gradient descent
		self.regularization_strength = regularization_strength 	#Regularization parameter, to control bias-variance tradeoff
		self.max_iter = max_iter	#Maximum Number of iterations to run gradient descent
		self.cost_threshold = 0.1 * learning_rate  #stopping criterion for gradien descent

	def fit(self, X, y):

		""" Adjust weights to training data """

		self.train_size = X.shape[0]
		self.num_feats = X.shape[1]
		self.input_matrix = np.append(X, np.ones(self.train_size).reshape(-1, 1), axis = 1)   #Add Column with Ones for intercept term 
		self.y_train = np.where(y == 0, -1, 1)
		self.weights = np.zeros(self.num_feats + 1) #Extra +1 for the intercept


		#optimize weights
		prev_cost = float("inf")
		for i in range(self.max_iter):
			cost = self._update_weights()
			
			if i%100 ==0 or i == self.max_iter:
				print("Cost after {} iterations is: {}".format(i, cost))
			if abs(prev_cost -cost) < self.cost_threshold*prev_cost:
				print("Cost after {} iterations is: {}".format(i, cost))
				break
			prev_cost = cost

	def _update_weights(self):

		"""
			Cost Function:
				l(w) = sum(max(0, 1 - y(wX + b))) + (lambda/2)(||w||)^2
				First Term is Hinge Loss, Second is Regularization term

			Gradient:
			    delta_w = dl/dw = (1/n) * ( if y(wX+b) < 1: -yX + lambda*w else: lambda*w )

			Gradient Descent
				w = w - (learning_rate * delta_w)

		"""
		y_pred = (self.weights * self.input_matrix).sum(axis = 1) # y_pred = wX+b

		dist = (1 - (self.y_train * y_pred))
		dist[dist < 0] = 0
		hinge_loss = np.sum(dist)/self.train_size
		regularization_term = (1/2) * self.regularization_strength * (np.dot(self.weights, self.weights))
		cost = hinge_loss  + regularization_term

		delta_w =  self.regularization_strength * (self.weights)

		for y, yhat, X in zip(self.y_train, y_pred, self.input_matrix):

			if y*yhat < 1:
				delta_w -= y*X

		delta_w /= self.train_size
		self.weights = self.weights - (self.learning_rate * delta_w)

		return cost

	def predict(self, X):

		""" Make predictions on given X using trained model """

		size = X.shape[0]
		X = np.append(X, np.ones(size).reshape(-1, 1), axis = 1)

		y_pred = np.sign((self.weights * X).sum(axis = 1))

		y_pred[np.where(y_pred == -1)] = 0.0

		return y_pred 

if __name__ == '__main__':

	#Synthetic Dataset
	print("\nSynthetic Dataset...")
	data = pd.read_table("Data/classification_data.txt")

	"""
		Dataset:
			Two input features and one binary target 
	"""

	#Separate Features and Target
	X, y = pre_processing(data)

	#Split data into Training and Testing Sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

	svm_clf = SVM(learning_rate = 1e-2, regularization_strength = 1.0)

	svm_clf.fit(X_train, y_train)

	print("Coefficients: {}".format(svm_clf.weights[:-1]))
	print("Intercept: {}".format(svm_clf.weights[-1]))

	print("Train Accuracy: {}".format(accuracy_score(y_train, svm_clf.predict(X_train))))
	print("Test Accuracy: {}".format(accuracy_score(y_test, svm_clf.predict(X_test))))


	####################################################################################################################

	#Real World Dataset
	print("\nCancer Dataset...")
	df = pd.read_csv("Data/datasets_cancer.csv")

	#pre-processing
	df.reset_index(inplace = True)
	lookup_map = {'M': 1.0, 'B': 0.0}

	df['diagnosis'] = df['diagnosis'].map(lookup_map)
	X = df.drop([df.columns[1], df.columns[2], df.columns[-1]], axis = 1)
	y = df[df.columns[2]]


	#Split data into Training and Testing Sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

	svm_clf = SVM(learning_rate = 1e-5, regularization_strength = 10.5)

	svm_clf.fit(X_train, y_train)

	print("Coefficients: {}".format(svm_clf.weights[:-1]))
	print("Intercept: {}".format(svm_clf.weights[-1]))

	print("Train Accuracy: {}".format(accuracy_score(y_train, svm_clf.predict(X_train))))
	print("Test Accuracy: {}".format(accuracy_score(y_test, svm_clf.predict(X_test))))