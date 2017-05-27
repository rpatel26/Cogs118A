''' Importing python packages '''
import scipy.io as sio
import matplotlib.pyplot as plt 
import numpy as np

''' Loading Data '''
n = 1
data = sio.loadmat('data.mat')
data2 = sio.loadmat('train.mat')
data3 = sio.loadmat('modified_data.mat')
x = data['x'].reshape([-1, 1])
y = data['y'].reshape([-1, 1])
trainX1 = data2['x1'].reshape([-1, 1])
trainX2 = data2['x2'].reshape([-1, 1])
trainY = data2['y'].reshape([-1, 1])
modified_X = data3['x'].reshape([-1,1])
modified_Y = data3['y'].reshape([-1,1])

''' Organizing Data '''
X = np.hstack((np.ones((len(x),1)), np.power(x,1)))
X = np.hstack((X, np.power(x,2)))

trainX = np.hstack( (trainX1, trainX2) )
trainX = np.hstack( (np.ones((len(trainX1), 1)), trainX) )

modifiedX = np.hstack( ((np.ones((len(modified_X),1)), np.power(x,1))) )
modifiedX = np.hstack( ((modifiedX, np.power(x,2))) )

class assignment4(object):

	'''
	Function Name: leastSquareParabola()

	Description: This function computes and plots the least square barabola
		on the given dataset

	Parameters: none

	Return Value: none
	'''
	def leastSquareParabola(self):
		theta = np.zeros([3,1])
		temp1 = np.matmul(np.transpose(X), X)
		temp1 = np.linalg.inv(temp1)
		temp2 = np.matmul(np.transpose(X), y)
		theta = np.matmul(temp1, temp2)

		plt.plot(x,y)
		plt.title("Lease Square Parobola")
		plt.xlabel("x")
		plt.ylabel("y")
		plt.grid(True)
		plt.hold(True)
		plt.plot(x, theta[0] + (theta[1]*x) + (theta[2] * np.power(x,2)), linewidth=2.0)
		plt.show()


	'''
	Function Name: regression()

	Description: this function preforms regression on the modified data set via
		the combination of two-norm and one-norm loss function by utilizing the
		gradient descent algorithm

	Parameters:
		tradeoff -- weights of the one-norm and two-norm loss function

	Return Value: none
	'''
	def regression(self, tradeoff):
		W = np.zeros([3,1])
		stepSize = 0.0001
		newW = stepSize * self.gradientRegression(modifiedX, modified_Y, W, tradeoff)

		while np.linalg.norm( (newW - W), ord=1) >= 0.001:
			W = newW
			newW = W - stepSize * self.gradientRegression(modifiedX, modified_Y, W, tradeoff)

		print "tradepff = ", tradeoff
		print "Optimal parameter = "
		print newW
		plt.scatter(modified_X, modified_Y)
		plt.title('Regression: L1-norm and L2-norm \n tradeoff = %s'%(tradeoff))
		plt.xlabel("Modified X")
		plt.ylabel("Modified Y")
		plt.hold(True)
		plt.grid(True)
		plt.plot(modified_X, newW[0] + newW[1]*modified_X + newW[2]*np.power(modified_X,2))
		plt.show()

 
 	'''
 	Function Name: gradientRegression()

 	Description: this function compute the gradient of the loss function
 		comprised of both one-norm and two-norm

 	Parameters: 
 		X -- input variables, may include bias
 		Y -- output variables
 		W -- the prameter matrix
 		tradeoff -- weights of one-norm and two-norm

 	Retrun Vale: gradient of the loss function
 	'''
	def gradientRegression(self, X, Y, W, tradeoff):
		twoNorm = np.matmul(np.matmul(np.transpose(X),X),W) - (np.matmul(np.transpose(X),Y))
		twoNorm = tradeoff * twoNorm

		err = Y - np.matmul(X, W)
		oneNorm = -1 *  np.matmul(np.transpose(X), np.sign(err))
		tradeoff = 1 - tradeoff
		oneNorm = tradeoff * oneNorm

		grad = oneNorm + twoNorm
		return grad

	'''
	Function Name: logisticRegression()

	Description: this function approximates the decision boundary of the
		training date using logistic regression as the loss function --
		computed via gradient descent

	Parameters: none

	Return Value: none
	'''
	def logisticRegression(self):
		theta = np.zeros([3])
		stepSize = 0.001
		newTheta = stepSize * self.gradientLogisticRegression(trainX, trainY, theta)


		for i in range(30):
		# while np.linalg.norm((newTheta - theta), ord=1) >= 0.001:
			theta = newTheta
			update = stepSize * self.gradientLogisticRegression(trainX, trainY, theta)
			newTheta = theta - update



		print newTheta
		temp = np.ones([70])
		a = newTheta[0]
		b = newTheta[1]
		c = newTheta[2]
		# plt.figure(n)
		plt.scatter(trainX1, trainX2)
		plt.title("Logistic Regression")
		plt.xlabel("x1")
		plt.ylabel("x2")
		plt.hold(True)
		# plt.plot(trainX1 , -(b/a)*trainX1 - (c/a)*temp)
		# n = n + 1
		plt.show()

	'''
	Function Name: gradientLogisticRegression()

	Description: this function computs the gradient of the logistic loss
		function

	Parameters:
		X -- input variables, may include bias terms
		Y -- output variables
		theta -- parameter matrix

	Return Value: gradient of the logistic loss function
	'''
	def gradientLogisticRegression(self, X, Y, theta):
		h = self.sigmoid(X, theta)
		h = h.reshape([70,1])
		h = Y - h

		temp = np.hstack((h,h))
		h = np.hstack((temp,h))
		grad = np.multiply(h,X)
		grad = -1 * np.sum(grad, axis=0)

		return grad


	def sigmoid(self, X, theta):
		z = -1 * np.matmul( X, theta)
		e = np.exp(z)
		e = 1 + e
		return (1/e)

	'''
	Function Name: linearDiscriminateAnalysis()

	Description: this function performs linear discriminate analysis on the
		training data to project the data on the optimal axis

	Parameters: none

	Return Value: none
	'''
	def linearDiscriminateAnalysis(self):
		trainX = np.hstack( (trainX1, trainX2))
		class0 = []
		class1 = []
		''' Separating training data into respective classes '''
		for i in range(trainY.shape[0]):
			if trainY[i,0] == 1:
				class1.append(trainX[i,:])
			else:
				class0.append(trainX[i,:])

		class0 = np.array(class0)
		class1 = np.array(class1)

		''' Computing class mean and class covariance '''
		mean0 = np.mean(class0, axis=0)
		mean1 = np.mean(class1, axis=0)
		cov0 = np.cov(class0[:,0], class0[:,1])
		cov1 = np.cov(class1[:,0], class1[:,1])

		''' Calculating the optimal projecting direction '''
		S = np.linalg.inv((cov0 + cov1))
		W = np.matmul(S, (mean1 - mean0))
		W = W / np.linalg.norm(W)
		W = np.reshape(W, [1,2])

		''' Transformation on the optimal projection'''
		projection = np.matmul(W, np.transpose(trainX))
		projection = np.matmul(np.transpose(projection), W)

		plt.scatter(trainX1, trainX2)
		plt.title("Linear Discriminative Analysis")
		plt.grid(True)
		plt.hold(True)
		plt.scatter(projection[:,0], projection[:,1])
		plt.show()

''' Problem Solution '''

problem1 = assignment4()
problem1.leastSquareParabola()

problem2_1 = assignment4()
problem2_1.regression(0.9)

problem2_2 = assignment4()
problem2_2.regression(0.5)

problem2_3 = assignment4()
problem2_3.regression(0.1)

problem3 = assignment4()
problem3.logisticRegression()

problem4 = assignment4()
problem4.linearDiscriminateAnalysis()