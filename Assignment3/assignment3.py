''' Importing python packages '''
import scipy.io as sio
import matplotlib.pyplot as plt 
import numpy as np 

''' Loading Data '''
data = sio.loadmat('data.mat')
x = data['x'].reshape([-1, 1])
y = data['y'].reshape([-1, 1])

X = np.hstack((np.ones((len(x),1)), np.power(x,1)))

class assignment3(object):
	'''
	Function Name: twoNormGradientDescent

	Function Declaration: def twoNormGradientDescent(self)

	Function Description: This function the parameters for linear regression 
		with the method of gradient descent using two norm loss function

	Function Parameters: none

	Return Vales: none
	'''
	def twoNormGradientDescent(self):

		W = np.zeros([2,1])
		stepSize = 0.00001

		dW = self.gradient2Norm(W,X,y)
		Wnew = stepSize*dW			

		# computing the optimal parameters
		while np.linalg.norm( (Wnew - W), ord  = 1 ) >= 0.0001:
			W = Wnew
			dW = self.gradient2Norm(W, X, y)
			Wnew = W - stepSize* dW

		print "Two Norm Gradient Descent: W ="
		print Wnew
		# Ploting the two norm curve to the data
		plt.plot(x,y)
		plt.hold(True)
		plt.plot(x, Wnew[0] + Wnew[1]*x)
		plt.title('Gradient Descent: Two Norm')
		plt.grid()
		plt.show()

	'''
	Function Name: gradient2Norm

	Function Declaration: def gradient2Norm(self, W, X, Y)

	Function Description: This computes the two norm of the loss function.

	Function Parameters:
		W: the paraeter matrix
		X: input variables, may include bias
		Y: output variables

	Return Values: gradient of the loss function
	'''
	def gradient2Norm(self, W, X, Y):
		return 2*np.matmul(np.matmul(np.transpose(X),X),W) - 2*(np.matmul(np.transpose(X),Y))


	'''
	Function Name: oneNormGradientDescent

	Function Declaration: def oneNormGradientDescent(self)

	Function Description: This function the parameters for linear regression 
		with the method of gradient descent using one norm loss function

	Function Parameters: none

	Return Vales: none
	'''
	def oneNormGradientDescent(self):
		W = np.zeros([2,1])
		# W = np.ones([2,1])
		stepSize = 0.0001

		dW = self.gradient1Norm(W, X, y)
		Wnew = stepSize * dW

		# Computing the optimal parameters
		while np.linalg.norm( (Wnew - W), ord = 1) >= 0.001:
			W = Wnew
			dW = self.gradient1Norm(W, X, y)
			Wnew = W - stepSize * dW
		
		print "One Norm Gradient Descent: W = "
		print Wnew
		# Ploting the one norm curve to data
		plt.plot(x,y)
		plt.hold(True)
		plt.plot(x, Wnew[0] + Wnew[1]*x)
		plt.title('Gradient Descent: One Norm')
		plt.grid()
		plt.show()

	'''
	Function Name: gradient1Norm

	Function Declaration: def gradient1Norm(self, W, X, Y)

	Function Description: This computes the one norm of the loss function.

	Function Parameters:
		W: the paraeter matrix
		X: input variables, may include bias
		Y: output variables

	Return Values: gradient of the loss function
	'''
	def gradient1Norm(self, W, X, Y):
		err = Y - np.matmul(X, W)
		return -1.0*np.matmul(np.transpose(X), np.sign(err))

	'''
	Function Name: normalDistribution

	Function Declaration: def normalDistribution()

	Function Description: This function generates two normally distributed random
		variables, each consisting of 1000 datapoints. It then perform arithmetic
		on those distribution and graphically compares the effect of arithmetic on
		random variables.

	Function Parameters: none

	Return Values: none
	'''
	def normalDistribution(self):
		n = 1
		X = np.random.randn(1000)
		Y = np.random.randn(1000)
		newX = (3*X) + Y
		newY = X - (2*Y)

		plt.figure(n)
		plt.hist(X)
		plt.title('X')
		n = n + 1

		plt.figure(n)
		plt.scatter(X,Y)
		plt.title('X vs Y')
		plt.xlabel('X')
		plt.ylabel('Y')
		n = n + 1

		plt.figure(n)
		plt.hist(newX)
		plt.title("X' = 3X + Y")
		n = n + 1

		plt.figure(n)
		plt.hist(newY)
		plt.title("Y' = X - 2Y")
		n = n + 1

		plt.figure(n)
		plt.scatter(newX, newY)
		plt.title("X' vs Y'")
		plt.xlabel("X'")
		plt.ylabel("Y'")
		n = n + 1

		plt.show()

problem3 = assignment3()
problem3.normalDistribution()

problem5 = assignment3()
problem5.twoNormGradientDescent()

problem6 = assignment3()
problem6.oneNormGradientDescent()