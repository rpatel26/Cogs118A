'''
Author: Ravi Patel
Date: 05/11/2017
PID: A11850926
'''

''' Importing python packages '''
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

''' Loading Data '''
data1 = sio.loadmat( 'ionosphere.mat' )
data2 = sio.loadmat( 'fisheriris.mat' )
data3 = sio.loadmat( 'arrhythmia.mat' )

''' Organizing Data '''
X1 = data1[ 'X' ].reshape( [351,34] )
Y1 = data1[ 'Y' ].reshape( [-1,1] )
X2 = data2[ 'meas' ].reshape( [150,4] )
Y2 = data2[ 'species' ].reshape( [-1,1] )
X3 = data3[ 'X' ].reshape( [452,279] )
Y3 = data3[ 'Y' ].reshape( [-1,1] )

print "Starting Assignment\n"

'''
This function replaces any NaN value of input data with zeros
'''
def fixNan( X ):
	print "Cleaning Data"
	X_shape = X.shape
	newX = np.zeros( X_shape )

	for i in range( X_shape[0] ):
		for j in range( X_shape[1] ):
			if np.isnan( X[i,j] ):
				continue
			else:
				newX[i,j] = X[i,j]
	
	return newX

'''
This function converts all the categorical labels into numerical values
'''
def convertLabels( Y, label ):
	Y_shape = Y.shape
	newLabels = np.zeros( Y_shape )
	
	for i in range( Y_shape[0] ):
		if Y[i] == label:
			newLabels[i] = -1
		else:
			newLabels[i] = 1

	newLabels = np.ravel( newLabels )
	return newLabels

'''
This function splits the input data into testing and training sets
'''
def splitData( X, Y, testSize ):
	Xtrain, Xtest, Ytrain, Ytest = train_test_split( X, Y, test_size = testSize )
	print "Splitting Data"
	print "Train Size = ", (1 - testSize)
	print "Test Size = ", testSize
	return Xtrain, Xtest, Ytrain, Ytest

'''
This function performs the linear kernel classification
'''
def linearTrain( X, Y, testX, testY ):
	print "Begin Training: kernel = 'linear'"
	optimal_C = 1
	old_meanScore = 0
	new_meanScore = old_meanScore
	
	mat = np.linspace( 0.1, 10, num = 100 )
	for i in mat:
		clf = SVC( kernel = 'linear', C = i)
		scores = cross_val_score( clf, X, Y , cv = 5)
		old_meanScore = np.mean( scores )
		if old_meanScore >= new_meanScore:
			new_meanScore = old_meanScore
			optimal_C = i
			print i
	
	print "Optimal C = ", optimal_C
	clf = SVC( kernel = 'linear', C = optimal_C )
	clf.fit( X, Y )
	print "Classificatio Error = ", (1 - clf.score( testX, testY ) )
	print "End Training: kernel = 'linear'"


'''
This function performs rbf kernel classification
'''
def rbfTrain( X, Y, testX, testY ):
	print "Begin Training: kernel = 'rbf'"
	optimal_C = 1
	old_meanScore = 0
	new_meanScore = old_meanScore
	
	mat = np.linspace( 0.1, 10, num = 100 )
	for i in mat:
		clf = SVC( kernel = 'rbf', C = i)
		scores = cross_val_score( clf, X, Y , cv = 5)
		old_meanScore = np.mean( scores )
		if old_meanScore >= new_meanScore:
			new_meanScore = old_meanScore
			optimal_C = i
			print i

	print "Optimal C = ", optimal_C
	clf = SVC( kernel = 'rbf', C = optimal_C )
	clf.fit( X, Y )
	print "Classificatio Error = ", (1 - clf.score( testX, testY ) )
	print "End Training: kernel = 'rbf'"


''' Problem 3 '''
print "Begin Problem 3"
X3 = fixNan( X3 )
Y3 = convertLabels( Y3, 1 )

''' 20% Testing Set '''
X3train, X3test, Y3train, Y3test = splitData( X3, Y3, 0.2 )
linearTrain( X3train, Y3train, X3test, Y3test )
rbfTrain( X3train, Y3train, X3test, Y3test )
print "\n"

''' 40% Testing Set '''
X3train, X3test, Y3train, Y3test = splitData( X3, Y3, 0.4 )
linearTrain( X3train, Y3train, X3test, Y3test )
rbfTrain( X3train, Y3train, X3test, Y3test )
print "\n"

''' 60% Testing Set '''
X3train, X3test, Y3train, Y3test = splitData( X3, Y3, 0.6 )
linearTrain( X3train, Y3train, X3test, Y3test )
rbfTrain( X3train, Y3train, X3test, Y3test )
print "End Problem 3"
print "\n"

print "Done!"
