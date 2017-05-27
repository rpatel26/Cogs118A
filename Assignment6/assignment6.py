'''
Author: Ravi Patel
Date: 5/27/2017
PID: A11850926
'''

''' Importing python packages '''
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
##from sklearn.svm import SVC
##from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

''' Loading Data '''
data1 = sio.loadmat( 'ionosphere.mat' )

''' Organizing Data '''
X1 = data1[ 'X' ].reshape( [351,34] )
Y1 = data1[ 'Y' ].reshape( [-1,1] )

print "Starting Assignment\n"

'''-------------------------------------- fixNan() -----------------------------------------'''
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

'''---------------------------------- convertLabels() --------------------------------------'''
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

'''------------------------------------- splitData() ---------------------------------------'''
'''
This function splits the input data into testing and training sets
'''
def splitData( X, Y, testSize ):
	Xtrain, Xtest, Ytrain, Ytest = train_test_split( X, Y, test_size = testSize, random_state = 42 )
	print "Splitting Data"
	print "Train Size = ", (1 - testSize)
	print "Test Size = ", testSize
	return Xtrain, Xtest, Ytrain, Ytest


'''------------------------------------- KNNTrain() ----------------------------------------'''
def KNNTrain( Xtrain, Xtest, K):
	print "K = ", K
	print "Training Data..."
	trainShape = Xtrain.shape
	testShape = Xtest.shape
	distance = np.zeros( [trainShape[0], testShape[0]] )
	for i in range( trainShape[0] ):
		for j in range( testShape[0] ):
			distance[ i, j] = np.linalg.norm( Xtrain[ i, : ] - Xtest[ j , : ] )
	
	return distance

'''------------------------------------- KNNClassify() -----------------------------------------'''
def KNNClassify( distance, Ytrain, K ):
	print "Classificating Test Data..."
	distShape = distance.shape
	nearestNbr = np.zeros( [K, distShape[1]] )
	
	for i in range( distShape[1] ):
		ind  = (distance[:,i]).argsort()[:K]
		nearestNbr[:, i] = Ytrain[ind]
	
	classification = np.sign( np.sum( nearestNbr, axis = 0 ) )
	return classification

'''------------------------------------- KNNTest() ---------------------------------------- '''
def KNNTest( classification, Ytest, K ):
	print "Testing Data..."
	result = classification + Ytest
	numOfTestData = float(result.size)
	misClassified = 0
	for i in result:
		if i == 0:
			misClassified = misClassified + 1
	error = misClassified / numOfTestData
	print error

''' crossValidation() '''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
X1 = fixNan( X1 )
Y1 = convertLabels( Y1, 'b' )

'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
''' ------------------------------ KNN Classification ------------------------------------- '''
print "Starting KNN Classification"
''' K = 1 '''
Xtrain, Xtest, Ytrain, Ytest = splitData( X1, Y1, 0.2 )

K = 9
distance = KNNTrain( Xtrain, Xtest, K )
classification = KNNClassify( distance, Ytrain, K )
KNNTest( classification, Ytest, K )
