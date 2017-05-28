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
data = sio.loadmat( 'ionosphere.mat' )

''' Organizing Data '''
X = data[ 'X' ].reshape( [351,34] )
Y = data[ 'Y' ].reshape( [-1,1] )

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
This function converts all the categorical labels into numerical values of binary class
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
	#print "Training Data..."
	#print "K = ", K
	trainShape = Xtrain.shape
	testShape = Xtest.shape
	distance = np.zeros( [trainShape[0], testShape[0]] )
	for i in range( trainShape[0] ):
		for j in range( testShape[0] ):
			distance[ i, j] = np.linalg.norm( Xtrain[ i, : ] - Xtest[ j , : ] )
	
	return distance

'''------------------------------------- KNNClassify() -----------------------------------------'''
def KNNClassify( distance, Ytrain, K ):
	#print "Classificating Test Data..."
	distShape = distance.shape
	nearestNbr = np.zeros( [K, distShape[1]] )
	
	for i in range( distShape[1] ):
		ind  = (distance[:,i]).argsort()[:K]
		nearestNbr[:, i] = Ytrain[ind]
	
	classification = np.sign( np.sum( nearestNbr, axis = 0 ) )
	return classification

'''------------------------------------- KNNTest() ---------------------------------------- '''
def KNNTest( classification, Ytest ):
	#print "Testing Data..."
	misClassified = 0
	result = classification + Ytest
	numOfTestData = float(result.size)
	for i in result:
		if i == 0:
			misClassified = misClassified + 1
	error = misClassified / numOfTestData
	return error

'''---------------------------------------- KNN() ----------------------------------------- '''
def KNN( Xtrain, Ytrain, Xtest, Ytest, K ):
	distance = KNNTrain( Xtrain, Xtest, K )
	classification = KNNClassify( distance, Ytrain, K )
	error = KNNTest( classification, Ytest )
	#print "Classification Error = ", error
	return error

'''----------------------------  K_Fold_crossValidation() ----------------------------------'''
def K_Fold_crossValidation( Xtrain, Ytrain, Xtest, Ytest, num_folds = 5 ):
	print( "Cross Valdating, %d folds" %(num_folds) )
	K_range = np.array( [1, 3, 5, 7] )
	err =  -1 * np.ones( [1, num_folds] )
	min_error = 9999999.0
	K = 0
	Xtrain_shape = Xtrain.shape
	for j in K_range:
		for i in range( num_folds ):
			lower = i * Xtrain_shape[0] / num_folds
			upper = lower + ( Xtrain_shape[0] / num_folds)
			newX_train = Xtrain[ lower : upper, : ]
			newY_train = Ytrain[ lower : upper ]
			newX_test = np.delete( Xtrain, range( lower, (upper + 1) ), 0 )
			newY_test = np.delete( Ytrain, range( lower, (upper + 1) ), 0 )
			err[0,i] = KNN( newX_train, newY_train, newX_test, newY_test, j )
		avg = (np.sum( err ) / float( num_folds ) )
		if avg <= min_error:
			min_error = avg
			K = j
	
	return min_error, err, K



'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
X = fixNan( X )
Y = convertLabels( Y, 'b' )

'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
''' ------------------------------ KNN Classification --------------------------------------'''
print "Starting KNN Classification"
Xtrain, Xtest, Ytrain, Ytest = splitData( X, Y, 0.2 )

validation_err, train_err, K = K_Fold_crossValidation( Xtrain, Ytrain, Xtest, Ytest )
test_err = KNN( Xtrain, Ytrain, Xtest, Ytest, 1 )
print "Validation error = ", validation_err
print "Training Error = ", train_err
print "Optimal K = ", K
print "Testing data..."
print "Test error = ", test_err
print "\n\n"

'''---------------------------------  60% Training, 40% Testing ----------------------------'''
Xtrain, Xtest, Ytrain, Ytest = splitData( X, Y, 0.4 )

validation_err, train_err, K = K_Fold_crossValidation( Xtrain, Ytrain, Xtest, Ytest )
test_err = KNN( Xtrain, Ytrain, Xtest, Ytest, 1 )
print "Validation error = ", validation_err
print "Training Error = ", train_err
print "Optimal K = ", K
print "Testing data..."
print "Test error = ", test_err

