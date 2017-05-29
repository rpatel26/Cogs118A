'''
Author: Ravi Patel
Date: 5/27/2017
PID: A11850926
'''

''' Importing python packages '''
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

''' Loading Data '''
data = sio.loadmat( 'ionosphere.mat' )

''' Organizing Data '''
X = data[ 'X' ].reshape( [351,34] )
Y = data[ 'Y' ].reshape( [-1,1] )

print "Starting Assignment\n"

'''-------------------------------------- fixNan() -----------------------------------------'''
'''
Function Name: fixNan()
Function Prototype: def fixNan( X )
Description: this function replaces any NaN data values with zeros. If there are no NaN values
	then the original data set is returned
Parameters: 
	X - arg1 -- data to fix
Return Value: modified data with NaN values replaced wth zeros or original data set
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
Function Name: convertLabels()
Function Prototype: def convertLabels( Y, labels )
Description: this fuction convert all the categorical labels into numerical values of binary
	class. If the categorical label is labeled as 'labels' then its set to -1, if not
	then its set to +1.
Parameters:
	Y - arg1 -- dataset containing all the labels for the training and testing set
	labels - arg2 -- categorical label that needs to be taken into account
Return Value: dataset with the modified label values
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
Function Name: splitData()
Function Prototype: def splitData( X, Y, testSize )
Description: this function splits the input data into testing and training sets
Parameters:
	X - arg1 -- data containing all the features
	Y - arg2 -- data containing all the labels
	testSize - arg3 -- size of the testing data, in the range of (0,1) exclusive
Return Value: this function will return the following four datasets in this order
	Xtrain -- new training set containing all the features
	Xtest -- new testing set containing all the features
	Ytrain -- new training set containin all the labels of the corresponding training set
	Ytest -- new testing set containing all the labels of the corresponding testing set
'''
def splitData( X, Y, testSize ):
	Xtrain, Xtest, Ytrain, Ytest = train_test_split( X, Y, test_size = testSize, random_state = 42 )
	print "Splitting Data"
	print "Train Size = ", (1 - testSize)
	print "Test Size = ", testSize
	return Xtrain, Xtest, Ytrain, Ytest


'''------------------------------------- KNNTrain() ----------------------------------------'''
'''
Function Name: KNNTrain()
Function Prototype: def KNNTrain( Xtrain, Xtest, K )
Description: this function trains and fits the K Nearest Neighbor classifier
Parameters: 
	Xtrain - arg1 -- training set containing all the features
	Xtest - arg2 -- testing set containing all the features
	K -- arg3 -- number of nearest neighbor to classify
Return Value: Eucledian distance betweent the training set and testing set
'''
def KNNTrain( Xtrain, Xtest, K):
	trainShape = Xtrain.shape
	testShape = Xtest.shape
	distance = np.zeros( [trainShape[0], testShape[0]] )
	for i in range( trainShape[0] ):
		for j in range( testShape[0] ):
			distance[ i, j] = np.linalg.norm( Xtrain[ i, : ] - Xtest[ j , : ] )
	
	return distance

'''------------------------------------- KNNClassify() -----------------------------------------'''
'''
Function Name: KNNClassify()
Function Prototype: def KNNClassify( distance, Ytrain, K )
Description: this function classify the K nearest neighbor of the testing set
Parameters:
	distance - arg1 -- data set containing the distance measure returned by KNNTrain()
	Ytrain - arg2 -- data set containing labels for the training set
	K - arg3 -- number of nearest neighbor to classify, should be the as KNNTrain()
Return Value: classification matrix containing K nearest neighbor of each data in testing set
'''
def KNNClassify( distance, Ytrain, K ):
	distShape = distance.shape
	nearestNbr = np.zeros( [K, distShape[1]] )
	
	for i in range( distShape[1] ):
		ind  = (distance[:,i]).argsort()[:K]
		nearestNbr[:, i] = Ytrain[ind]
	
	classification = np.sign( np.sum( nearestNbr, axis = 0 ) )
	return classification

'''------------------------------------- KNNTest() ---------------------------------------- '''
'''
Function Name: KNNTest()
Function Prototype: def KNNTest( classification, Ytest )
Description: this function test the K nearest neighbor classifier and report the
	classification error
Parameter:
	classification - arg1 -- clasification matrix returned by KNNClassify()
	Ytest - agr2 -- data set containing labels for the testing set
Return Value: classification error of the K Nearest Neighbor classifier
'''
def KNNTest( classification, Ytest ):
	misClassified = 0
	result = classification + Ytest
	numOfTestData = float(result.size)
	for i in result:
		if i == 0:
			misClassified = misClassified + 1
	error = misClassified / numOfTestData
	return error

'''---------------------------------------- KNN() ----------------------------------------- '''
'''
Function Name: KNN()
Function Prototype: def KNN( Xtrain, Ytrain, Xtest, Ytest, K )
Description: this function runs K Nearest Neighbor classifier by calling the KNNTrain(),
	KNNClassify() and KNNTest() function one after the other
Parameter:
	Xtrain - arg1 -- data set containing all the features of the training set
	Ytrain - arg2 -- data set containing all the labels of the training set
	Xtest - arg3 -- data set containing all the features of the testing set
	Ytest - arg4 -- data set containing all the labels of the testing set
	K - arg4 -- number of nearest neighbor to take into consideration
Return Vale: classification error rate to of the K Nearest Neighbor classifier	
'''
def KNN( Xtrain, Ytrain, Xtest, Ytest, K ):
	distance = KNNTrain( Xtrain, Xtest, K )
	classification = KNNClassify( distance, Ytrain, K )
	error = KNNTest( classification, Ytest )
	return error

'''----------------------------  K_Fold_crossValidation() ----------------------------------'''
'''
Function Name: K_Fold_crossValidation()
Function Prototype: def K_Fold_crossValidation( Xtrain, Ytrain, Xtest, Ytest, num_folds = 5 )
Description: this function performs K Fold Cross-Validation on the K Nearest Neighbor
	classifier with K = [ 1, 3, 5, 7 ]
Parameter:
	Xtrain - arg1 -- data set containing all the features of the training set
	Ytrain - arg2 -- data set containing all the labels of the training set
	num_folds - opt arg5 -- number of folds to use
Return Value: this function will return the following in this order:
	validation_err -- minimum validation error of the classifier
	train_err -- training error on the non-validating set
	K -- optimal value of K that provides the minimum validation error
'''
def K_Fold_crossValidation( Xtrain, Ytrain, num_folds = 5 ):
	print( "Cross Valdating, %d folds" %(num_folds) )
	K_range = np.array( [1, 3, 5, 7] )
	err =  -1 * np.ones( [1, num_folds] )
	err_train = -1 * np.ones( [1, num_folds] )
	validation_err = 9999999.0
	train_err = 9999999.0
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
			
			err_train[0,i] = KNN( newX_train, newY_train, newX_train, newY_train, j )
			err[0,i] = KNN( newX_train, newY_train, newX_test, newY_test, j )
		
		avg = (np.sum( err ) / float( num_folds ) )
		if avg <= validation_err:
			validation_err = avg
			train_err = (np.sum(err_train) / float( num_folds ) )
			K = j
	
	return validation_err, train_err, K


'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''------------------------ Testing K Nearest Neighbor Classifier --------------------------'''
X = fixNan( X )
Y = convertLabels( Y, 'b' )

''' ------------------------------ KNN Classification --------------------------------------'''
print "Starting KNN Classification"

'''---------------------------- 80% Training, 20% Testing ----------------------------------'''
Xtrain, Xtest, Ytrain, Ytest = splitData( X, Y, 0.2 )

validation_err, train_err, K = K_Fold_crossValidation( Xtrain, Ytrain, num_folds = 5 )
test_err = KNN( Xtrain, Ytrain, Xtest, Ytest, 1 )
print "Validation error = ", validation_err
print "Training Error = ", train_err
print "Optimal K = ", K
print "Testing data..."
print "Test error = ", test_err
print "\n\n"

'''------------------------------  60% Training, 40% Testing -------------------------------'''
Xtrain, Xtest, Ytrain, Ytest = splitData( X, Y, 0.4 )

validation_err, train_err, K = K_Fold_crossValidation( Xtrain, Ytrain )
test_err = KNN( Xtrain, Ytrain, Xtest, Ytest, 1 )
print "Validation error = ", validation_err
print "Training Error = ", train_err
print "Optimal K = ", K
print "Testing data..."
print "Test error = ", test_err

