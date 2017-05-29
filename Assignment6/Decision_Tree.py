'''
Author: Ravi Patel
Date: 05/28/2017
PID: A11850926
'''

''' Importing python packages '''
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree

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

'''--------------------------------------- splitData() -----------------------------------------'''
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

'''--------------------------------- decision_tree_fit() ---------------------------------------'''
'''
Function Name: decision_tree_fit()
Function Prototype: def decision_tree_fit( Xtrain, Ytrain, depth = None )
Description: this function fit a decision tree model with a training set
Parameters:
	Xtrain - arg1 -- training dataset containing all the features
	Ytrain - arg2 -- training dataset containing all the labels 
	depth - opt agr3 -- depth of the decision tree (default = None )
Return Value: object containing the fitted model of the training set
'''
def decision_tree_fit( Xtrain, Ytrain, depth = None ):
	clf = tree.DecisionTreeClassifier( max_depth = depth )
	clf = clf.fit( Xtrain, Ytrain )
	return clf

'''-------------------------------- decision_tree_predict() ------------------------------------'''
'''
Function Name: decision_tree_predict()
Function Prototype: def decision_tree_predict( Xtest, classifier )
Description: this function predicts on a testing set using a decision tree classifier
Parameters:
	Xtest - arg1 -- testing dataset containing all the features
	classifier - arg2 -- classificating object that is returned from decision_tree_fit()
Return Value: object containing the predicted result of the classifier
'''
def decision_tree_predict( Xtest, classifier ):
	predict = classifier.predict( Xtest )
	return predict

'''-------------------------------- decision_tree_score() --------------------------------------'''
'''
Function Name: decision_tree_score( Ytest, predict )
Function Prototype: def decision_tree_score( Ytest, predict )
Description: this function calculates the accuracy of the testing set based on the Decision Tree
	classifier.
Parameters:
	Ytest - arg1 -- testing dataset containing all the labels
	predict - arg2 -- prediction object that is returned from decision_tree_predict()
Return Value: classification error of the Decision Tree Classifier
'''
def decision_tree_score( Ytest, predict ):
	misClassified = 0
	result = predict + Ytest
	numOfTestData = float( result.size )
	for i in result:
		if i == 0:
			misClassified = misClassified + 1
	error = misClassified / numOfTestData
	return error

'''------------------------------------ decision_tree() ----------------------------------------'''
'''
Function Name: decision_tree()
Function Prototype: def decision_tree( Xtrain, Ytrain, Xtest, Ytest, depth = None )
Description: this function runs decision_tree_fit(), decision_tree_predict() and 
	decision_tree_score() functions in that order
Parameters:
	Xtrain - arg1 -- training dataset containing all the features
	Ytrain - arg2 -- training dataset containing all the labels
	Xtest - agr3 -- testing dataset containing all the features
	Ytest - arg4 -- testing dataset containing all the labels
	depth - opt arg5 -- depth of the Decsion Tree (Default = None )
Return Value: classification error of the Decision Tree classifier
'''
def decision_tree( Xtrain, Ytrain, Xtest, Ytest, depth = None ):
	clf = decision_tree_fit( Xtrain, Ytrain, depth )
	predict = decision_tree_predict( Xtest, clf )
	error = decision_tree_score( Ytest, predict )
	return error

'''------------------------------- K_Fold_crossValidation() ------------------------------------'''
'''
Function Name: K_Fold_crossValidation()
Function Prototype: def K_Fold_crossValidation( Xtrain, Ytrain, num_folds = 5 )
Description: this function performs K Fold Cross-Validation on Decision Classifier to find the
	optimal depth of the classifier
Parameters:
	Xtrain - arg1 -- training set containing the features
	Ytrain - arg2 -- training set containing the labels
	num_folds - opt arg3 -- number of folds to make (Default = 5 )
Return Value: this function returns the follwing values in this order
	validation_err -- validation error corresponding to the optimal depth
	train_err -- training error corresponding to the optimal depth
	optimal_depth -- optimal depth for the Decision Tree classifier
'''
def K_Fold_crossValidation( Xtrain, Ytrain, num_folds = 5 ):
	print "Cross Validating, %s folds..." %(num_folds)
	err =  -1 * np.ones( [1, num_folds] )
	err_train = -1 * np.ones( [1, num_folds] )
	validation_err = 9999999.0
	train_err = 9999999.0
	optimal_depth = 0
	Xtrain_shape = Xtrain.shape
	
	for j in range( 1,  Xtrain_shape[0] ):
		for i in range( num_folds ):
			lower = i * Xtrain_shape[0] / num_folds
			upper = lower + ( Xtrain_shape[0] / num_folds)
			
			newX_test = Xtrain[ lower : upper, : ]
			newY_test = Ytrain[ lower : upper ]
			
			newX_train = np.delete( Xtrain, range( lower, (upper + 1) ), 0 )
			newY_train = np.delete( Ytrain, range( lower, (upper + 1) ), 0 )
			
			err_train[0,i] = decision_tree( newX_train, newY_train, newX_train, newY_train, j )
			err[0,i] = decision_tree( newX_train, newY_train, newX_test, newY_test, j )
		
		avg = (np.sum( err ) / float( num_folds ) )
		if avg < validation_err:
			validation_err = avg
			optimal_depth = j
			train_err = (np.sum(err_train)/ float( num_folds ) )

	return validation_err, train_err, optimal_depth

	
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''---------------------------- Testing Decision Tree Classifier -------------------------------'''
X = fixNan( X )
Y = convertLabels( Y, 'b' )

'''------------------------------ Decision Tree Classification ---------------------------------'''
print "Starting Decision Tree Classifier..."

'''------------------------------ 80% Training, 20% Testing ------------------------------------'''
Xtrain, Xtest, Ytrain, Ytest = splitData( X, Y, 0.2 )

validation_err, train_err, optimal_depth = K_Fold_crossValidation( Xtrain, Ytrain )
print "Validation error = ", validation_err
print "Training error = ", train_err
print "Optimal depth = ", optimal_depth
print "Testing on the optimal parameter..."
test_err = decision_tree( Xtrain, Ytrain, Xtest, Ytest, optimal_depth )
print "Test error = ", test_err
print "\n\n"


'''------------------------------ 60% Training, 40% Testing ------------------------------------'''
Xtrain, Xtest, Ytrain, Ytest = splitData( X, Y, 0.4 )

validation_err, train_err, optimal_depth = K_Fold_crossValidation( Xtrain, Ytrain )
print "Validation error = ", validation_err
print "Training error = ", train_err
print "Optimal depth = ", optimal_depth
print "Testing on the optimal parameter..."
test_err = decision_tree( Xtrain, Ytrain, Xtest, Ytest, optimal_depth )
print "Test error = ", test_err
