
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

'''---------------------------------- decision_tree() ------------------------------------------'''
def decision_tree( Xtrain, Ytrain, Xtest, depth = None ):
	clf = tree.DecisionTreeClassifier( max_depth = depth )
	clf = clf.fit( Xtrain, Ytrain )
	predict = clf.predict( Xtest )
	return predict	

'''--------------------------------- err_decisionTree() ----------------------------------------'''
def err_decisionTree( predict, Ytest ):
	misClassified = 0
	result = predict + Ytest
	numOfTestData = float( result.size )
	for i in result:
		if i == 0:
			misClassified = misClassified + 1
	error = misClassified / numOfTestData
	return error

'''------------------------------- decision_tree_classification() ------------------------------'''
def decision_tree_classification( Xtrain, Ytrain, Xtest, Ytest, depth = None ):
	predict = decision_tree( Xtrain, Ytrain, Xtest, depth )
	error = err_decisionTree( predict, Ytest )
	return error

'''------------------------------- K_Fold_crossValidation() ------------------------------------'''
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
			
			err_train[0,i] = decision_tree_classification( newX_train, newY_train, newX_train, newY_train, j )
			err[0,i] = decision_tree_classification( newX_train, newY_train, newX_test, newY_test, j )
		
		avg = (np.sum( err ) / float( num_folds ) )
		if avg <= validation_err:
			validation_err = avg
			optimal_depth = j
			train_err = (np.sum(err_train)/ float( num_folds ) )

	return validation_err, train_err, optimal_depth

	
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
X = fixNan( X )
Y = convertLabels( Y, 'b' )

'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''------------------------------ Decision Tree Classification ---------------------------------'''
Xtrain, Xtest, Ytrain, Ytest = splitData( X, Y, 0.2 )
validation_err, train_err, optimal_depth = K_Fold_crossValidation( Xtrain, Ytrain )
print "Validation error = ", validation_err
print "Training error = ", train_err
print "Optimal depth = ", optimal_depth
print "Testing on the optimal parameter..."
test_err = decision_tree_classification( Xtrain, Ytrain, Xtest, Ytest, optimal_depth )
print "Test error = ", test_err
print "\n\n"


'''------------------------------- 60% training 40% testing ------------------------------------'''
Xtrain, Xtest, Ytrain, Ytest = splitData( X, Y, 0.4 )
validation_err, train_err, optimal_depth = K_Fold_crossValidation( Xtrain, Ytrain )
print "Validation error = ", validation_err
print "Training error = ", train_err
print "Optimal depth = ", optimal_depth
print "Testing on the optimal parameter..."
test_err = decision_tree_classification( Xtrain, Ytrain, Xtest, Ytest, optimal_depth )
print "Test error = ", test_err


