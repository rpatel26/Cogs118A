'''
Author: Ravi Patel
Date: 06/9/2017
'''

''' Importing python packages '''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''----------------------------------------------- splitData() -------------------------------------------------'''
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
	return Xtrain, Xtest, Ytrain, Ytest

'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''----------------------------------------- decision_tree_fit() -----------------------------------------------'''
'''
Function Name: decision_tree_train()
Function Prototype: def decision_tree_fit( Xtrain, Ytrain, depth = None )
Description: this function fit a decision tree model with a training set
Parameters:
	Xtrain - arg1 -- training dataset containing all the features
	Ytrain - arg2 -- training dataset containing all the labels 
	depth - opt agr3 -- depth of the decision tree (default = None )
	view - opt arg4 -- boolean value specifying whether to view the classifier of not (default = False )
	fileName - opt arg5 -- file name to be used for creating visualization
				must be ".dot" filename (default = "tree.dot" )
Return Value: object containing the fitted model of the training set
'''
def decision_tree_train( Xtrain, Ytrain, depth = None, view = False, fileName = "tree.dot" ):
	clf = tree.DecisionTreeClassifier( max_depth = depth )
	clf = clf.fit( Xtrain, Ytrain )
	if view == True:
		visualize_data( clf, fileName )
	return clf

'''---------------------------------------- decision_tree_predict() --------------------------------------------'''
'''
Function Name: decision_tree_classify()
Function Prototype: def decision_tree_predict( Xtest, classifier )
Description: this function predicts on a testing set using a decision tree classifier
Parameters:
	Xtest - arg1 -- testing dataset containing all the features
	classifier - arg2 -- classificating object that is returned from decision_tree_fit()
Return Value: object containing the predicted result of the classifier
'''
def decision_tree_classify( classifier, Xtest ):
	predict = classifier.predict( Xtest )
	return predict

'''---------------------------------------- decision_tree_score() ----------------------------------------------'''
'''
Function Name: decision_tree_score( Ytest, predict )
Function Prototype: def decision_tree_score( Ytest, predict )
Description: this function calculates the accuracy of the testing set based on the Decision Tree classifier.
Parameters:
	Ytest - arg1 -- testing dataset containing all the labels
	predict - arg2 -- prediction object that is returned from decision_tree_predict()
Return Value: classification error of the Decision Tree Classifier
'''
def decision_tree_score( predict, Ytest ):
	misClassified = 0
	result = predict + Ytest
	numOfTestData = float( result.size )
	for i in result:
		if i == 0:
			misClassified = misClassified + 1
	error = misClassified / numOfTestData
	return error

'''-------------------------------------------- decision_tree() ------------------------------------------------'''
'''
Function Name: decision_tree()
Function Prototype: def decision_tree( Xtrain, Ytrain, Xtest, Ytest, depth = None )
Description: this function runs decision_tree_fit(), decision_tree_predict() and decision_tree_score() functions
		in that order
Parameters:
	Xtrain - arg1 -- training dataset containing all the features
	Ytrain - arg2 -- training dataset containing all the labels
	Xtest - agr3 -- testing dataset containing all the features
	Ytest - arg4 -- testing dataset containing all the labels
	depth - opt arg5 -- depth of the Decsion Tree (Default = None )
	view - opt arg4 -- boolean value specifying whether to view the classifier of not (default = False )
	fileName - opt arg5 -- file name to be used for creating visualization
				must be ".dot" filename (default = "tree.dot" )
Return Value: classification error of the Decision Tree classifier
'''
def decision_tree( Xtrain, Ytrain, Xtest, Ytest, depth = None, view = False, fileName = "tree.dot" ):
	clf = decision_tree_train( Xtrain, Ytrain, depth, view, fileName )
	predict = decision_tree_classify( clf, Xtest )
	error = decision_tree_score( predict, Ytest )
	return error

'''--------------------------------------- K_Fold_crossValidation() --------------------------------------------'''
'''
Function Name: K_Fold_crossValidation()
Function Prototype: def K_Fold_crossValidation( Xtrain, Ytrain, num_folds = 5 )
Description: this function performs K Fold Cross-Validation on Decision Classifier to find th optimal depth of the
		classifier
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


'''------------------------------------------- visualize_data()  -----------------------------------------------'''
'''
Function Name: visualize_data()
Function Prototype: def visualize_data( clf, fileName = "tree.dot" )
Description: this function created a file containing a visualization of the Decision Tree classifier in the 
	directory containing the source file
Parameters:
	clf - arg1 -- classifier object with an already fitted training data
	fileName - opt arg2 -- filename to store the visualization
Return Value: none
'''
def visualize_data( clf, fileName = "tree.dot" ):
	print "Creating visualization..."
	tree.export_graphviz( clf, out_file = fileName, filled = True, class_names = True )
	print "Look for a file called %s in the directory containing the source file" %(fileName)

'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" '''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
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
			temp = Xtrain[ i, : ] - Xtest[ j, : ]
			temp = temp.reshape( [trainShape[1] ,1] )
			distance[ i, j] = np.linalg.norm( temp )
	
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
Function Name: KNNScore()
Function Prototype: def KNNTest( classification, Ytest )
Description: this function test the K nearest neighbor classifier and report the
	classification error
Parameter:
	classification - arg1 -- clasification matrix returned by KNNClassify()
	Ytest - agr2 -- data set containing labels for the testing set
Return Value: classification error of the K Nearest Neighbor classifier
'''
def KNNScore( classification, Ytest ):
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

'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
''' SVM_train() '''
'''
Kernels:linear
	polynomial degree 2, 3,...
	rbf, with width (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2)
C: factor of ten from 10^(-7) to 10^3
'''
def SVM_train( Xtrain, Ytrain, ker = 'linear', C_valule = 1 ):
	clf = SVC( kernel = ker, C = C_value )
	clf.fit( Xtrain, Ytrain )
	return clf

''' SVM_classify() '''
def SVM_classify( clf, Xtest ):
	classification = clf.predict( Xtest )
	return classification

''' SVM_score() '''
def SVM_score( clf, Xtest, Ytest ):
	accuracy = clf.score( Xtest, Ytest )
	return ( 1 - accuracy )

'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
''' logistic_regression_train() '''
'''
C: inverse of the regularization (10^(-8) to 10^4, change by factor of 10 )
'''
def logistic_regression_train( Xtrain, Ytrain, C_value = 1 ):
	clf = LogisticRegression( C = C_value )
	clf.fit( Xtrain, Ytrain )
	return clf

''' logistic_regression_classify() '''
def logistic_regression_classify( clf, Xtest ):
	classification = clf.predict( Xtest )
	return classification

''' logistic_regression_score() '''
def logistic_regression_score( clf, Xtest, Ytest ):
	accuracy = clf.score( Xtest, Ytest )
	return ( 1 - accuracy )

'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
''' random_forest_train() '''
def random_forest_train( Xtrain, Ytrain, num_of_trees = 10, depth = None ):
	clf = RandomForestClassifier( n_estimator = num_of_trees, max_depth = depth )
	clf.fit( Xtrain, Ytrain )
	return clf

''' random_forest_classify() '''
def random_forest_classify( clf, Xtest ):
	classification = clf.predict( Xtest )
	return classification

''' random_forest_score() '''
def random_forest_score( clf, Xtest, Ytest ):
	accuracy = clf.score( Xtest,Ytest )
	return ( 1 - accuracy )

'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
''' adaboost_train( ) '''
def adaboost_train( Xtrain, Ytrain, num_of_estimator = 50, learningRate = 1.0 ):
	clf = AdaBoostClassifier( n_estimator = num_of_estimator, learning_rate = learningRate )
	clf.fit( Xtrain, Ytrain )
	return clf

'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
'''"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'''
''' general_classify() '''
def general_classify( classifier, Xtest ):
	classification = classifier.predict( Xtest )
	return classification

''' general_score() '''
def general_score( classifier, Xtest, Ytest ):
	accuracy = classifier.score( Xtest, Ytest )
	return ( 1 - accuracy )

''' testFunction() '''
def testFunction():
	print "\nTest Passed\n"
