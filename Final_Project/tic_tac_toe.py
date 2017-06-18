import csv
import numpy as np

''' readFile() '''
'''
This function reads from a file annd converts each line into a list and returns
the list as numpy array
'''
def readFile( fileName ):
	with open( fileName, 'r' ) as f:
		reader = csv.reader( f )
		my_list = list( reader )
	my_list = np.asarray( my_list )
	return my_list

''' getLabel '''
'''
This funciton extracts the labels from the original list and modifies the
features and returns the updated features as well as the labels for each
feature vector
'''
def getLabel( orig_list ):
	list_shape = orig_list.shape
	Y = np.zeros( list_shape[0] )
	for row in range( list_shape[ 0 ] ):
		for col in orig_list[ row, : ]:
			if col == 'negative': 
				Y[ row ] = 0
			elif col == 'positive':
				Y[ row ] = 1

	X = np.delete( orig_list, -1, 1 )
	Y = Y.astype( int )
	return X, Y	

''' convertFeatures() '''
'''
This function converts all the categorical features into numerical features
'''
def convertFeatures( X ):
	newX = X
	XShape = X.shape

	for row in range( XShape[0] ):
		for col in range( XShape[1] ):
			if newX[ row, col ] == 'x':
				newX[ row, col ] = 1
			elif newX[ row, col ] == 'o':
				newX[ row, col ] = 2
			elif newX[ row, col ] == 'b':
				newX[ row, col ] = 3
			else:
				print "Error converting labels"
				break

	return newX

my_list = readFile( 'tic-tac-toe.csv' )
X, Y = getLabel( my_list )

X = convertFeatures( X )
for i in X:
	print i
