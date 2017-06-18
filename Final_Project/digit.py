import scipy.io as sio
import numpy as np

data = sio.loadmat( 'imageTrain.mat' )

imageTrain = data[ 'imageTrain' ].reshape( [ 784, 5000 ] )
labelTrain = data[ 'labelTrain' ].reshape( [ -1, 1 ] )
imageTest = data[ 'imageTest' ].reshape( [ 784, 500 ] )
labelTest = data[ 'labelTest' ].reshape( [ -1, 1 ] )

''' convertLabels() '''
def convertLabels( Y, label ):
	Y_shape = Y.shape
	newLabels = np.zeros( Y_shape )
	
	for i in range( Y_shape[0] ):
		if Y[i] == label:
			newLabels[i] = 1

	#newLabels = np.ravel( newLabels )
	return newLabels
		

labelTest = convertLabels( labelTest, 1 )
labelTrain = convertLabels( labelTrain, 1 )

''' Starting Classification '''
