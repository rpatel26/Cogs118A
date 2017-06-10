import test2
import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

'''
This function reads from a file and convert each line into a list and returns
the list as numpy array
'''
def readFile( fileName ):
	with open( fileName, 'r') as f:
		reader = csv.reader(f)
		my_list = list(reader)
	my_list = np.asarray( my_list )
	return my_list

#my_data = np.genfromtxt( 'census.csv', delimiter=',', dtype="|S5", autostrip=True )

'''
This function exracts the labels from the original list and modifies the 
features and return the updated features as well as the labels for each
feature vector
'''
def getLabel( orig_list ):
	list_shape = orig_list.shape
	Y = np.zeros( list_shape[0] )
	for row in range( list_shape[ 0 ] ):
		for col in orig_list[ row, : ]:
			if col == ' <=50K': 
				Y[ row ] = 0
			elif col == ' >50K':
				Y[ row ] = 1

	X = np.delete( my_list, -1, 1 )
	return X, Y	

'''
This function converts all the caterogical features into numerical features
the are read for one-hot-encoding, it returns the nemurical feature matrix 
'''
def convertFeatures( X ):
	print "Converting Features"
	#newX = np.zeros( X.shape )
	newX = X
	XShape = X.shape
	count = np.zeros( [30] )
	
	for row in range( XShape[0] ):
	
		''' Fixing workclass '''
		temp = newX[ row, 1 ]
		if temp == ' ?':
			newX[ row, 1 ] = 0
		elif temp == ' Private':
			newX[ row, 1 ] = 1
		elif temp == ' Self-emp-not-inc':
			newX[ row, 1 ] = 2
		elif temp == ' Self-emp-inc':
			newX[ row, 1 ] = 3
		elif temp == ' Federal-gov':
			newX[ row, 1 ] = 4
		elif temp == ' Local-gov':
			newX[ row, 1 ] = 5
		elif temp == ' State-gov':
			newX[ row, 1 ] = 6
		elif temp == ' Without-pay':
			newX[ row, 1 ] = 7
		elif temp == ' Never-worked':
			newX[ row, 1 ] = 8
		else:
			print "Invalid workclass label: ", temp
			break

		''' Fixing Education '''
		temp = newX[ row, 3 ]
		if temp == ' ?':
			newX[ row, 3 ] = 0
		elif temp == ' Bachelors':
			newX[ row, 3 ] = 1
		elif temp == ' Some-college':
			newX[ row, 3 ] = 2
		elif temp == ' 11th':
			newX[ row, 3 ] = 3
		elif temp == ' HS-grad':
			newX[ row, 3 ] = 4
		elif temp == ' Prof-school':
			newX[ row, 3 ] = 5
		elif temp == ' Assoc-acdm':
			newX[ row, 3 ] = 6
		elif temp == ' Assoc-voc':
			newX[ row, 3 ] = 7
		elif temp == ' 9th':
			newX[ row, 3 ] = 8
		elif temp == ' 7th-8th':
			newX[ row, 3 ] = 9
		elif temp == ' 12th':
			newX[ row, 3 ] = 10
		elif temp == ' Masters':
			newX[ row, 3 ] = 11
		elif temp == ' 1st-4th':
			newX[ row, 3 ] = 12
		elif temp == ' 10th':
			newX[ row, 3 ] = 13
		elif temp == ' Doctorate':
			newX[ row, 3 ] = 14
		elif temp == ' 5th-6th':
			newX[ row, 3 ] = 15
		elif temp == ' Preschool':
			newX[ row, 3 ] = 16
		else:
			print "Invalid education label: ", temp
			break

		''' Fixing marital-status '''
		temp = newX[ row, 5 ]
		if temp == ' ?':
			newX[ row, 5 ] = 1
		elif temp == ' Married-civ-spouse':
			newX[ row, 5 ] = 2
		elif temp == ' Divorced':
			newX[ row, 5 ] = 3
		elif temp == ' Never-married':
			newX[ row, 5 ] = 4
		elif temp == ' Separated':
			newX[ row, 5 ] = 5
		elif temp == ' Widowed':
			newX[ row, 5 ] = 6
		elif temp == ' Married-spouse-absent':
			newX[ row, 5 ] = 7
		elif temp == ' Married-AF-spouse':
			newX[ row, 5 ] = 8
		else:
			print "Invalid marital-status label: ", temp
			break

		'''  Fixing occupation'''
		temp = newX[ row, 6 ]
		if temp == ' ?':
			newX[ row, 6 ] = 0
		elif temp == ' Tech-support':
			newX[ row, 6 ] = 1
		elif temp == ' Craft-repair':
			newX[ row, 6 ] = 2
		elif temp == ' Other-service':
			newX[ row, 6 ] = 3
		elif temp == ' Sales':
			newX[ row, 6 ] = 4
		elif temp == ' Exec-managerial':
			newX[ row, 6 ] = 5
		elif temp == ' Prof-specialty':
			newX[ row, 6 ] = 6
		elif temp == ' Handlers-cleaners':
			newX[ row, 6 ] = 7
		elif temp == ' Machine-op-inspct':
			newX[ row, 6 ] = 8
		elif temp == ' Adm-clerical':
			newX[ row, 6 ] = 9
		elif temp == ' Farming-fishing':
			newX[ row, 6 ] = 10
		elif temp == ' Transport-moving':
			newX[ row, 6 ] = 11
		elif temp == ' Priv-house-serv':
			newX[ row, 6 ] = 12
		elif temp == ' Protective-serv':
			newX[ row, 6 ] = 13
		elif temp == ' Armed-Forces':
			newX[ row, 6 ] = 14
		
		else:
			print "Invalid occupation label: ", temp
			break		
		
		''' Fixing relationship '''
		temp = newX[ row, 7 ]
		if temp == ' ?':
			newX[ row, 7 ] = 0
		elif temp == ' Wife':
			newX[ row, 7 ] = 1
		elif temp == ' Own-child':
			newX[ row, 7 ] = 2
		elif temp == ' Husband':
			newX[ row, 7 ] = 3
		elif temp == ' Not-in-family':
			newX[ row, 7 ] = 4
		elif temp == ' Other-relative':
			newX[ row, 7 ] = 5
		elif temp == ' Unmarried':
			newX[ row, 7 ] = 6
		else:
			print "Invalid relationship label: ", temp
			break
	
		''' Fixing race '''
		temp = newX[ row, 8 ]
		if temp == ' ?':
			newX[ row, 8 ] = 0
		elif temp == ' White':
			newX[ row, 8 ] = 1
		elif temp == ' Asian-Pac-Islander':
			newX[ row, 8 ] = 2
		elif temp == ' Amer-Indian-Eskimo':
			newX[ row, 8 ] = 3
		elif temp == ' Other':
			newX[ row, 8 ] = 4
		elif temp == ' Black':
			newX[ row, 8 ] = 5
		else:
			print "Invalid race label: ", temp
			break

		''' Fixing sex '''
		temp = newX[ row, 9 ]
		if temp == ' ?':
			newX[ row, 9 ] = 0
		elif temp == ' Female':
			newX[ row, 9 ] = 1
		elif temp == ' Male':
			newX[ row, 9 ] = 2
		else:
			print "Invalid sex label: ", temp
			break

		''' Fixing native-country '''
		temp = newX[ row, 13 ]
		if temp == ' ?':
			newX[ row, 13 ] = 0
		elif temp == ' United-States':
			newX[ row, 13 ] = 1
		elif temp == ' Cambodia':
			newX[ row, 13 ] = 2
		elif temp == ' England':
			newX[ row, 13 ] = 3
		elif temp == ' Puerto-Rico':
			newX[ row, 13 ] = 4
		elif temp == ' Canada':
			newX[ row, 13 ] = 5
		elif temp == ' Germany':
			newX[ row, 13 ] = 6
		elif temp == ' Outlying-US(Guam-USVI-etc)':
			newX[ row, 13 ] = 7
		elif temp == ' India':
			newX[ row, 13 ] = 8
		elif temp == ' Japan':
			newX[ row, 13 ] = 9
		elif temp == ' Greece':
			newX[ row, 13 ] = 10
		elif temp == ' South':
			newX[ row, 13 ] = 11
		elif temp == ' China':
			newX[ row, 13 ] = 12		
		elif temp == ' Cuba':
			newX[ row, 13 ] = 13
		elif temp == ' Iran':
			newX[ row, 13 ] = 14
		elif temp == ' Honduras':
			newX[ row, 13 ] = 15
		elif temp == ' Philippines':
			newX[ row, 13 ] = 16
		elif temp == ' Italy':
			newX[ row, 13 ] = 17
		elif temp == ' Poland':
			newX[ row, 13 ] = 18
		elif temp == ' Jamaica':
			newX[ row, 13 ] = 19
		elif temp == ' Vietnam':
			newX[ row, 13 ] = 20
		elif temp == ' Mexico':
			newX[ row, 13 ] = 21
		elif temp == ' Portugal':
			newX[ row, 13 ] = 22
		elif temp == ' Ireland':
			newX[ row, 13 ] = 23
		elif temp == ' France':
			newX[ row, 13 ] = 24
		elif temp == ' Dominican-Republic':
			newX[ row, 13 ] = 25
		elif temp == ' Laos':
			newX[ row, 13 ] = 26
		elif temp == ' Ecuador':
			newX[ row, 13 ] = 27
		elif temp == ' Taiwan':
			newX[ row, 13 ] = 28
		elif temp == ' Haiti':
			newX[ row, 13 ] = 29
		elif temp == ' Columbia':
			newX[ row, 13 ] = 30
		elif temp == ' Hungary':
			newX[ row, 13 ] = 31
		elif temp == ' Guatemala':
			newX[ row, 13 ] = 32
		elif temp == ' Nicaragua':
			newX[ row, 13 ] = 33
		elif temp == ' Scotland':
			newX[ row, 13 ] = 34
		elif temp == ' Thailand':
			newX[ row, 13 ] = 35
		elif temp == ' Yugoslavia':
			newX[ row, 13 ] = 36
		elif temp == ' El-Salvador':
			newX[ row, 13 ] = 37
		elif temp == ' Trinadad&Tobago':
			newX[ row, 13 ] = 38
		elif temp == ' Peru':
			newX[ row, 13 ] = 39
		elif temp == ' Hong':
			newX[ row, 13 ] = 40
		elif temp == ' Holand-Netherlands':
			newX[ row, 13 ] = 41
		else:
			print "Invalid native-country label: ", temp
			break
	return newX


my_list = readFile( 'census.csv' )

X, Y = getLabel( my_list )

'''
enc = OneHotEncoder()
enc.fit([[-1,0,3]])
print enc.n_values_
print enc.feature_indices_
'''

newX = convertFeatures( X )

enc = OneHotEncoder()
enc.fit( newX )
test2.testFunction()
