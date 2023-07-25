#This weeks code focuses on understanding basic functions of pandas and numpy 
#This will help you complete other lab experiments


# Do not change the function definations or the parameters
import numpy as np
import pandas as pd

#input: tuple (x,y)    x,y:int 
def create_numpy_ones_array(shape):
        #return a numpy array with one at all index
	array = np.ones([int(shape[0]), int(shape[1])], dtype = int)
	return array

#input: tuple (x,y)    x,y:int 
def create_numpy_zeros_array(shape):
	#return a numpy array with zeros at all index
	array = np.zeros([int(shape[0]), int(shape[1])], dtype = int)
	return array

#input: int  
def create_identity_numpy_array(order):
	#return a identity numpy array of the defined order
	array=None
	array = np.identity(order, dtype = int)
	return array

#input: numpy array
def matrix_cofactor(array):
	#return cofactor matrix of the given array
        cofactor = np.linalg.inv(array).T * (np.linalg.det(array))
        return array

#Input: (numpy array, int ,numpy array, int , int , int , int , tuple,tuple)
#tuple (x,y)    x,y:int 
def f1(X1,coef1,X2,coef2,seed1,seed2,seed3,shape1,shape2):
    #note: shape is of the forst (x1,x2)
	#return W1 x (X1 ** coef1) + W2 x (X2 ** coef2) +b
	# where W1 is random matrix of shape shape1 with seed1
	# where W2 is random matrix of shape shape2 with seed2
	# where B is a random matrix of compatible shape with seed3
	# if dimension mismatch occur return -1
    np.random.seed(seed1)
    W1 = np.random.randn(*shape1)
    np.random.seed(seed2)
    W2 = np.random.randn(*shape2)
    powertuple1= np.power(X1,coef1)
    powertuple2= np.power(X2,coef2)
    multuple1 = np.matmul(W1,powertuple1)
    multuple2 = np.matmul(W2,powertuple2)
    res = np.add(multuple1,multuple2)
    shape3=res.shape
    np.random.seed(seed3)
    b=np.random.randn(*shape3)
    ans = np.add(res,b)
    if (shape1==shape2):
            return ans
    else:
            return -1

def fill_with_mode(filename, column):
    df=pd.read_csv(filename)
    df[column] = df[column].fillna(df[column].mode()[0])
    return df

def fill_with_group_average(df, group, column):
    df[column] = df.groupby(group)[column].transform(lambda x: x.fillna(x.mean()))    
    return df

def get_rows_greater_than_avg(df, column):
    mask = df[column] > df[column].mean()
    df=df[mask]
    return df

