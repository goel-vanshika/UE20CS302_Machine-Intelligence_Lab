'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd
import random


'''Calculate the entropy of the entire dataset'''
# input:pandas_dataframe
# output:int/float
def get_entropy_of_dataset(df):
	column_end=list(df.columns)[-1]
	l=[]
	for i in df[column_end].unique():
		l.append(i)
	sum=0
	entropy=0
	for i in list(df[column_end]):
		if i in l:
			sum=sum+1
	for i in l:
		dataFrame=df[df.iloc[:,-1:]==i].count()[-1]
		if(dataFrame!=0):
			entropy = entropy-dataFrame/sum*np.log2(dataFrame/sum)
	return entropy
				
'''Return average_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float
def get_avg_info_of_attribute(df, attribute):
	average_info =0
	for n in set(df[attribute]):
		dataFrame_n = df[df[attribute]==n]
		dataFrame = dataFrame_n.shape[0]/df.shape[0]
		E = get_entropy_of_dataset(dataFrame_n)
		average_info += (dataFrame*E)    	
	return average_info


'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float
def get_information_gain(df, attribute):
	gain = get_entropy_of_dataset(df)-get_avg_info_of_attribute(df,attribute)
	return gain




#input: pandas_dataframe
#output: ({dict},'str')
def get_selected_attribute(df):
	'''
    Return a tuple with the first element as a dictionary which has IG of all columns 
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''
    # TODO
	d=dict()
	coln=list(df.columns)
	coln.pop()
	maxi=-1
	selected_col=''
	for i in coln:
		d[i]=get_information_gain(df,i)
		if d[i]>maxi:
			maxi=d[i]
			selected_col=i
	return (d,selected_col)
	
	pass
	
	
	
	
	
	
