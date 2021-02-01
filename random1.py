#First part is completed 
import random
import sys
interrupted=False
import csv
import time
from itertools import zip_longest
name=[]
frequency=[]
j=0
sys.setrecursionlimit(10**8)

def check(freq):
	if(freq<4):
		print("Delta waves:%s"%freq)

	if(freq>=4 and freq<=8):
		print("Theta waves:%s"%freq)

	if(freq>=9 and freq<=13):
		print("Alpha waves:%s"%freq)

	if(freq>=14 and freq<=30):
		print("Beta waves:%s"%freq)


	if(freq>30 and freq<=100):
		print("Gamma waves:%s"%freq)


	
def freq_checker(freq):
	if(freq<4):
		godelta(freq)

	if(freq>=4 and freq<=8):
		gotheta(freq)

	if(freq>=9 and freq<=13):
		goalpha(freq)

	if(freq>=14 and freq<=30):
		gobeta(freq)
	

	if(freq>30 and freq<=100):
		gogamma(freq)
	



def csv_complete(name,frequency):
	fields=['name','frequency']
	record=[name,frequency]
	export_data = zip_longest(*record, fillvalue = '')
	with open('Frequency.csv',mode='w',newline='') as file:
		csv_writer=csv.writer(file)
		csv_writer.writerow(fields)
		csv_writer.writerows(export_data)

def norm_csv_complete(name,frequency):
	fields=['name','frequency']
	record=[name,frequency]
	export_data = zip_longest(*record, fillvalue = '')
	with open('normfrequency.csv',mode='w',newline='') as file:
		csv_writer=csv.writer(file)
		csv_writer.writerow(fields)
		csv_writer.writerows(export_data)



def godelta(freq):
	freq1=[]
	freq1.append(freq)
	name=[]
	name.append("Delta")
	i=0
	while(i<=100):
		new_frequency=round(random.uniform(freq,4),2)
		freq1.append(new_frequency)
		name.append("Delta")
		time.sleep(300)
		print("Delta:%s"%new_frequency)
		i=i+1
	x=max(freq1)
	y=min(freq1)
	findnormalize(name,x,y,freq1)
	csv_complete(name,freq1)
	if(len(freq1)==101):
		sys.exit()



def gotheta(freq):
	freq1=[]
	freq1.append(freq)
	name=[]
	name.append('Theta')
	i=0
	while(i<=100):
		new_frequency=round(random.uniform(freq,9),2)
		freq1.append(new_frequency)
		name.append("Theta")
		print("Theta:%s"%new_frequency)
		i=i+1
	x=max(freq1)
	y=min(freq1)
	findnormalize(name,x,y,freq1)
	csv_complete(name,freq1)
	if(len(freq1)==101):
		sys.exit()

def goalpha(freq):
	freq1=[]
	freq1.append(freq)
	name=[]
	name.append('Alpha')

	i=0
	while(i<=9):
		new_frequency=round(random.uniform(freq,14),2)
		freq1.append(new_frequency)
		name.append('Alpha')
		print("Alpha:%s"%new_frequency)
		i=i+1
	x=max(freq1)
	y=min(freq1)
	findnormalize(name,x,y,freq1)
	csv_complete(name,freq1)
	if(len(freq1)==101):
		sys.exit()


def gobeta(freq):
	freq1=[]
	name=[]
	i=0
	freq1.append(freq)
	name.append('Beta')
	while(i<=100):
		new_frequency=round(random.uniform(freq,31),2)
		freq1.append(new_frequency)

		name.append('Beta')
		print("Beta:%s"%new_frequency)
		i=i+1
	x=max(freq1)
	y=min(freq1)
	findnormalize(name,x,y,freq1)
	csv_complete(name,freq1)
	if(len(freq1)==101):
		sys.exit()


def gogamma(freq):
	freq1=[]
	name=[]
	i=0
	freq1.append(freq)
	name.append('Gamma')
	
	while(i<=100):
		new_frequency=round(random.uniform(freq,100),2)
		freq1.append(new_frequency)
		name.append('Gamma')
		print("Gamma:%s" %new_frequency)
		i=i+1
	x=max(freq1)
	y=min(freq1)
	findnormalize(name,x,y,freq1)
	print(x)
	print(y)
	csv_complete(name,freq1)
	percent=((sum(freq1)/len(freq1))-min(freq1))*100/(max(freq1)-min(freq1))
	print(percent)

	
	
	if(len(freq1)==101):
		sys.exit()



def findnormalize(name,x,y,freq1):
	normalize=[]
	print(freq1)
	for i in range(0,len(freq1)):
		norm=freq1[i]
		find_norm=(x-norm)/(x-y)
		addnorm=find_norm+freq1[i]
		normalize.append(addnorm)
	print(normalize)
	norm_csv_complete(name,normalize)





	


freq=round(random.uniform(0.5,100),2)
freq_checker(freq)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# Importing the dataset
dataset = pd.read_csv('Frequency.csv')
y = dataset.iloc[:,-1].values
X=pd.get_dummies(dataset['name'])
print(X.head())

norm_dataset=pd.read_csv('normfrequency.csv')
norm_y = dataset.iloc[:,-1].values
norm_X=pd.get_dummies(dataset['name'])



"""
# Training the Random Forest Regression model on the whole dataset
# from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
# regressor.fit(X, y)
# y_grid = np.arange(min(y), max(y), 0.01)
# y_grid = y_grid.reshape((len(y_grid), 1))
# plt.scatter(X, y, color = 'red')
# plt.plot(y_grid, regressor.predict(y_grid), color = 'blue')
# plt.title('(Random Forest Regression)')
# plt.xlabel('Name of frequency')
# plt.ylabel('Frequency')
# plt.show()
"""


import seaborn as sns

x=list(range(0,len(y)))
print(x)
print(len(y))
fig,axes=plt.subplots(1,1,figsize=(12,6))
sns.lineplot(y=y,x=x)
axes.set_title(datetime.datetime.now())
axes.set_xlabel('Time(s)')
axes.set_ylabel('Frequency(Hertz)')
#axes.text(x[1],y[1],str(x[1])+', '+ str(y[1]))
#axes.annotate(str(x[1])+', '+ str(y[1]), xy=(x[1],y[1]),xytext=(x[1]+5,y[1]+max(y)))
plt.show()


norm_x=list(range(0,len(norm_y)))
fig,axes=plt.subplots(1,1,figsize=(12,6))
sns.lineplot(y=norm_y,x=norm_x)
axes.set_title(datetime.datetime.now())
axes.set_xlabel('Time(s)')
axes.set_ylabel('Frequency(Hertz)')
#axes.text(x[1],y[1],str(x[1])+', '+ str(y[1]))
#axes.annotate(str(x[1])+', '+ str(y[1]), xy=(x[1],y[1]),xytext=(x[1]+5,y[1]+max(y)))
plt.show()













