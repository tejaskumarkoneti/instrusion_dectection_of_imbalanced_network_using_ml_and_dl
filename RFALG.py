
#--------------------------------------------------------------
# Include Libraries
#--------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np
import re
import csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score


def process():
	dataset=pd.read_csv('KDDTrain+.csv').values
	dataset1=pd.read_csv('KDDTest+.csv').values
	
	X_train = dataset[:,0:41]
	y_train = dataset[:,41] 
	
	X_test = dataset1[:,0:41]
	y_test = dataset1[:,41]

	model = RandomForestClassifier(max_depth=50, random_state=0,n_estimators=25)
	model.fit(X_train,y_train)
	y_pred = model.predict(X_test)

	mse=mean_squared_error(y_test, y_pred)
	mae=mean_absolute_error(y_test, y_pred)
	r2=r2_score(y_test, y_pred)
	
	
	print("---------------------------------------------------------")
	print("MSE VALUE FOR Random Forest IS %f "  % mse)
	print("MAE VALUE FOR Random Forest IS %f "  % mae)
	print("R-SQUARED VALUE FOR Random Forest IS %f "  % r2)
	rms = np.sqrt(mean_squared_error(y_test, y_pred))
	print("RMSE VALUE FOR Random Forest IS %f "  % rms)
	ac=accuracy_score(y_test,y_pred.round())
	print ("ACCURACY VALUE Random Forest IS %f" % ac)
	print("---------------------------------------------------------")
	

	result2=open("results/RFMetrics.csv", 'w')
	result2.write("Parameter,Value" + "\n")
	result2.write("MSE" + "," +str(mse) + "\n")
	result2.write("MAE" + "," +str(mae) + "\n")
	result2.write("R-SQUARED" + "," +str(r2) + "\n")
	result2.write("RMSE" + "," +str(rms) + "\n")
	result2.write("ACCURACY" + "," +str(ac*100) + "\n")
	result2.close()
	
	
	df =  pd.read_csv("results/RFMetrics.csv")
	acc = df["Value"]
	alc = df["Parameter"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
	
	fig = plt.figure()
	plt.bar(alc, acc,color=colors)
	plt.xlabel('Parameter')
	plt.ylabel('Value')
	plt.title("Random Forest Metrics Value")
	fig.savefig("results/RFMetricsValue.png") 
	plt.pause(5)
	plt.show(block=False)
	plt.close()

