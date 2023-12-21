import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn import linear_model
#from sklearn import cross_validation
from scipy.stats import norm

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.svm import SVC
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

#from sklearn import cross_validation

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from random import seed
from random import randrange
from csv import reader
import csv
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

mse=[]
mae=[]
rsq=[]
rmse=[]
acy=[]

np.random.seed(0)




data = pd.read_csv('KDDTrain+.csv')
print(data.head())

names=list(data.columns)

correlations = data.corr()
# plot correlation matrix
fig = plt.figure()
fig.canvas.set_window_title('Correlation Matrix')
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
#fig.savefig('Correlation Matrix.png')
    
 
#scatterplot
#scatter_matrix(data)
#    
plt.show()

ncols=3
plt.clf()
f = plt.figure(1)
f.suptitle(" Data Histograms", fontsize=12)
vlist = list(data.columns)
nrows = len(vlist) // ncols
if len(vlist) % ncols > 0:
	nrows += 1
for i, var in enumerate(vlist):
	plt.subplot(nrows, ncols, i+1)
	plt.hist(data[var].values, bins=15)
	plt.title(var, fontsize=10)
	plt.tick_params(labelbottom='off', labelleft='off')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()


def process(X_train,X_test,y_train,y_test): 
	#X_train, X_test, y_train, y_test = train_test_split(x1, y1)    
	model3=LogisticRegression()
	model3.fit(X_train,y_train)
	y = model3.predict(X_test)
	print("MSE VALUE FOR LogisticRegression IS %f "  % mean_squared_error(y_test,y))
	print("MAE VALUE FOR LogisticRegression IS %f "  % mean_absolute_error(y_test,y))
	print("R-SQUARED VALUE FOR LogisticRegression IS %f "  % r2_score(y_test,y))
	rms = np.sqrt(mean_squared_error(y_test,y))
	print("RMSE VALUE FOR LogisticRegression IS %f "  % rms)
	ac=accuracy_score(y_test,y) * 100
	print ("ACCURACY VALUE LogisticRegression IS %f" % ac)
	print("------------------------------------------------------------------")	
	mse.append(mean_squared_error(y_test,y))
	mae.append(mean_absolute_error(y_test,y))
	rsq.append(r2_score(y_test,y))
	rmse.append(rms)
	acy.append(ac)
	result2=open("results/resultLogisticRegression.csv","w")
	result2.write("ID,Predicted Value" + "\n")
	for j in range(len(y)):
	    result2.write(str(j+1) + "," + str(y[j]) + "\n")
	result2.close()
    
    

	model4 = DecisionTreeClassifier()
	model4.fit(X_train, y_train)
	y = model4.predict(X_test)
	print("MSE VALUE FOR DecisionTree IS %f "  % mean_squared_error(y_test,y))
	print("MAE VALUE FOR DecisionTree IS %f "  % mean_absolute_error(y_test,y))
	print("R-SQUARED VALUE FOR DecisionTree IS %f "  % r2_score(y_test,y))
	rms = np.sqrt(mean_squared_error(y_test,y))
	print("RMSE VALUE FOR DecisionTree IS %f "  % rms)
	ac=accuracy_score(y_test,y) * 100
	print ("ACCURACY VALUE DecisionTree IS %f" % ac)
	print("------------------------------------------------------------------")	
	mse.append(mean_squared_error(y_test,y))
	mae.append(mean_absolute_error(y_test,y))
	rsq.append(r2_score(y_test,y))
	rmse.append(rms)
	acy.append(ac)
	result2=open("results/resultDecisionTree.csv","w")
	result2.write("ID,Predicted Value" + "\n")
	for j in range(len(y)):
	    result2.write(str(j+1) + "," + str(y[j]) + "\n")
	result2.close()
    

	model5 = RandomForestClassifier()
	model5.fit(X_train, y_train)
	y = model5.predict(X_test)
	

	print("------------------------------------------------------------------")	
	print("MSE VALUE FOR Random Forest IS %f "  % mean_squared_error(y_test,y))
	print("MAE VALUE FOR Random Forest IS %f "  % mean_absolute_error(y_test,y))
	print("R-SQUARED VALUE FOR Random Forest IS %f "  % r2_score(y_test,y))
	rms = np.sqrt(mean_squared_error(y_test,y))
	print("RMSE VALUE FOR Random Forest IS %f "  % rms)
	ac=accuracy_score(y_test,y) * 100
	print ("ACCURACY VALUE Random Forest IS %f" % ac)
	print("------------------------------------------------------------------")	
	mse.append(mean_squared_error(y_test,y))
	mae.append(mean_absolute_error(y_test,y))
	rsq.append(r2_score(y_test,y))
	rmse.append(rms)
	acy.append(ac)
	result2=open("results/resultRandomForest.csv","w")
	result2.write("ID,Predicted Value" + "\n")
	for j in range(len(y)):
	    result2.write(str(j+1) + "," + str(y[j]) + "\n")
	result2.close()

	model6 = KNeighborsClassifier(n_neighbors=5)
	model6.fit(X_train, y_train)
	y = model6.predict(X_test)
	

	print("------------------------------------------------------------------")	
	print("MSE VALUE FOR KNN IS %f "  % mean_squared_error(y_test,y))
	print("MAE VALUE FOR KNN IS %f "  % mean_absolute_error(y_test,y))
	print("R-SQUARED VALUE FOR KNN IS %f "  % r2_score(y_test,y))
	rms = np.sqrt(mean_squared_error(y_test,y))
	print("RMSE VALUE FOR KNN IS %f "  % rms)
	ac=accuracy_score(y_test,y) * 100
	print ("ACCURACY VALUE KNN IS %f" % ac)
	print("------------------------------------------------------------------")	
	mse.append(mean_squared_error(y_test,y))
	mae.append(mean_absolute_error(y_test,y))
	rsq.append(r2_score(y_test,y))
	rmse.append(rms)
	acy.append(ac)
	result2=open("results/resultKNNresult.csv","w")
	result2.write("ID,Predicted Value" + "\n")
	for j in range(len(y)):
	    result2.write(str(j+1) + "," + str(y[j]) + "\n")
	result2.close()
	
	

	al = ['Logistic Regression','DecisionTree', 'Random Forest', 'KNN']
	    
	result2=open('results/MSE.csv', 'w')
	result2.write("Algorithm,MSE" + "\n")
	for i in range(0,len(mse)):
	    result2.write(al[i] + "," +str(mse[i]) + "\n")
	result2.close()
	    
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
       
    
	#Barplot for the dependent variable
	fig = plt.figure(0)
	df =  pd.read_csv('results/MSE.csv')
	acc = df["MSE"]
	alc = df["Algorithm"]
	plt.bar(alc,acc,align='center', alpha=0.5,color=colors)
	plt.xlabel('Algorithm')
	plt.ylabel('MSE')
	plt.title("MSE Value");
	fig.savefig('results/MSE.png')
	plt.show()
    
    
    
	result2=open('results/MAE.csv', 'w')
	result2.write("Algorithm,MAE" + "\n")
	for i in range(0,len(mae)):
	    result2.write(al[i] + "," +str(mae[i]) + "\n")
	result2.close()
                
	fig = plt.figure(0)            
	df =  pd.read_csv('results/MAE.csv')
	acc = df["MAE"]
	alc = df["Algorithm"]
	plt.bar(alc,acc,align='center', alpha=0.5,color=colors)
	plt.xlabel('Algorithm')
	plt.ylabel('MAE')
	plt.title('MAE Value')
	fig.savefig('results/MAE.png')
	plt.show()
	    
	result2=open('results/R-SQUARED.csv', 'w')
	result2.write("Algorithm,R-SQUARED" + "\n")
	for i in range(0,len(rsq)):
	    result2.write(al[i] + "," +str(rsq[i]) + "\n")
	result2.close()
            
	fig = plt.figure(0)        
	df =  pd.read_csv('results/R-SQUARED.csv')
	acc = df["R-SQUARED"]
	alc = df["Algorithm"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
	plt.bar(alc,acc,align='center', alpha=0.5,color=colors)
	plt.xlabel('Algorithm')
	plt.ylabel('R-SQUARED')
	plt.title('R-SQUARED Value')
	fig.savefig('results/R-SQUARED.png')
	plt.show()
    
	result2=open('results/RMSE.csv', 'w')
	result2.write("Algorithm,RMSE" + "\n")
	for i in range(0,len(rmse)):
	    result2.write(al[i] + "," +str(rmse[i]) + "\n")
	result2.close()
      
	fig = plt.figure(0)    
	df =  pd.read_csv('results/RMSE.csv')
	acc = df["RMSE"]
	alc = df["Algorithm"]
	plt.bar(alc, acc, align='center', alpha=0.5,color=colors)
	plt.xlabel('Algorithm')
	plt.ylabel('RMSE')
	plt.title('RMSE Value')
	fig.savefig('results/RMSE.png')
	plt.show()
    
	result2=open('results/Accuracy.csv', 'w')
	result2.write("Algorithm,Accuracy" + "\n")
	for i in range(0,len(acy)):
	    result2.write(al[i] + "," +str(acy[i]) + "\n")
	result2.close()
    
	fig = plt.figure(0)
	df =  pd.read_csv('results/Accuracy.csv')
	acc = df["Accuracy"]
	alc = df["Algorithm"]
	plt.bar(alc, acc, align='center', alpha=0.5,color=colors)
	plt.xlabel('Algorithm')
	plt.ylabel('Accuracy')
	plt.title('Accuracy Value')
	fig.savefig('results/Accuracy.png')
	plt.show()
    



dataset=pd.read_csv('KDDTrain+.csv').values
dataset1=pd.read_csv('KDDTest+.csv').values



X_train = dataset[:,0:41]
y_train = dataset[:,41] 

X_test = dataset1[:,0:41]
y_test = dataset1[:,41]

dos=[]
u2r=[]
r2l=[]
probe=[]
normal= []
unknown=[]


for c in y_test:
	print(c)
	if c==1.0:
		dos.append(c)
	if c==2.0:
		u2r.append(c)
	if c==3.0:
		r2l.append(c)
	if c==4.0:
		probe.append(c)
	if c==5.0:
		normal.append(c)

#X_train, X_test, y_train, y_test
process(X_train,X_test,y_train,y_test)

tot=len(dos)+len(u2r)+len(r2l)+len(probe)+len(normal)
d=(len(dos)/tot)*100
u=(len(u2r)/tot)*100
r=(len(r2l)/tot)*100
p=(len(probe)/tot)*100
n=(len(normal)/tot)*100

print(len(dos))
print(len(u2r))
print(len(r2l))
print(len(probe))
print(len(normal))

print(d)
print(u)
print(r)
print(p)
print(n)

colors = ["#FF0000", "#FF0000", "#FF0000", "#FF0000", "#008000"]
explode = (0.1, 0, 0, 0, 0)  


alc = ["dos","U2R","R2L","PROBE","NORMAL"]
acc = [d,u,r,p,n]


fig = plt.figure(0)    
plt.bar(alc, acc, align='center', alpha=0.5,color=colors)
plt.xlabel('Methods')
plt.ylabel('Rate in %')
plt.title('Detection Rate')
plt.show()
 