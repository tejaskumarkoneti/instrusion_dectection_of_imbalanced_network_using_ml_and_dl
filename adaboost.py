import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from random import seed
from random import randrange
from csv import reader
import csv
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import re

def process():
    dataset=pd.read_csv('KDDTrain+.csv').values
    dataset1=pd.read_csv('KDDTest+.csv').values
    X_train = dataset[:,0:41]
    y_train = dataset[:,41]
    X_test = dataset1[:,0:41]
    y_test = dataset1[:,41]
    #Fitting the KNNClassifier to the training set

    rfc = AdaBoostClassifier(n_estimators=100, base_estimator= None,learning_rate=1, random_state = 1)
    rfc.fit(X_train, y_train)

    #Making prediction and checking the test set

    y_pred = rfc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    result2=open("results/resultadaboost.csv","w")
    result2.write("ID,Predicted Value" + "\n")
    for j in range(len(y_pred)):
        result2.write(str(j+1) + "," + str(y_pred[j]) + "\n")
    result2.close()

    mse=mean_squared_error(y_test, y_pred)
    mae=mean_absolute_error(y_test, y_pred)
    r2=r2_score(y_test, y_pred)


    print("---------------------------------------------------------")
    print("MSE VALUE FOR adaboost IS %f "  % mse)
    print("MAE VALUE FOR adaboost IS %f "  % mae)
    print("R-SQUARED VALUE FOR adaboost IS %f "  % r2)
    rms = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE VALUE FOR adaboost IS %f "  % rms)
    ac=accuracy_score(y_test,y_pred)
    print ("ACCURACY VALUE adaboost IS %f" % ac)
    print("---------------------------------------------------------")


    result2=open('results/adaboostMetrics.csv', 'w')
    result2.write("Parameter,Value" + "\n")
    result2.write("MSE" + "," +str(mse) + "\n")
    result2.write("MAE" + "," +str(mae) + "\n")
    result2.write("R-SQUARED" + "," +str(r2) + "\n")
    result2.write("RMSE" + "," +str(rms) + "\n")
    result2.write("ACCURACY" + "," +str(ac*100) + "\n")
    result2.close()


    df =  pd.read_csv('results/adaboostMetrics.csv')
    acc = df["Value"]
    alc = df["Parameter"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
    explode = (0.1, 0, 0, 0, 0)  

    fig = plt.figure()
    plt.bar(alc, acc,color=colors)
    plt.xlabel('Parameter')
    plt.ylabel('Value')
    plt.title('adaboost Metrics Value')
    fig.savefig('results/adaboostMetricsValue.png') 
    plt.pause(5)
    plt.show(block=False)
    plt.close()
#process()
