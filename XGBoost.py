# -*- coding: utf-8 -*-

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
#from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score

def process():
        #importing the dataset
        dataset=pd.read_csv('KDDTrain+.csv').values
        dataset1=pd.read_csv('KDDTest+.csv').values
        
        X_train = dataset[:,0:41]
        y_train = dataset[:,41] 
        
        X_test = dataset1[:,0:41]
        y_test = dataset1[:,41]

        classifier = XGBClassifier(objective="multi:softmax" , max_depth=6, n_estimators=10)
        classifier.fit(X_train, y_train)

        #sorted_idx=np.argsort(classifier.feature_importances_)[::-1]
        #for index in sorted_idx:
                #print([x_train.columns[index], classifier.feature_importances_[index]])

        


        #predicting the tests set result
        y_pred=classifier.predict(X_test)
        


        result2=open("results/resultXGBClassifier.csv","w")
        result2.write("ID,Predicted Value" + "\n")
        for j in range(len(y_pred)):
            result2.write(str(j+1) + "," + str(y_pred[j]) + "\n")
        result2.close()
        
        mse=mean_squared_error(y_test, y_pred)
        mae=mean_absolute_error(y_test, y_pred)
        r2=r2_score(y_test, y_pred)
         
        
        print("---------------------------------------------------------")
        print("MSE VALUE FOR XGBClassifier IS %f "  % mse)
        print("MAE VALUE FOR XGBClassifier IS %f "  % mae)
        print("R-SQUARED VALUE FOR XGBClassifier IS %f "  % r2)
        rms = np.sqrt(mean_squared_error(y_test, y_pred))
        print("RMSE VALUE FOR XGBClassifier IS %f "  % rms)
        ac=accuracy_score(y_test,y_pred)
        print ("ACCURACY VALUE XGBClassifier IS %f" % ac)
        print("---------------------------------------------------------")
        

        result2=open('results/XGBClassifierMetrics.csv', 'w')
        result2.write("Parameter,Value" + "\n")
        result2.write("MSE" + "," +str(mse) + "\n")
        result2.write("MAE" + "," +str(mae) + "\n")
        result2.write("R-SQUARED" + "," +str(r2) + "\n")
        result2.write("RMSE" + "," +str(rms) + "\n")
        result2.write("ACCURACY" + "," +str(ac*100) + "\n")
        result2.close()


         
        
        df =  pd.read_csv('results/XGBClassifierMetrics.csv')
        acc = df["Value"]
        alc = df["Parameter"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
        explode = (0.1, 0, 0, 0, 0)  
        
        fig = plt.figure()
        plt.bar(alc, acc,color=colors)
        plt.xlabel('Parameter')
        plt.ylabel('Value')
        plt.title('XGBClassifier Metrics Value')
        fig.savefig('results/XGBClassifierMetricsValue.png') 
        plt.show()
        


         
        
       
#process()
