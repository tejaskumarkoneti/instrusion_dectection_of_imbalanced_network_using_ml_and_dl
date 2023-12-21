#Neural Network
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras 
from keras.models import Sequential
from keras.layers import Dense 
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

def process():
        # importing the dataset
        dataset=pd.read_csv('KDDTrain+.csv').values
        dataset1=pd.read_csv('KDDTest+.csv').values
        
        X_train = dataset[:,0:41]
        y_train = dataset[:,41] 
        
        X_test = dataset1[:,0:41]
        y_test = dataset1[:,41]

        # Splitting the dataset into the Training set and Test set
        #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Intinialising the NN
        classifier = Sequential()

        # Adding the input layer and the first Hidden layer 
        classifier.add(Dense(activation="relu", input_dim=41, units=7, kernel_initializer="uniform"))

        # Adding the output layer 
        classifier.add(Dense(activation="sigmoid", input_dim=41, units=6, kernel_initializer="uniform"))

        # Compiling the NN
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        
        y_binary = to_categorical(y_train)
        
        # Fitting the ANN to the training set
        history=classifier.fit(X_train, y_binary, batch_size=10, epochs=30)

        # Fitting classifier to the Training set
        # Create your classifier here

        #history = model.fit(X_train, y_train, batch_size=64, epochs=50, verbose=2)
        classifier.save('results/alexnet.h5');
        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train'],loc='upper left')
        plt.savefig('results/Alexnet Loss.png') 
        plt.pause(5)
        plt.show(block=False)
        plt.close()

        #train and validation accuracy
        plt.plot(history.history['accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train'],loc='upper left')
        plt.savefig('results/Alexnet Accuracy.png') 
        plt.pause(5)
        plt.show(block=False)
        plt.close()
#process()
