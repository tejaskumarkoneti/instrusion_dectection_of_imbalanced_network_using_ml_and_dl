import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import *
from keras import callbacks
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
def process():
        #importing the dataset
        
        dataset=pd.read_csv('KDDTrain+.csv').values
        dataset1=pd.read_csv('KDDTest+.csv').values
        
        X_train = dataset[:,0:41]
        y_train = dataset[:,41] 
        
        X_test = dataset1[:,0:41]
        y_test = dataset1[:,41]
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.random.seed(7))
        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        model = Sequential()

        model.add(Dense(41, activation='relu',
                  kernel_initializer='uniform',input_dim=X_train.shape[1]))
        model.add(Dense(31, activation='relu',
                  kernel_initializer='uniform'))
        model.add(Dense(21, activation='relu',
                  kernel_initializer='uniform'))
        model.add(Dense(11, activation='relu',
                  kernel_initializer='uniform'))
        model.add(Dense(6,  activation='sigmoid', 
                  kernel_initializer='uniform'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        print(model.summary())
        es_cb = callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=5)
        history = model.fit(X_train, y_train, batch_size=64, epochs=30, verbose=2)
        model.save('results/CNNLSTM.h5');
        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train'],loc='upper left')
        plt.savefig('results/CNNLSTM Loss.png') 
        plt.pause(5)
        plt.show(block=False)
        plt.close()

        #train and validation accuracy
        plt.plot(history.history['accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train'],loc='upper left')
        plt.savefig('results/CNNLSTM Accuracy.png') 
        plt.pause(5)
        plt.show(block=False)
        plt.close()
#process()
       

        
        
