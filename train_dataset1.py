import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import kerastuner as kt
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Dense, Flatten, Activation, ZeroPadding2D
from tensorflow.keras.layers import LSTM, Embedding, Bidirectional, TimeDistributed, Reshape, Dropout
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import random
from numpy.random import seed
import tensorflow
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.merge import Concatenate
import argparse

def train(args):

    #read in the reference data
    DF = pd.read_csv('datasets/Dataset 1.csv') #to be comparable with the size of other datasets, select ~5000 data points from the reference
    DF['TRIMER_mol'] = DF['TRIMER'].apply(Chem.MolFromSmiles)
    DF = DF.dropna()

    
    #fingperints featurization
    fp = DF['TRIMER_mol'].apply(lambda m: AllChem.GetMorganFingerprint(m, radius=3))
    
    #Recognize all n substructures found in the datasets
    fp_n = fp.apply(lambda m: m.GetNonzeroElements())
    HashCode = []
    for i in fp_n:
        for j in i.keys():
            HashCode.append(j)
    unique_set = set(HashCode)
    unique_list = list(unique_set)
    Corr_df = pd.DataFrame(unique_list).reset_index()

    #Construct feature matrix
    MY_finger = []
    for polymer in fp_n:
        my_finger = [0] * len(unique_list)
        for key in polymer.keys():
            index = Corr_df[Corr_df[0] == key]['index'].values[0]
            my_finger[index] = polymer[key]
        MY_finger.append(my_finger)
    MY_finger_dataset = pd.DataFrame(MY_finger)  

    
    # filter feature matrix using only the most dominant substructures in the dataset
    Zero_Sum = (MY_finger_dataset == 0).astype(int).sum()
    NumberOfZero = 4670 # a optimized parameter for feature filter: if more than 4670 of the 5000 samples do not pocess one substrcture, then remove this substructure from the feature matrix
    X = MY_finger_dataset[Zero_Sum[Zero_Sum < NumberOfZero].index]
    
    # CNN model
    if args.model == 'CNN':
        # input for CNN
        Mix_X_100Block = []
        for i in range(len(DF)):
            Sequency_X = [0 for a in range(100)]
            for j in range(100):
                    if j%2 == 0:
                        Sequency_X[j] = list(X.iloc[i].values)
                    else:
                        Sequency_X[j] = list(X.iloc[i].values)
            Mix_X_100Block.append(Sequency_X)   
        
        Mix_X_100Block = np.array(Mix_X_100Block)
        
        Mix_X_100Block = Mix_X_100Block.reshape((len(X) , 100, 132, 1))
        Mix_X_100Block.shape

        # data split into train/test sets for property IP
        X_train, X_test, y_train, y_test = train_test_split(Mix_X_100Block, DF['IP (eV)'], test_size=0.2, random_state=42)
        
        # model setup using the optimized architecture for IP
        model = Sequential()
        model.add(Conv2D(32, (10, 10), activation='relu', input_shape=X_train.shape[1:]))
        model.add(Conv2D(32, (8, 8), activation='relu'))
        model.add(Conv2D(32, (4, 4), activation='relu'))
        model.add(Flatten())
        model.add(Dense(1))
        optimizer=keras.optimizers.Adam(lr=0.001)
        model.compile(optimizer=optimizer, loss='mean_absolute_error')
        model.fit(x=X_train,y=y_train,epochs=200,
                        batch_size=32,
                        validation_split=0.2)

        filepath = 'Binary_IP_CNN.model'
        save_model(model, filepath, save_format='h5')

        # model evaluation
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        print("model performance (IP)")
        print("Train set R^2: %.2f" % r2_score(y_train, y_pred_train))
        print("Train MAE score: %.2f" % mean_absolute_error(y_train, y_pred_train))
        print("Train RMSE score: %.2f" % np.sqrt(mean_squared_error(y_train, y_pred_train)))
        print("Test set R^2: %.2f" % r2_score(y_test, y_pred_test))
        print("Test MAE score: %.2f" % mean_absolute_error(y_test, y_pred_test))
        print("Test RMSE score: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred_test)))

        # data split into train/test sets for property EA
        X_train, X_test, y_train, y_test = train_test_split(Mix_X_100Block, DF['EA (eV)'], test_size=0.2, random_state=42)
        
        # model setup using the optimized architecture for EA
        model = Sequential()
        model.add(Conv2D(24, (10, 10), activation='relu', input_shape=X_train.shape[1:]))
        model.add(Conv2D(32, (4, 4), activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))  
        model.add(Flatten())
        model.add(Dense(1))
        optimizer=keras.optimizers.Adam(lr=0.001)
        model.compile(optimizer=optimizer, loss='mean_absolute_error')
        model.fit(x=X_train,y=y_train,epochs=200,
                        batch_size=32,
                        validation_split=0.2)

        filepath = 'Binary_EA_CNN.model'
        save_model(model, filepath, save_format='h5')

        # model evaluation
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        print("model performance (EA)")
        print("Train set R^2: %.2f" % r2_score(y_train, y_pred_train))
        print("Train MAE score: %.2f" % mean_absolute_error(y_train, y_pred_train))
        print("Train RMSE score: %.2f" % np.sqrt(mean_squared_error(y_train, y_pred_train)))
        print("Test set R^2: %.2f" % r2_score(y_test, y_pred_test))
        print("Test MAE score: %.2f" % mean_absolute_error(y_test, y_pred_test))
        print("Test RMSE score: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred_test)))


    # Fusion model
    if args.model == 'Fusion':
        # input for Fusion
        Mix_X_100Block = []
        for i in range(len(DF)):
            Sequency_X = [0 for a in range(100)]
            for j in range(100):
                    if j%2 == 0:
                        Sequency_X[j] = 1
                    else:
                        Sequency_X[j] = 1
            Mix_X_100Block.append(Sequency_X)   
        
        Mix_X_100Block = np.array(Mix_X_100Block)
        
        Mix_X_100Block = Mix_X_100Block.reshape((4914, 100, 1))
        Mix_X_100Block.shape
        
        LSTMunits = 20 # hyperprameter for LSTM
        # define two sets of inputs
        inputA = Input(shape=(100,1))
        inputB = Input(shape=(132))

        # model setup using the optimized architecture for IP
        # the first branch operates on the first input
        RNNmodel = Sequential()
        RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True), input_shape=(100,1)))
        RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True)))
        RNNmodel.add(TimeDistributed(Dense(int(LSTMunits/2), activation="relu")))
        RNNmodel.add(Reshape((int(LSTMunits/2*100),)))
        
        # the second branch opreates on the second input
        y = Dense(8, activation="relu")(inputB)
        y = Dense(8, activation="relu")(y)
        y = Model(inputs=inputB, outputs=y)

        # fuse two branches
        combined = Concatenate()([RNNmodel.output, y.output])
        z = Dense(8, activation="relu")(combined)
        z = Dense(1, activation="linear")(z)
        model = Model(inputs=[RNNmodel.input, y.input], outputs=z)

        # data split into train/test sets for property IP
        xtrain_A, xtest_A, ytrain_A, ytest_A=train_test_split(Mix_X_100Block, DF['IP (eV)'], test_size=0.20, random_state=200)
        xtrain_B, xtest_B, ytrain_B, ytest_B=train_test_split(X, DF['IP (eV)'], test_size=0.20, random_state=200)

        optimizer=keras.optimizers.Adam(lr=0.001)
        model.compile(optimizer=optimizer, loss='mean_absolute_error')
        model.fit(x=[xtrain_A, xtrain_B], y=ytrain_B,epochs=200,
                        batch_size=32,
                        validation_split=0.2)

        filepath = 'Binary_IP_Fusion.model'
        save_model(model, filepath, save_format='h5')

        # model evaluation
        print("model performance (IP)")
        ytrain = ytrain_B
        ytest = ytest_B
        y_pred_train = model.predict([xtrain_A, xtrain_B])
        print("Train set R^2: ", r2_score(ytrain, y_pred_train))
        print("Train MAE score: %.2f" % mean_absolute_error(ytrain, y_pred_train))
        print("Train RMSE score: %.2f" % np.sqrt(mean_squared_error(ytrain, y_pred_train)))
        y_pred_test = model.predict([xtest_A, xtest_B])
        print("Test set R^2: ", r2_score(ytest, y_pred_test))
        print("Test MAE score: %.2f" % mean_absolute_error(ytest, y_pred_test))
        print("Test RMSE score: %.2f" % np.sqrt(mean_squared_error(ytest, y_pred_test)))


        # model setup using the optimized architecture for EA
        # the first branch operates on the first input
        RNNmodel = Sequential()
        RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True), input_shape=(100,1)))
        RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True)))
        RNNmodel.add(TimeDistributed(Dense(int(LSTMunits/2), activation="relu")))
        RNNmodel.add(Reshape((int(LSTMunits/2*100),)))
        
        # the second branch opreates on the second input
        y = Dense(8, activation="relu")(inputB)
        y = Dense(8, activation="relu")(y)
        y = Model(inputs=inputB, outputs=y)

        # fuse two branches
        combined = Concatenate()([RNNmodel.output, y.output])
        z = Dense(8, activation="relu")(combined)
        z = Dense(1, activation="linear")(z)
        model = Model(inputs=[RNNmodel.input, y.input], outputs=z)

        # data split into train/test sets for property EA
        xtrain_A, xtest_A, ytrain_A, ytest_A=train_test_split(Mix_X_100Block, DF['EA (eV)'], test_size=0.20, random_state=200)
        xtrain_B, xtest_B, ytrain_B, ytest_B=train_test_split(X, DF['EA (eV)'], test_size=0.20, random_state=200)

        optimizer=keras.optimizers.Adam(lr=0.001)
        model.compile(optimizer=optimizer, loss='mean_absolute_error')
        model.fit(x=[xtrain_A, xtrain_B], y=ytrain_B,epochs=200,
                        batch_size=32,
                        validation_split=0.2)

        filepath = 'Binary_EA_Fusion.model'
        save_model(model, filepath, save_format='h5')

        # model evaluation
        print("model performance (EA)")
        ytrain = ytrain_B
        ytest = ytest_B
        y_pred_train = model.predict([xtrain_A, xtrain_B])
        print("Train set R^2: ", r2_score(ytrain, y_pred_train))
        print("Train MAE score: %.2f" % mean_absolute_error(ytrain, y_pred_train))
        print("Train RMSE score: %.2f" % np.sqrt(mean_squared_error(ytrain, y_pred_train)))
        y_pred_test = model.predict([xtest_A, xtest_B])
        print("Test set R^2: ", r2_score(ytest, y_pred_test))
        print("Test MAE score: %.2f" % mean_absolute_error(ytest, y_pred_test))
        print("Test RMSE score: %.2f" % np.sqrt(mean_squared_error(ytest, y_pred_test)))


    # FFNN model
    if args.model == 'FFNN':
        
        # data split into train/test sets for property IP
        x_train, x_test, y_train, y_test = train_test_split(X, DF['IP (eV)'], test_size=0.2, random_state=11)
        
        # model setup using the optimized architecture for IP
        model = keras.models.Sequential()
        model.add(Dense(units = 24, input_dim = x_train.shape[1],activation='relu'))
        model.add(Dense(units = 64, activation='relu'))
        model.add(Dense(units = 1))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss="mean_squared_error")
        model.fit(x_train, y_train, epochs = 300, batch_size = 128,
                  validation_split=0.2)
        
        filepath = 'Binary_IP_FFNN.model'
        save_model(model, filepath, save_format='h5')
        
        # model evaluation
        print("model performance (IP)")
        y_pred_train = model.predict((x_train))
        print("Train set R^2: %.2f" % r2_score(y_train, y_pred_train))
        print("Train MAE score: %.2f" % mean_absolute_error(y_train, y_pred_train))
        print("Train RMSE score: %.2f" % np.sqrt(mean_squared_error(y_train, y_pred_train)))
        y_pred_test = model.predict((x_test))
        print("Test set R^2: %.2f" % r2_score(y_test, y_pred_test))
        print("Test MAE score: %.2f" % mean_absolute_error(y_test, y_pred_test))
        print("Test RMSE score: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred_test)))


        # data split into train/test sets for property EA
        x_train, x_test, y_train, y_test = train_test_split(X, DF['EA (eV)'], test_size=0.2, random_state=11)
        
        # model setup using the optimized architecture for EA
        model = keras.models.Sequential()
        model.add(Dense(units = 24, input_dim = x_train.shape[1],activation='relu'))
        model.add(Dense(units = 64, activation='relu'))
        model.add(Dense(units = 1))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss="mean_squared_error")
        model.fit(x_train, y_train, epochs = 300, batch_size = 128,
                  validation_split=0.2)
        
        filepath = 'Binary_EA_FFNN.model'
        save_model(model, filepath, save_format='h5')
        
        # model evaluation
        print("model performance (EA)")
        y_pred_train = model.predict((x_train))
        print("Train set R^2: %.2f" % r2_score(y_train, y_pred_train))
        print("Train MAE score: %.2f" % mean_absolute_error(y_train, y_pred_train))
        print("Train RMSE score: %.2f" % np.sqrt(mean_squared_error(y_train, y_pred_train)))
        y_pred_test = model.predict((x_test))
        print("Test set R^2: %.2f" % r2_score(y_test, y_pred_test))
        print("Test MAE score: %.2f" % mean_absolute_error(y_test, y_pred_test))
        print("Test RMSE score: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred_test)))


    # RNN model
    if args.model == 'RNN':
        # input for RNN
        Mix_X_100Block = []
        for i in range(len(DF)):
            Sequency_X = [0 for a in range(100)]
            for j in range(100):
                    if j%2 == 0:
                        Sequency_X[j] = list(X.iloc[i].values)
                    else:
                        Sequency_X[j] = list(X.iloc[i].values)
            Mix_X_100Block.append(Sequency_X)   
        Mix_X_100Block = np.array(Mix_X_100Block)
        
        # data split into train/test sets for property IP
        X_train, X_test, y_train, y_test = train_test_split(Mix_X_100Block, DF['IP (eV)'], test_size=0.2, random_state=11)

        # model setup using the optimized architecture for IP
        LSTMunits = 20 # hyperprameter for LSTM
        RNNmodel = Sequential()
        RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True), input_shape=(100,132)))
        RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True)))
        RNNmodel.add(TimeDistributed(Dense(int(LSTMunits/2), activation="relu")))
        RNNmodel.add(Reshape((int(LSTMunits/2*100),)))
        RNNmodel.add(Dense(1))

        RNNmodel.compile(loss='mse', optimizer='adam')
        RNNmodel.fit(X_train, y_train, validation_split=0.2, epochs=120, batch_size=64)

        filepath = 'Binary_IP_RNN.model'
        save_model(model, filepath, save_format='h5')

        # model evaluation
        print("model performance (IP)")
        y_pred_train = RNNmodel.predict((X_train))
        print("Train set R^2: %.2f" % r2_score(y_train, y_pred_train))
        print("Train MAE score: %.2f" % mean_absolute_error(y_train, y_pred_train))
        print("Train RMSE score: %.2f" % np.sqrt(mean_squared_error(y_train, y_pred_train)))
        y_pred_test = RNNmodel.predict((X_test))
        print("Test set R^2: %.2f" % r2_score(y_test, y_pred_test))
        print("Test MAE score: %.2f" % mean_absolute_error(y_test, y_pred_test))
        print("Test RMSE score: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred_test)))


        # data split into train/test sets for property EA
        X_train, X_test, y_train, y_test = train_test_split(Mix_X_100Block, DF['EA (eV)'], test_size=0.2, random_state=11)

        # model setup using the optimized architecture for EA
        LSTMunits = 20 # hyperprameter for LSTM
        RNNmodel = Sequential()
        RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True), input_shape=(100,132)))
        RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True)))
        RNNmodel.add(TimeDistributed(Dense(int(LSTMunits/2), activation="relu")))
        RNNmodel.add(Reshape((int(LSTMunits/2*100),)))
        RNNmodel.add(Dense(1))

        RNNmodel.compile(loss='mse', optimizer='adam')
        RNNmodel.fit(X_train, y_train, validation_split=0.2, epochs=120, batch_size=64)

        filepath = 'Binary_EA_RNN.model'
        save_model(model, filepath, save_format='h5')

        # model evaluation
        print("model performance (EA)")
        y_pred_train = RNNmodel.predict((X_train))
        print("Train set R^2: %.2f" % r2_score(y_train, y_pred_train))
        print("Train MAE score: %.2f" % mean_absolute_error(y_train, y_pred_train))
        print("Train RMSE score: %.2f" % np.sqrt(mean_squared_error(y_train, y_pred_train)))
        y_pred_test = RNNmodel.predict((X_test))
        print("Test set R^2: %.2f" % r2_score(y_test, y_pred_test))
        print("Test MAE score: %.2f" % mean_absolute_error(y_test, y_pred_test))
        print("Test RMSE score: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred_test)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required = True, 
    	help='Choose either "CNN", "DNN", "RNN", or "Fusion" for model architecture')
   
    parsed_args = parser.parse_args()

    train(parsed_args)