## near one grid + 4 boundary condition grid
## target grid : 서울지역 0.1*0.1 grid 8개
## boundary condition : grid 11개

# method
## 8개 grid별 모델학습(동서남북중앙 값 시계열 변화 학습) 이후 앙상블학습 (bagging)을 통해 모델 학습
## data overfitting예방

## model grid target location
## (37.6,126.9),(37.6,127.0),(37.5,126.9),(37.5,127.0),(37.5,127.1),(37.4,126.9),(37.4,127.0),(37.4,127.1)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import CNSEW
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from keras.utils.vis_utils import plot_model
from keras import layers

def preprocess(lat, lon) :
    df = CNSEW.df_CNSEW(lat, lon)
    df.dropna(axis=0, inplace=True)
    print(df)
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    df_std = (df - train_mean) / train_std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(df.keys(), rotation=90)
    print(train_mean, train_std)
    plt.show()
    return train_df, val_df, test_df, train_mean, train_std

######## split into input x and output y
def targetmodel(train_df, val_df, test_df) :
    chemicals = ['Nt_CPM25','Nt_CPM10','Nt_CSO2','Nt_CNO2','Nt_CCO','Nt_CO3']
    test_x = test_df.drop(columns=chemicals, axis=1)
    test_y = test_df.loc[:,chemicals]

    models = []
    for target in chemicals :
        train_x = train_df.drop(columns=chemicals, axis=1)
        train_y = train_df.loc[:,[target]]

        val_x = val_df.drop(columns=chemicals,axis=1)
        val_y = val_df.loc[:,[target]]

        # create model
        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            model = Sequential()
            model.add(Dense(20, activation="relu", input_dim=55, kernel_initializer="uniform"))
            model.add(Dense(1, activation="linear", kernel_initializer="uniform"))

        # Compile model
            model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        # Fit the model
            model.fit(train_x, train_y, epochs=100, batch_size=5,  verbose=1)

        # Save model
            models.append(model)

        # Calculate predictions
            PredTrainSet = model.predict(train_x)
            PredValSet = model.predict(val_x)

        plt.plot(train_y,PredTrainSet,'ro')
        plt.title(target +'Training Set')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.show()
        #Compute R-Square value for training set
        TestR2Value = r2_score(train_y,PredTrainSet)
        print("Training Set R-Square=", TestR2Value)

        plt.plot(val_y,PredValSet,'ro')
        plt.title(target + 'Validation Set')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.show()
        #Compute R-Square value for validation set
        ValR2Value = r2_score(val_y,PredValSet)
        print("Validation Set R-Square=",ValR2Value)

    return models, test_x, test_y

def allchemmodel(models, test_x, test_y) :

    df_PM25 = []
    df_PM10 = []
    df_SO2 = []
    df_NO2 = []
    df_CO = []
    df_O3 = []

    for i in range(0,len(test_x)-1):
        pred_next = []
        for model in models :
           pred_next.append(model.predict(test_x[i:i+1]))

        df_PM25.append([test_y.Nt_CPM25.iloc[i],pred_next[0]])
        df_PM10.append([test_y.Nt_CPM10.iloc[i],pred_next[1]])
        df_SO2.append([test_y.Nt_CSO2.iloc[i],pred_next[2]])
        df_NO2.append([test_y.Nt_CNO2.iloc[i],pred_next[3]])
        df_CO.append([test_y.Nt_CCO.iloc[i],pred_next[4]])
        df_O3.append([test_y.Nt_CO3.iloc[i],pred_next[5]])

        test_x.iloc[i + 1].CPM25 = pred_next[0]
        test_x.iloc[i + 1].CPM10 = pred_next[1]
        test_x.iloc[i + 1].CSO2 = pred_next[2]
        test_x.iloc[i + 1].CNO2 = pred_next[3]
        test_x.iloc[i + 1].CCO = pred_next[4]
        test_x.iloc[i + 1].CO3 = pred_next[5]
    return df_PM25, df_PM10, df_SO2, df_NO2, df_CO, df_O3

def plot(df, timestep, lat, lon, target, mean, std) :
    df = pd.DataFrame(df)
    df.columns = ["Obsv", "Pred"]
    plt.figure(figsize=(20, 15))
    plt.plot(df.Obsv, 'r', label='Obsv')
    plt.plot(df.Pred, 'b', label='Pred')
    plt.title(str(timestep) + ' hours '+target+' Prediction at ' + str(lat) + ',' + str(lon))
    plt.xlabel('timestep')
    PredR2Value = r2_score(df.Obsv, df.Pred)
    font = {'color': 'black', 'size': 14}
    plt.legend()
    plt.text(0, -1, "Prediction R-Square=" + str(round(PredR2Value, 4)), fontdict=font)
    plt.savefig(str(timestep) + 'h_'+target+' Pred_' + str(lat) + '_' + str(lon) + '.png')
    plt.show()

def run(lat, lon) :
    train_df, val_df, test_df , train_mean, train_std = preprocess(lat,lon)
    models, test_x, test_y = targetmodel(train_df, val_df, test_df)

    df_PM25, df_PM10, df_SO2, df_NO2, df_CO, df_O3 = allchemmodel(models, test_x, test_y)

    plot(df_PM25,len(test_x), lat, lon,'PM25')
    plot(df_PM10, len(test_x), lat, lon, 'PM10')
    plot(df_SO2, len(test_x), lat, lon, 'SO2')
    plot(df_NO2, len(test_x), lat, lon, 'NO2')
    plot(df_CO, len(test_x), lat, lon, 'CO')
    plot(df_O3, len(test_x), lat, lon, 'O3')

run(37.6,126.9)
