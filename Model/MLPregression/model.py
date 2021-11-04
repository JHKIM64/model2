## near one grid + 4 boundary condition grid
## target grid : 서울지역 0.1*0.1 grid 8개
## boundary condition : grid 11개

# method
## 8개 grid별 모델학습(동서남북중앙 값 시계열 변화 학습) 이후 앙상블학습을 통해 모델 학습
## data overfitting예방

## model grid target location
## (37.6,126.9),(37.6,127.0),(37.5,126.9),(37.5,127.0),(37.5,127.1),(37.4,126.9),(37.4,127.0),(37.4,127.1)

import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import CNSEW
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import r2_score
import Plot.timeseriesplot as tsplot
from keras.utils.vis_utils import plot_model
from Plot.traininghistory import plot_history

def copynext(df_bd,t):
    Nt_CPM25 = df_bd.CPM25.shift(-t)
    Nt_CPM10 = df_bd.CPM10.shift(-t)
    Nt_CSO2 = df_bd.CSO2.shift(-t)
    Nt_CNO2 = df_bd.CNO2.shift(-t)
    Nt_CCO = df_bd.CCO.shift(-t)
    Nt_CO3 = df_bd.CO3.shift(-t)

    df_bd['Nt_CPM25'] = Nt_CPM25
    df_bd['Nt_CPM10'] = Nt_CPM10
    df_bd['Nt_CSO2'] = Nt_CSO2
    df_bd['Nt_CNO2'] = Nt_CNO2
    df_bd['Nt_CCO'] = Nt_CCO
    df_bd['Nt_CO3'] = Nt_CO3

    df_bd = df_bd.iloc[:-t]
    return df_bd

def preprocess(lat, lon, t) :
    df = CNSEW.df_CNSEW(lat, lon)
    df.dropna(axis=0, inplace=True)
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = copynext((train_df - train_mean) / train_std,t)
    val_df = copynext ((val_df - train_mean) / train_std,t)
    test_df = copynext((test_df - train_mean) / train_std,t)

    df_std = (df - train_mean) / train_std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(df.keys(), rotation=90)
    plt.show()
    print(test_df.CPM25, test_df.Nt_CPM25)
    return train_df, val_df, test_df, train_mean, train_std

######## split into input x and output y
def targetmodel(train_df, val_df, test_df) :
    chemicals = ['Nt_CPM25','Nt_CPM10','Nt_CSO2','Nt_CNO2','Nt_CCO','Nt_CO3']
    test_x = test_df.drop(columns=chemicals, axis=1)
    test_y = test_df.loc[:,chemicals]

    models = []
    for target in chemicals :
        train_x = train_df.drop(columns=chemicals, axis=1)
        train_y = train_df.loc[:,chemicals]

        val_x = val_df.drop(columns=chemicals,axis=1)
        val_y = val_df.loc[:,chemicals]

        # create model
        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            model = Sequential()
            model.add(Dense(55, activation="relu", input_dim=55, kernel_initializer="uniform"))
            model.add(Dense(11, activation="relu", input_dim=55, kernel_initializer="uniform"))
            model.add(Dense(1, activation="linear", kernel_initializer="uniform"))

        # Compile model
            model.compile(loss='mse', optimizer='adam', metrics=['accuracy','mse'])

        # Fit the model
            model_history = model.fit(train_x, train_y, epochs=100, batch_size=5, validation_data=(val_x,val_y),verbose=1)
            plot_history([(target,model_history)],target)

            # Save model
            model.save(target+'model')
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

def allchemmodel(model, test_x, test_y,t) :
    df_PM25 = []
    df_PM10 = []
    df_SO2 = []
    df_NO2 = []
    df_CO = []
    df_O3 = []

    for i in range(0,len(test_x)-t):
        pred_next = []
        pred_next.append(model.predict(test_x[i:i+1]))

        df_PM25.append([test_y.Nt_CPM25.iloc[i],pred_next[0]])
        df_PM10.append([test_y.Nt_CPM10.iloc[i],pred_next[1]])
        df_SO2.append([test_y.Nt_CSO2.iloc[i],pred_next[2]])
        df_NO2.append([test_y.Nt_CNO2.iloc[i],pred_next[3]])
        df_CO.append([test_y.Nt_CCO.iloc[i],pred_next[4]])
        df_O3.append([test_y.Nt_CO3.iloc[i],pred_next[5]])

        test_x.iloc[i + t].CPM25 = pred_next[0]
        test_x.iloc[i + t].CPM10 = pred_next[1]
        test_x.iloc[i + t].CSO2 = pred_next[2]
        test_x.iloc[i + t].CNO2 = pred_next[3]
        test_x.iloc[i + t].CCO = pred_next[4]
        test_x.iloc[i + t].CO3 = pred_next[5]

    return df_PM25, df_PM10, df_SO2, df_NO2, df_CO, df_O3

def run(lat, lon, t) :
    train_df, val_df, test_df , train_mean, train_std = preprocess(lat,lon,t)
    models, test_x, test_y = targetmodel(train_df, val_df, test_df)
    df_PM25, df_PM10, df_SO2, df_NO2, df_CO, df_O3 = allchemmodel(models, test_x, test_y,t)
    tsplot.actual_plot(df_PM25,len(test_x), lat, lon,'PM25',train_mean.CPM25,train_std.CPM25,'ug/m^3',t)
    tsplot.actual_plot(df_PM10, len(test_x), lat, lon, 'PM10',train_mean.CPM10,train_std.CPM10,'ug/m^3',t)
    tsplot.actual_plot(df_SO2, len(test_x), lat, lon, 'SO2',train_mean.CSO2,train_std.CSO2,'ug/m^3',t)
    tsplot.actual_plot(df_NO2, len(test_x), lat, lon, 'NO2',train_mean.CNO2,train_std.CNO2,'ug/m^3',t)
    tsplot.actual_plot(df_CO, len(test_x), lat, lon, 'CO',train_mean.CCO,train_std.CCO,'ug/m^3',t)
    tsplot.actual_plot(df_O3, len(test_x), lat, lon, 'O3',train_mean.CO3,train_std.CO3,'ug/m^3',t)

run(37.6,126.9, 1)