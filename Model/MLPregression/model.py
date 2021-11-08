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
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import r2_score
import Plot.timeseriesplot as tsplot
from keras.utils.vis_utils import plot_model
from Plot.traininghistory import plot_history
from Plot.trainval import cp_predict

def copynext(df_bd):
    Nt_CPM25 = df_bd.CPM25.shift(-1)
    Nt_CPM10 = df_bd.CPM10.shift(-1)
    Nt_CSO2 = df_bd.CSO2.shift(-1)
    Nt_CNO2 = df_bd.CNO2.shift(-1)
    Nt_CCO = df_bd.CCO.shift(-1)
    Nt_CO3 = df_bd.CO3.shift(-1)

    df_bd['Nt_CPM25'] = Nt_CPM25
    df_bd['Nt_CPM10'] = Nt_CPM10
    df_bd['Nt_CSO2'] = Nt_CSO2
    df_bd['Nt_CNO2'] = Nt_CNO2
    df_bd['Nt_CCO'] = Nt_CCO
    df_bd['Nt_CO3'] = Nt_CO3

    df_bd = df_bd.iloc[:-1]
    return df_bd

def preprocess(lat, lon, IsTrained) :
    df = CNSEW.df_CNSEW(lat, lon)
    df.dropna(axis=0, inplace=True)
    n = len(df)

    if IsTrained :
        train_df = df
        val_df = df
        test_df = df
    else :
        train_df = df[0:int(n * 0.7)]
        val_df = df[int(n * 0.7):int(n * 0.9)]
        test_df = df[int(n * 0.9):]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = copynext((train_df - train_mean) / train_std)
    val_df = copynext ((val_df - train_mean) / train_std)
    test_df = copynext((test_df - train_mean) / train_std)

    df_std = (df - train_mean) / train_std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(df.keys(), rotation=90)
    plt.show()
    return train_df, val_df, test_df, train_mean, train_std

######## split into input x and output y
def targetmodel(train_df, val_df, test_df, isTrained) :
    chemicals = ['Nt_CPM25','Nt_CPM10','Nt_CSO2','Nt_CNO2','Nt_CCO','Nt_CO3']
    test_x = test_df.drop(columns=chemicals, axis=1)
    test_y = test_df.loc[:,chemicals]

    models = []
    for target in chemicals :
        if isTrained :
            models.append(keras.models.load_model(target + 'model'))
        else :
            train_x = train_df.drop(columns=chemicals, axis=1)
            train_y = train_df.loc[:,chemicals]

            val_x = val_df.drop(columns=chemicals,axis=1)
            val_y = val_df.loc[:,chemicals]

            # create model
            strategy = tf.distribute.MirroredStrategy()

            with strategy.scope():
                model = Sequential()
                model.add(Dense(11, activation="relu", input_dim=55, kernel_initializer="uniform"))
                model.add(Dense(5, activation="relu", input_dim=55, kernel_initializer="uniform"))
                model.add(Dense(6, activation="linear", kernel_initializer="uniform"))

            # early stop epoch
                early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

            # Compile model
                model.compile(loss='mse', optimizer='adam', metrics=['accuracy','mse'])

            # Fit the model
                model_history = model.fit(train_x, train_y, epochs=100, batch_size=5, validation_data=(val_x,val_y),verbose=1)
                plot_history([(target,model_history)],target)

                # Save model
                model.save(target+'model')
                models.append(model)


            # Calculate predictions
                PredTrainSet = pd.DataFrame(model.predict(train_x))
                PredValSet = pd.DataFrame(model.predict(val_x))
                idx = 0
                for chemical in chemicals :
                    print(train_y)
                    cp_predict(chemical.replace('NT_C',''),train_y.iloc[:,idx], PredTrainSet.iloc[:,idx])
                    cp_predict(chemical.replace('NT_C',''),val_y.iloc[:,idx], PredValSet.iloc[:,idx])
                    idx += 1

    return models, test_x, test_y

def ensemblemodel() :
    return 1

def allchemmodel(models, test_x, test_y,t) :
    global pred_next
    df_PM25 = []
    df_PM10 = []
    df_SO2 = []
    df_NO2 = []
    df_CO = []
    df_O3 = []

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        for i in range(0,len(test_x)-t):
            split_test = test_x.iloc[i:i+t].copy()
            for step in range(0,t-1) :
                input = split_test[step:step + 1]
                pred_next = models[5].predict(input)
                split_test.iloc[step+1].CPM25 = pred_next[:, 0]
                split_test.iloc[step+1].CPM10 = pred_next[:, 1]
                split_test.iloc[step+1].CSO2 = pred_next[:, 2]
                split_test.iloc[step+1].CNO2 = pred_next[:, 3]
                split_test.iloc[step+1].CCO = pred_next[:, 4]
                split_test.iloc[step+1].CO3 = pred_next[:, 5]

            input = split_test[t-1:t]
            pred_next = models[5].predict(input)

            df_PM25.append([test_y.Nt_CPM25.iloc[i+t-1],float(pred_next[:,0])])
            df_PM10.append([test_y.Nt_CPM10.iloc[i+t-1],float(pred_next[:,1])])
            df_SO2.append([test_y.Nt_CSO2.iloc[i+t-1],float(pred_next[:,2])])
            df_NO2.append([test_y.Nt_CNO2.iloc[i+t-1],float(pred_next[:,3])])
            df_CO.append([test_y.Nt_CCO.iloc[i+t-1],float(pred_next[:,4])])
            df_O3.append([test_y.Nt_CO3.iloc[i+t-1],float(pred_next[:,5])])

            #### consecutive prediction code ####
            # pred_next = models[5].predict(test_x[i:i + 1])
            #
            # df_PM25.append([test_y.Nt_CPM25.iloc[i], float(pred_next[:,0])])
            # df_PM10.append([test_y.Nt_CPM10.iloc[i], float(pred_next[:,1])])
            # df_SO2.append([test_y.Nt_CSO2.iloc[i], float(pred_next[:,2])])
            # df_NO2.append([test_y.Nt_CNO2.iloc[i], float(pred_next[:,3])])
            # df_CO.append([test_y.Nt_CCO.iloc[i], float(pred_next[:,4])])
            # df_O3.append([test_y.Nt_CO3.iloc[i], float(pred_next[:,5])])
            #
            # test_x.iloc[i + t].CPM25 = pred_next[:,0]
            # test_x.iloc[i + t].CPM10 = pred_next[:,1]
            # test_x.iloc[i + t].CSO2 = pred_next[:,2]
            # test_x.iloc[i + t].CNO2 = pred_next[:,3]
            # test_x.iloc[i + t].CCO = pred_next[:,4]
            # test_x.iloc[i + t].CO3 = pred_next[:,5]

    return df_PM25, df_PM10, df_SO2, df_NO2, df_CO, df_O3

def run(lat, lon, t) :
    train_df, val_df, test_df , train_mean, train_std = preprocess(lat,lon,True)
    models, test_x, test_y = targetmodel(train_df, val_df, test_df, True)
    df_PM25, df_PM10, df_SO2, df_NO2, df_CO, df_O3 = allchemmodel(models, test_x, test_y,t)
    tsplot.actual_plot(df_PM25,len(df), lat, lon,'PM25',train_mean.CPM25,train_std.CPM25,'ug/m^3',t)
    tsplot.actual_plot(df_PM10, len(test_x), lat, lon, 'PM10',train_mean.CPM10,train_std.CPM10,'ug/m^3',t)
    tsplot.actual_plot(df_SO2, len(test_x), lat, lon, 'SO2',train_mean.CSO2,train_std.CSO2,'ug/m^3',t)
    tsplot.actual_plot(df_NO2, len(test_x), lat, lon, 'NO2',train_mean.CNO2,train_std.CNO2,'ug/m^3',t)
    tsplot.actual_plot(df_CO, len(test_x), lat, lon, 'CO',train_mean.CCO,train_std.CCO,'ug/m^3',t)
    tsplot.actual_plot(df_O3, len(test_x), lat, lon, 'O3',train_mean.CO3,train_std.CO3,'ug/m^3',t)

run(37.6,126.9, 24)

# 12, 24, 48, 72, 96, 120, 144, 168
# 1day 2day 3day 4day 5day 6day 7day