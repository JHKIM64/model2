import datetime
import numpy as np
import pandas as pd

import Grid.selectGrid as selgrid
import Model.ensemble as ensemble
import Plot.timeseriesplot as tsplot
import ray

data = selgrid.toxarray()
time = np.array(data.time)
models = ensemble.get_trained_model('/home/intern01/jhk/model2/Model/MLPregression/')

ray.init(num_cpus=8)
inner_grid = [(37.6,126.9),(37.6,127.0),(37.5,126.9),(37.5,127.0),(37.5,127.1),(37.4,126.9),(37.4,127.0),(37.4,127.1)]

def prediction(data, time) :
    df_mean = data.mean()
    df_std = data.std()

    normalized_df = (data - df_mean) / df_std
    print(len(time))

    for tick in range(0,168) :
        pred_res = [gridcalculation(lat,lon,normalized_df.loc[time[tick]]) for lat,lon in inner_grid]
        idx = 0
        for lat, lon in inner_grid :
            normalized_df.loc[time[tick+1], lat, lon].PM25 = float(pred_res[idx][:,0])
            normalized_df.loc[time[tick+1], lat, lon].PM10 = float(pred_res[idx][:,1])
            normalized_df.loc[time[tick+1], lat, lon].SO2 = float(pred_res[idx][:,2])
            normalized_df.loc[time[tick+1], lat, lon].NO2 = float(pred_res[idx][:,3])
            normalized_df.loc[time[tick+1], lat, lon].CO = float(pred_res[idx][:,4])
            normalized_df.loc[time[tick+1], lat, lon].O3 = float(pred_res[idx][:,5])
            idx += 1
        print(time[tick], ' : done')
    pred_df = normalized_df*df_std + df_mean

    choosechem(37.6,126.9,data.loc[time[0]:time[82]],pred_df.loc[time[0]:time[82]], 'PM25')
    choosechem(37.6,127.0, data.loc[time[0]:time[82]], pred_df.loc[time[0]:time[82]], 'PM25')
    choosechem(37.5,126.9, data.loc[time[0]:time[82]], pred_df.loc[time[0]:time[82]], 'PM25')
    choosechem(37.6, 127.0, data.loc[time[0]:time[82]], pred_df.loc[time[0]:time[82]], 'PM25')
    choosechem(37.6, 127.1, data.loc[time[0]:time[82]], pred_df.loc[time[0]:time[82]], 'PM25')
    choosechem(37.4, 126.9, data.loc[time[0]:time[82]], pred_df.loc[time[0]:time[82]], 'PM25')
    choosechem(37.4, 127.0, data.loc[time[0]:time[82]], pred_df.loc[time[0]:time[82]], 'PM25')
    choosechem(37.4, 127.1, data.loc[time[0]:time[82]], pred_df.loc[time[0]:time[82]], 'PM25')

def gridcalculation(lat, lon, data) :
    c = data.loc[lat,lon]
    n = data.loc[round(lat+0.1,1),lon]
    s = data.loc[round(lat-0.1,1),lon]
    e = data.loc[lat,round(lon+0.1,1)]
    w = data.loc[lat,round(lon-0.1,1)]
    input = np.concatenate((c,n,s,e,w),axis=None).reshape((1,-1))

    result=np.zeros(6)
    for model in models :
        pred_res = ensembleresult(model, input)
        result = result + pred_res

    return result/len(models)

def ensembleresult(model,input) :
    return model.predict(input)

def choosechem(lat, lon, obsv, pred, target) :
    pred = pred.swaplevel(0, 2, axis=0).loc[(lon,lat),target]
    obsv = obsv.swaplevel(0, 2, axis=0).loc[(lon,lat),target]
    data = pd.concat([obsv, pred],axis=1)


    tsplot.normalized_plot(data,len(pred), lat, lon, target)

prediction(data.to_dataframe(),time)