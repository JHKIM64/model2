import stationToGrid
from dateutil.parser import parse
import pandas as pd
import pandas_multiprocess.multiprocess as mt
import numpy as np
import time
import datetime
import xarray as xr
# import os
#
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

korea_dir_path ='/home/intern01/jhk/AirKorea/'
china_dir_path ='/home/intern01/jhk/China/'
weather_data = xr.open_dataset("/home/intern01/jhk/ECMWF_Land/ECMWR_EA_CLCOND_1920.nc")

#path < want to make .nc file
def EA_AQdata(begin,end,top,bottom,left,right,lat_step,lon_step) :
    t_begin = parse(begin)
    t_end = parse(end)
    kr_dic, ch_dic = stationToGrid.toGrid(top, bottom, left, right, lon_step, lat_step)
    AQdata = pd.DataFrame()

    while t_begin <= t_end :
        Ymd = t_begin.strftime("%Y.%m.%d").split('.')
        if t_begin.day==1 :
            kr_path = korea_dir_path + Ymd[0] + '/' + Ymd[0] + '년 ' + Ymd[1] + '월.csv'
            df_krAQ = pd.read_csv(kr_path)  # Read file
            kr_data_1m=KRAQtoDF(df_krAQ,kr_dic)
            m_data_kr = convert(kr_data_1m,weather_data)

            AQdata=pd.concat([AQdata,m_data_kr])

        # ch_path = china_dir_path + 'data/china_aqi_' + Ymd[0]+Ymd[1] +Ymd[2]+ '.csv'
        # df_chAQ = pd.read_csv(ch_path)  # Read file
        # if not df_chAQ.empty :
        #     m_data_ch = CHAQtoDF(df_chAQ,ch_dic)
        #     #m_data_ch = mt.multi_process(func=CHAQtoDF(df_chAQ,ch_dic),data=df_chAQ,num_process=8)
        #     AQdata=pd.concat([AQdata,m_data_ch]).sort_index()

        print(t_begin.strftime("%Y%m%d")+"// done")
        t_begin += datetime.timedelta(days=1)
    return AQdata

def convert(data, weather):
    for (time,lat,lon),value in data.iterrows() :
        if time > datetime.datetime(year=2020,month=12,day=31) : continue
        temp = weather.t2m.loc[time,lat,lon].values
        pres = weather.sp.loc[time,lat,lon].values
        if temp == np.nan or pres == np.nan : continue
        value['O3'] = value['O3']*48*pres*1000/(62.4*(237.15+temp)*132.322)
        value['NO2'] = value['NO2']*46.01*pres*1000/(62.4*(237.15+temp)*132.322)
        value['SO2'] = value['SO2']*64.06*pres*1000/(62.4*(237.15+temp)*132.322)
        value['CO'] = value['CO']*28.01*pres/(62.4*(237.15+temp)*132.322)
    return data

def KRAQtoDF(df_krAQ, kr_dic) :

    df_krAQ = df_krAQ.drop(['지역','망','측정소코드','주소'],axis=1)
    df_krAQ.rename(columns={'측정일시':'datetime','측정소명':'location'},inplace=True) # drop unnecessary col and rename col
    df_krAQ['datetime']=df_krAQ['datetime'].astype(str)
    df_krAQ['date'] = df_krAQ.datetime.str[:8]
    df_krAQ['time'] = df_krAQ.datetime.str[8:]
    df_krAQ['date'] = pd.to_datetime(df_krAQ['date'],format="%Y%m%d")
    df_krAQ.loc[(df_krAQ.time=='24'),'date'] += datetime.timedelta(days=1)
    df_krAQ['date']=df_krAQ['date'].apply(lambda  x: x.strftime('%Y%m%d'))
    df_krAQ['time'].replace({'24':'00'}, inplace=True)
    df_krAQ['datetime'] = df_krAQ[['date','time']].apply(lambda row:''.join(row.values.astype(str)),axis=1)
    df_krAQ = df_krAQ.drop(['date', 'time'], axis=1)
    df_krAQ['datetime'] = pd.to_datetime(df_krAQ['datetime'],format='%Y%m%d%H')
    df_krAQ.replace(kr_dic,inplace=True) # change station name to location
    df_krAQ=df_krAQ[df_krAQ['location']<'가'] # drop station which is not change to location

    df_krAQ=df_krAQ.groupby(['datetime','location'],as_index=False).mean() # to grid value

    df_krAQ['Lat'] = df_krAQ.location.str.split(',').str[0]
    df_krAQ['Lon'] = df_krAQ.location.str.split(',').str[1]
    df_krAQ = df_krAQ.drop(['location'],axis=1) # location to lat, lon value

    df_krAQ['datetime'] += datetime.timedelta(hours=9)
    df_krAQ = df_krAQ.set_index(['datetime', 'Lat', 'Lon']) # indexing
    df_krAQ = df_krAQ[['PM25','PM10','SO2','NO2','CO','O3']] # reorder chemicals
    df_krAQ.columns.name = 'type'
    return df_krAQ

def CHAQtoDF(df_chAQ, ch_dic) :
    df_chAQ = df_chAQ[df_chAQ['type'].isin(['PM2.5','PM10','SO2','NO2','CO','O3'])] #drop unnecessary chemical

    df_chAQ['datetime'] = df_chAQ['date'].astype('str') + df_chAQ['hour'].map('{:02}'.format)
    df_chAQ = df_chAQ.drop(['date', 'hour'], axis=1) #concat date and time
    df_chAQ = pd.melt(df_chAQ, id_vars=['datetime','type'], var_name='location')
    df_chAQ = df_chAQ.pivot_table(index=['datetime','location'], columns='type',values='value')
    df_chAQ.reset_index(level=['datetime','location'],inplace=True)
    df_chAQ.replace(ch_dic, inplace=True)  # change station name to location
    df_chAQ = df_chAQ[df_chAQ['location'].str.contains(',')]  # drop station which is not change to location

    df_chAQ = df_chAQ.groupby(['datetime', 'location'], as_index=False).mean()

    df_chAQ['Lat'] = df_chAQ.location.str.split(',').str[0]
    df_chAQ['Lon'] = df_chAQ.location.str.split(',').str[1]
    df_chAQ = df_chAQ.drop(['location'], axis=1)  # location to lat, lon value
    df_chAQ['datetime'] = pd.to_datetime(df_chAQ['datetime'], format="%Y%m%d%H") + datetime.timedelta(hours=8)
    df_chAQ = df_chAQ.set_index(['datetime', 'Lat', 'Lon'])  # indexing

    df_chAQ.rename(columns={'PM2.5':'PM25'}, inplace = True)
    df_chAQ = df_chAQ[['PM25', 'PM10', 'SO2', 'NO2', 'CO', 'O3']] # reorder chemicals

    return df_chAQ



data=EA_AQdata('20210101','20210430',39,34,124,133,0.1,0.1)
data.to_csv('EA_AQ_2021_1_4.csv')