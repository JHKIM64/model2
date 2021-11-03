from Preprocessing import stationToGrid
from dateutil.parser import parse
import pandas as pd
import numpy as np
import datetime
import xarray as xr
from dateutil.relativedelta import relativedelta

korea_dir_path ='/home/intern01/jhk/AirKorea/'
weather_data = xr.open_dataset("/home/intern01/jhk/ECMWF_Land/ECMWR_EA_CLCOND_21_0104.nc")

#path < want to make .nc file
def EA_AQdata(begin,end,top,bottom,left,right) :
    t_begin = parse(begin)
    t_end = parse(end)
    kr_dic, _ = stationToGrid.toGrid(top, bottom, left, right)
    AQdata = pd.DataFrame()

    while t_begin <= t_end :
        Ymd = t_begin.strftime("%Y.%m.%d").split('.')
        kr_path = korea_dir_path + Ymd[0] + '/' + Ymd[0] + '년 ' + Ymd[1] + '월.csv'
        df_krAQ = pd.read_csv(kr_path)  # Read file
        kr_data_1m=KRAQtoDF(df_krAQ,kr_dic)
        m_data_kr = convert(kr_data_1m,weather_data)
        AQdata=pd.concat([AQdata,m_data_kr])

        t_begin += relativedelta(months=1)
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

def run(begin,end,top,bottom,left,right) :
    #'20210101','20210430',39,34,124,133,0.1,0.1
    data = EA_AQdata(begin, end, top, bottom, left, right)
    data.to_csv('EA_AQ_2021_0104.csv')
