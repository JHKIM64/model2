from Preprocessing import stationToGrid
from dateutil.parser import parse
import pandas as pd
import datetime
import xarray as xr

china_dir_path ='/home/intern01/jhk/China/'
weather_data = xr.open_dataset("/home/intern01/jhk/ECMWF_Land/ECMWR_EA_CLCOND_1920.nc")

#path < want to make .nc file
def EA_AQdata(begin,end,top,bottom,left,right) :
    t_begin = parse(begin)
    t_end = parse(end)
    _, ch_dic = stationToGrid.toGrid(top, bottom, left, right)
    AQdata = pd.DataFrame()

    while t_begin <= t_end :
        Ymd = t_begin.strftime("%Y.%m.%d").split('.')
        ch_path = china_dir_path + 'data/china_aqi_' + Ymd[0]+Ymd[1] +Ymd[2]+ '.csv'
        df_chAQ = pd.read_csv(ch_path)  # Read file
        if not df_chAQ.empty :
            m_data_ch = CHAQtoDF(df_chAQ,ch_dic)
            AQdata=pd.concat([AQdata,m_data_ch]).sort_index()
        t_begin += datetime.timedelta(days=1)
    return AQdata

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

def run(begin,end,top,bottom,left,right) :
    #'20210101','20210430',39,34,124,133,0.1,0.1
    data = EA_AQdata(begin, end, top, bottom, left, right)
    return data