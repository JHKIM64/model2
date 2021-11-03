import math

import pandas as pd
import numpy as np
stations_korea = pd.read_csv("/home/intern01/jhk/AirKorea/Korea_stninfo.csv")
stations_china = pd.read_csv("/home/intern01/jhk/China/China_stninfo.csv")

def station_index(df) :
    df.columns=['Station','Lon','Lat']
    df = df.set_index('Station')
    return df

def boolgrid(top, bottom, left, right, lon_step, lat_step) :
    longitude = []
    latitude = []
    i=bottom
    j=left
    while i < top :
        latitude.append(i)
        i = round(i + lat_step,1)
    while j < right :
        longitude.append(j)
        j = round(j + lon_step,1)
    grid = pd.DataFrame(False,index=latitude,columns=longitude)
    return grid

def modLocation(df) :
    df['Lon'] = np.floor(df['Lon'].values * 10) / 10
    df['Lat'] = np.floor(df['Lat'].values * 10) / 10
    return df

def toGrid(top,bottom,left,right,lon_step,lat_step) :
    korea = modLocation(station_index(stations_korea))
    china = modLocation(station_index(stations_china))
    stnToLocDict_KR = {}
    stnToLocDict_CH = {}

    for nk,rk in korea.iterrows() :
        grid_col=rk['Lon']
        grid_row=rk['Lat']
        if grid_col < left or grid_col > right or grid_row > top or grid_row < bottom :
            continue

        stnToLocDict_KR[nk] = str(grid_row)+','+str(grid_col)

    for nc, rc in china.iterrows():
        grid_col = rc['Lon']
        grid_row = rc['Lat']
        if grid_col < left or grid_col > right or grid_row > top or grid_row < bottom:
            continue

        stnToLocDict_CH[str(nc)+'A'] = str(grid_row)+','+str(grid_col)

    return stnToLocDict_KR,stnToLocDict_CH