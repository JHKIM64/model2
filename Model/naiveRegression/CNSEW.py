## 시계열 데이터를 t -> t+1.png 데이터의 집합으로 바꿔야함

import pandas as pd
import Grid.selectGrid as selgrid
import xarray
xr = selgrid.toxarray()

def toDF(xr,str) :
    df = xr.to_dataframe().drop(columns=['latitude','longitude'])
    df = df.rename(columns={'PM25':str+'PM25', 'PM10':str+'PM10', 'SO2':str+'SO2', 'NO2':str+'NO2', 'CO':str+'CO', 'O3':str+'O3', 'u10':str+'u10', 'v10':str+'v10', 'd2m':str+'d2m', 't2m':str+'t2m',
       'sp':str+'sp'})
    return df

def df_CNSEW(lat,lon) :
    print(xr)
    C = xr.sel(latitude=lat, longitude=lon)
    N = xr.sel(latitude=round(lat + 0.1, 1), longitude=lon)
    S = xr.sel(latitude=round(lat - 0.1, 1), longitude=lon)
    E = xr.sel(latitude=lat, longitude=round(lon + 0., 1))
    W = xr.sel(latitude=lat, longitude=round(lon - 0.1, 1))
    df_C = toDF(C,'C')
    df_N = toDF(N,'N')
    df_S = toDF(S,'S')
    df_E = toDF(E,'E')
    df_W = toDF(W,'W')

    df_bd = pd.concat([df_C,df_N,df_S,df_E,df_W],axis=1).iloc[2:]

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
    print(df_bd)
    return df_bd

df_CNSEW(37.5,127.0)
# china 117.2, 31.8 // 28.2, 112.9