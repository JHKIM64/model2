import datetime
import xarray as xr
import numpy as np
import pandas as pd

aerosol = xr.open_dataset("/home/intern01/jhk/Observation/EA_AQ_21_0104.nc")
weather = xr.open_dataset("/home/intern01/jhk/ECMWF_Land/ECMWR_EA_CLCOND_21_0104.nc")

##boundary grid - outside of inner grid(+모양으로 둘러쌓음)
boundary = [(37.7,126.9),(37.7,127.0),(37.6,127.1),(37.5,127.2),(37.4,127.2),(37.3,127.1),(37.3,127.0),(37.3,126.9),(37.4,126.8),(37.5,126.8),(37.6,126.8)]
#boundary = [(31.7,117.2),(31.9,117.2),(31.8,117.1),(31.8,117.3), (28.1, 112.9), (28.3, 112.9), (28.2, 112.8), (28.2, 113.0)]
##inner grid - seoul 8 grid
inner_grid = [(37.6,126.9),(37.6,127.0),(37.5,126.9),(37.5,127.0),(37.5,127.1),(37.4,126.9),(37.4,127.0),(37.4,127.1)]
#inner_grid = [(31.8,117.2), (28.2, 112.9)]
def to_dic(array) :
    dic = np.array([])
    for i in array :
        dic = np.append(dic,{'latitude':i[0],'longitude':i[1]})
    return dic

def from_xarray(dic, xr) :
    ns_aq_data = pd.DataFrame()
    for loc in dic :
        data = xr.sel(loc).expand_dims(["latitude","longitude"]).to_dataframe()
        ns_aq_data = pd.concat([ns_aq_data,data]).sort_index()

    return ns_aq_data

def gridData() :
    ae_bd = from_xarray(to_dic(boundary), aerosol)
    ae_ig = from_xarray(to_dic(inner_grid), aerosol)
    wt_bd = reset_Loc(from_xarray(to_dic(boundary), weather))
    wt_ig = reset_Loc(from_xarray(to_dic(inner_grid), weather))

    df_bd = pd.merge(ae_bd, wt_bd, left_index=True, right_index=True, how='left').sort_index()
    df_ig = pd.merge(ae_ig, wt_ig, left_index=True, right_index=True, how='left').sort_index()
    df_all = pd.concat([df_bd,df_ig]).sort_index()

    return df_bd, df_ig, df_all

def reset_Loc(df) :
    df = df.reset_index(['time','latitude','longitude'])
    df.latitude = np.round(df.latitude,1)
    df.longitude = np.round(df.longitude, 1)
    df = df.set_index(['time','latitude','longitude'])
    return df

def toxarray() :
    _,_,data = gridData()
    data.reset_index(inplace=True)
    mask1 = (data.time < datetime.datetime(year=2021, month=5, day=1, hour=0))
    data = data.loc[mask1,:]
    data.set_index(['time','latitude','longitude'],inplace=True)
    data.sort_index(inplace=True)

    xr_data = data.to_xarray()
    # print(data)
    return xr_data

# to_xarray()
# china 117.2, 31.8 // 28.2, 112.9