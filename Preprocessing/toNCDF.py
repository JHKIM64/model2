import pandas as pd
from Preprocessing.AirKorea import getKRAQ
from Preprocessing.AirChina import getCHAQ
#'20210101', '20210430', 39, 34, 124, 133
def run(begin, end, top, bottom, left, right) :
    getKRAQ.run(begin, end, top, bottom, left, right)
    df = pd.read_csv('EA_AQ_2021_0104.csv')
    print(df)
    # chdf = getCHAQ.run(begin, end, top, bottom, left, right)
    # df = pd.concat([krdf,chdf]).sort_index()
    # df.reset_index(inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"],format="%Y-%m-%d %H:%M:%S")
    df = df.rename(columns={"datetime":"time","Lat":"latitude","Lon":"longitude"})
    df = df.set_index(['time', 'latitude', 'longitude'])

    aqxarray = df.to_xarray()

    aqxarray.to_netcdf("/home/intern01/jhk/Observation/EA_AQ_21_0104.nc",mode='w')

run('20210101', '20210430', 39, 34, 124, 133)