
import matplotlib.pyplot as plt
import matplotlib.animation as amp
import numpy as np
import Grid.selectGrid as selgrid
from timeit import default_timer as timer
import cartopy.crs as ccrs

xr_near_seoul = selgrid.toxarray()

def update(t):
    U = np.array(xr_near_seoul.u10.isel(time=t).values)
    V = np.array(xr_near_seoul.v10.isel(time=t).values)
    C = np.array(xr_near_seoul.PM25.isel(time=t).values)

    c.set_array(C)
    Q.set_UVC(U, V)
    ax.set_title("Time="+np.datetime_as_string(xr_near_seoul.time[t].values, unit="h"))
    print(t)
    return Q,c,


index = xr_near_seoul.indexes

time = np.array(index.__getitem__('time'))
lon =  np.array(index.__getitem__('longitude'))
lat =  np.array(index.__getitem__('latitude'))

X,Y = np.meshgrid(lon,lat)
U = np.array(xr_near_seoul.u10.isel(time=2).values)
V = np.array(xr_near_seoul.v10.isel(time=2).values)
C = np.array(xr_near_seoul.PM25.isel(time=2).values)

fig = plt.figure(figsize =(6,6))
box = [126.5, 127.5, 37, 38]
scale = '50m'
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(box,crs=ccrs.PlateCarree())
ax.coastlines(scale)
ax.set_xticks(np.arange(box[0], box[1],0.1), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(box[2], box[3],0.1), crs=ccrs.PlateCarree())
ax.grid(b=True)

c = ax.pcolormesh(X,Y,C,cmap="gist_ncar")
Q = ax.quiver(X,Y,U,V,scale=30, scale_units='inches')

start = timer()

anim = amp.FuncAnimation(fig, update, frames=100,interval=300)
anim.save('seoul_data.gif')
print(timer()-start)
plt.show()

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.utils.vis_utils import plot_model
# model = Sequential()
# model.add(Dense(2, input_dim=1, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)