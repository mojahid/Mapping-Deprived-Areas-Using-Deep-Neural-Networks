import rasterio
import numpy as np
import pandas as pd
from rasterio.plot import show, show_hist
from pyproj import Proj, transform


LABEL_PATH = r'C:/Users/minaf/Documents/GWU/Capstone/Data/lagos/lag_training_data/lag_training_2021.tif'

# Basic exploration and meta data
mapData = rasterio.open(LABEL_PATH)
print(mapData.meta)
show_hist(mapData)


# Read data from training tif and extract log, lat and label
val = mapData.read(1)
no_data = mapData.nodata
data = [(mapData.xy(x,y)[0],mapData.xy(x,y)[1],val[x,y]) for x,y in np.ndindex(val.shape) if val[x,y] != no_data]
lon = [i[0] for i in data]
lat = [i[1] for i in data]
d = [i[2] for i in data]
res = pd.DataFrame({"long":lon,'lat':lat, 'Label':d})

#print(res)

# transform the log and lat format from 3857 to 4326
inProj = Proj(init='epsg:3857')
outProj = Proj(init='epsg:4326')

res["new_long"],res["new_lat"] = transform(inProj,outProj,res["long"],res["lat"])

# Save coordinates
res.to_csv('coordinates.csv' , index=False)
