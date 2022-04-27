import rasterio
import numpy as np
import pandas as pd
from rasterio.plot import show, show_hist
from pyproj import Proj, transform
from sklearn.model_selection import train_test_split

# This code runs after having training file which is a Geotiff file that contains
# the labeled areas on 100m2 level using the survey framework arranged by Idea Maps Network (https://ideamapsnetwork.org/lagos-aos/)

# This file will then loop through all the meta data in the file and extract the coordinates
# with its corresponding label in a csv


BASE_PATH = r"1.Data"

# update for lagos for default pathing------------------------------------------------------------------------------------
LABEL_PATH = BASE_PATH + r'\training_data\acc_training_2021.tif'

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
res.to_csv(BASE_PATH + r"\coordinates.csv", index=False)

res = pd.read_csv(BASE_PATH + r"\coordinates.csv")

# Split the data and reserve 20% as test that will never be used in the model training or validation
# use stratify to main the distribution
train, test = train_test_split(res, test_size = 0.2, stratify=res.Label, random_state=42)
train.to_csv(BASE_PATH + r"\train_ac42.csv")
test.to_csv(BASE_PATH + r"\test_ac42.csv")
