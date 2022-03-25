import os
os.system("sudo pip install geopandas")
os.system("sudo pip install rasterio")

#import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
from rasterio.plot import show, show_hist
#from pyproj import Proj, transform


LABEL_PATH = r'lag_covariates_compilation.tif'

# Generating data from tif file


# Basic exploration and meta data
#mapData = rasterio.open(LABEL_PATH)


# Read data from training tif and extract log, lat and label
#for i in range(63):
  #  if i > 0:
   #     val = mapData.read(i)

    #    no_data = mapData.nodata

      #  data = [(mapData.xy(x,y)[0],mapData.xy(x,y)[1],val[x,y]) for x,y in np.ndindex(val.shape) if val[x,y] != no_data]

       # lon = [i[0] for i in data]
        #lat = [i[1] for i in data]
        #d = [i[2] for i in data]



        #res = pd.DataFrame({"long":lon,'lat':lat, 'Band_{}'.format(i):d})
        #print(len(res))
        #res.to_csv('covariate_band_{}.csv'.format(i), index=False)

#------------------------------------------------------------------------------------------------------------
# Using Coordinates from training data to extract all band values  rather than just reading file


data_2= pd.read_csv("coordinates.csv")

data_2["Coordinates"] = [(x,y) for x, y in zip(data_2["long"], data_2["lat"])]
src = rasterio.open(LABEL_PATH)


# Sample the raster at every point location and store values in DataFrame
data_2['Raster Value'] = [x for x in src.sample(data_2["Coordinates"])]
x=[]
#for i in src.sample(data_2["Coordinates"]):
    #if i.isnumeric() == False:
        #y= 99999999
        #x.append(y)
    #else:
        #x.append(i)

#for i in range(63):
   # if i > 0:
       #val = src.read(i)
       #data_2["Band_{}".format(i)] = val.sample(data_2["Coordinates"])
#print(data_2)
#data_2['Raster Value'] = data_2['Raster Value'].apply(lambda x: x['Raster Value'][0], axis=1)


# write dataframe to csv
data_2.to_csv('covariate.csv' , index=False)


#--------------------------------------------------------------------------------------------
# Extracting covariate bands into a data frame

# Raster value column is a string containing all band values.
# input Raster value column to a list
raster_lst = data_2["Raster Value"].tolist()

# Remove "[" characters and convert all values to float
new_lst = []
for z in raster_lst:

    new_str = ""

    for i in range(len(z)):
        if i != 0:
            if i != (len(z) - 1):
                new_str = new_str + z[i]

    new_str = [float(i) for i in new_str.split()]
    new_lst.append(new_str)

# Extract each band value for all coordinates  and create a new column in dataframe
# There are 61 bands
x=0
band_lst=[]
for z in range(61):
    for i in new_lst:
        band_lst.append(i[x])
    data_2["Band_{}".format(x+1)]=band_lst
    x +=1
    band_lst=[]

data_2= data_2.rename({"Data":"Label"}, axis=1)
data_2.to_csv('Covariate_Features.csv' , index=False)



