import os
os.system("sudo pip install geopandas")
os.system("sudo pip install rasterio")
#import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
from rasterio.plot import show, show_hist
#from pyproj import Proj, transform

from project_root import get_project_root
root = get_project_root()

############################################# COVARIATE DATA EXTRACTION ############################################################


LABEL_PATH = r'lag_covariates_compilation.tif'

# METHOD 1 :
# Can be used to generate all data within the covariate tif file.



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

#---------------------------------------------------------------------------------------------------------------------

# METHOD 2:

# Using Coordinates from training data to extract all band values  rather than just reading file


data_2 = pd.read_csv(root / '1.Data' / 'coordinates.csv')

data_2["Coordinates"] = [(x,y) for x, y in zip(data_2["long"], data_2["lat"])]
src = rasterio.open(LABEL_PATH)


# Sample the raster at every point location and store values in DataFrame
data_2['Raster Value'] = [x for x in src.sample(data_2["Coordinates"])]

# write dataframe to csv
save_path = r'1.Data'
filename = 'covariate.csv'
data_2.to_csv(f'{save_path}/{filename}', index=False)


################################### PROCESSING COVARIATE FEATURES #####################################################
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

data_2 = data_2.rename({"Data":"Label"}, axis=1)

# write dataframe to csv
filename = 'Covariate_Features.csv'
data_2.to_csv(root / '1.Data' / f'{filename}', index=False)

