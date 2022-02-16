import os
#os.system("sudo pip install rasterio")
#os.system("sudo pip install pyproj")

#os.system("sudo pip install geopandas")
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
from rasterio.plot import show, show_hist
#from pyproj import Proj, transform


#--------------------------------------------------------------------------------------------------------------

# This portion should include the code to generate 100 points for each coordinate from the training data

#----------------------------------------------------------------------------------------------------------------

# read file that included extracted coordinates , this can be replaced with any coordinates ( the ones generated in previous section)

data_2 = pd.read_csv("coordinates_fourier.csv") # file that had 65 million rows as an example

# create new column that zips long and lat coordinates into tuples
data_2["coords"] = [(x,y) for x, y in zip(data_2["long"], data_2["lat"])]

print(data_2)


# open tif file
src = rasterio.open('fourier_sc31_mean.tif')


# Sample the raster at every point location and store values in DataFrame
data_2['Raster Value'] = [x for x in src.sample(data_2["coords"])]
data_2['Raster Value'] = data_2['Raster Value'].apply(lambda x: x['Raster Value'][0], axis=1)


# write dataframe to csv
data_2.to_csv('coordinates_fourier_sc31_test3.csv' , index=False)
