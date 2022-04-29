import os
#os.system("sudo pip install rasterio")
#os.system("sudo pip install pyproj")

#os.system("sudo pip install geopandas")
#import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
from rasterio.plot import show, show_hist
#from pyproj import Proj, transform

from project_root import get_project_root
root = get_project_root()

#--------------------------------------------------------------------------------------------------------------

# This portion should include the code to generate 100 points for each coordinate from the training data

# The following code can generate 100 points from a given point
# "data" dataframe contains the coordinates from the training data
data = pd.read_csv(root / '1.Data' / 'coordinates.csv.csv')


# create a column and put coordinates in tuple, Shift center of the label to the top left coordinates.

data["Long_new"]= data["long"] -  5*0.00008333
data["Lat_new"]= data["lat"] +  5*0.00008333
data["coords"] = [(x,y) for x, y in zip(data["Long_new"], data["Lat_new"])]

print(data)

# Coordinate tuple
#x= (long, lat)
coord_lst=[]
coordinates=data["coords"].tolist()
# step size ( distance between labels ( 10 pixels) )
for i in coordinates:
    p=0.000833333

    long_lst=[]
    lat_lst=[]
    long=i[0]
    lat=i[1]
    # first long and lat
    long_lst.append(long+ (p/20))
    lat_lst.append(lat - (p/20))

    long = long + (p / 20)
    lat = lat - (p / 20)

# create long list
    for v in range(9):
        new_long= long + (p/10)
        long_lst.append(new_long)
        long= new_long


# create lat list
    for w in range(9):
        new_lat= lat - (p/10)
        lat_lst.append(new_lat)
        lat= new_lat



# generate coordinates
    z = [ (a,b) for a in long_lst for b in lat_lst ]
    coord_lst.append(z)
    print(len(z))

print("Number of points per row: ", len(coord_lst[0]))
print("Total points: " ,len(coord_lst)*100)

coord_final=[]
for i in coord_lst:
    for x in i:
        coord_final.append(x)

print(len(coord_final))

# Dataframe containing all coordinates
data_2= pd.DataFrame({"Coordinates": coord_final})
print(data_2)
data_2.to_csv('contextual_coordinates.csv' , index=False)

######################################### CONTETXUAL FEATURES CSV GENERATION #########################################

# Generate csv files for each contextual feature
directory_in_str= "/home/ubuntu/context/"

directory = os.fsencode(directory_in_str)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".tif"):
        # print(os.path.join(directory, filename))
        # open tif file
        src = rasterio.open(filename)


        # Sample the raster at every point location and store values in DataFrame
        data_2['Raster Value'] = [x for x in src.sample(data_2["Coordinates"])]

        # write dataframe to csv
        data_2.to_csv(root / '1.Data' / f'{filename}', index=False)





