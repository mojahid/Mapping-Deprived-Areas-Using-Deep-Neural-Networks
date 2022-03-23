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


#--------------------------------------------------------------------------------------------------------------

# This portion should include the code to generate 100 points for each coordinate from the training data

# The following code can generate 100 points from a given point
# "data" dataframe contains the coordinates from the training data
data = pd.read_csv("coordinates.csv")
# create a column and put coordinates in tuple
data["coords"] = [(x,y) for x, y in zip(data["long"], data["lat"])]

print(data)

# Coordinate tuple
#x= (3.343333335, 6.402500176)
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

# create long list
    for v in range(9):
        new_long= long + (p/10)
        long_lst.append(new_long)
        long= new_long

#print(long_lst)

# create lat list
    for w in range(9):
        new_lat= lat - (p/10)
        lat_lst.append(new_lat)
        lat= new_lat

#print(lat_lst)


#print(len(lat_lst))
#print(len(long_lst))

# generate coordinates
    z = [ (a,b) for a in long_lst for b in lat_lst ]
    coord_lst.append(z)
#print(z)
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
#----------------------------------------------------------------------------------------------------------------


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
#print(data_2)
#data_2['Raster Value'] = data_2['Raster Value'].apply(lambda x: x['Raster Value'][0], axis=1)


# write dataframe to csv
        data_2.to_csv('{}.csv' .format(filename), index=False)




