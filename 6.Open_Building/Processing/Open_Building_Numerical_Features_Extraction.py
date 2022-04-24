# Import Libraries
import pandas as pd
import os
import glob

###########################################OPEN BUILDING NUMERICAL FEATURES EXTRACTION######################################################3

# This python file contains code used to extract numerical features from Lagos open building dataset

# Read Lagos open building dataset
lagos=pd.read_csv("Lagos_OpenBuilding.csv")

# Choose subset of needed columns
lagos= lagos[["latitude", "longitude", "area_in_meters", "confidence", "geometry"]]


# Display dataframe in chunks

#chunksize = 10 ** 6
#for chunk in pd.read_csv("Lagos_OpenBuilding.csv", chunksize=chunksize):
    #print(chunk)



# Read Training labels coordinates file
coord=pd.read_csv("coordinates.csv")

# Remove non-built up label
coord= coord[coord["Data"] < 2]
coord= coord.rename({"Data":"Label"}, axis=1)

# Create Point Column
num_lst=[]
for i in range(15472):
    if i >0 :
        num_lst.append(i)
coord["Point"]= num_lst
coord=coord.reset_index(drop=True)
coord= coord[["long", "lat", "Label", "Point"]]


# Loop through coordinates and extract polygon coordinates for each label

for index, row in coord.iterrows():
    # Get pixel coordinates and define the 4 boundary coordinates
    py, px = row['long'], row['lat']
    py1 = round(py + 5*0.00008333, 6)
    py2 = round(py - 5*0.00008333, 6)
    px1 = round(px + 5*0.00008333, 6)
    px2 = round(px - 5*0.00008333, 6)

    # Query text to bound longitude and latitude within the boundaries
    query_text = 'latitude  <' +str(px1)+ ' & latitude > '+str(px2)+ ' & longitude < ' + str(py1)+' & longitude > ' +str(py2)

    zone = lagos.query(query_text)

    # Perform aggregations for each training data point
    zone["Point"]=index+1
    zone["Mean_Area"]= zone.groupby("Point")["area_in_meters"].transform("mean")
    zone["Median_Area"]= zone.groupby("Point")["area_in_meters"].transform("median")
    zone["Building_Count"]= zone.groupby("Point")["Point"].transform("count")
    zone["Max_Area"]= zone.groupby("Point")["area_in_meters"].transform("max")
    zone["Min_Area"]= zone.groupby("Point")["area_in_meters"].transform("min")



    # Create csv file for each training data point
    zone.to_csv("Lagos_polygons_OpenBuilding_{}.csv".format(index+1))






# Read and merge all generated csv files


# setting the path for joining multiple files
files = os.path.join(r"C:\Users\abdul\capstone\Open_Building_2", "Lagos_polygons*.csv")

# list of merged files returned
files = glob.glob(files)


# joining files with concat and read_csv
area_df = pd.concat(map(pd.read_csv, files), ignore_index=True)

# Sort  generated open building dataset by Point , rest index and drop duplicates
area_df =area_df.sort_values(by=["Point"])
area_df= area_df.reset_index(drop= True)
area_df= area_df.drop_duplicates("Point")




area_final= area_df[["Point", "Mean_Area", "Median_Area", "Building_Count", "Max_Area", "Min_Area"]]



# Merge open building generated dataset with training coordinates data
coord = coord.merge(area_final, on="Point", how="left")



# Wrtie final dataframe to csv
coord.to_csv("Area_OpenBuilding.csv")












