import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from project_root import get_project_root
root = get_project_root()

######################################### CONTEXTUAL FEATURES MERGING############################################


# Merge all contextual features csv files and aggregate them to the training data level


# Read training file
training_df = pd.read_csv(root / '1.Data' / 'coordinates.csv')

# Change Label column to "Label"
training_df= training_df.rename({"Data":"Label"}, axis=1)

# List for creating point column in training_df
num_lst=[]
for i in range(47561):
    if i >0 :
        num_lst.append(i)

training_df["Point"]= num_lst
training_df= training_df[["long", "lat", "Label", "Point"]]

print(training_df)
print(100*"--")

# List for creating point column in context feature dataframe
num_lst2= []
for i in range(47561):
    for z in range(100):
        if i >0 :
            num_lst2.append(i)

print(num_lst2[-1])

# Directory containing all contextual features csv files
directory_in_str= "/home/ubuntu/context/"

directory = os.fsencode(directory_in_str)

# Looping through directory ( file must be csv)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv") and filename != "coordinates.csv":

        # Read contextual feature csv file
        context_df= pd.read_csv(filename)

        # Remove [] from all extracted values
        context_df['Raster Value'] = context_df['Raster Value'].str.strip('[]').astype(float)
        # Adds point number column
        context_df["Point"]= num_lst2
        # Calculate Average for each point ( 100 contextual feature datapoints)
        context_df[filename]= context_df.groupby("Point")["Raster Value"].transform("mean")
        # Drop Duplicates since the avergae values is computed ( leaving dataframe row number equals to traininf_df)
        context_df= context_df.drop_duplicates("Point")
        # Remove "Raster Value" Column
        context_df= context_df[["Point", filename]]
        # Remove ".tif.csv" from column name
        context_df.columns= context_df.columns.str.replace(r".tif.csv", "")

        # Merge context_df with training_df
        training_df = training_df.merge(context_df, on="Point")



print(training_df)


# write dataframe to csv
filename = 'Contextual_Features.csv'
training_df.to_csv(root / '1.Data' / f'{filename}', index=False)
