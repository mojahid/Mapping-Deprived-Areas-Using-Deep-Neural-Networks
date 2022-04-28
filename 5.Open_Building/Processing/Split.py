from sklearn.model_selection import train_test_split
import pandas as pd
import os
import shutil

def get_next_image_name(global_index):
    """
    provide next name for the image that keep files sorted properly and avoid Lexicographic orders
    Inputs:
        int: global index in the file
    return:
        string new file name
    """
    #Get the number of digits the number is and add preceeding zeros
    length = len(str(global_index))
    zeros = ""
    for i in range(7-length):
        zeros = zeros + "0"
    return zeros + str(global_index)

# Flag to whether a split is needed
DO_SPLIT = False

if DO_SPLIT:
    res = pd.read_csv(r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Open_building\mixed\OB_Coords_ordered_All.csv")

    train, test = train_test_split(res, test_size = 0.2, stratify=res.Label, random_state=42)
    train.to_csv(r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Open_building\mixed\OB_Coordinates_training.csv")
    test.to_csv(r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Open_building\mixed\OB_Coordinates_validation.csv")
    print("*********Done Split*********")

# After the split, this code will setup the parameters for
validation = pd.read_csv(r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Test_Ensemble\OB_Coordinates_test.csv")
#validation = validation[validation['Label']==1]
source_raw_images = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Test_Ensemble\raw_test\{}.png"
source_OB_images  = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Test_Ensemble\OB_test\{}.png"

destination_raw = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Test_Ensemble\raw_test\1\{}.png"
destination_OB  = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Test_Ensemble\OB_test\1\{}.png"



for index, row in validation.iterrows():
    fileName = get_next_image_name(int(row['name']))
    shutil.move(source_raw_images.format(fileName), destination_raw.format(fileName))
    shutil.move(source_OB_images.format(fileName), destination_OB.format(fileName))
    print("test")