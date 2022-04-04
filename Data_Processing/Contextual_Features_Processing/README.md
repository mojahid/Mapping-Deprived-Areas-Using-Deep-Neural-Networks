# Contextual Features Processing

The contextual Features consists of 144 tiff files. Each file represent one contextual feature. Each pixel in a tiff file has a unique value. 


## Contextual_Features_Extraction:

This python file contains the code used to extract all contextual features. The training coordinates did not cover the 100 pixels within one label. Therefore, We first calculate the center coordinates of the 100 pixels for each label. Then, this is followed by using the rasterio library to extract the contextual feature value for the corresponding coordinates. This is conducted to all the contextual features tiff files. This file will output 144 csv files


     
## Contextual_Features_Merging:

This python file contains the code used to merge and aggregate all processed contextual features. the output is then merged into one dataframe. Each 100 points correspond to one labeled data. Before merging with the labeled data file, We label each 100 points with the corresponding point number. The mean of the values for these point is then computed. Finally, We merge the contextual features dataframe with the labeled dataframe. The output of this code is one csv file containing the labeled training data and the corresponding 144 contextual features columns. 




