
# The data
All data was collected with joint efforts led by [Idea Maps Network](https://ideamapsnetwork.org/lagos-aos/).  
IdeaMaps network mapped 100x100 m2 areas to against the following three labels (across multiple major cities in Africa):  
1- Built-up areas with label 0  
2- Deprived area with label 1  
3 - Non-built-up area with label 2 


For the raw image handing, the given data was two geoTiff files:  
1- Map image extracted from Google earth engine using [this branch of the project](https://github.com/arathinair11/Satellite-Imagery)  
2- Labeled image that is mapping three labels:  
    2.1 - Built-up areas with label 0  
    2.2 - Deprived area with label 1  
    2.3 - Non-built-up area with label 2  

![image](https://user-images.githubusercontent.com/34656794/161972674-fe31679d-8ca9-451b-b603-6cee83e8e759.png)
![image](https://user-images.githubusercontent.com/34656794/161972686-3409fd16-40f3-4e58-8ee2-44bd0f458003.png)


# Downloading the data
Data can be downloaded as per the instructions [here](https://github.com/mojahid/Mapping-Deprived-Areas-Using-Deep-Neural-Networks/tree/main/Data)


# Extracting coordinates
The first step in all the data processing was to extract all the data from the labeled image and end up with a csv file that contains coordinates of the labeled box and the corresponding label.    
![image](https://user-images.githubusercontent.com/34656794/161973643-d21d341c-2fff-44ff-ac90-ce7a516a6d19.png)  

# Using the code
With data downloaded, **Extract_Coordinates.py** will need the raw satellite image file and the training label file to generate the csv that contains all the coordinates of the labeled data. This will be the first step in all further data processing.

# Data Processing


Depending on the processing path, one of these folders can be selected:

1) Contextual_Features_Processing:
    
       Contextual_Feature_Extraction.py: Python file to extarct all data from 144 tiff files. Each file contains one contextual feature data.
    
       Contextual_Feature_Merging.py: Python file to merge and aggregate the output of the extracted data.
     
2) Covariate_Features_Processing:

       Covariate_Feature_Extraction.py: Python file to extarct all data from one tiff files. The tiff file contains 61 covariate feature data.


3) Raw_Images_Processing:

       Clipping_Images.py: Python file to clip labeled satellite images used in training and testing deep learning models 
       
       Image_Rolling.py:  Python file to perform image augmentation to handle class imbalance and more files as per the instruction inside

  




