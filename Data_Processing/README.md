# Data Processing

This folder contains python code used to extract, process of data used in this project

1) Contextual_Features_Processing:
    
       Contextual_Feature_Extraction.py: Python file to extarct all data from 144 tiff files. Each file contains one contextual feature data.
    
       Contextual_Feature_Merging.py: Python file to merge and aggregate the output of the extracted data.
     
2) Covariate_Features_Processing:

       Covariate_Feature_Extraction.py: Python file to extarct all data from one tiff files. The tiff file contains 61 covariate feature data.


3) Raw_Images_Processing:

       Clipping_Images.py: Python file to clip labeled satellite images used in training and testing deep learning models 
       
       Image_Rolling.py:  Python file to perform image augmentation to handle class imbalance. 

  
For the raw image handing, the give data was two geoTiff files:  
1- Map image extracted from Google earth engine using the [this branch of the project](https://github.com/arathinair11/Satellite-Imagery)  
2- Labeled images that is mapping three labels:  
    2.1 - Built-up areas with label 0  
    2.2 - Deprived area with label 1  
    2.3 - Non-built-up area with label 2  

![image](https://user-images.githubusercontent.com/34656794/161972674-fe31679d-8ca9-451b-b603-6cee83e8e759.png)
![image](https://user-images.githubusercontent.com/34656794/161972686-3409fd16-40f3-4e58-8ee2-44bd0f458003.png)

