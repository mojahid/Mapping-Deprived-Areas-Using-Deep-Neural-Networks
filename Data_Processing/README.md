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
