
# Using Open Building Data for classification

The aim is to leverage Open Building data to map deprived areas by two approaches:  
1- Genearting images from Open Building Data to train different models  
2- Generating numercial data that can be used to train hybrid model (images + numerical data)    

For image generation, the following technique was used:  

1. For each label coordinate in the training dataset, map all buildings that falls within the boundary of that label (100m2) 
2. Construct an image using the geometry column that defines the shape of the mapped buildings  

![image](https://user-images.githubusercontent.com/34656794/164742475-84b99530-a9dc-4f6b-8952-8cec027dc253.png)

## Using the code for Open Building Classification

1. The dataset can be downloaded from [here](https://sites.research.google/open-buildings/), the full dataset or just certain regions can be downloaded separately
2. Run **Extract_Open_Building_Subset.py** to narrow down the subset further and get the data for a specific boundary long/lat coordinates (current values in the cde corresponds to Lagos) **NOTE:** Lagos building subset should be part of the downloaded data
3. Run **Open_Building_Testing.py** which will use the deprived labels and iterate through the Open Building subset to generate corresponding image for each label using functions in **Generate_OB_Images.py**   
4. To augment the images, the **Open_Building_Augmentation.py** to generate more images from the minority class 

At the end of these four steps, there will be two folders (corresponds to the labels) and each contains the generated images that corresponds to that label.  
<img src="https://user-images.githubusercontent.com/34656794/164747500-80940dd4-7c50-4e7f-9693-dab271c2da82.png" width="400" height="250"/>   
  


# Using Open Building Data combined with satellite images for classification 
The other use of Google Open building is to use it along with satellite images to train a hybrid model. To do so, the following steps were used:
1. Generate numerical data (number of building, min/max/mean/avg area) from Google Open Building for the hybrid model

![image](https://user-images.githubusercontent.com/34656794/164884508-6af76e40-4292-4638-81f7-7920adde7218.png)


2. To maintain the same order, each record in the numerical data will be used to generate an image from Google Open Building

![image](https://user-images.githubusercontent.com/34656794/164884520-4e5bdfa8-b7a4-462f-abc9-fe9adcb4ce71.png)


3. Augmentation will be applied to by generating more images (shifting and rotating images) and also adding the corresponding numerical data

![image](https://user-images.githubusercontent.com/34656794/164884536-0e022be1-658e-4bab-b18f-25017930f9cf.png)

4. Use the coordinates in the augmented data to generate the corresponding satellite images

![image](https://user-images.githubusercontent.com/34656794/164884570-32e0cd91-64a5-4e14-ada7-b12fadefed99.png)

## Using the code for combined data

To combine the three data (raw satelitte images, open building generated images and the numerical data) the code flow is as follows:
1. With the Open Building subset file available (outcome of step two in the above code flow) and using the original coordinates file that contains all the labeled coordinates, run the **Open_Building_Numerical_Features_Extraction.py**
2. After this step a csv with all numerical data will be available, copy the raw images (no rolling) folder and the open building folder under the "mixed_data" folder. There should be two separate folders one for "raw_images" and the other is for open building "ob_images"
3. Since some open building images are empty and might not be generated also raw satelitte images has empty images, we have to match the two datasets and rename the files to align and this is done by running: **match_datasets.py**. This file will also keep train data mixed (not split in two different folders per labels).
4. Run the **Rotate_Mixed_Data.py** to add more images to both datasets
5. Run **Split.py** to split the validation from the test
