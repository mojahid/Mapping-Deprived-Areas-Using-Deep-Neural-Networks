
# Using Open Building Data for classification

The aim is to leverage Open Building data to map deprived areas by two approaches:  
1- Genearting images from Open Building Data to train different models  
2- Generating numercial data that can be used to train hybride model (images + numerical data)    

For image generation, the following technique was used:  

1. For each label coordinate in the training dataset, map all buildings that falls within the boundary of that label (100m2) 
2. Construct an image using the geometry column that defines the shape of the mapped buildings  

![image](https://user-images.githubusercontent.com/34656794/164742475-84b99530-a9dc-4f6b-8952-8cec027dc253.png)

## Using the code for Open Building Classification

1. The dataset can be downloaded from [here](https://sites.research.google/open-buildings/), the full dataset or just certain regions can be downloaded separately
2. Run **Extract_Open_Building_Subset.py** to narrow down the subset further and get the data for a specific boundary long/lat coordinates (current values in the cde corresponds to Lagos)
3. Run **Open_Building_Testing.py** which will use the deprived labels and iterate through the Open Building subset to generate corresponding image for each label using functions in **Generate_OB_Images.py**   
4. To augment the images, the **Open_Building_Augmentation.py** to generate more images from the minority class 

At the end of these four steps, there will be two folders (corresponds to the labels) and each contains the generated images that corresponds to that label.  
<img src="https://user-images.githubusercontent.com/34656794/164747500-80940dd4-7c50-4e7f-9693-dab271c2da82.png" width="400" height="250"/>   
  


# Using Open Building Data combined with satellite images for classificatin 
The other use of Google Open building is to use it along with satellite images to train a hybrid model. To do so, the following steps were used:
1. Generate numerical data from Google Open Building for the hybrid model

![image](https://user-images.githubusercontent.com/34656794/164884508-6af76e40-4292-4638-81f7-7920adde7218.png)


2. To maintain the same order, each record in the numerical data will be used to generate an image from Google Open Building

![image](https://user-images.githubusercontent.com/34656794/164884520-4e5bdfa8-b7a4-462f-abc9-fe9adcb4ce71.png)


3. Augmentation will be applied to by generating more images (shifting and rotating images) and also adding the corresponding numerical data

![image](https://user-images.githubusercontent.com/34656794/164884536-0e022be1-658e-4bab-b18f-25017930f9cf.png)

4. Use the coordinates in the augmented data to generate the corresponding satellite images

![image](https://user-images.githubusercontent.com/34656794/164884570-32e0cd91-64a5-4e14-ada7-b12fadefed99.png)

## Using the code for combined data

