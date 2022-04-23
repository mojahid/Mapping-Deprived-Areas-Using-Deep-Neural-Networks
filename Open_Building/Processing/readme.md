# Google Open Building 

As per [Google Open Building research website](https://sites.research.google/open-buildings/), this large-scale open dataset contains the outlines of buildings 
derived from high-resolution satellite imagery in order to support these types of uses. The project being based in Ghana, the current focus is on the continent of Africa.  

For each building in this dataset we include the polygon describing its footprint on the ground, a confidence score indicating how sure we are that this is a building, 
and a Plus Code corresponding to the centre of the building. There is no information about the type of building, its street address, or any details other than its geometry.

The following shows a snapshot of the data that can be donwloaded from Google Open Building website.
![image](https://user-images.githubusercontent.com/34656794/164737827-42ae80bc-8c6e-4c9e-9c3c-c66fa021549d.png)  
  
   
Google offers also a hybrid map option through Google Earth Enginer to overlay the building on the high resolution map:  

<img src="https://user-images.githubusercontent.com/34656794/164739572-576e179f-16d9-4c4a-a5f5-8e72ee2fff95.png" width="400" height="250"/>  



# Using Open Building Data

The aim is to leverage Open Building data to map deprived areas by two approaches:  
1- Genearting images from Open Building Data to train different models  
2- Generating numercial data that can be used to train hybride model (images + numerical data)    

For image generation, the following technique was used:
1- For each label coordinate in the training dataset, map all buildings that falls within the boundary of that label (100m2) 
2- Construct an image using the geometry column that defines the shape of the mapped buildings  

![image](https://user-images.githubusercontent.com/34656794/164742475-84b99530-a9dc-4f6b-8952-8cec027dc253.png)

# Using the code

1. The dataset can be downloaded from [here](https://sites.research.google/open-buildings/), the full dataset or just certain regions can be downloaded separately
2. Run **Extract_Open_Building_Subset.py** to narrow down the subset further and get the data for a specific boundary long/lat coordinates (current values in the cde corresponds to Lagos)
3. Run **Open_Building_Testing.py** which will use the deprived labels and iterate through the Open Building subset to generate corresponding image for each label using functions in **Generate_OB_Images.py**   
4. To augment the images, the **Open_Building_Augmentation.py** to generate more images from the minority class 

At the end of these four steps, there will be two folders (corresponds to the labels) and each contains the generated images that corresponds to that label.  
<img src="https://user-images.githubusercontent.com/34656794/164747500-80940dd4-7c50-4e7f-9693-dab271c2da82.png" width="400" height="250"/> 



