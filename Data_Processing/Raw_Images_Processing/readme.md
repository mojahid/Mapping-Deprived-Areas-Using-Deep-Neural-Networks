# Clipping Image  
This code loop through all the coordinates extracted from the training raster and extract their corresponding clipped image (GeoTiff) from the satellite image and store them in separate folder based on their label
![image](https://user-images.githubusercontent.com/34656794/161365206-53dee169-7082-4b97-8d05-9e364293c695.png)

# Convert to PNG
This code converts all the geoTiff to PNG that can be used with Convolutional Neural Network training in the next project phases.  
To do so, the following steps are followed:  
1- Retrieve the minimum and maximum values of each band in the original satellite image (we used QGIS to capture these information):   
![image](https://user-images.githubusercontent.com/34656794/161365482-711ea02c-7143-4023-9254-40d21b5be606.png)

2- Loop through the clipped GeoTiff and for each image normalize the value of each band based on minimum and maximum values, this will end up having a normalized array for each band with values 0-1  

3- Multiply x 255 which will transform the values to RGB  

4- As per the recommendation from GIS experts we only used bands B3,B4,B8 (Green, Red and NIR) as explained [here](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR#bands) these bands are bands 0,1 and 2 in the provided image. Band B2 (which is the blue band) is excluded from the photo and this will make the photo more red due to the NIR (Near Infra-red) which provide more information about the vegetation assessment of an area.
![image](https://user-images.githubusercontent.com/34656794/161365738-7157c6ff-a7fa-43a3-8556-3f4237ec4e15.png)

5- Use PIL image library to construct an image out of the 3 bands extracted in the previous step and save the new file as PNG  


# Image Rolling
Due to the imbalanced data and the very few images of the deprived images (label "1"), this code parse the coordinates dataframe and for label 1 and for each labeled box, it tries to find if it has an adjacent box horizontally or vertically by calclauting the difference between longitude and latitude and see if it equal to 10 pixel long. (Note: the pixel size was known from the image resolution in QGIS and the value 0.0000833 was used in our case). If an adjacent image was found, the code will roll across the two images and generate 9 intersecting images between these two boxes as per the following figure:  

<img src="https://user-images.githubusercontent.com/34656794/161366001-03d28f36-a579-44a7-a012-78cbf4f1491a.png" data-canonical-src="https://user-images.githubusercontent.com/34656794/161366001-03d28f36-a579-44a7-a012-78cbf4f1491a.png" width="800" height="500"/>

Due to the limited number of labels in this minority class, only 30 adjacent images were found and this generated 270 extra images.

# Image Shifting
The idea of image shifting is to take all images in the minority class (including the ones generated in the image rolling) and generate 8 new images by shifting the center of each image by 1 pixel in 4 directions and then 2 pixels in 4 directions as per the following image:  
  

 
<img src="https://user-images.githubusercontent.com/34656794/161366476-77da4e1f-42b4-4e3c-ae00-630da4373f08.png" data-canonical-src="https://user-images.githubusercontent.com/34656794/161366476-77da4e1f-42b4-4e3c-ae00-630da4373f08.png" width="800" height="400" />

# Over Sample by rotation
Final step to have enough labels for minority class is to take all images (original, rolled and shifted images) and rotate them by 90,180 and 270 degrees.  
<img src="https://user-images.githubusercontent.com/34656794/161366147-20782f9c-7827-4564-90c2-d17b48d9a975.png" data-canonical-src="https://user-images.githubusercontent.com/34656794/161366147-20782f9c-7827-4564-90c2-d17b48d9a975.png"/>


  
 
At the end, we will have enough labels from the minority class to proceed with a good training dataset to train different model in the next phase of the raw image processig and analysis:  
  
<img src="https://user-images.githubusercontent.com/34656794/161366201-598b737f-8354-4d0b-95e6-897b252ce808.png" data-canonical-src="https://user-images.githubusercontent.com/34656794/161366201-598b737f-8354-4d0b-95e6-897b252ce808.png" width="750" height="300" />  

# Working with the code:

The code in this folder should be executed after completing the coordinates extraction in the Extract_Coordinates.py which will end up with a csv file with all the coordinates and the corresponding label from the training data. Here should be the flow:  

1- Generate/download the map tiff file (in our case the map created from the code [here](https://github.com/arathinair11/Satellite-Imagery/tree/main/Lagos)  
2- Place it under your base path /Maps/  
3- **Run Clipping** Images file in both modes TEST and TRAIN  and use only two labels   
4- Now we have tif images extracted for both TEST and TRAIN under two different folders (each label has its own folder under Train or Test)  
5- Now we can **Image Rolling** which will scan the coordinates (mainly for the train csv) and detect adjacent images and generate new images by rolling through them  
6- After Image Rolling, run the **Image Shifting** which will also generate new images by scanning the coordinates for the minority class and shift the center by 1 or 2 pixel and generate new images from the original image  
7- Now run the **convet to PNG** To convert all images to png to prepare for training  







