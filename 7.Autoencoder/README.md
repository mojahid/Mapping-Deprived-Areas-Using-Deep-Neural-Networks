## Mapping Deprived Areas In Low & Middle Income Countries Using Satellite Images

## About
This repository consist of Python code for extracting cloud-free Sentinel-2 images from Google Earth Engine and implementation of CNN & MLP Autoencoder for image reconstruction and training image classification usng Pytorch.

#### Dependencies
```
Python package installations
- Python3, Scikit-learn, Pytorch, Gdal
- PIL, Rasterio, Scikit-image
```
```
To run Google Earth Engine on Google Colab
- pip install earthengine-api
- pip install folium
```
#### Remote Server
```
All files are saved on World Bank server and can be accessed from (/home/ubuntu/Autoencoder/Final) path.
```
## Files 

#### **1. Cloud Free Sentinel-2 Image Extraction**
- Create a [Google Earth Engine](https://earthengine.google.com) account before running the code.

  - Lagos_Cloud_Free_Satellite_Image.ipynb :  Code for extracting cloud free Sentinel-2 image for Lagos in TIF format.
  - Accra_Cloud_Free_Satellite_Image.ipynb : Code for extracting cloud free Sentinel-2 image for Accra in TIF format.
  - Nairobi_Cloud_Free_Satellite_Image.ipynb : Code for extracting cloud free Sentinel-2 image for Nairobi in TIF format
   - Parameters chosen for image fine-tuning : *START_DATE,END_DATE,CLOUD_FILTER,CLD_PRB_THRESH,CLD_PRJ_DIST,BUFFER*
- Image_Extraction.py : Clipping 10 x 10 pixels from TIF files.
- ConvertToPNG.py : Converting 10 x 10 pixel TIF images to PNG format.
- Image_Filteration.py : Flitering images less than 10 x 10 px or blank images.
- Combine_Satellite_PNG.py : Combines all final PNG images for Lagos, Accra, Nairobi into a single folder.
- [TIF Files](https://drive.google.com/drive/folders/1y-t8iV_hT73FOQrflBfAui3L1wc6osST?usp=sharing) : Link to download the final Cloud-Free TIF images for Lagos, Accra and Nairobi. The files can also be downloaded from the World Bank server *(/home/ubuntu/Autoencoder/Final/TIF_Files)*.

#### **2. Image Reconstruction**
- CNN_Autoencoder.py : Code for image reconstruction using Convolutional Autoencoder
- MLP_Autoencoder.py : Code for image reconstruction using  Multilayer Perceptron Autoencoder
- Satellite Images : The clipped 1030089 satellite images (10 x 10 px) in formats TIF & PNG can be dowloaded from the World Bank server *(/home/ubuntu/Autoencoder/Final/Satellite_Images)*.


#### **3. Image Classification** 
- Image_Classification_CNN.py : Classification of labeled images in the training dataset using the saved CNN encoder model
- Image_Classification_MLP.py : Classification of labeled images in the training dataset using the saved MLP encoder model
- Training Dataset : The labeled dataset for Lagos and Accra can be downloaded from the World Bank server *(/home/ubuntu/Autoencoder/Final/Training_Dataset)*.
