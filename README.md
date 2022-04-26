# Mapping-Deprived-Areas-Using-Deep-Neural-Networks

* [According to the UN](https://unstats.un.org/sdgs/report/2019/goal-11/) more than 1 billion people are living in deprived areas

* Policy makers, government and global organizations are seeking detailed identification of deprived areas to enhance assignment of resources and track progress of development projects 

The goal of this project is to develop a methodology to map of deprived areas using a range of geospatial data at
approximately 100m grid cells. 

The project is divided in several sub-streams where each is focused on certain aspect of the data and applies relevant analysis and machine learning techniques.

<img src = "https://user-images.githubusercontent.com/34656794/165091702-398c5a32-69bf-4d95-b376-15c093dc0cf9.png" width="600" hight="450">  

## The data
<img src = "https://user-images.githubusercontent.com/34656794/165094527-8ae4b4a6-3567-4136-a0c4-6408548cb570.png" width="800" hight="600">  



## Navigating through the repo:

* The data folder contains necessary instrcutions to download the data
* Data Processing folders contains the label extraction code and pre-processing of the data for each of the sub-streams (raw images, contextual and covariate)
* Contextual features modeling: contains modeling coding and results based on the contextual features
* Covariate features modeling: contains modeling coding and results based on covariate features
* Raw images modeling: contains modeling coding and results based on raw images 
* Open Building: contains modeling and special data processing based on google open building dataset
* Ensemble model: ensembling best models in raw image modeling in one ensemble model to boost the performance



