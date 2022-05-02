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

## Steps for Processing and Running Covariate and Contextual feature Importance and Modeling

1. In folder ‘1.Data’  run download.py
    * This will create coordinates.csv file labeled data of ‘Deprived’, ‘Built-up’, ‘Not-Built-up’
2. For contextual features
    * In folder ‘2.Data_Processing/Contextual_features_Processing’
        *  run Contextual_Feature_Extraction.py
        * Then run Contextual_Feature_Merging.py
            * This code extracts the contextual features that match with the labeled coordinates and merges them all into one csv file 
    * In folder ‘3.Contextual_and_Covariate_Features_Modeling/Code/
        * Run Contextual_Feature_Importance_0_1.py
            * This will run feature importance methods for the contextual features and store results in ‘3.Contextual_and_Covariate_Features_Modeling/feature_selection/Contextual’
  3. For Covariate Features
    * In folder ‘2.Data_processing/Covariate_Features_Processing’ run covariate_extraction.py
        * This code will process the covariate data into a csv
    * In folder ‘3.Contextual_and_Covariate_Features_Modeling.py’  run Covariate_Feature_Importance_0_1.py
            * This will run feature importance methods for the covariate features and store results in ‘3.Contextual_and_Covariate_Features_Modeling/feature_selection/Covariate’
  4. Run Contextual_Ensemble_Model.py
    * This file will run ensemble modeling for the contextual features
  5. RunModels.py
    * This file will run modeling for both Contextual and Covariate Features

