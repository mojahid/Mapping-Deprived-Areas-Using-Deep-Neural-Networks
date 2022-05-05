# Contextual Features Modeling

This section contains three folders: 
  1) Code - the analysis for the contextual and covariate features
  2) Results - model results for the contextual features
  3) feature_selection - contains two folders 'Contextual' and 'Covariate'. Each folder stores the five different feature importance methods
  4) Saved Models - All saved trained models for contextual and covariate features.

## Steps for Processing and Running Covariate and Contextual feature Importance and Modeling

1. In folder ‘1.Data’  run download.py
    * This will create coordinates.csv file labeled data of ‘Deprived’, ‘Built-up’, ‘Not-Built-up’
2. For contextual features
    * In folder ‘2.Data_Processing/Contextual_features_Processing’
        *  run Contextual_Feature_Extraction.py
            * This code will create 144 csv files for each of the contextual features
        * Then run Contextual_Feature_Merging.py
            * This code merges all contextual csv files into one csv file 
    * In folder ‘3.Contextual_and_Covariate_Features_Modeling/Code'
        * Run Contextual_Feature_Importance_0_1.py
            * This code will run feature importance methods for the contextual features and store results in ‘3.Contextual_and_Covariate_Features_Modeling/feature_selection/Contextual’
  3. For Covariate Features
      * In folder ‘2.Data_processing/Covariate_Features_Processing’ run covariate_extraction.py
        * This code will process the covariate data into a csv
      * In folder ‘3.Contextual_and_Covariate_Features_Modeling/Code'  run Covariate_Feature_Importance_0_1.py
        * This code will run feature importance methods for the covariate features and store results in ‘3.Contextual_and_Covariate_Features_Modeling/feature_selection/Covariate’
  4. Run Contextual_Ensemble_Model.py
      * This code will run ensemble modeling for the contextual features
  5. RunModels.py
      * This code will run modeling for both contextual and covariate features
