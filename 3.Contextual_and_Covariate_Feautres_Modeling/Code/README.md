# Contextual and Covariate Feature Folder

Description of each file:
  1. Contextual_Feature_Importance_0_1.jpynb
      
      * This is a jupyter notebook that details EDA and feature importance analysis for Contextual Features

  2. Contextual_Feature_Importance_0_1.py
      * this is the .py version of the jupyter notebook file (Contextual_Feature_Importance_0_1.jpynb)

  3. Contextual_Features_modeling_update.py
    
      * This is additional analysis for contextual features that was not used in the final report
  4. Covariate_Data_Plot.jpynb 
      * creates a file 'Covariate_Predictions.csv' that is stored in the 1.Data folder of the coordinates for the MLP predictions on the Covariate data
  
  5. Covariate_Data_Plot.py
      * Same thing as Covariate_Data_Plot.jpynb just in .py format

  6. Models.py
    
      * Contains all models to be run for analyzing the Contextual Features
    
  7. Contextual_Ensemble_Model.py

      * Ensembles desired models for the contextual features dataset testing.
      * Stores model in Saved_Models directory 
      * Run before 'Run_Model.py' to include in results output table.
      * creates 'Contextual_Ensemble_Predictions.csv' which has the coordiantes for model predictions to be inputted into QGIS
    
  8. Run_Models.py
   
      * Runs the file Models.py1
     
  9. Covariate_Feature_Importance_0_1.py
     
     * .py version of jupyter notebook file

  10. Covariate_Features_Importance_0_1.jpynb
     
    * jupyter notebook of feature importance methodology on covariate features
     
  11.  project_root:
     * This python file contains the code to path throughout the project for all users, regardless of system used to operate the code.


## Pipeline for Contextual Feature Importance
<img width="628" alt="Contextual_diagram" src="https://user-images.githubusercontent.com/60163434/165119981-01fb84d6-42ea-40fc-b147-c45991ae2185.png">

## Pipeline for Covariate Feature Importance
<img width="625" alt="Covariate_diagram" src="https://user-images.githubusercontent.com/60163434/165120931-8a15a858-6812-43e5-9eb5-3d12d6ef5442.png">



