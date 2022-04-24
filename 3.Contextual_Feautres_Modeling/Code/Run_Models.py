# Script for running the run_model function

import pandas as pd
from Models import run_model


# all potential parameters for model testing
datasets = ['contextual', 'covariate']
models = ['MLP', 'Gradient_Boosting', 'Logistic_Regression', 'Random_Forest']
features = ['All_Features', 'PCA_Features', 'Random_Forest_Features']
# feature_count = 2 #Any integer. Not required if using full feature set(use '' to remove count from titles)
classes = ['all_classes', 'classes_0&1']
pretrained = False #{True: Use if the model has been trained and saved as a .sav, False: Use if model needs to be created/trained and saved as a .sav]

# desired model parameter combinations
datasets = ['contextual']
models = ['MLP', 'Gradient_Boosting', 'Random_Forest', 'Logistic_Regression']
features = ['All_Features', 'PCA_Features', 'Random_Forest_Features']
# feature_count = 50 # only applies when features == 'PCA_Features' or 'Random_Forest_Features'
classes = ['classes_0&1']
# pretrained = False #{True: Use if the model has been trained and saved as a .sav, False: Use if model needs to be created/trained and saved as a .sav]

dataset_list = []
model_list = []
features_list = []
feat_count_list = []
classes_list = []
deprived_list = []
macro_list = []


# iterate through desired models
for dataset in datasets:
    for model in models:
        for feat in features:
            if feat == 'All_Features':
                feature_count = 144
            else:
                feature_count = 50
            for class_count in classes:
                stored_model_info = run_model(dataset=dataset,
                                              model=model,
                                              features=feat,
                                              feature_count=feature_count,
                                              classes=class_count)
                stored_model_info # prints classification report table and confusion matrix image
                dataset_list.append(dataset) # dataset used
                model_list.append(model) # model name
                features_list.append(feat) # feature set used
                feat_count_list.append(feature_count) # number of features trained on
                classes_list.append(class_count) # classes trained and tested on
                deprived_list.append(stored_model_info[1]) # f1 score for class 1 (deprived)
                macro_list.append(stored_model_info[2]) # macro f1 score

# store data in dataframe
data = {'Dataset': dataset_list,
        'Model': model_list,
        'Feature Set': features_list,
        'Feature Count': feat_count_list,
        'Classes': classes_list,
        'F1 - Class 1 (Deprived)': deprived_list,
        'F1 - Macro': macro_list}
comparison_table = pd.DataFrame(data)
print(comparison_table)

# write dataframe to csv
comparison_table.to_csv('Model_Results_Comparison_Table.csv', index=False)

