# Script for running the run_model function

from Models_Contextual import run_model


# Parameter options
# models: ['MLP', 'Gradient_Boosting', 'Logistic_Regression', 'Random_Forest'
# features: ['All_Features', 'PCA_Features', 'Random_Forest_Features', 'Logistic_Regression_Features']
# feature_count: Any integer. Not required if using full feature set(use '' to remove count from titles)
# classes: ['all_classes', 'classes_0&1']
# pretrained: {True: Use if the model has been trained and saved as a .sav, False: Use if model needs to be created/trained and saved as a .sav]


# MLP - All Features - All Classes
run_model(model='MLP', features='All_Features', feature_count='', classes='classes_0&1', pretrained=True)

# Gradient Boosting - All Features - All Classes
run_model(model='Gradient_Boosting', features='All_Features', feature_count='', classes='classes_0&1', pretrained=False)

# Logistic Regression - All Features - All Classes
run_model(model='Logistic_Regression', features='All_Features', feature_count='', classes='classes_0&1', pretrained=False)

# Random Forest - All Features - All Classes
run_model(model='Random_Forest', features='All_Features', feature_count='', classes='classes_0&1', pretrained=False)