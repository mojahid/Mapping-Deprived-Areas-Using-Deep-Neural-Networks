import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
import warnings
warnings.filterwarnings("ignore")

def run_model(dataset ='', model='', features='', feature_count=50, classes = ''):

    if dataset == 'contextual': 
        # Read in dataframe and remove merged columns
        df = pd.read_csv(r'1.Data/Contextual_Features.csv')
        df = df.drop(columns=['long', 'lat', 'Point'])
        cols_to_move = ['Label']
        df = df[cols_to_move + [col for col in df.columns if col not in cols_to_move]]

        # Move Target to first column
        target = 'Label'
        first_col = df.pop(target)
        df.insert(0, target, first_col)

    elif dataset == 'covariate':
        # Read in dataframe and remove merged columns
        df = pd.read_csv(r'1.Data/Covariate_Features.csv')
        df.drop(['long', 'lat'], axis=1, inplace=True)
        df.dropna(inplace=True)
        cols_to_move = ['Label']
        df = df[cols_to_move + [col for col in df.columns if col not in cols_to_move]]

        # Move Target to first column
        target = 'Label'
        first_col = df.pop(target)
        df.insert(0, target, first_col)

    if classes == 'all_classes':
        df = df
    elif classes == 'classes_0&1':
        # set label column equal to only 0 and 1 classes
        df = df[df['Label'].isin([0, 1])]

    # define target and independent features
    if features == 'All_Features': # full dataset, all features

        feature_count = ''
        X = df.values[:, 1:]
        y = df.values[:, 0]

    elif features == 'ADA_Features': # PCA feature selection

        if dataset == 'contextual':
            ada_features = pd.read_csv(r'3.Contextual_and_Covariate_Feautres_Modeling/feature_selection/Contextual/Contextual_best_ada_boosting_features_0_1.csv')
        else:
            ada_features = pd.read_csv(r'3.Contextual_and_Covariate_Feautres_Modeling/feature_selection/Covariate/Covariate_best_ada_boosting_features_0_1.csv')


        ada = ['Label']
        for row in range(feature_count):
            ada.append(ada_features.iloc[row, 0])

        df_ada = df[ada]
        X = df_ada.values[:, 1:]
        y = df_ada.values[:, 0]

    elif features == 'Random_Forest_Features':  # Random Forest feature selection

        if dataset == 'contextual':
            rf_features = pd.read_csv(r'3.Contextual_and_Covariate_Feautres_Modeling/feature_selection/Contextual/Contextual_best_random_forest_features_0_1.csv')
        else:
            rf_features = pd.read_csv(r'3.Contextual_and_Covariate_Feautres_Modeling/feature_selection/Covariate/Covariate_best_random_forest_features_0_1.csv')

        rf = ['Label']
        for row in range(feature_count):
            rf.append(rf_features.iloc[row,0])

        df_rf = df[rf]

        X = df_rf.values[:, 1:]
        y = df_rf.values[:, 0]

    elif features == 'Gradient_Boosting_Features':  # Random Forest feature selection

        if dataset == 'contextual':
            gb_features = pd.read_csv(r'C:\Users\brear\OneDrive\Desktop\Grad School\Data-Science-Capstone\Contextual_Feautres_Modeling\feature_selection\Contextual_features\Contextual_best_gradient_boosting_features_0_1.csv')
        else:
            gb_features = pd.read_csv(r'C:\Users\brear\OneDrive\Desktop\Grad School\Data-Science-Capstone\Contextual_Feautres_Modeling\feature_selection\Covariate_features\Covariate_best_gradient_boosting_features_0_1.csv')

        gb = ['Label']
        for row in range(feature_count):
            gb.append(gb_features.iloc[row,0])

        df_gb = df[gb]

        X = df_gb.values[:, 1:]
        y = df_gb.values[:, 0]

    elif features == 'Logistic_Features':  # Random Forest feature selection

        if dataset == 'contextual':
            log_features = pd.read_csv(r'3.Contextual_and_Covariate_Feautres_Modeling/feature_selection/Contextual/Contextual_best_gradient_boosting_features_0_1.csv')
        else:
            log_features = pd.read_csv(r'3.Contextual_and_Covariate_Feautres_Modeling/feature_selection/Covariate/Covariate_best_gradient_boosting_features_0_1.csv')

        log = ['Label']
        for row in range(feature_count):
            log.append(log_features.iloc[row,0])

        df_log = df[log]

        X = df_log.values[:, 1:]
        y = df_log.values[:, 0]

    elif features == 'Minfo_Features':  # Random Forest feature selection

        if dataset == 'contextual':
            minfo_features = pd.read_csv(r'3.Contextual_and_Covariate_Feautres_Modeling/feature_selection/Contextual/Contextual_minfo_features_0_1.csv')
        else:
            minfo_features = pd.read_csv(r'3.Contextual_and_Covariate_Feautres_Modeling/feature_selection/Covariate/Covariate_minfo_features_0_1.csv')

        minfo = ['Label']
        for row in range(feature_count):
            minfo.append(minfo_features.iloc[row,0])

        df_minfo = df[minfo]

        X = df_minfo.values[:, 1:]
        y = df_minfo.values[:, 0]


    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,
                                                      random_state=42)  # 0.25 x 0.8 = 0.2

    # Feature Scaling
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    X_val = sc.transform(X_val)


    # Model file saved and exists externally

    # directory path models are saved in
    directory = r'3.Contextual_and_Covariate_Feautres_Modeling/Saved_Models'
    # model filename
    filename = f'{dataset}_{model}_model_{feature_count}{features}_{classes}.sav'

    if not os.path.exists(directory + '\\' + filename):
        if model == 'MLP':
            # Hyper-parameter space
            '''
            parameter_space = {
                'hidden_layer_sizes': [c],
                'activation': ['identity', 'relu', 'logistic', 'tanh'],
                'solver': ['sgd', 'adam', 'lbfgs'],
                'alpha': [0.0001, 0.00001, 0.000001],
                'learning_rate': ['constant', 'adaptive', 'invscaling'],
            }
            '''

            parameter_space = {
                'hidden_layer_sizes': [(60, 100, 60), (100, 100, 100), (50, 100, 50)],
                'activation': ['identity', 'relu', 'logistic', 'tanh'],
                'solver': ['adam'],
                'alpha': [0.0001],
                'learning_rate': ['invscaling'],
            }

            # Create network
            clf = MLPClassifier(max_iter=1000000)

            # Run Gridsearch
            clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3)

            clf.fit(X_train, y_train)
            clf_pred = clf.predict(X_test)
            print(f"Test Results Using {dataset} dataset {model} Best Params, {feature_count}{features}, and {classes}: \n")
            print("Classification Report: ")
            print(classification_report(y_test, clf_pred))

            # Best parameter set
            print(f'Best parameters found for {model}:\n', clf.best_params_)

            # Save model
            filename = f'{dataset}_{model}_model_{feature_count}{features}_{classes}.sav'
            pickle.dump(clf, open(filename, 'wb'))

        elif model == "Gradient_Boosting":
            # Hyper-parameter space
            '''
            parameter_space = {
                'loss': ['deviance'],
                'criterion': ['friedman_mse', 'squared_error', 'mse'],
                'n_estimators': [100, 200, 50],
                'subsample': [1.0, 0.8, 0.6],
                "learning_rate": [0.01, 0.025, 0.05],
                "min_samples_split": np.linspace(0.1, 0.5, 12),
                "min_samples_leaf": np.linspace(0.1, 0.5, 12),
                "max_depth": [3, 5, 8],
                "max_features": ["log2", "sqrt"],
            }
            

            parameter_space = {
                'loss': ['deviance'],
                'criterion': ['friedman_mse', 'mse'],
                'n_estimators': [100],
                'subsample': [1.0, 0.6],
                "learning_rate": [0.01, 0.05],
                "min_samples_split": np.linspace(0.1, 0.5, 3),
                "min_samples_leaf": np.linspace(0.1, 0.5, 3),
                "max_depth": [3, 8],
                "max_features": ["log2", "sqrt"],
            }
            '''
            clf = GradientBoostingClassifier()

            # Run Gridsearch
            # clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3)

            clf.fit(X_train, y_train)
            clf_pred = clf.predict(X_test)
            print(f"Test Results Using {model}, {feature_count}{features}, and {classes}: \n")
            print("Classification Report: ")
            print(classification_report(y_test, clf_pred))

            # Best parameter set
            # print(f'Best parameters found for {model}:\n', clf.best_params_)

            # Save model
            filename = f'{dataset}_{model}_model_{feature_count}{features}_{classes}.sav'
            pickle.dump(clf, open(filename, 'wb'))

        elif model == "Logistic_Regression":
            # Logistic Regression Hyper-parameter space
            parameter_space = {
                'penalty': ['l1', 'l2','elasticnet', 'none'],
                'dual': [True, False],
                'C': [0.001, 0.01, 0.1, 1, 10],
                'class_weight': ['dict', 'balanced', None],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            }

            clf = LogisticRegression()

            # Run Gridsearch
            clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3)

            clf.fit(X_train, y_train)
            clf_pred = clf.predict(X_test)
            print(f"Test Results Using {dataset} dataset {model} Best Params, {feature_count}{features}, and {classes}: \n")
            print("Classification Report: ")
            print(classification_report(y_test, clf_pred))

            # Best parameter set
            print(f'Best parameters found for {model}:\n', clf.best_params_)

            # Save model
            filename = f'{dataset}_{model}_model_{feature_count}{features}_{classes}.sav'
            pickle.dump(clf, open(filename, 'wb'))

        elif model == "Random_Forest":
            # Hyper-parameter space
            '''
            parameter_space = {
                'criterion': ['gini', 'entropy'],
                'n_estimators': [100, 200],
                "min_samples_split": np.linspace(0.1, 0.5, 3),
                "min_samples_leaf": np.linspace(0.1, 0.5, 3),
                "max_depth": [2, 10, 20],
                "max_features": ["log2", "sqrt", 'auto'],
            }
            '''
            clf = RandomForestClassifier()

            # Run Gridsearch
            # clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3)

            clf.fit(X_train, y_train)
            clf_pred = clf.predict(X_test)
            print(f"Test Results Using {dataset} dataset {model}, {feature_count}{features}, and {classes}: \n")
            print("Classification Report: ")
            print(classification_report(y_test, clf_pred))

            # Best parameter set
            # print(f'Best parameters found for {model}:\n', clf.best_params_)

            # Save model
            filename = f'{dataset}_{model}_model_{feature_count}{features}_{classes}.sav'
            pickle.dump(clf, open(filename, 'wb'))

        # Load Model
        stored_path = r'3.Contextual_and_Covariate_Feautres_Modeling/Saved_Models'
        filename = f'{dataset}_{model}_model_{feature_count}{features}_{classes}.sav'
        loaded_model = pickle.load(open(f'{stored_path}/{filename}', 'rb'))

        # Predict on validation set
        val_pred = loaded_model.predict(X_val)
        print(f"Validation Results Using {dataset} dataset, {model}, {feature_count}{features}, and {classes}: \n")
        print("Classification Report: ")
        print(classification_report(y_val, val_pred))
        cf_matrix = confusion_matrix(y_val, val_pred)
        print(cf_matrix)
        sns.heatmap(cf_matrix, annot=True, fmt="d")
        plt.title(f'{model} Confusion Matrix - {feature_count}{features}, {classes}')
        plt.show()

        # f1 scores for comparison table output
        f1_micro_class0 = f1_score(y_val, val_pred, average=None)[0]
        f1_micro_class1 = f1_score(y_val, val_pred, average=None)[1]
        f1_macro = f1_score(y_val, val_pred, average='macro')

        return f1_micro_class0, f1_micro_class1, f1_macro


    # Model already saved as external file
    else:
        # Load Model
        stored_path = r'3.Contextual_and_Covariate_Feautres_Modeling/Saved_Models'
        filename = f'{dataset}_{model}_model_{feature_count}{features}_{classes}.sav'
        loaded_model = pickle.load(open(f'{stored_path}/{filename}', 'rb'))

        # Predict on validation set
        val_pred = loaded_model.predict(X_val)
        print(f"Validation Results Using {dataset} dataset, {model}, {feature_count}{features}, and {classes}: \n")
        print("Classification Report: ")
        print(classification_report(y_val, val_pred))
        cf_matrix = confusion_matrix(y_val, val_pred)
        print(cf_matrix)
        sns.heatmap(cf_matrix, annot=True, fmt="d")
        plt.title(f'{model} - {feature_count} {features}, {classes}')
        plt.tight_layout()
        plt.show()

        if model == 'MLP' or model == 'Logistic_Regression':
            print(f'Best parameters found for {model}:\n', loaded_model.best_params_)

        # f1 scores for comparison table output
        f1_micro_class0 = f1_score(y_val, val_pred, average=None)[0]
        f1_micro_class1 = f1_score(y_val, val_pred, average=None)[1]
        f1_macro = f1_score(y_val, val_pred, average='macro')

        return f1_micro_class0, f1_micro_class1, f1_macro


