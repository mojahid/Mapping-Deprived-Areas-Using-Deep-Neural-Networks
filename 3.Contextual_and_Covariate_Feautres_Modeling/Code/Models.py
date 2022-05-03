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
from project_root import get_project_root
warnings.filterwarnings("ignore")

root = get_project_root()

def run_model(dataset ='', model='', features='', feature_count=50, classes = ''):

    if dataset == 'contextual': 
        # Read in dataframe and remove merged columns
        df = pd.read_csv(root / '1.Data' / 'Contextual_Features.csv')
        df = df.drop(columns=['long', 'lat', 'Point'])
        cols_to_move = ['Label']
        df = df[cols_to_move + [col for col in df.columns if col not in cols_to_move]]

        # Move Target to first column
        target = 'Label'
        first_col = df.pop(target)
        df.insert(0, target, first_col)

    elif dataset == 'covariate':
        # Read in dataframe and remove merged columns
        df = pd.read_csv(root / '1.Data' / 'Covariate_Features.csv')
        df.drop(['long','lat','Coordinates','Transformed_Long','Transformed_Lat','new_long','new_lat','Raster Value'],axis=1,inplace=True)
        df.dropna(inplace=True)
        df.rename(columns={'Band_1': ' fs_dist_fs_2020',
                           'Band_2': ' fs_dist_hf_2019',
                           'Band_3': ' fs_dist_hf1_2020',
                           'Band_4': ' fs_dist_market_2020',
                           'Band_5': ' fs_dist_mosques_2017',
                           'Band_6': ' fs_dist_school_2020',
                           'Band_7': ' fs_dist_school1_2018',
                           'Band_8': ' fs_dist_well_2018',
                           'Band_9': ' fs_electric_dist_2020',
                           'Band_10': ' in_dist_rd_2016',
                           'Band_11': ' in_dist_rd_intersect_2016',
                           'Band_12': ' in_dist_waterway_2016',
                           'Band_13': ' in_night_light_2016',
                           'Band_14': ' ph_base_water_2010',
                           'Band_15': ' ph_bio_dvst_2015',
                           'Band_16': ' ph_climate_risk_2020',
                           'Band_17': ' ph_dist_aq_veg_2015',
                           'Band_18': ' ph_dist_art_surface_2015',
                           'Band_19': ' ph_dist_bare_2015',
                           'Band_20': ' ph_dist_cultivated_2015',
                           'Band_21': ' ph_dist_herb_2015',
                           'Band_22': ' ph_dist_inland_water_2018',
                           'Band_23': ' ph_dist_open_coast_2020',
                           'Band_24': ' ph_dist_riv_network_2007',
                           'Band_25': ' ph_dist_shrub_2015',
                           'Band_26': ' ph_dist_sparse_veg_2015',
                           'Band_27': ' ph_dist_woody_tree_2015',
                           'Band_28': ' ph_gdmhz_2005',
                           'Band_29': ' ph_grd_water_2000',
                           'Band_30': ' ph_hzd_index_2011',
                           'Band_31': ' ph_land_c1_2019',
                           'Band_32': ' ph_land_c2_2020',
                           'Band_33': ' ph_max_tem_2019',
                           'Band_34': ' ph_ndvi_2019',
                           'Band_35': ' ph_pm25_2016',
                           'Band_36': ' ph_slope_2000',
                           'Band_37': ' po_pop_fb_2018',
                           'Band_38': ' po_pop_un_2020',
                           'Band_39': ' ses_an_visits_2016',
                           'Band_40': ' ses_child_stunted_2018',
                           'Band_41': ' ses_dpt3_2018',
                           'Band_42': ' ses_hf_delivery_2018',
                           'Band_43': ' ses_impr_water_src_2016',
                           'Band_44': ' ses_ITN_2016',
                           'Band_45': ' ses_m_lit_2018',
                           'Band_46': ' ses_measles_2018',
                           'Band_47': ' ses_odef_2018',
                           'Band_48': ' ses_pfpr_2017',
                           'Band_49': ' ses_preg_2017',
                           'Band_50': ' ses_unmet_need_2018',
                           'Band_51': ' ses_w_anemia_2018',
                           'Band_52': ' ses_w_lit_2018',
                           'Band_53': ' sh_dist_conflict_2020',
                           'Band_54': ' sh_dist_mnr_pofw_2019',
                           'Band_55': ' sh_dist_pofw_2019',
                           'Band_56': ' sh_ethno_den_2020',
                           'Band_57': ' sh_pol_relev_ethnic_gr_2019',
                           'Band_58': ' uu_bld_count_2020',
                           'Band_59': ' uu_bld_den_2020',
                           'Band_60': ' uu_impr_housing_2015',
                           'Band_61': ' uu_urb_bldg_2018'}, inplace=True)
        df.reset_index(inplace=True)
        del df['index']

        # removed Band 28 as there were nan values
        df.drop([' ph_gdmhz_2005'], axis=1, inplace=True)

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
            filename = 'Contextual_best_ada_boosting_features_0_1.csv'
            ada_features = pd.read_csv(root / '3.Contextual_and_Covariate_Feautres_Modeling'/ 'feature_selection' / 'Contextual' / f'{filename}')
        else:
            filename = 'Covariate_best_ada_boosting_features_0_1.csv'
            ada_features = pd.read_csv(root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'feature_selection' / 'Covariate' / f'{filename}')

        ada = ['Label']
        for row in range(feature_count):
            ada.append(ada_features.iloc[row, 0])

        df_ada = df[ada]
        X = df_ada.values[:, 1:]
        y = df_ada.values[:, 0]

    elif features == 'Random_Forest_Features':  # Random Forest feature selection

        if dataset == 'contextual':
            filename = 'Contextual_best_random_forest_features_0_1.csv'
            rf_features = pd.read_csv(root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'feature_selection' / 'Contextual' / f'{filename}')
        else:
            filename = 'Covariate_best_random_forest_features_0_1.csv'
            rf_features = pd.read_csv(root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'feature_selection' / 'Covariate' / f'{filename}')

        rf = ['Label']
        for row in range(feature_count):
            rf.append(rf_features.iloc[row,0])

        df_rf = df[rf]

        X = df_rf.values[:, 1:]
        y = df_rf.values[:, 0]

    elif features == 'Gradient_Boosting_Features':  # Random Forest feature selection

        if dataset == 'contextual':
            filename = 'Contextual_best_gradient_boosting_features_0_1.csv'
            gb_features = pd.read_csv(root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'feature_selection' / 'Contextual' / f'{filename}')
        else:
            filename = 'Covariate_best_gradient_boosting_features_0_1.csv'
            gb_features = pd.read_csv(root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'feature_selection' / 'Covariate' / f'{filename}')

        gb = ['Label']
        for row in range(feature_count):
            gb.append(gb_features.iloc[row,0])

        df_gb = df[gb]

        X = df_gb.values[:, 1:]
        y = df_gb.values[:, 0]

    elif features == 'Logistic_Features':  # Random Forest feature selection

        if dataset == 'contextual':
            filename = 'Contextual_best_logistic_features_0_1.csv'
            log_features = pd.read_csv(root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'feature_selection' / 'Contextual' / f'{filename}')
        else:
            filename = 'Covariate_best_logistic_features_0_1.csv'
            log_features = pd.read_csv(root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'feature_selection' / 'Covariate' / f'{filename}')

        log = ['Label']
        for row in range(feature_count):
            log.append(log_features.iloc[row,0])

        df_log = df[log]

        X = df_log.values[:, 1:]
        y = df_log.values[:, 0]

    elif features == 'Minfo_Features':  # Random Forest feature selection

        if dataset == 'contextual':
            filename = 'Contextual_minfo_features_0_1.csv'
            minfo_features = pd.read_csv(root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'feature_selection' / 'Contextual' / f'{filename}')
        else:
            filename = 'Covariate_minfo_features_0_1.csv'
            minfo_features = pd.read_csv(root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'feature_selection' / 'Covariate' / f'{filename}')

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
    directory = root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'Saved_Models'
    # model filename
    filename = f'{dataset}_{model}_model_{feature_count}{features}_{classes}.sav'

    if not os.path.exists(directory / filename):
        if model == 'MLP':
            # Hyper-parameter space
            '''
            parameter_space = {
                'hidden_layer_sizes': [(60, 100, 60), (100, 100, 100), (50, 100, 50)],
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
            pickle.dump(clf, open(root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'Saved_Models' / f'{filename}','wb'))

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
            pickle.dump(clf, open(root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'Saved_Models' / f'{filename}','wb'))

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
            pickle.dump(clf,open(root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'Saved_Models' / f'{filename}','wb'))

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
            pickle.dump(clf, open(root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'Saved_Models' / f'{filename}', 'wb'))

        # Load Model
        filename = f'{dataset}_{model}_model_{feature_count}{features}_{classes}.sav'
        loaded_model = pickle.load(open(root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'Saved_Models' / f'{filename}', 'rb'))

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
        filename = f'{dataset}_{model}_model_{feature_count}{features}_{classes}.sav'
        loaded_model = pickle.load(open(root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'Saved_Models' / f'{filename}', 'rb'))

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


