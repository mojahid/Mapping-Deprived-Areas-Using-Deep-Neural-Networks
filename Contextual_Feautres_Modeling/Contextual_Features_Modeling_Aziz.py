


# Base Logistic Regression and Random Forest Models
# Things to work on: 
#----------PCA and Feature Reduction
# ---------Apply different oversampling Techniques 




# Data Manuiplation dependencies
# https://numpy.org/
import numpy as np
# https://pandas.pydata.org/
import pandas as pd 
#----------------------------------
# Visualisations dependencies
# https://matplotlib.org/
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#https://seaborn.pydata.org/
import seaborn as sns
#-------------------------------------
import warnings
import itertools
import os
#--------------------------------
sns.set(style="darkgrid")
# Modelling dependencies
#sklearn Dependencies
# All dependencies can be found here https://scikit-learn.org/stable/
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score,confusion_matrix, classification_report

# Ignore warnings
warnings.filterwarnings('ignore')

#-----------------------------------------------------------------------




# Some Useful Functions
def Distribution_plot(dataframe,feature, target, title, xlabel):
    """"
    Plot Distribution of feature for both classes
    
    Parameters
    ------------
    dataframe: Training dataframe
    feature: Variable to find distribution
    target: Dependent variable column containing all classes
    title: Plot title
    """
    
    # set the figure size
    plt.figure(figsize=(10,8))

    ax = plt.subplot()
    #distribution plot for all classes
    sns.distplot(dataframe[feature][dataframe[target] == 0], bins=50, label="Deprived Area")
    sns.distplot(dataframe[feature][dataframe[target] ==1], bins=50, label="Developed Area")
    sns.distplot(dataframe[feature][dataframe[target] ==2], bins=50, label="Water Area")

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)

    # Add legend and show plot
    ax.legend(title="Area classification")
    plt.show()


# In[57]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)        


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




#-------------------------------------------------------------------------------------------




context_df= pd.read_csv("Contextual_Features_final.csv")
print(context_df)




context_df= context_df.drop(columns=["Point","long", "lat", "long_x", "lat_x", "Label_x", "long_y","Label_y","lat_y"])




print(context_df.head())




#Label Distribution plot
sns.countplot('Label', data=context_df,)
plt.title('Label Distribution', fontsize=14)
plt.xlabel("Label")
plt.show()




print(context_df["Label"].value_counts())




# Feature Distribution for all classes  ( a good way to check which features have different distribution for classes)
# Example: Check ndvi fourier, gabor features. they seem like good predictors
# this can also be used to compare features within the same group of features
for i in context_df.columns:
    Distribution_plot(context_df, "{}".format(i), "Label", "{}".format(i), "{}".format(i))




# Checking for missing values
print(context_df.isna().sum())




# Split the data into training and test datasets while keeping the same class distribution
print('Developed Area', round(context_df['Label'].value_counts()[0]/len(context_df) * 100,2), '% of the dataset')
print('Deprived Area', round(context_df['Label'].value_counts()[1]/len(context_df) * 100,2), '% of the dataset')
print('Water Area', round(context_df['Label'].value_counts()[2]/len(context_df) * 100,2), '% of the dataset')


X = context_df.drop('Label', axis=1)
y = context_df['Label']
#using StratifiedKfold to generate (Train dataframe and  Test dataframe ) having the same class distribution that is similar to the orginial data

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
    


# Verify the Distribution of the labels


# Turn into an array
original_Xtrain_array = original_Xtrain.values
original_Xtest_array = original_Xtest.values
original_ytrain_array = original_ytrain.values
original_ytest_array = original_ytest.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_ytrain_array, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest_array, return_counts=True)
print('-' * 100)

print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain_array))
print(test_counts_label/ len(original_ytest_array))




print(original_Xtrain)


#-----------------------------------------------------------------------------------------------------------

# Standardize Features for training and test set
# The StandardScaler
ss = StandardScaler()
# Standardize the training data
X_train = ss.fit_transform(original_Xtrain)

# Standardize the test data
X_test = ss.transform(original_Xtest)

#--------------------------------------------------------------------------------------------


# Applying smote to oversample and balance classes
smote = SMOTE(random_state=42)

# Augment the training data
X_smote_train, y_smote_train = smote.fit_resample(X_train, original_ytrain)




print("Label Counts after oversampling")
y_smote_train.value_counts()


#-------------------------------------------------------------------------------------------------------

# Fitting Logistice Regression model
models = {'lr': LogisticRegression(class_weight='balanced', random_state=42)}

pipes = {}

for acronym, model in models.items():
    pipes[acronym] = Pipeline([('model', model)])
param_grids = {}

# Logistic Regression hyperparamters

# The parameter grid of tol
tol_grid = [10 ** -5, 10 ** -4, 10 ** -3]

# The parameter grid of C
C_grid = [0.1, 1, 10]

# Update param_grids
param_grids['lr'] = [{'model__tol': tol_grid,
                      'model__C': C_grid}]


# In[50]:


# Make directory to save results 
directory = os.path.dirname('result/cv_results/GridSearchCV/')
if not os.path.exists(directory):
    os.makedirs(directory)


# In[51]:


# The list of [best_score_, best_params_, best_estimator_] obtained by GridSearchCV
best_score_params_estimator_gs = []
# GridSearchCV
gs = GridSearchCV(estimator=pipes[acronym],
                      param_grid=param_grids[acronym],
                      scoring='f1_macro',
                      n_jobs=2,
                      cv=5,
                      return_train_score=True)
        
# Fit the pipeline
gs = gs.fit(X_smote_train, y_smote_train)
    
# Update best_score_params_estimator_gs
best_score_params_estimator_gs.append([gs.best_score_, gs.best_params_, gs.best_estimator_])
    
# Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
cv_results = pd.DataFrame.from_dict(gs.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])
    
# Get the important columns in cv_results
important_columns = ['rank_test_score',
                         'mean_test_score', 
                         'std_test_score', 
                         'mean_train_score', 
                         'std_train_score',
                         'mean_fit_time', 
                         'std_fit_time',                        
                         'mean_score_time', 
                         'std_score_time']
    
# Move the important columns ahead
cv_results = cv_results[important_columns + sorted(list(set(cv_results.columns) - set(important_columns)))]

# Write cv_results file
cv_results.to_csv(path_or_buf='result/cv_results/GridSearchCV/' + acronym + '.csv', index=False)

# Sort best_score_params_estimator_gs in descending order of the best_score_
best_score_params_estimator_gs = sorted(best_score_params_estimator_gs, key=lambda x : x[0], reverse=True)

# Print best_score_params_estimator_gs
pd.DataFrame(best_score_params_estimator_gs, columns=['best_score', 'best_param', 'best_estimator'])




best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]




# Predict using test data
y_test_pred = best_estimator_gs.predict(X_test)

# Get the precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(original_ytest, y_test_pred)


# Get the dataframe of precision, recall, fscore and auc
print(pd.DataFrame([[precision, recall, fscore]], columns=['Precision', 'Recall', 'F1-score']))




# Plotting confusion matrix obtained from the testing data predictions
sns.set(style="white")
cnf_matrix = confusion_matrix(original_ytest,y_test_pred)
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
class_names = ["0","1","2"]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()




print(classification_report(original_ytest,y_test_pred))




# Evaluate odds of each variable and sort by odds value
odds = np.exp(best_estimator_gs["model"].coef_[0])
pd.DataFrame(odds, original_Xtrain.columns, columns=['odds']).sort_values(by='odds', ascending=False)


#---------------------------------------------------------------------------------------------------------


models = {'rfc': RandomForestClassifier(class_weight='balanced', random_state=42)}
pipes = {}

for acronym, model in models.items():
    pipes[acronym] = Pipeline([('model', model)])
param_grids = {}
# Random Forest Hyper Parameters
# The grids for min_samples_split
min_samples_split_grids = [2, 20, 100]

# The grids for min_samples_leaf
min_samples_leaf_grids = [1, 20, 100]

# Update param_grids
param_grids['rfc'] = [{'model__min_samples_split': min_samples_split_grids,
                       'model__min_samples_leaf': min_samples_leaf_grids}]




# The list of [best_score_, best_params_, best_estimator_] obtained by GridSearchCV
best_score_params_estimator_gs = []
# GridSearchCV
gs = GridSearchCV(estimator=pipes[acronym],
                      param_grid=param_grids[acronym],
                      scoring='f1_macro',
                      n_jobs=2,
                      cv=5,
                      return_train_score=True)
        
# Fit the pipeline
gs = gs.fit(X_smote_train, y_smote_train)
    
# Update best_score_params_estimator_gs
best_score_params_estimator_gs.append([gs.best_score_, gs.best_params_, gs.best_estimator_])
    
# Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
cv_results = pd.DataFrame.from_dict(gs.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])
    
# Get the important columns in cv_results
important_columns = ['rank_test_score',
                         'mean_test_score', 
                         'std_test_score', 
                         'mean_train_score', 
                         'std_train_score',
                         'mean_fit_time', 
                         'std_fit_time',                        
                         'mean_score_time', 
                         'std_score_time']
    
# Move the important columns ahead
cv_results = cv_results[important_columns + sorted(list(set(cv_results.columns) - set(important_columns)))]

# Write cv_results file
cv_results.to_csv(path_or_buf='result/cv_results/GridSearchCV/' + acronym + '.csv', index=False)

# Sort best_score_params_estimator_gs in descending order of the best_score_
best_score_params_estimator_gs = sorted(best_score_params_estimator_gs, key=lambda x : x[0], reverse=True)

# Print best_score_params_estimator_gs
print(pd.DataFrame(best_score_params_estimator_gs, columns=['best_score', 'best_param', 'best_estimator']))




best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]




# Predict using test data
y_test_pred = best_estimator_gs.predict(X_test)

# Get the precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(original_ytest, y_test_pred)


# Get the dataframe of precision, recall, fscore and auc
print(pd.DataFrame([[precision, recall, fscore]], columns=['Precision', 'Recall', 'F1-score']))




# Plotting confusion matrix obtained from the testing data predictions
sns.set(style="white")
cnf_matrix = confusion_matrix(original_ytest,y_test_pred)
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
class_names = ["0","1","2"]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()




print(classification_report(original_ytest,y_test_pred))

