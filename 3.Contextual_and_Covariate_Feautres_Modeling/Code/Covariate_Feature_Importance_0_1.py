#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif

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
pd.set_option('display.max_columns', 90) # display all column of dataframe
pd.set_option('display.max_row', 100)
pd.set_option('display.max_colwidth', -1) 
#-----------------------------------------------------------------------


# # Table of Contents

#     * EDA
#     
#     * Correlation Map
#      
#     * Splitting and standardizing data for analysis
#     
#     * Mutual Feature Selection
# 
#     * Random Forest Model with Test Data
#     
#     * Random Forest Model with Validation Data
#        
#     * Random Forest Feature selection
#         
#     * Logistic Model with Test Data
# 
#     * Logistic Model with Validation Data
#     
#     * Logistic Feature selection
#     
#     * Gradient Boosting Model with Test Data     
# 
#     * Gradient Boosting Model with Validation Data
#     
#     * Gradient Boosting Feature selection
#     
#     * AdaBoosting Model with Test Data     
# 
#     * AdaBoosting Model with Validation Data
#     
#     * AdaBoosting Feature selection
#     
#     * Comparing Features
#     

# # EDA 

# In[2]:


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


# In[3]:


def count_values_in_column(data,feature):
    total=data.loc[:,feature].value_counts(dropna=False)
    percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])


# In[4]:


# import data and clean it
df = pd.read_csv(r'1.Data/Covariate_Features.csv')
#df = pd.read_csv('Covariate_Features.csv')
df.drop(['long','lat','Coordinates','Transformed_Long','Transformed_Lat','new_long','new_lat','Raster Value'],axis=1,inplace=True)
print('there are', df.shape[1], 'columns in the original dataframe')
print('there are', df.shape[0],'values in the original dataframe')
df.head()


# In[5]:


df.rename(columns= {'Band_1': ' fs_dist_fs_2020',
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
 'Band_61': ' uu_urb_bldg_2018'},inplace=True)
df.reset_index(inplace=True)
del df['index']
df.head()
#df.to_csv('Covariates_w_names.csv',index=False)


# In[6]:


# create pie chart data
pie_data= count_values_in_column(df,"Label") # save data aS a dataframe
count_values_in_column(df,"Label")


# In[7]:


plt.figure(figsize=(10,10))
labels = ['not-built-up','Built-up', 'Deprived']
plt.title('Overview of Labeled Covariate Coordinates before Removing NaN Values', fontsize=30)
plt.pie(pie_data['Total'], labels=labels, autopct='%1.1f%%', textprops={'fontsize': 20})
plt.rcParams["axes.labelweight"] = "bold"
plt.tight_layout()
plt.show


# In[8]:


#checking NAN on Covariate data values
null_values = df[df.isnull().any(axis=1)]
print('there are',df[df.isnull().any(axis=1)].shape[0], 'nan values in the dataframe')
print(null_values['Label'].value_counts())
df.dropna(inplace=True)
# removed Band 28 as there were nan values
df.drop([' ph_gdmhz_2005'], axis=1,inplace = True)

print('there are ',df.shape[0],'rows of data after removing nan values')


# In[109]:


nan_values = pd.DataFrame(null_values.index,columns=['index_values'])
nan_values.to_csv('covariate_null_values.csv',index=False)


# In[9]:


pie_data= count_values_in_column(df,"Label") # save data aS a dataframe
count_values_in_column(df,"Label")


# In[10]:


plt.figure(figsize=(10,10))
labels = ['not-built-up','Built-up', 'Deprived']
plt.title('Overview of Labeled Covariate Coordinates after Removing NaN Values', fontsize=30)
plt.pie(pie_data['Total'], labels=labels, autopct='%1.1f%%', textprops={'fontsize': 20})
plt.rcParams["axes.labelweight"] = "bold"
plt.tight_layout()
plt.show


# In[11]:


#heat map on Covariate Features
plt.figure(figsize=(150, 150))
plt.title('Correlation Heat Map of Covariate Features\n', fontsize= 250)
sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, cmap='YlOrRd', square=True)
plt.tight_layout()
plt.show()


# In[12]:



#df.drop('Label',axis=1).corr()
df.corr()
df.shape


# In[13]:


# create dataframe 'df_corr' of all correlation values
# correlation values on for 0 and 1
df_corr =df[df['Label'].isin([0,1])]
# remove 'label' column from correlation datafram
df_corr = df_corr.drop('Label',axis=1)
print(df_corr.shape)
# unstack correlation matrix
correlation = pd.DataFrame(df_corr.corr().unstack().sort_values(ascending=False).reset_index())
# remove rows that are correlated with themselves as the correlation values would be 1
corr = correlation.loc[lambda x : x['level_0'] != x['level_1']].reset_index(drop=True)
# rename correlation row
corr = corr.rename(columns={0: 'Correlation_Values'})
corr.sort_values(by = 'level_0',ascending=False).reset_index()
corr.drop_duplicates(subset='Correlation_Values',inplace=True)
# remove every odd numbered index as it is the same value of the even cell above it
print('the skew of the correlation coefficient values for covariate features is', corr.skew())
corr.shape


# In[14]:


#created Histogram of Correlation Values
plt.figure(figsize= (15,10), facecolor='white')
plt.rcParams["figure.figsize"] = (20,15)
corr.plot(kind='hist',bins=15, color ='#ff5349')
plt.title('Distribution of Correlation Values on Covariate Features',fontsize=30)
plt.xlabel('Correlation Coefficeint Values',fontsize=25)
plt.ylabel('Frequency',fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(False)
plt.legend().remove()
#plt.style.use('classic')
plt.tight_layout()
plt.show()


# # Splitting and standardizing data for analysis

# In[15]:


# Make directory to save results 
directory = os.path.dirname('result/cv_results/GridSearchCV/')
if not os.path.exists(directory):
    os.makedirs(directory)


# In[16]:


# select 0 and 1 classes
df =df[df['Label'].isin([0,1])]
X = df.drop('Label', axis=1)
y = df['Label']
# train, val, test split 60/20/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2


# In[17]:


# create pie chart data
pie_data= count_values_in_column(df,"Label") # save data aS a dataframe
count_values_in_column(df,"Label")


# In[18]:


#created Pie chart after removing Not-Built-up areas
plt.figure(figsize= (10,10))
labels = ['Built-up', 'Deprived']
plt.title("Overview of Area Descriptions After Removing 'Not-Built-up' Areas", fontsize=30)
plt.pie(pie_data['Total'], labels=labels, autopct='%1.1f%%', textprops={'fontsize': 20})
plt.rcParams["axes.labelweight"] = "bold"
plt.tight_layout()
plt.show


# In[19]:


# Standardize Features for training and test set
# The StandardScaler
ss = StandardScaler()
# Standardize the training data
X_train = ss.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train, columns= X.columns)
# Standardize Validation data
X_val = ss.fit_transform(X_val)
X_val_scaled = pd.DataFrame(X_val, columns= X.columns)
#Standardize Testing data
X_test = ss.fit_transform(X_test)
X_test_scaled = pd.DataFrame(X_test, columns= X.columns)


# In[20]:


# Check shape of split data 
print('There are', X_train_scaled.shape[0], 'rows in the train data')
print('There are', X_val_scaled.shape[0], 'rows in the validation data')
print('There are', X_test_scaled.shape[0], 'rows in the test data')


# In[21]:


X_train_scaled.head()


# # Mutual Infomration Feature Selection 

# In[92]:


#run select k best
np.random.seed(42)
fs_fit_fscore = SelectKBest(mutual_info_classif,  k='all')
fs_fit_fscore.fit_transform(X_train_scaled,y_train)
fs_indicies_fscore = np.argsort(np.nan_to_num(fs_fit_fscore.scores_))[::-1][0:60]
best_features_fscore = X.columns[fs_indicies_fscore].values
feature_importances_fscore = fs_fit_fscore.scores_[fs_indicies_fscore]
feature_importances_fscore

data_tuples = list(zip(best_features_fscore, feature_importances_fscore))
m_info_0_1 = pd.DataFrame(data_tuples,columns = ['Covariate_features','values'])
m_info_0_1.head()


# In[23]:


m_info_0_1.to_csv(path_or_buf='feature_selection/Covariate/' + 'Covariate_minfo_features_0_1.csv',index=False)


# In[24]:


#Create a figure for Random Forest Feature Importance
fig = plt.figure(figsize=(15, 10), facecolor='white')

# Implement me
# The bar plot of the top 5 feature importance
plt.bar(m_info_0_1['Covariate_features'][:50], m_info_0_1['values'][:50], color='orange')

# Set x-axis
plt.title('Mutual Information Feature Importance on Covariate Features for Classes 0 and 1', fontsize=30)
plt.xlabel('Features', fontsize = 20)
plt.xticks(rotation=90)

# Set y-axis
plt.ylabel('Importance', fontsize = 20)
plt.grid(False)
# Save and show the figure
plt.tight_layout()
plt.show()


# # Random Forest Model with Test Data

# In[25]:


models = {'rfc': RandomForestClassifier( random_state=42)}
pipes = {}

for acronym, model in models.items():
    pipes[acronym] = Pipeline([('model', model)])
param_grids = {}
# Random Forest Hyper Parameters
# The grids for min_samples_split
min_samples_split_grids = [2,10, 20, 50, 100]

# The grids for min_samples_leaf
min_samples_leaf_grids = [1,10, 20, 50, 100]

# Update param_grids
param_grids['rfc'] = [{'model__min_samples_split': min_samples_split_grids,
                       'model__min_samples_leaf': min_samples_leaf_grids}]


# In[26]:


# The list of [best_score_, best_params_, best_estimator_] obtained by GridSearchCV
best_score_params_estimator_gs = []
# GridSearchCV
gs = GridSearchCV(estimator=pipes[acronym],
                      param_grid=param_grids[acronym],
                      scoring='f1_macro',
                      n_jobs=1,
                      cv=StratifiedKFold(),
                      return_train_score=True)
        
# Fit the pipeline
gs = gs.fit(X_train_scaled, y_train)
    
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

# Sort best_score_params_estimator_gs in descending order of the best_score_
best_score_params_estimator_gs = sorted(best_score_params_estimator_gs, key=lambda x : x[0], reverse=True)

# Print best_score_params_estimator_gs
pd.DataFrame(best_score_params_estimator_gs, columns=['best_score', 'best_param', 'best_estimator'])


# In[27]:


best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]


# In[28]:


# Predict using test data
y_test_pred = best_estimator_gs.predict(X_test_scaled)

# Get the precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_test_pred)


# Get the dataframe of precision, recall, fscore and auc
pd.DataFrame([[precision, recall, fscore]], columns=['Precision', 'Recall', 'F1-score'])


# In[29]:


# Plotting confusion matrix obtained from the testing data predictions
sns.set(style="white")
cnf_matrix = confusion_matrix(y_test,y_test_pred)
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
class_names = ["0","1"]
plt.figure(figsize=(5,5))
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.title('Random Forest Model with Test Data\n on Covariate Features Confusion Matrix')
plt.tight_layout()
plt.show()


# In[30]:


print('Random Forest Model with Test Data\n on Covariate Features Classification Report')
print(classification_report(y_test,y_test_pred))


# # Random Forest Model with Validation Data

# In[31]:


# Predict using Validation data
y_val_pred = best_estimator_gs.predict(X_val_scaled)

# Get the precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y_val, y_val_pred)


# Get the dataframe of precision, recall, fscore and auc
pd.DataFrame([[precision, recall, fscore]], columns=['Precision', 'Recall', 'F1-score'])


# In[32]:


# Plotting confusion matrix obtained from the testing data predictions
sns.set(style="white")
cnf_matrix = confusion_matrix(y_val,y_val_pred)
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
class_names = ["0","1"]
plt.figure(figsize=(5,5))
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.title('Random Forest Model with Validation Data\n on Covariate Features Confusion Matrix')
plt.tight_layout()
plt.show()


# In[33]:


print('Random Forest Model with Validation Data\n on Covariate Features Classification Report')
print(classification_report(y_val,y_val_pred))


# # Random Forest Feature Importance

# In[34]:


target="label"
# Get the best_score, best_param and best_estimator of random forest obtained by GridSearchCV
best_score_rfc, best_param_rfc, best_estimator_rfc = best_score_params_estimator_gs[0]

# Get the dataframe of feature and importance
df_fi_rfc_0_1 = pd.DataFrame(np.hstack((np.setdiff1d(X.columns, [target]).reshape(-1, 1), best_estimator_rfc.named_steps['model'].feature_importances_.reshape(-1, 1))),
                         columns=['Features', 'Importance'])

# Sort df_fi_rfc in descending order of the importance
df_fi_rfc_0_1 = df_fi_rfc_0_1.sort_values(ascending=False, by='Importance').reset_index(drop=True)

# Print the first 5 rows of df_fi_rfc
df_fi_rfc_0_1.head()

#save results as csv
df_fi_rfc_0_1.to_csv(path_or_buf='feature_selection/Covariate/' + 'Covariate_best_random_forest_features_0_1.csv',index=False)


# In[35]:


#Create a figure for Random Forest Feature Importance
fig = plt.figure(figsize=(15, 10))

# Implement me
# The bar plot of the top 5 feature importance
plt.bar(df_fi_rfc_0_1['Features'][:50], df_fi_rfc_0_1['Importance'][:50], color='green')

# Set x-axis
plt.title('Random Forest Feature Importance on Covariate Features for Classes 0 and 1', fontsize=30)
plt.xlabel('Features', fontsize = 20)
plt.xticks(rotation=90)

# Set y-axis
plt.ylabel('Importance', fontsize = 20)

# Save and show the figure
plt.tight_layout()
plt.show()


# # Logistic Model with Testing Data

# In[36]:


# Logistic model construction
models = {'lr': LogisticRegression(solver= "lbfgs", random_state=42)}


pipes = {}

for acronym, model in models.items():
    pipes[acronym] = Pipeline([('model', model)])
param_grids = {}

# Logistic Regression hyperparamters

# The parameter grid of tol
tol_grid = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]

# The parameter grid of C
C_grid = [0.001, 0.0001, 0.1, 1, 10]

param_grids['lr'] = [{'model__tol': tol_grid,
                      'model__C': C_grid}]


# In[37]:


# The list of [best_score_, best_params_, best_estimator_] obtained by GridSearchCV
best_score_params_estimator_gs = []
# GridSearchCV
gs = GridSearchCV(estimator=pipes[acronym],
                      param_grid=param_grids[acronym],
                      scoring='f1_macro',
                      n_jobs=-1,
                      cv=StratifiedKFold(),
                      return_train_score=True)
        
# Fit the pipeline
gs = gs.fit(X_train_scaled, y_train)
    
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

# Sort best_score_params_estimator_gs in descending order of the best_score_
best_score_params_estimator_gs = sorted(best_score_params_estimator_gs, key=lambda x : x[0], reverse=True)

# Print best_score_params_estimator_gs
pd.DataFrame(best_score_params_estimator_gs, columns=['best_score', 'best_param', 'best_estimator'])


# In[38]:


best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]
# Predict using test data
y_test_pred = best_estimator_gs.predict(X_test_scaled)

# Get the precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_test_pred)


# Get the dataframe of precision, recall, fscore and auc
pd.DataFrame([[precision, recall, fscore]], columns=['Precision', 'Recall', 'F1-score'])


# In[39]:


# Plotting confusion matrix obtained from the testing data predictions
sns.set(style="white")
cnf_matrix = confusion_matrix(y_test,y_test_pred)
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
class_names = ["0","1"]
plt.figure(figsize=(5,5))
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.title('Logistic Model with Test Data\n on Covariate Features Confusion Matrix')
plt.show()


# In[40]:


print('Logistic Model with Test Data\n on Covariate Features Classification Report') 
print(classification_report(y_test,y_test_pred))

    


# # Logistic Model with Validation Data

# In[41]:


best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]
# Predict using test data
y_val_pred = best_estimator_gs.predict(X_val_scaled)

# Get the precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y_val, y_val_pred)


# Get the dataframe of precision, recall, fscore and auc
pd.DataFrame([[precision, recall, fscore]], columns=['Precision', 'Recall', 'F1-score'])


# In[42]:


# Plotting confusion matrix obtained from the testing data predictions
sns.set(style="white")
cnf_matrix = confusion_matrix(y_val,y_val_pred)
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
class_names = ["0","1"]
plt.figure(figsize=(5,5))
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.title('Logistic Model with Validation Data\n on Covariate Features Confusion Matrix')
plt.tight_layout()
plt.show()


# In[43]:


print('Logistic Model with Test Data\n on Covariate Features Classification Report')  
print(classification_report(y_val,y_val_pred))


# # Logistic Feature Importance

# In[44]:


# Evaluate odds of each variable and sort by odds value
odds = np.exp(best_estimator_gs["model"].coef_[0])
best_log =pd.DataFrame(odds, X_train_scaled.columns, columns=['odds']).sort_values(by='odds', ascending=False)
best_log.reset_index(inplace=True)
best_log.rename(columns={'index':'Covariate_features','odds':'values'},inplace=True)
best_log.head(10)


# In[45]:


#Create a figure
fig = plt.figure(figsize=(15, 10))

# Implement me
# The bar plot of the top 5 feature importance
plt.bar(best_log['Covariate_features'][:50], best_log['values'][:50], color='blue')

# Set x-axis
plt.title('Logistic Regression Feature Importance on Covariate Features for Classes 0 and 1', fontsize=20)
plt.xlabel('Features',fontsize=20)
plt.xticks(rotation=90)

# Set y-axis
plt.ylabel('Importance', fontsize=20)

# Save and show the figure
plt.tight_layout()
plt.show()


# In[46]:


#save best logistic features in csv file
best_log.to_csv(path_or_buf='feature_selection/Covariate/' + 'Covariate_best_logistic_features_0_1.csv',index=False)


# # Gradient Boosting with Testing Data

# In[47]:


# hyper parameters for testing
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

clf = GradientBoostingClassifier(random_state=42)

# The list of [best_score_, best_params_, best_estimator_] obtained by GridSearchCV
best_score_params_estimator_gs = []
# Run Gridsearch
gs = GridSearchCV(clf, parameter_space,
                      scoring='f1_macro',
                      n_jobs=-1,
                      cv=StratifiedKFold(),
                      return_train_score=True)


# run model
gs = gs.fit(X_train_scaled, y_train)

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


# Sort best_score_params_estimator_gs in descending order of the best_score_
best_score_params_estimator_gs = sorted(best_score_params_estimator_gs, key=lambda x : x[0], reverse=True)

# Print best_score_params_estimator_gs
pd.DataFrame(best_score_params_estimator_gs, columns=['best_score', 'best_param', 'best_estimator'])


# In[48]:


best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]


# In[49]:


#best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]
# Predict using test data
y_test_pred = best_estimator_gs.predict(X_test_scaled)

# Get the precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_test_pred)


# Get the dataframe of precision, recall, fscore and auc
pd.DataFrame([[precision, recall, fscore]], columns=['Precision', 'Recall', 'F1-score'])


# In[50]:


# Gradient Boosting Results
# create confusion matrix for Gradient Boosting test data
sns.set(style="white")
cnf_matrix = confusion_matrix(y_test, y_test_pred)
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
class_names = ["0","1"]
plt.figure(figsize=(5,5))
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.title('Gradient Boosting Model with Test Data\n on Covariate Features Confusion Matrix')
plt.tight_layout()
plt.show()


# In[51]:


print('Gradient Boosting Model with Testing Data\n on Covariate Features Classification Report') 
print(classification_report(y_test,y_test_pred))


# # Gradient Boosting with Validation Data

# In[52]:


best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]
# Predict using test data
y_val_pred = best_estimator_gs.predict(X_val_scaled)

# Get the precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y_val, y_val_pred)


# Get the dataframe of precision, recall, fscore and auc
pd.DataFrame([[precision, recall, fscore]], columns=['Precision', 'Recall', 'F1-score'])


# In[53]:


# Gradient Boosting Results
# create confusion matrix for Gradient Boosting test data
sns.set(style="white")
cnf_matrix = confusion_matrix(y_val, y_val_pred)
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
class_names = ["0","1"]
plt.figure(figsize=(5,5))
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.title('Gradient Boosting Model with Validation Data\n on Covariate Features Confusion Matrix')
plt.tight_layout()
plt.show()


# In[54]:


print('Gradient Boosting Model with Validation Data\n on Covariate Features Classification Report') 
print(classification_report(y_val,y_val_pred))


# In[55]:


target="label"
# Get the best_score, best_param and best_estimator of random forest obtained by GridSearchCV
best_score_gb, best_param_gb, best_estimator_gb = best_score_params_estimator_gs[0]

# Get the dataframe of feature and importance
df_fi_gb_0_1 = pd.DataFrame(np.hstack((np.setdiff1d(X.columns, [target]).reshape(-1, 1), best_estimator_gb.feature_importances_.reshape(-1, 1))),
                         columns=['Features', 'Importance'])


# In[56]:


# Sort df_fi_rfc in descending order of the importance
df_fi_gb_0_1 = df_fi_gb_0_1.sort_values(ascending=False, by='Importance').reset_index(drop=True)

#save results as csv
df_fi_gb_0_1.to_csv(path_or_buf='feature_selection/Covariate/' + 'Covariate_best_gradient_boosting_features_0_1.csv',index=False)


# # Gradient Boosting Feature Importance

# In[57]:


#Create a figure
fig = plt.figure(figsize=(15, 10))

# Implement me
# The bar plot of the top 5 feature importance
plt.bar(df_fi_gb_0_1['Features'][:50], df_fi_gb_0_1['Importance'][:50], color='red')

# Set x-axis
plt.title('Gradient Boosting Feature importance on Covariate Data for Classes 0 and 1', fontsize=30)
plt.xlabel('Features',fontsize=20)
plt.xticks(rotation=90)

# Set y-axis
plt.ylabel('Importance',fontsize=20)

# Save and show the figure
plt.tight_layout()
plt.show()


# # AdaBoosting on Testing Data

# In[58]:


# hyper parameters for testing
parameter_space = {
    'n_estimators': [50, 100, 150, 200],
    "learning_rate": [0.01, 0.05, 0.025]
}

clf = AdaBoostClassifier(random_state=42)

# The list of [best_score_, best_params_, best_estimator_] obtained by GridSearchCV
best_score_params_estimator_gs = []
# Run Gridsearch
gs = GridSearchCV(clf, parameter_space,
                      scoring='f1_macro',
                      n_jobs=-1,
                      cv=StratifiedKFold(),
                      return_train_score=True)


# run model
gs = gs.fit(X_train_scaled, y_train)

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


# Sort best_score_params_estimator_gs in descending order of the best_score_
best_score_params_estimator_gs = sorted(best_score_params_estimator_gs, key=lambda x : x[0], reverse=True)

# Print best_score_params_estimator_gs
pd.DataFrame(best_score_params_estimator_gs, columns=['best_score', 'best_param', 'best_estimator'])


# In[59]:


best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]


# In[60]:


#best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]
# Predict using test data
y_test_pred = best_estimator_gs.predict(X_test_scaled)

# Get the precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_test_pred)


# Get the dataframe of precision, recall, fscore and auc
pd.DataFrame([[precision, recall, fscore]], columns=['Precision', 'Recall', 'F1-score'])


# In[61]:


sns.set(style="white")
cnf_matrix = confusion_matrix(y_test, y_test_pred)
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
class_names = ["0","1"]
plt.figure(figsize=(5,5))
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.title('AdaBoost Model with Test Data\n on Covariate Features Confusion Matrix')
plt.tight_layout()
plt.show()


# In[62]:


print('AdaBoost Model with Testing Data\n on Covariate Features Classification Report') 
print(classification_report(y_test,y_test_pred))


# # AdaBoosting on Validation Data

# In[63]:


#best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]
# Predict using test data
y_val_pred = best_estimator_gs.predict(X_val_scaled)

# Get the precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y_val, y_val_pred)


# Get the dataframe of precision, recall, fscore and auc
pd.DataFrame([[precision, recall, fscore]], columns=['Precision', 'Recall', 'F1-score'])


# In[64]:


sns.set(style="white")
cnf_matrix = confusion_matrix(y_val, y_val_pred)
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
class_names = ["0","1"]
plt.figure(figsize=(5,5))
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.title('AdaBoost Model with Validation Data\n on Covariate Features Confusion Matrix')
plt.tight_layout()
plt.show()


# In[65]:


print('AdaBoost Model with Validation Data\n on Covariate Features Classification Report') 
print(classification_report(y_val,y_val_pred))


# # AdaBoosting Feature Importance

# In[66]:


target="label"
# Get the best_score, best_param and best_estimator of random forest obtained by GridSearchCV
best_score_ad, best_param_ad, best_estimator_ad = best_score_params_estimator_gs[0]

# Get the dataframe of feature and importance
df_fi_ad_0_1 = pd.DataFrame(np.hstack((np.setdiff1d(X.columns, [target]).reshape(-1, 1), best_estimator_ad.feature_importances_.reshape(-1, 1))),
                         columns=['Features', 'Importance'])


# In[67]:


# Sort df_fi_rfc in descending order of the importance
df_fi_ad_0_1 = df_fi_ad_0_1.sort_values(ascending=False, by='Importance').reset_index(drop=True)

# Print the first 5 rows of df_fi_rfc
df_fi_ad_0_1.head()

#save results as csv
df_fi_ad_0_1.to_csv(path_or_buf='feature_selection/Covariate/' + 'Covariate_best_ada_boosting_features_0_1.csv',index=False)


# In[68]:


#Create a figure
fig = plt.figure(figsize=(15, 10))

# Implement me
# The bar plot of the top 5 feature importance
plt.bar(df_fi_ad_0_1['Features'][:50], df_fi_ad_0_1['Importance'][:50], color='purple')

# Set x-axis
plt.title('AdaBoosting Feature importance on Covariate Data for Classes 0 and 1', fontsize=30)
plt.xlabel('Features',fontsize=20)
plt.xticks(rotation=90)

# Set y-axis
plt.ylabel('Importance',fontsize=20)

# Save and show the figure
plt.tight_layout()
plt.show()


# In[69]:


df_fi_ad_0_1.head()


# # Comparing Feature Selections of Different Models

# In[70]:


# Random Forest feature importance two classes
df_fi_rfc_0_1 = df_fi_rfc_0_1.rename(columns = {'Features':'Covariate_features','Importance':'values'})
df_fi_rfc_0_1['top_Random_Forest_0_1']= range(1,len(df_fi_rfc_0_1)+1)
#df_fi_rfc_0_1.drop(['values'],axis=1, inplace=True)
df_fi_rfc_0_1.head()


# In[71]:


# logisitc featue importance for two classes
best_log['top_logistic_0_1'] = range(1,len(best_log)+1)
#best_log.drop(['values'],axis=1, inplace=True)
best_log.head()


# In[72]:


# Gradient Boosting feature importance for two classes
df_fi_gb_0_1 = df_fi_gb_0_1.rename(columns = {'Features':'Covariate_features','Importance':'values'})
df_fi_gb_0_1['top_Gradient_Boosting_0_1']= range(1,len(df_fi_gb_0_1)+1)
#df_fi_gb_0_1.drop(['values'],axis=1, inplace=True)
df_fi_gb_0_1.head()


# In[73]:


df_fi_ad_0_1 = df_fi_ad_0_1.rename(columns = {'Features':'Covariate_features','Importance':'values'})
df_fi_ad_0_1['top_Ada_Boosting_0_1']= range(1,len(df_fi_ad_0_1)+1)
#df_fi_gb_0_1.drop(['values'],axis=1, inplace=True)
df_fi_ad_0_1.head()


# In[74]:


m_info_0_1['minfo_0_1'] = range(1,len(m_info_0_1)+1)
m_info_0_1.head()


# In[75]:


#merge best features for all three methods 
best_0 = best_log.merge(df_fi_rfc_0_1[['Covariate_features','top_Random_Forest_0_1']],how='inner', on = 'Covariate_features')
best_1 = best_0.merge(df_fi_gb_0_1[['Covariate_features','top_Gradient_Boosting_0_1']],how='inner',on = 'Covariate_features')
best_2 = best_1.merge(df_fi_ad_0_1[['Covariate_features','top_Ada_Boosting_0_1']],how='inner',on = 'Covariate_features')
best = best_2.merge(m_info_0_1[['Covariate_features','minfo_0_1']],how='inner',on = 'Covariate_features')
#create rank column
best['combined_rank'] = best['top_logistic_0_1'] + best['top_Random_Forest_0_1'] + best['top_Gradient_Boosting_0_1'] + best['top_Ada_Boosting_0_1'] + best['minfo_0_1']

best = best.sort_values(by= ['combined_rank'], ascending =True).reset_index(drop=True)
best['rank']= range(1,len(best)+1)
del best['values']
best.head(60)


# In[76]:


# save file 
best.to_csv(path_or_buf='feature_selection/Covariate/' + 'Covariate_Features_Ranking.csv',index=False)


# In[77]:


best['max'] = best[["top_logistic_0_1", "top_Random_Forest_0_1",
                    "top_Gradient_Boosting_0_1","top_Ada_Boosting_0_1","minfo_0_1"]].max(axis=1)
best['min'] = best[["top_logistic_0_1", "top_Random_Forest_0_1",
                    "top_Gradient_Boosting_0_1","top_Ada_Boosting_0_1","minfo_0_1"]].min(axis=1)
best.head(60)


# In[110]:


best[0:10]


# In[ ]:




