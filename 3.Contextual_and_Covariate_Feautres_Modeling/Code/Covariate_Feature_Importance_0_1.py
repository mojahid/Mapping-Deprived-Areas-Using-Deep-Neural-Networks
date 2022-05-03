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
from matplotlib.lines import Line2D 
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
from scipy.stats import kstest
import scipy.stats as stats

#-----------------------------------------------------------------------
# pathing
from project_root import get_project_root
root = get_project_root()

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
#     * Statistical Test
#     
#     * Chi Square test for cateogorical data
#     
#     * Kolmogorov–Smirnov Test for Normality
#     
#     * Leven Test for Equality of Variance 
#     
#     * Krskal-Wallis H-Test for non-parametric Version of ANOVA
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
df = pd.read_csv(root / '1.Data' / 'Covariate_Features.csv')
# df = pd.read_csv('Covariate_Features.csv')
print(df.head())
print(df.columns)
df.drop(['long','lat','Coordinates','new_long','new_lat','Raster Value'],axis=1,inplace=True)
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
#df.to_csv('Covariate_Features.csv',index=False)


# In[6]:


print(df.groupby('Label').mean()[' uu_bld_den_2020'])
df.groupby('Label').mean()[' uu_bld_den_2020'].plot(kind='bar')
plt.figure()


# In[7]:


# create pie chart data
pie_data= count_values_in_column(df,"Label") # save data aS a dataframe
count_values_in_column(df,"Label")


# In[8]:


plt.figure(figsize=(10,10))
labels = ['not-built-up','Built-up', 'Deprived']
plt.title('Overview of Labeled Covariate Coordinates before Removing NaN Values', fontsize=30)
plt.pie(pie_data['Total'], labels=labels, autopct='%1.1f%%', textprops={'fontsize': 20})
plt.rcParams["axes.labelweight"] = "bold"
plt.tight_layout()
plt.show


# In[9]:


#checking NAN on Covariate data values
null_values = df[df.isnull().any(axis=1)]
print('there are',df[df.isnull().any(axis=1)].shape[0], 'nan values in the dataframe')
print(null_values['Label'].value_counts())
df.dropna(inplace=True)
# removed Band 28 as there were nan values
df.drop([' ph_gdmhz_2005'], axis=1,inplace = True)

print('there are ',df.shape[0],'rows of data after removing nan values')


# In[10]:


nan_values = pd.DataFrame(null_values.index,columns=['index_values'])
nan_values.to_csv('covariate_null_values.csv',index=False)


# In[11]:


pie_data= count_values_in_column(df,"Label") # save data aS a dataframe
count_values_in_column(df,"Label")


# In[12]:


plt.figure(figsize=(10,10))
labels = ['not-built-up','Built-up', 'Deprived']
plt.title('Overview of Labeled Covariate Coordinates after Removing NaN Values', fontsize=30)
plt.pie(pie_data['Total'], labels=labels, autopct='%1.1f%%', textprops={'fontsize': 20})
plt.rcParams["axes.labelweight"] = "bold"
plt.tight_layout()
plt.show


# In[13]:


#heat map on Covariate Features
plt.figure(figsize=(150, 150))
plt.title('Correlation Heat Map of Covariate Features\n', fontsize= 250)
sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, cmap='YlOrRd', square=True)
plt.tight_layout()
plt.show()


# In[14]:


df.corr()


# In[15]:


# recognized values had high correlation with each other
df[[' sh_pol_relev_ethnic_gr_2019',' uu_urb_bldg_2018']].corr()


# In[16]:


# decided to remove uu_urb_bld_2018 as it had a lower average correlation with values than did sh_pol_relev_ethnic_gr_2019
ex = df.corr()
print(ex[[' sh_pol_relev_ethnic_gr_2019',' uu_urb_bldg_2018']].mean())

#df = df.drop(' uu_urb_bldg_2018', axis =1)
df.shape


# In[17]:


# create dataframe 'df_corr' of all correlation values
# correlation values on for 0 and 1
df_corr =df[df['Label'].isin([0,1])]


# In[18]:


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


# In[19]:


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


# In[20]:


# created function that creates histogram of correlation coefficients of data
def distribution_hist(data,title=''):
        correlation = pd.DataFrame(data.corr().unstack().sort_values(ascending=False).reset_index())
        corr = correlation.loc[lambda x : x['level_0'] != x['level_1']].reset_index(drop=True)
        corr = corr.rename(columns={0: 'Correlation_Values'})
        corr.sort_values(by = 'level_0',ascending=False).reset_index()
        corr.drop_duplicates(subset='Correlation_Values',inplace=True)
        print('the skew of the correlation coefficient values for covariate features is', corr.skew())
        plt.figure(figsize= (15,10), facecolor='white')
        plt.rcParams["figure.figsize"] = (20,15)
        corr.plot(kind='hist',bins=15, color ='#ff5349')
        plt.title('Distribution of Correlation Values ' + title,fontsize=30)
        plt.xlabel('Correlation Coefficeint Values',fontsize=25)
        plt.ylabel('Frequency',fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid(False)
        plt.legend().remove()
        plt.tight_layout()
        plt.show()


# # Splitting and standardizing data for analysis

# In[21]:


# Make directory to save results 
directory = os.path.dirname('result/cv_results/GridSearchCV/')
if not os.path.exists(directory):
    os.makedirs(directory)


# In[22]:


# select 0 and 1 classes
df =df[df['Label'].isin([0,1])]


# In[23]:


X = df.drop('Label', axis=1)
y = df['Label']
# train, val, test split 60/20/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2


# In[24]:


# create pie chart data
pie_data= count_values_in_column(df,"Label") # save data aS a dataframe
count_values_in_column(df,"Label")


# In[25]:


#created Pie chart after removing Not-Built-up areas
plt.figure(figsize= (10,10))
labels = ['Built-up', 'Deprived']
plt.title("Overview of Area Descriptions After Removing 'Not-Built-up' Areas", fontsize=30)
plt.pie(pie_data['Total'], labels=labels, autopct='%1.1f%%', textprops={'fontsize': 20})
plt.rcParams["axes.labelweight"] = "bold"
plt.tight_layout()
plt.show


# In[26]:


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


# In[27]:


# Check shape of split data 
print('There are', X_train_scaled.shape[0], 'rows in the train data')
print('There are', X_val_scaled.shape[0], 'rows in the validation data')
print('There are', X_test_scaled.shape[0], 'rows in the test data')


# In[28]:


X_train_scaled.head()


# # Mutual Infomration Feature Selection 

# In[29]:


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


# In[30]:




filename = 'Covariate_minfo_features_0_1.csv'
m_info_0_1.to_csv(root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'feature_selection' / 'Covariate'/ f'{filename}', index=False)


#m_info_0_1.to_csv(path_or_buf='feature_selection/Covariate/' + 'Covariate_minfo_features_0_1.csv',index=False)



# In[31]:


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

# In[32]:


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


# In[33]:


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


# In[34]:


best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]


# In[35]:


# Predict using test data
y_test_pred = best_estimator_gs.predict(X_test_scaled)

# Get the precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_test_pred)


# Get the dataframe of precision, recall, fscore and auc
pd.DataFrame([[precision, recall, fscore]], columns=['Precision', 'Recall', 'F1-score'])


# In[36]:


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


# In[37]:


print('Random Forest Model with Test Data\n on Covariate Features Classification Report')
print(classification_report(y_test,y_test_pred))


# # Random Forest Model with Validation Data

# In[38]:


# Predict using Validation data
y_val_pred = best_estimator_gs.predict(X_val_scaled)

# Get the precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y_val, y_val_pred)


# Get the dataframe of precision, recall, fscore and auc
pd.DataFrame([[precision, recall, fscore]], columns=['Precision', 'Recall', 'F1-score'])


# In[39]:


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


# In[40]:


print('Random Forest Model with Validation Data\n on Covariate Features Classification Report')
print(classification_report(y_val,y_val_pred))


# # Random Forest Feature Importance

# In[41]:


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


filename = 'Covariate_best_random_forest_features_0_1.csv'
df_fi_rfc_0_1.to_csv(root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'feature_selection' / 'Covariate'/ f'{filename}', index=False)



#df_fi_rfc_0_1.to_csv(path_or_buf='feature_selection/Covariate/' + 'Covariate_best_random_forest_features_0_1.csv',index=False)


# In[42]:


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

# In[43]:


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


# In[44]:


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


# In[45]:


best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]
# Predict using test data
y_test_pred = best_estimator_gs.predict(X_test_scaled)

# Get the precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_test_pred)


# Get the dataframe of precision, recall, fscore and auc
pd.DataFrame([[precision, recall, fscore]], columns=['Precision', 'Recall', 'F1-score'])


# In[46]:


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


# In[47]:


print('Logistic Model with Test Data\n on Covariate Features Classification Report') 
print(classification_report(y_test,y_test_pred))

    


# # Logistic Model with Validation Data

# In[48]:


best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]
# Predict using test data
y_val_pred = best_estimator_gs.predict(X_val_scaled)

# Get the precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y_val, y_val_pred)


# Get the dataframe of precision, recall, fscore and auc
pd.DataFrame([[precision, recall, fscore]], columns=['Precision', 'Recall', 'F1-score'])


# In[49]:


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


# In[50]:


print('Logistic Model with Test Data\n on Covariate Features Classification Report')  
print(classification_report(y_val,y_val_pred))


# # Logistic Feature Importance

# In[51]:


# Evaluate odds of each variable and sort by odds value
odds = np.exp(best_estimator_gs["model"].coef_[0])
best_log =pd.DataFrame(odds, X_train_scaled.columns, columns=['odds']).sort_values(by='odds', ascending=False)
best_log.reset_index(inplace=True)
best_log.rename(columns={'index':'Covariate_features','odds':'values'},inplace=True)
best_log.head(10)


# In[52]:


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


# In[53]:


#save best logistic features in csv file

filename = 'Covariate_best_logistic_features_0_1.csv'
best_log.to_csv(root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'feature_selection' / 'Covariate'/ f'{filename}', index=False)


#best_log.to_csv(path_or_buf='feature_selection/Covariate/' + 'Covariate_best_logistic_features_0_1.csv',index=False)


# # Gradient Boosting with Testing Data

# In[54]:


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


# In[55]:


best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]


# In[56]:


#best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]
# Predict using test data
y_test_pred = best_estimator_gs.predict(X_test_scaled)

# Get the precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_test_pred)


# Get the dataframe of precision, recall, fscore and auc
pd.DataFrame([[precision, recall, fscore]], columns=['Precision', 'Recall', 'F1-score'])


# In[57]:


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


# In[58]:


print('Gradient Boosting Model with Testing Data\n on Covariate Features Classification Report') 
print(classification_report(y_test,y_test_pred))


# # Gradient Boosting with Validation Data

# In[59]:


best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]
# Predict using test data
y_val_pred = best_estimator_gs.predict(X_val_scaled)

# Get the precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y_val, y_val_pred)


# Get the dataframe of precision, recall, fscore and auc
pd.DataFrame([[precision, recall, fscore]], columns=['Precision', 'Recall', 'F1-score'])


# In[60]:


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


# In[61]:


print('Gradient Boosting Model with Validation Data\n on Covariate Features Classification Report') 
print(classification_report(y_val,y_val_pred))


# In[62]:


target="label"
# Get the best_score, best_param and best_estimator of random forest obtained by GridSearchCV
best_score_gb, best_param_gb, best_estimator_gb = best_score_params_estimator_gs[0]

# Get the dataframe of feature and importance
df_fi_gb_0_1 = pd.DataFrame(np.hstack((np.setdiff1d(X.columns, [target]).reshape(-1, 1), best_estimator_gb.feature_importances_.reshape(-1, 1))),
                         columns=['Features', 'Importance'])


# In[63]:


# Sort df_fi_rfc in descending order of the importance
df_fi_gb_0_1 = df_fi_gb_0_1.sort_values(ascending=False, by='Importance').reset_index(drop=True)

#save results as csv

filename = 'Covariate_best_gradient_boosting_features_0_1.csv'
df_fi_gb_0_1.to_csv(root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'feature_selection' / 'Covariate'/ f'{filename}', index=False)



#df_fi_gb_0_1.to_csv(path_or_buf='feature_selection/Covariate/' + 'Covariate_best_gradient_boosting_features_0_1.csv',index=False)


# # Gradient Boosting Feature Importance

# In[64]:


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

# In[65]:


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


# In[66]:


best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]


# In[67]:


#best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]
# Predict using test data
y_test_pred = best_estimator_gs.predict(X_test_scaled)

# Get the precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_test_pred)


# Get the dataframe of precision, recall, fscore and auc
pd.DataFrame([[precision, recall, fscore]], columns=['Precision', 'Recall', 'F1-score'])


# In[68]:


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


# In[69]:


print('AdaBoost Model with Testing Data\n on Covariate Features Classification Report') 
print(classification_report(y_test,y_test_pred))


# # AdaBoosting on Validation Data

# In[70]:


#best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_gs[0]
# Predict using test data
y_val_pred = best_estimator_gs.predict(X_val_scaled)

# Get the precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y_val, y_val_pred)


# Get the dataframe of precision, recall, fscore and auc
pd.DataFrame([[precision, recall, fscore]], columns=['Precision', 'Recall', 'F1-score'])


# In[71]:


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


# In[72]:


print('AdaBoost Model with Validation Data\n on Covariate Features Classification Report') 
print(classification_report(y_val,y_val_pred))


# # AdaBoosting Feature Importance

# In[73]:


target="label"
# Get the best_score, best_param and best_estimator of random forest obtained by GridSearchCV
best_score_ad, best_param_ad, best_estimator_ad = best_score_params_estimator_gs[0]

# Get the dataframe of feature and importance
df_fi_ad_0_1 = pd.DataFrame(np.hstack((np.setdiff1d(X.columns, [target]).reshape(-1, 1), best_estimator_ad.feature_importances_.reshape(-1, 1))),
                         columns=['Features', 'Importance'])


# In[74]:


# Sort df_fi_rfc in descending order of the importance
df_fi_ad_0_1 = df_fi_ad_0_1.sort_values(ascending=False, by='Importance').reset_index(drop=True)

# Print the first 5 rows of df_fi_rfc
df_fi_ad_0_1.head()

#save results as csv

filename = 'Covariate_best_ada_boosting_features_0_1.csv'
df_fi_ad_0_1.to_csv(root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'feature_selection' / 'Covariate'/ f'{filename}', index=False)



#df_fi_ad_0_1.to_csv(path_or_buf='feature_selection/Covariate/' + 'Covariate_best_ada_boosting_features_0_1.csv',index=False)


# In[75]:


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


# In[76]:


df_fi_ad_0_1.head()


# # Comparing Feature Selections of Different Models

# In[77]:


# Random Forest feature importance two classes
df_fi_rfc_0_1 = df_fi_rfc_0_1.rename(columns = {'Features':'Covariate_features','Importance':'values'})
df_fi_rfc_0_1['top_Random_Forest_0_1']= range(1,len(df_fi_rfc_0_1)+1)
#df_fi_rfc_0_1.drop(['values'],axis=1, inplace=True)
df_fi_rfc_0_1.head()


# In[78]:


# logisitc featue importance for two classes
best_log['top_logistic_0_1'] = range(1,len(best_log)+1)
#best_log.drop(['values'],axis=1, inplace=True)
best_log.head()


# In[79]:


# Gradient Boosting feature importance for two classes
df_fi_gb_0_1 = df_fi_gb_0_1.rename(columns = {'Features':'Covariate_features','Importance':'values'})
df_fi_gb_0_1['top_Gradient_Boosting_0_1']= range(1,len(df_fi_gb_0_1)+1)
#df_fi_gb_0_1.drop(['values'],axis=1, inplace=True)
df_fi_gb_0_1.head()


# In[80]:


df_fi_ad_0_1 = df_fi_ad_0_1.rename(columns = {'Features':'Covariate_features','Importance':'values'})
df_fi_ad_0_1['top_Ada_Boosting_0_1']= range(1,len(df_fi_ad_0_1)+1)
#df_fi_gb_0_1.drop(['values'],axis=1, inplace=True)
df_fi_ad_0_1.head()


# In[81]:


m_info_0_1['minfo_0_1'] = range(1,len(m_info_0_1)+1)
m_info_0_1.head()


# In[82]:


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


# In[83]:


# save file 

filename = 'Covariate_Features_Ranking.csv'
best.to_csv(root / '3.Contextual_and_Covariate_Feautres_Modeling' / 'feature_selection' / 'Covariate'/ f'{filename}', index=False)


#best.to_csv(path_or_buf='feature_selection/Covariate/' + 'Covariate_Features_Ranking.csv',index=False)


# # Statistical tests for confirm difference between Deprived and Built-up

# # Chi Square test of Independence for Cateogircal features

# In[84]:


# introduced Chi Square test on six categorical variables to see if they were statistically significant 
# in showing a difference between 'Deprived' and 'Built-up' areas


# In[85]:


# identified categorical features
best_cat = best.set_index('Covariate_features')
best_cat = best_cat.loc[[ ' fs_electric_dist_2020', ' ph_hzd_index_2011', ' ph_land_c1_2019',
                        ' ph_land_c2_2020', ' sh_pol_relev_ethnic_gr_2019', ' uu_urb_bldg_2018']]
best_cat


# In[86]:


#created dataframe for categorical features with Label data

df_cat = df[['Label',' fs_electric_dist_2020',' ph_hzd_index_2011',' ph_land_c1_2019',
                  ' ph_land_c2_2020', ' sh_pol_relev_ethnic_gr_2019',' uu_urb_bldg_2018']]


# In[87]:


# convert values to categorical
df_cat['Label'] = pd.Categorical(df_cat['Label'])
df_cat[' fs_electric_dist_2020'] = pd.Categorical(df_cat[' fs_electric_dist_2020'])
df_cat[' ph_hzd_index_2011'] = pd.Categorical(df_cat[' ph_hzd_index_2011'])
df_cat[' ph_land_c1_2019'] = pd.Categorical(df_cat[' ph_land_c1_2019'])
df_cat[' ph_land_c2_2020'] = pd.Categorical(df_cat[' ph_land_c2_2020'])
df_cat[' sh_pol_relev_ethnic_gr_2019'] = pd.Categorical(df_cat[' sh_pol_relev_ethnic_gr_2019'])
df_cat[' uu_urb_bldg_2018'] = pd.Categorical(df_cat[' uu_urb_bldg_2018'])
df_cat.info()


# In[88]:


df_cat.head()


# In[89]:


# run chi square test on categorical variables
crosstab_1 = pd.crosstab(df_cat["Label"], df_cat[" fs_electric_dist_2020"], margins=True)
c,p,dof, ex = stats.chi2_contingency(crosstab_1)

print('chi2 statistic for fs_electric_dist_2020 was' , round(c,4))
print('p-value for fs_electric_dist_2020 was' , round(p,4))
print('The contingency table was\n ')

#convert expected value to dataframe
expected_1 = pd.DataFrame(ex)
expected_1 = expected_1.iloc[:-1 , :-1]
expected_1.rename(columns={0:'0_expected',1:'1_expected'},inplace=True)

# clean up observed dataframe
crosstab_1 = crosstab_1.iloc[:-1 , :-1]
crosstab_1.rename(columns={0:'0_observed',1:'1_observed'},inplace=True)  
crosstab_1

#concatonate observed and expected tables
table_1 = pd.concat([crosstab_1,expected_1], axis=1)
table_1 = table_1.reindex(sorted(table_1.columns), axis=1)
table_1


# In[90]:


table_1 = pd.concat([crosstab_1,expected_1], axis=1)
table_1 = table_1.reindex(sorted(table_1.columns), axis=1)
table_1


# In[91]:


# plot data for first 
plt.figure()
table_1.plot.bar(color = ['blue','red','blue','red','blue','red','blue','red'],xticks=[] ,rot= 90)
plt.title('Bar Chart on fs_electric_dist_2020 Variable\n ',fontsize= 50)
plt.ylabel('Count',fontsize= 40)
plt.xticks([0, 1], ['Built-up', 'Deprived'], fontsize= 30,rotation=450)
plt.yticks(fontsize=30)
plt.xlabel('\nArea Descriptions', fontsize=40)
plt.grid(False)
colors = ['blue','red']
lines = [Line2D([0], [0], color=c, linewidth=4) for c in colors]
labels = ['Expected','Observed']
plt.legend(lines,labels,prop={'size': 30})
plt.tight_layout()
plt.show()


# In[92]:


# run chi square test on categorical variables
crosstab_2 = pd.crosstab(df_cat["Label"], df_cat[" ph_hzd_index_2011"], margins= True)
c,p,dof,ex = stats.chi2_contingency(crosstab_2)
print('chi2 statistic for ph_hzd_index_2011 was' , round(c,4))
print('The contingency table is\n ', ex)
print('p-value for ph_hzd_index_2011 was' , round(p,4))

#convert expected value to dataframe
expected_2 = pd.DataFrame(ex)
expected_2 = expected_2.iloc[:-1 , :-1]
expected_2.rename(columns={0:'0_expected',1:'1_expected', 2: '2_expected',
                          3: '3_expected', 4:'4_expected',5:'5_expected'},inplace=True)
expected_2


# In[93]:


# clean up observed dataframe
crosstab_2 = crosstab_2.iloc[:-1 , :-1]
crosstab_2.rename(columns={0:'0_observed',1:'1_observed', 2: '2_observed',
                          3: '3_observed', 4:'4_observed',5:'5_observed'},inplace=True) 
crosstab_2

#concatonate observed and expected tables
table_2 = pd.concat([crosstab_2,expected_2], axis=1)
table_2 = table_2.reindex(sorted(table_2.columns), axis=1)
table_2


# In[94]:


# plot data for first 
plt.figure()
table_2.plot.bar(color = ['blue','red','blue','red','blue','red','blue','red',
                         'blue','red','blue','red'],xticks=[] ,rot= 90)
plt.title('Bar Chart on ph_hzd_index_2011 Variable\n ',fontsize= 50)
plt.ylabel('Count',fontsize= 40)
plt.xticks(rotation=90)
plt.xticks([0, 1], ['Built-up', 'Deprived'], fontsize= 30)
plt.yticks(fontsize=30)
plt.xlabel('\nArea Descriptions', fontsize=40)
plt.grid(False)
colors = ['blue','red']
lines = [Line2D([0], [0], color=c, linewidth=4) for c in colors]
labels = ['Expected','Observed']
plt.legend(lines,labels,prop={'size': 30})
plt.tight_layout()
plt.show()


# In[95]:


crosstab_3 = pd.crosstab(df_cat["Label"], df_cat[" ph_land_c1_2019"], margins= True)
c,p,dof,ex = stats.chi2_contingency(crosstab_3)
print('chi2 statistic for ph_land_c1_2019 was' , round(c,4))
print('The contingency table is\n ', ex)
print('p-value for ph_land_c1_2019 was' , round(p,4))
crosstab_3


# In[96]:


crosstab_3 = crosstab_3.iloc[:-1 , :-1]
crosstab_3.rename(columns= {20:'20_observed',30:'30_observed',40:'40_observed',
                           50:'50_observed',60:'60_observed',80:'80_observed',
                           90:'90_observed',112:'112_observed',116:'116_observed',
                           126:'126_observed',200:'200_observed'},inplace=True)
crosstab_3


# In[97]:


#convert expected value to dataframe
expected_3 = pd.DataFrame(ex)
expected_3 = expected_3.iloc[:-1 , :-1]
expected_3.rename(columns= {0:'20_expected',1:'30_expected',2:'40_expected',
                           3:'50_expected',4:'60_expected',5:'80_expected',
                           6:'90_expected',7:'112_expected',8:'116_expected',
                           9:'126_expected',10:'200_expected'},inplace=True)
expected_3


# In[98]:


#concatonate observed and expected tables
table_3 = pd.concat([crosstab_3,expected_3], axis=1)
#table_3 = table_3.reindex(sorted(table_3.columns), axis=1)
table_3 = table_3[['20_expected','20_observed', '30_expected','30_observed','40_expected','40_observed',
                  '50_expected','50_observed','60_expected','60_observed','80_expected','80_observed',
                  '90_expected','90_observed','112_expected','112_observed','116_expected','116_observed',
                  '126_expected','126_observed','200_expected','200_observed']]
table_3


# In[99]:


# plot data for first 
plt.figure()
table_3.plot.bar(color = ['blue','red','blue','red','blue','red','blue','red',
                         'blue','red','blue','red','blue','red','blue','red','blue','red','blue','red',
                         'blue','red'],xticks=[] ,rot= 90)
plt.title('Bar Chart on ph_land_c1_2019 Variable\n ',fontsize= 50)
plt.ylabel('Count',fontsize= 40)
plt.xticks([0, 1], ['Built-up', 'Deprived'], fontsize= 30)
plt.yticks(fontsize=30)
plt.xlabel('\nArea Descriptions', fontsize=40)
plt.grid(False)
colors = ['blue','red']
lines = [Line2D([0], [0], color=c, linewidth=4) for c in colors]
labels = ['Expected','Observed']
plt.legend(lines,labels,prop={'size': 30})
plt.tight_layout()
plt.show()


# In[100]:


crosstab_4 = pd.crosstab(df_cat["Label"], df_cat[" ph_land_c2_2020"], margins= True)
c,p,dof,ex = stats.chi2_contingency(crosstab_4)
print('chi2 statistic for ph_land_c2_2020 was' , round(c,4))
print('The contingency table is\n ', ex)
print('p-value for ph_land_c2_2020 was' , round(p,4))
crosstab_4


# In[101]:


crosstab_4 = crosstab_4.iloc[:-1 , :-1]
crosstab_4.rename(columns= {10:'10_observed',20:'20_observed',30:'30_observed',
                           40:'40_observed',50:'50_observed',60:'60_observed',
                           80:'80_observed',255.0:'255_observed'},inplace=True)
crosstab_4


# In[102]:


#convert expected value to dataframe
expected_4 = pd.DataFrame(ex)
expected_4 = expected_4.iloc[:-1 , :-1]
expected_4.rename(columns= {0:'10_expected',1:'20_expected',2:'30_expected',
                           3:'40_expected',4:'50_expected',5:'60_expected',
                           6:'80_expected',7:'255_expected'},inplace=True)
expected_4


# In[103]:


#concatonate observed and expected tables
table_4 = pd.concat([crosstab_4,expected_4], axis=1)

table_4 = table_4[['10_expected','10_observed', '20_expected','20_observed','30_expected','30_observed',
                  '40_expected','40_observed','50_expected','50_observed','60_expected','60_observed',
                  '80_expected','80_observed','255_expected','255_observed']]
table_4


# In[104]:


# plot data for first 
plt.figure()
table_4.plot.bar(color = ['blue','red','blue','red','blue','red','blue','red',
                         'blue','red','blue','red','blue','red','blue','red'],xticks=[] ,rot= 90)
plt.title('Bar Chart on ph_land_c2_2020 Variable\n ',fontsize= 50)
plt.ylabel('Count',fontsize= 40)
plt.xticks([0, 1], ['Built-up', 'Deprived'], fontsize= 30)
plt.yticks(fontsize=30)
plt.xlabel('\nArea Descriptions', fontsize=40)
plt.grid(False)
colors = ['blue','red']
lines = [Line2D([0], [0], color=c, linewidth=4) for c in colors]
labels = ['Expected','Observed']
plt.legend(lines,labels,prop={'size': 30})
plt.tight_layout()
plt.show()


# In[105]:


crosstab_5 = pd.crosstab(df_cat["Label"], df_cat[" sh_pol_relev_ethnic_gr_2019"], margins= True)
c,p,dof,ex = stats.chi2_contingency(crosstab_5)
print('chi2 statistic for sh_pol_relev_ethnic_gr_2019 was' , round(c,4))
print('The contingency table is\n ', ex)
print('p-value for sh_pol_relev_ethnic_gr_2019 was' , round(p,4))
crosstab_5


# In[106]:


crosstab_5 = crosstab_5.iloc[:-1 , :-1]
crosstab_5.rename(columns= {0:'0_observed',1:'1_observed'},inplace=True)
crosstab_5


# In[107]:


#convert expected value to dataframe
expected_5 = pd.DataFrame(ex)
expected_5 = expected_5.iloc[:-1 , :-1]
expected_5.rename(columns= {0:'0_expected',1:'1_expected'},inplace=True)
expected_5


# In[108]:


#concatonate observed and expected tables
table_5 = pd.concat([crosstab_5,expected_5], axis=1)

table_5 = table_5[['0_expected','0_observed','1_expected','1_observed']]
table_5


# In[109]:


# plot data for first 
plt.figure()
table_5.plot.bar(color = ['blue','red','blue','red','blue','red'],xticks=[] ,rot= 90)
plt.title('Bar Chart on sh_pol_relev_ethnic_gr_2019 Variable\n ',fontsize= 50)
plt.ylabel('Count',fontsize= 40)
plt.xticks([0, 1], ['Built-up', 'Deprived'], fontsize= 30)
plt.yticks(fontsize=30)
plt.xlabel('\nArea Descriptions', fontsize=40)
plt.grid(False)
colors = ['blue','red']
lines = [Line2D([0], [0], color=c, linewidth=4) for c in colors]
labels = ['Expected','Observed']
plt.legend(lines,labels,prop={'size': 30})
plt.tight_layout()
plt.show()


# In[110]:


crosstab_6 = pd.crosstab(df_cat["Label"], df_cat[" uu_urb_bldg_2018"], margins= True)
c,p,dof,ex = stats.chi2_contingency(crosstab_6)
print('chi2 statistic for uu_urb_bldg_2018 was' , round(c,4))
print('The contingency table is\n ', ex)
print('p-value for uu_urb_bldg_2018 was' , round(p,4))
crosstab_6


# In[111]:


crosstab_6 = crosstab_6.iloc[:-1 , :-1]
crosstab_6.rename(columns= {-1:'-1_observed',0:'0_observed',1:'1_observed'},inplace=True)
crosstab_6


# In[112]:


#convert expected value to dataframe
expected_6 = pd.DataFrame(ex)
expected_6 = expected_6.iloc[:-1 , :-1]
expected_6.rename(columns= {0:'-1_expected',1:'0_expected',2:'1_expected'},inplace=True)
expected_6


# In[113]:


#concatonate observed and expected tables
table_6 = pd.concat([crosstab_6,expected_6], axis=1)

table_6 = table_6[['-1_expected','1_observed', '0_expected','0_observed','1_expected','1_observed']]
table_6


# In[118]:


# plot data for first 
plt.figure()
table_6.plot.bar(color = ['blue','red','blue','red','blue','red'], xticks=[] ,rot= 90)
plt.title('Bar Chart on uu_urb_bldg_2018 Variable\n ',fontsize= 50)
plt.ylabel('Count',fontsize= 40)
plt.xticks([0, 1], ['Built-up', 'Deprived'], fontsize= 30)
plt.yticks(fontsize=30)
plt.xlabel('\nArea Descriptions', fontsize=40)
plt.grid(False)
colors = ['blue','red']
lines = [Line2D([0], [0], color=c, linewidth=4) for c in colors]
labels = ['Expected','Observed']
plt.legend(lines,labels,prop={'size': 30})
plt.tight_layout()
plt.show()


# In[119]:


df_cat.columns
print(df_cat[' sh_pol_relev_ethnic_gr_2019'].value_counts())
print(df_cat[' uu_urb_bldg_2018'].value_counts())
df_cat[[' sh_pol_relev_ethnic_gr_2019',' uu_urb_bldg_2018']][300:310]


# #  Kolmogorov–Smirnov - Test for Independence on Class 0 and 1 

# used the Kolmogorov–Smirnov test to see if distribution of Covariate features follows normal distribution

# In[120]:


# independence test on continuous data
df_continuous = df.loc[:, ~df.columns.isin([' fs_electric_dist_2020',' ph_hzd_index_2011',' ph_land_c1_2019',
                  ' ph_land_c2_2020', ' sh_pol_relev_ethnic_gr_2019',' uu_urb_bldg_2018'])]
df_continuous.head()


# In[121]:


deprived = df_continuous[df_continuous['Label']==1]
deprived = deprived.drop('Label',axis=1)
built_up = df_continuous[df_continuous['Label']==0]
built_up = built_up.drop('Label',axis=1)


# In[125]:


# Check Normality of Covariate features for deprived area
Norm= []
Norm_col = []
for col in deprived.columns: 
    Norm_col.append(col)
    Norm.append(kstest(deprived[col],'norm'))
norm = pd.DataFrame (Norm, columns = ['Statistics','p-value'])
norm_col= pd.DataFrame (Norm_col, columns = ['Covariate_features'])
normal_check_deprived = norm_col.merge(norm, left_index=True, right_index=True)
normal_check_deprived['p-value']= round(normal_check_deprived['p-value'],4)
normal_check_deprived.head()


# In[126]:


# Check Normality of Built-up variable
Norm= []
Norm_col = []
for col in built_up.columns: 
    Norm_col.append(col)
    Norm.append(kstest(built_up[col],'norm'))
norm = pd.DataFrame (Norm, columns = ['Statistics','p-value'])
norm_col= pd.DataFrame (Norm_col, columns = ['Covariate_features'])
normal_check_built_up = norm_col.merge(norm, left_index=True, right_index=True)
normal_check_built_up['p-value']= round(normal_check_built_up['p-value'],4)
normal_check_built_up.head()


# # Levene Test - Equal Variance Test for Class 0 and 1

# Conducted levene test for equality of variance amongst the Covariate features to confirm assumption that there is a difference in variance between the deprived and built up areas

# In[127]:


# checking equality of variance 
levene = []
f_value = []
p_value = []
for col in df_continuous.columns[1:]:   
    den = df_continuous[['Label',col]]
    den_0 = den[den['Label']==0]
    den_1 = den[den['Label']==1]
    den_full = [den_0,den_1]
    fvalue, pvalue = stats.levene(den_0[col], den_1[col])
    levene.append(col)
    f_value.append(fvalue)
    p_value.append(pvalue)
df_levene = pd.DataFrame({'Covariate_features': levene, 
                   'fvalue': f_value,
                    'p_value': p_value})
df_levene['p_value'] = round(df_levene['p_value'],4)

# compare ranks for levene score
df_levene = df_levene.merge(best[['Covariate_features','rank']], on = 'Covariate_features')
df_levene.p_value = round(df_levene.p_value,4)
df_levene = df_levene.sort_values('rank').reset_index(drop=True)
df_levene.head(60)


# # Kruskal-Wallis H-Test 

# In[128]:


# create dataframe that runs Kruskal-Wallis H test test for each covariate feature on each value
b = []
f_value = []
p_value = []
for col in df_continuous.columns[1:]:   
    den = df_continuous[['Label',col]]
    den_0 = den[den['Label']==0]
    den_1 = den[den['Label']==1]
    den_full = [den_0,den_1]
    fvalue, pvalue = stats.kruskal(den_0[col], den_1[col])
    b.append(col)
    f_value.append(fvalue)
    p_value.append(pvalue)
fd = pd.DataFrame({'Covariate_features': b, 
                   'fvalue': f_value,
                    'p_value': p_value})


# In[129]:


fd.p_value = round(fd.p_value,4)
fd = fd.sort_values('p_value').reset_index(drop=True)
fd.head(60)


# In[130]:


FD = fd.merge(best[['Covariate_features','rank']], on = 'Covariate_features')
FD.p_value = round(FD.p_value,4)
FD = FD.sort_values('rank').reset_index(drop=True)
FD.head(60)


# In[131]:


#Create a figure
fig = plt.figure(figsize=(15, 10))

# Implement me
# The bar plot of the top 5 feature importance
plt.bar(FD['Covariate_features'], FD['p_value'], color='grey')

# Set x-axis
plt.title('Kruskal-Wallis p_value on Covariate Data for Classes 0 and 1', fontsize=30)
plt.xlabel('Features',fontsize=20)
plt.xticks(rotation=90)

# Set y-axis
plt.ylabel('p_value',fontsize=20)

# Save and show the figure
plt.tight_layout()
plt.show()


# In[132]:


#FD['group'] = FD['1'].iloc
FD.reset_index(inplace=True,drop=True)
ex_1 = ['group_1']*27
ex_2 = ['group_2']*28
ex = ex_1 + ex_2

ex = pd.DataFrame(ex,columns=['groups'])


# In[133]:


FD = FD.merge(ex, left_index= True, right_index= True)
FD.head()


# # Boxplot

# In[134]:


# created boxplots in this section for deprived and built up for selected covariate features


# In[135]:


def boxplot_graph(data, feature = ''):
    den = data[['Label',feature]]
    den_0 = den[den['Label']==0]
    den_0.reset_index(inplace=True,drop=True)
    den_1 = den[den['Label']==1]
    den_1.reset_index(inplace=True,drop=True)
    den_full = [den_0[feature],
                den_1[feature]]

    # stats f_oneway functions takes the groups as input and returns ANOVA F and p value
    fvalue, pvalue = stats.kruskal(den_0[feature], den_1[feature])
    print('the Kruskal-wallis test statistic is', round(fvalue,4), 'with a p-value of',round(pvalue,4))

    plt.figure()
    plt.boxplot(den_full)
    plt.title('Box Plot on' +feature+ ' Variable\n ',fontsize= 60)
    plt.ylabel('Count of Building Density',fontsize= 40)
    plt.xticks([1, 2], ['Built-up', 'Deprived'], fontsize= 30)
    plt.yticks(fontsize=30)
    plt.xlabel('Area Descriptions', fontsize=40)
    plt.grid(False)
    plt.tight_layout()
    plt.show()


# In[136]:


boxplot_graph(df,feature= ' ses_odef_2018')


# In[137]:


boxplot_graph(df,feature= ' uu_bld_den_2020')


# In[138]:


boxplot_graph(df,feature= ' ses_odef_2018')


# In[139]:


boxplot_graph(df,feature= ' ses_impr_water_src_2016')


# In[140]:


boxplot_graph(df,feature= ' ph_dist_aq_veg_2015')


# In[141]:


boxplot_graph(df,feature= ' ses_measles_2018')


# In[142]:


boxplot_graph(df,feature= ' uu_bld_count_2020')


# In[143]:


boxplot_graph(df,feature= ' fs_dist_well_2018')


# In[ ]:




