import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
import warnings
warnings.filterwarnings("ignore")

# data
df = pd.read_csv(r'C:\Users\brear\OneDrive\Desktop\Grad School\Mapping-Deprived-Areas-Using-Deep-Neural-Networks\1.Data\Contextual_Features_final.csv')
df = df.drop(columns=['long', 'lat', 'Point'])
cols_to_move = ['Label']
df = df[cols_to_move + [col for col in df.columns if col not in cols_to_move]]
df = df[df['Label'].isin([0, 1])]

# Move Target to first column
target = 'Label'
first_col = df.pop(target)
df.insert(0, target, first_col)

# set features and target
X = df.values[:, 1:]
y = df.values[:, 0]

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


# Load Models

# model 1
stored_path = r'3.Contextual_and_Covariate_Feautres_Modeling/Saved_Models'
filename = 'contextual_Gradient_Boosting_model_50ADA_Features_classes_0&1.sav'
# loaded_model1 = pickle.load(open(f'{stored_path}/{filename}', 'rb'))
loaded_model1 = pickle.load(open(r'C:\Users\brear\OneDrive\Desktop\Grad School\Mapping-Deprived-Areas-Using-Deep-Neural-Networks\3.Contextual_and_Covariate_Feautres_Modeling\Saved_Models\contextual_Logistic_Regression_model_All_Features_classes_0&1.sav', 'rb'))

lm1_pred = loaded_model1.predict(X_val)

# model 2
stored_path = r'3.Contextual_and_Covariate_Feautres_Modeling/Saved_Models'
filename = 'contextual_Gradient_Boosting_model_50Gradient_Boosting_Features_classes_0&1.sav'
# loaded_model2 = pickle.load(open(f'{stored_path}/{filename}', 'rb'))
loaded_model2 = pickle.load(open(r'C:\Users\brear\OneDrive\Desktop\Grad School\Mapping-Deprived-Areas-Using-Deep-Neural-Networks\3.Contextual_and_Covariate_Feautres_Modeling\Saved_Models\contextual_MLP_model_All_Features_classes_0&1.sav', 'rb'))


lm2_pred = loaded_model2.predict(X_val)

# model 3
stored_path = r'3.Contextual_and_Covariate_Feautres_Modeling/Saved_Models'
filename = 'contextual_Gradient_Boosting_model_50Minfo_Features_classes_0&1.sav'
# loaded_model3 = pickle.load(open(f'{stored_path}/{filename}', 'rb'))
loaded_model3 = pickle.load(open(r'C:\Users\brear\OneDrive\Desktop\Grad School\Mapping-Deprived-Areas-Using-Deep-Neural-Networks\3.Contextual_and_Covariate_Feautres_Modeling\Saved_Models\contextual_Random_Forest_model_All_Features_classes_0&1.sav', 'rb'))


lm3_pred = loaded_model3.predict(X_val)

# model 4
stored_path = r'3.Contextual_and_Covariate_Feautres_Modeling/Saved_Models'
filename = 'contextual_Gradient_Boosting_model_All_Features_classes_0&1.sav'
# loaded_model4 = pickle.load(open(f'{stored_path}/{filename}', 'rb'))
loaded_model4 = pickle.load(open(r'C:\Users\brear\OneDrive\Desktop\Grad School\Mapping-Deprived-Areas-Using-Deep-Neural-Networks\3.Contextual_and_Covariate_Feautres_Modeling\Saved_Models\contextual_Gradient_Boosting_model_All_Features_classes_0&1.sav', 'rb'))


lm4_pred = loaded_model4.predict(X_val)

# model 5
stored_path = r'3.Contextual_and_Covariate_Feautres_Modeling/Saved_Models'
filename = 'contextual_Random_Forest_model_50ADA_Features_classes_0&1.sav'
# loaded_model5 = pickle.load(open(f'{stored_path}/{filename}', 'rb'))
# loaded_model5 = pickle.load(open(r'C:\Users\brear\OneDrive\Desktop\Grad School\Mapping-Deprived-Areas-Using-Deep-Neural-Networks\3.Contextual_and_Covariate_Feautres_Modeling\Saved_Models\contextual_Random_Forest_model_50ADA_Features_classes_0&1.sav', 'rb'))
loaded_model5 = GaussianNB()

loaded_model5.fit(X_train, y_train)
lm5_pred = loaded_model5.predict(X_val)

data = {'Model1_pred1': lm1_pred,
        'Model1_pred2': lm2_pred,
        'Model1_pred3': lm3_pred,
        'Model1_pred4': lm4_pred,
        'Model1_pred5': lm5_pred}
df_pred = pd.DataFrame(data)
df_pred['Vote'] = df_pred.mean(axis=1).round(0)


clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GradientBoostingClassifier()
clf4 = GaussianNB()
clf5 = MLPClassifier()

# ensemble models
eclf1 = VotingClassifier(estimators=[('model1', loaded_model1), ('model2', loaded_model2), ('model3', loaded_model3), ('model4', loaded_model4), ('model5', loaded_model5)], voting='hard')
# eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3), ('gnb', clf4), ('mlp', clf5)], voting='hard')

# Fit model
eclf1.fit(X_train, y_train)



# Predict on validation set
val_pred = eclf1.predict(X_val)
print(f"Validation Results Using Ensembled Contextual Models: \n")
print("Classification Report: ")
print(classification_report(y_val, val_pred))
cf_matrix = confusion_matrix(y_val, val_pred)
print(cf_matrix)
sns.heatmap(cf_matrix, annot=True, fmt="d")
plt.title(f'Confusion Matrix - Contextual Ensemble Model')
plt.show()

# f1 scores for comparison table output
f1_micro_class0 = f1_score(y_val, val_pred, average=None)[0]
f1_micro_class1 = f1_score(y_val, val_pred, average=None)[1]
f1_macro = f1_score(y_val, val_pred, average='macro')

val_pred = df_pred['Vote']
print(f"Validation Results Using Ensembled Contextual Models: \n")
print("Classification Report: ")
print(classification_report(y_val, val_pred))
cf_matrix = confusion_matrix(y_val, val_pred)
print(cf_matrix)
sns.heatmap(cf_matrix, annot=True, fmt="d")
plt.title(f'Confusion Matrix - Contextual Ensemble Model')
plt.show()

# f1 scores for comparison table output
f1_micro_class0 = f1_score(y_val, val_pred, average=None)[0]
f1_micro_class1 = f1_score(y_val, val_pred, average=None)[1]
f1_macro = f1_score(y_val, val_pred, average='macro')


