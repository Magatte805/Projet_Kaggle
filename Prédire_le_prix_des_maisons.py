#!/usr/bin/env python
# coding: utf-8

# # Import the libraries

# In[5]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.filterwarnings("ignore")


# # Exploration of data

# In[6]:


#Data importation
data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv", index_col="Id")

#Displaying the first elements of dataset
data.head()


# In[7]:


data_submission=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
data_submission.head()



# In[8]:


#Displaying the last elements of dataset
data.tail()


# In[9]:


# Overview of general dataset characteristics
data.info()


# #### The dataset consists of 1460 rows and 80 columns, each representing a specific characteristic of each house, such as lotArea,street, etc. Some columns, like 'LotFrontage,' 'Alley,' etc., have missing values.

# In[10]:


# Overview of descriptive statistics on numeric columns
data.describe()


# # Cleanning the data

# In[11]:


# Calculate the percentage of missing values for each column
missing_percentage = (data.isnull().sum() / data.shape[0]) * 100

# Filter columns with missing values
columns_with_missing_values = missing_percentage[missing_percentage > 0]

# Show percentage of missing values for each column
print(columns_with_missing_values)


# In[12]:


#removing columns
data.drop(columns=['MasVnrType','PoolQC', 'PoolArea'],inplace=True)


# In[13]:


# Handling missing values 
data['LotFrontage'] = data['LotFrontage'].fillna(0)
data['Electrical'] = data['Electrical'].fillna( data['Electrical'].mode()[0])
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(0)
data['MasVnrArea'] = data['MasVnrArea'].fillna( data['MasVnrArea'].mean())

data['Alley'] = data['Alley'].fillna("No Alley")
data['BsmtQual'] = data['BsmtQual'].fillna("NO Bsmt")
data['BsmtCond'] = data['BsmtCond'].fillna("NO Bsmt")
data['BsmtExposure'] = data['BsmtExposure'].fillna("NO Bsmt")
data['BsmtFinType1'] = data['BsmtFinType1'].fillna("NO Bsmt")
data['BsmtFinType2'] = data['BsmtFinType2'].fillna("NO Bsmt")
data['FireplaceQu'] = data['FireplaceQu'].fillna("NO Fireplace")
data['GarageType'] = data['GarageType'].fillna("NO Garage")
data['GarageFinish'] = data['GarageFinish'].fillna("NO Garage")
data['GarageQual'] = data['GarageQual'].fillna("NO Garage")
data['GarageCond '] = data['GarageCond'].fillna("NO Garage")
data['Fence'] = data['Fence'].fillna("NO Fence")
data['MiscFeature'] = data['MiscFeature'].fillna("NO MiscFeature")


# In[14]:


data.info()


# # Visualization

# In[15]:


plt.figure(figsize=(10, 6))
sns.histplot(data['SalePrice'], kde=True, color='blue')
plt.title("Distribution of sales prices")
plt.xlabel("sales prices")
plt.ylabel("frequency")
plt.show()


# In[16]:


plt.figure(figsize=(12, 6))
sns.barplot(x='MoSold', y='SalePrice', data=data)
plt.title('Prix de vente moyen en fonction du mois de vente')
plt.show()


# In[17]:


data.hist(figsize=(25,25), bins=20)
plt.show()


# In[18]:


numeric_data = data.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_data.corr(), cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Heatmap of Correlation between numeric variables')
plt.show()


# In[19]:


# Creation of the bar chart
plt.figure(figsize=(12, 8))
sns.countplot(x='MSZoning', data=data)
plt.title('Distribution of houses by classification zone')
plt.xlabel('MSZoning')
plt.ylabel('Number of houses')
plt.show()



# In[20]:


#visualiser la relation entre 'SalePrice' et 'BldgType'
plt.figure(figsize=(12, 8))
sns.boxplot(x='BldgType', y='SalePrice', data=data)
plt.title('Prix de vente moyen en fonction du type de logement')
plt.xlabel('Type de logement')
plt.ylabel('Prix de vente')
plt.xticks(rotation=45)
plt.show()


# In[21]:


categorical_data = data.select_dtypes(include=['object'])
for column in categorical_data.columns:
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=column, y='SalePrice', data=data)
    plt.title(f'Boxplot for SalePrice by {column}')
    plt.xticks(rotation=45)
    plt.show()



# In[22]:


# select categorical variables
categorical_vars = data.select_dtypes(include=['object']).columns

# Create the subplot
plt.figure(figsize=(15, 8))

# Browse categorical variables and create bar charts
for var in categorical_vars:
    sns.barplot(x=var, y='SalePrice', data=data, palette='viridis')
    plt.title(f'Mean SalePrice by {var}')
    plt.xlabel(var)
    plt.ylabel('Mean SalePrice')
    plt.show()


# # Prepare the dataset

# In[23]:


df = pd.get_dummies(data,dtype=float,drop_first=True)
df.head()


# In[24]:


index_to_exclude = df.columns.get_loc('SalePrice')
X = df.iloc[:, [i for i in range(df.shape[1]) if i != index_to_exclude]].values
y = df.iloc[:,index_to_exclude].values


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=329) 


# ## Dataset Normalization

# In[26]:


X_train_N= X_train.copy()
X_test_N = X_test.copy()
y_train_N = y_train.copy()

sc_X = StandardScaler()
X_train[:, :34] = sc_X.fit_transform(X_train[:, :34])
X_test[:, :34] = sc_X.transform(X_test[:, :34])
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1)).flatten()


# ## Multi LINEAR REGRESSION

# ### Train the model

# In[27]:


linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train_N, y_train_N)


# ### Evaluate the model

# In[28]:


# Predictions on the test Set
y_pred_linear = linear_reg_model.predict(X_test_N)



# Calculation of root mean square error (RMSE) on the test data
rmse_test_linear = mean_squared_error(y_test, y_pred_linear, squared=False)
print("RMSE_Linear : ", rmse_test_linear )


# Calculation of the coefficient of determination (R2) on the test data
r2_test_linear = r2_score(y_test, y_pred_linear)
print("R2_linear : ", r2_test_linear )


# 
# 

# ## Random Forest

# ### Train the model 

# In[29]:


random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train_N, y_train_N)


# ### Evaluate the model

# In[30]:


# Predictions on the test Set
y_pred_rf = random_forest_model.predict(X_test_N)



# Calculation of root mean square error (RMSE) on the test data
rmse_test_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
print("RMSE_rf : ", rmse_test_rf )



# Calculation of the coefficient of determination (R2) on the test data 
r2_test_rf = r2_score(y_test, y_pred_rf)
print("R2_rf : ", r2_test_rf )


# ## RIDGE

# ### Train the model

# In[31]:


ridge_model = Ridge()  
ridge_model.fit(X_train_N, y_train_N)


# ### Evaluate the model 

# In[32]:


# Predictions on the test Set
y_pred_ridge = ridge_model.predict(X_test_N)

# Calculation of root mean square error (RMSE) on the test data 
rmse_test_ridge = mean_squared_error(y_test, y_pred_ridge, squared=False)
print("RMSE_Ridge : ", rmse_test_ridge)


# Calculation of the coefficient of determination (R2) on the test data
r2_test_ridge = r2_score(y_test, y_pred_ridge)
print("R2_Ridge : ", r2_test_ridge )


# 

# ## LASSO

# ### Train the model

# In[33]:


lasso_model = Lasso() 
lasso_model.fit(X_train_N, y_train_N)


# ### Evaluate the model

# In[34]:


# Predictions on the test Set
y_pred_lasso = lasso_model.predict(X_test_N)


# Calculation of root mean square error (RMSE) on the test data 
rmse_test_lasso = mean_squared_error(y_test, y_pred_lasso, squared=False)
print("LASSO - RMSE_Test : ", rmse_test_lasso)

# Calculation of the coefficient of determination (R2) on the test data
r2_test_lasso = r2_score(y_test, y_pred_lasso)
print("LASSO - R2_Test : ", r2_test_lasso)


# ## Linear SVR 

# ### Train the model 

# In[35]:


svr_model = SVR(kernel='linear')  
svr_model.fit(X_train_N, y_train_N)


# ### Evaluate the model 

# In[36]:


# Predictions on the test Set
y_pred_svr = svr_model.predict(X_test_N)


# Calculation of root mean square error (RMSE) on the test data 
rmse_test_svr = mean_squared_error(y_test, y_pred_svr, squared=False)
print("SVR - RMSE_Test : ", rmse_test_svr)


# Calculation of the coefficient of determination (R2) on the test data
r2_test_svr = r2_score(y_test, y_pred_svr)
print("SVR - R2_Test : ", r2_test_svr)


# # Gradient Boosting

# ### Train the model

# In[37]:


gradient_boosting_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)  
gradient_boosting_model.fit(X_train_N, y_train_N)


# ### Evaluate the model 

# In[38]:


# Predictions on the test Set
y_pred_gradient_boosting = gradient_boosting_model.predict(X_test_N)


# Calculation of root mean square error (RMSE) on the test data 
rmse_test_gradient_boosting = mean_squared_error(y_test, y_pred_gradient_boosting, squared=False)
print("Gradient Boosting - RMSE_Test : ", rmse_test_gradient_boosting)

# Calculation of the coefficient of determination (R2) on the test data
r2_test_gradient_boosting = r2_score(y_test, y_pred_gradient_boosting)
print("Gradient Boosting - R2_Test : ", r2_test_gradient_boosting)



# In[39]:


# Nombre d'arbres dans le modèle Gradient Boosting
num_trees = 100  

# Entraînement du modèle avec différentes quantités d'arbres
logs = np.zeros((num_trees, len(y_test)))

for i, y_pred in enumerate(gradient_boosting_model.staged_predict(X_test_N)):
    logs[i, :] = mean_squared_error(y_test, y_pred, squared=False)

# Création du diagramme RMSE en fonction du nombre d'arbres
plt.plot(range(1, num_trees + 1), logs)
plt.xlabel("Nombre d'arbres")
plt.ylabel("RMSE")
plt.title("Évolution du RMSE avec le nombre d'arbres pour le modèle Gradient Boosting")
plt.show()


# ## Decision Tree

# ### Train the model

# In[40]:


decision_tree_model = DecisionTreeRegressor(random_state=42)  
decision_tree_model.fit(X_train_N, y_train_N)


# ### Evaluate the model 

# In[41]:


# Predictions on the test Set
y_pred_decision_tree = decision_tree_model.predict(X_test_N)

# Calculation of root mean square error (RMSE) on the test data 
rmse_test_decision_tree = mean_squared_error(y_test, y_pred_decision_tree, squared=False)
print("Decision Tree - RMSE_Test : ", rmse_test_decision_tree)


# Calculation of the coefficient of determination (R2) on the test data
r2_test_decision_tree = r2_score(y_test, y_pred_decision_tree)
print("Decision Tree - R2_Test : ", r2_test_decision_tree)


# ## compare models

# In[42]:


models = pd.DataFrame({
    'Model': [
        'Multiple Linear Regression',  'Random Forest', 'Ridge', 'Lasso', 'Linear SVR','Gradient Boosting', 'Decision Tree',
       
    ],

    'Testing R2 Score': [
       r2_test_linear, r2_test_rf, r2_test_ridge, r2_test_lasso, r2_test_svr, r2_test_gradient_boosting, r2_test_decision_tree 
    ],
    
    'Testing Mean Square Error': [
       rmse_test_linear, rmse_test_rf, rmse_test_ridge, rmse_test_lasso, rmse_test_svr, rmse_test_gradient_boosting, rmse_test_decision_tree 
    ]
})


# In[43]:


models.sort_values(by='Testing R2 Score', ascending=False).style.background_gradient(
        cmap='Greens')


# In[45]:


# Paramètres du graphique
bar_width = 0.35
index = np.arange(len(models['Model']))

# Création du graphique
fig, ax1 = plt.subplots(figsize=(12, 6))

# Barres pour le R2
bar_r2 = ax1.bar(index, models['Testing R2 Score'], bar_width, label='R2', color='b')

# Configurations pour le R2
ax1.set_xlabel('Modèles')
ax1.set_ylabel('R2 Score', color='b')
ax1.tick_params('y', colors='b')
ax1.set_xticks(index)
ax1.set_xticklabels(models['Model'], rotation=45, ha='right')

# Création d'un axe y supplémentaire pour le RMSE
ax2 = ax1.twinx()
bar_rmse = ax2.bar(index + bar_width, models['Testing Mean Square Error'], bar_width, label='RMSE', color='r')

# Configurations pour le RMSE
ax2.set_ylabel('Mean Square Error (RMSE)', color='r')
ax2.tick_params('y', colors='r')


fig.tight_layout()
fig.legend([bar_r2, bar_rmse], ['R2', 'RMSE'], loc='upper left')
plt.title('Comparaison des Modèles - R2 et RMSE')

# Affichage du graphique
plt.show()


# # SOUMMISSION
# 

# In[46]:


#Data test importation
test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv", index_col="Id")

#Displaying the first elements of data test
test_data.head()




# In[47]:


# Calculate the percentage of missing values for each column
missing_percentage_test = (test_data.isnull().sum() / test_data.shape[0]) * 100

# Filter columns with missing values
columns_with_missing_values_test = missing_percentage_test[missing_percentage_test > 0]

# Show percentage of missing values for each column
print(columns_with_missing_values_test)

# supprimer les collones dans le fichier test.csv
test_data.drop(columns=['MasVnrType','PoolQC', 'PoolArea'],inplace=True)

#Remplacer les données manquantes
test_data['LotFrontage'] = test_data['LotFrontage'].fillna(0)
test_data['Alley'] = test_data['Alley'].fillna("No Alley")
test_data['MasVnrArea'] = test_data['MasVnrArea'].fillna(test_data['MasVnrArea'].mean())
test_data['BsmtQual'] = test_data['BsmtQual'].fillna("NO Bsmt")
test_data['BsmtCond'] = test_data['BsmtCond'].fillna("NO Bsmt")
test_data['BsmtExposure'] = test_data['BsmtExposure'].fillna("NO Bsmt")
test_data['BsmtFinType1'] = test_data['BsmtFinType1'].fillna("NO Bsmt")
test_data['BsmtFinType2'] = test_data['BsmtFinType2'].fillna("NO Bsmt")
test_data['Electrical'] = test_data['Electrical'].fillna(test_data['Electrical'].mode()[0])
test_data['FireplaceQu'] = test_data['FireplaceQu'].fillna("NO Fireplace")
test_data['GarageType'] = test_data['GarageType'].fillna("NO Garage")
test_data['GarageYrBlt'] = test_data['GarageYrBlt'].fillna(0)
test_data['GarageFinish'] = test_data['GarageFinish'].fillna("NO Garage")
test_data['GarageQual'] = test_data['GarageQual'].fillna("NO Garage")
test_data['GarageCond '] = test_data['GarageCond'].fillna("NO Garage")
test_data['Fence'] = test_data['Fence'].fillna("NO Fence")
test_data['MiscFeature'] = test_data['MiscFeature'].fillna("NO MiscFeature")
test_data['Exterior1st'] = test_data['Fence'].fillna("NO Exterior1st")
test_data['Exterior2nd'] = test_data['Fence'].fillna("NO Exterior2nd")
test_data['BsmtFinSF1'] = test_data['BsmtFinSF1'].fillna(0)
test_data['BsmtFinSF2'] = test_data['BsmtFinSF2'].fillna(0)
test_data['BsmtUnfSF'] = test_data['BsmtUnfSF'].fillna(0)
test_data['TotalBsmtSF'] = test_data['TotalBsmtSF'].fillna(0)
test_data['BsmtFullBath'] = test_data['BsmtFullBath'].fillna(0)
test_data['BsmtHalfBath'] = test_data['BsmtHalfBath'].fillna(0)

test_data['KitchenQual'] = test_data['KitchenQual'].fillna("NO KitchenQual")
test_data['Functional'] = test_data['Functional'].fillna("NO Functional")

test_data['GarageCars'] = test_data['GarageCars'].fillna(0)
test_data['GarageArea'] = test_data['GarageArea'].fillna(0)
test_data['GarageCond'] = test_data['GarageCond'].fillna("NO GarageCond")
test_data['SaleType'] = test_data['SaleType'].fillna("NO SaleType")
test_data['Utilities'] = test_data['Utilities'].fillna("NO Utilities")
test_data['MSZoning'] = test_data['MSZoning'].fillna("NO MSZoning")
 


# In[48]:


test_data.info()


# In[49]:


df_test = pd.get_dummies(test_data,dtype=float,drop_first=True)
df_test.head()


# In[51]:


#Correction des Colonnes Manquantes dans le Jeu de Données de Test
missing_columns = set(df.columns) - set(df_test.columns)
for col in missing_columns:
    df_test[col] = 0


# In[52]:


# Réorganiser les colonnes dans l'ordre de l'ensemble d'entraînement
df_test = df_test[df.columns]


# In[53]:


# Supprimez la colonne 'SalePrice' de l'ensemble de test
df_test = df_test.drop(['SalePrice'], axis=1)


# In[54]:


random_forest_model.fit(df.drop(['SalePrice'],axis=1),df['SalePrice'])

# Effectuez les prédictions  avec rf
test_predictions = random_forest_model.predict(df_test)

submission_df = pd.DataFrame({
    'Id': df_test.index, 
    'SalePrice': test_predictions
})



submission_df.to_csv('submission_rf.csv', index=False)
df_sub = pd.read_csv("/kaggle/working/submission_rf.csv")
df_sub.head()


# In[55]:


# Effectuez les prédictions  avec gradient Boosting
test_predictions = gradient_boosting_model.predict(df_test)

#Création du datafram
submission_df = pd.DataFrame({
    'Id': df_test.index, 
    'SalePrice': test_predictions
})




#création du fichier de soumission sur kaggle 
submission_df.to_csv('submission.csv', index=False)
#Afficher les premiers éléments du dataframe 
df_sub_grad = pd.read_csv("/kaggle/working/submission.csv")
df_sub_grad.head()

