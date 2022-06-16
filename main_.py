#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This is a prediction model for the california housing project
#A project on prediction of median housing price for districts
#The error rate of this model should at least be less than 15%


# In[2]:


#data manipulation and linear algebra libraries
import pandas as pd
import numpy as np

#visualization libraries
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns

#preprocessing libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

#feature engineering
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

#Model Selection and Training


# Loading Data and Initial Exploratory Data Analysis

# In[3]:


housing = pd.read_csv('housing.csv')
housing.info()
housing.describe() #statistical measures for the numerical values


# In[4]:


housing['ocean_proximity'].value_counts() 
#exploring the categorical variable


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
housing.hist(bins=20, figsize=(20,15), color ='#b492ff')
plt.show()


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')

sns.jointplot(x='median_income',y='median_house_value',data= housing,kind = 'scatter')


# Training and Testing Set Splits

# In[7]:


train_set,test_set = train_test_split(housing, test_size= 0.2, random_state=42)
print(train_set.shape, test_set.shape)


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.histplot(housing['median_income'], bins= 10)
sns.histplot(test_set['median_income'], bins= 10) ##checking if the test set is representative of the data


# In[9]:


#fig, ax 


# In[10]:


#creating a discrete income category attribute from median income
housing['income_cat'] = np.ceil(housing['median_income']/1.5) #rounding up
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace= True)
housing['income_cat'].value_counts()


# In[11]:


sss = StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=42)
#This sss class creates a set of list of indexes given n_splits number of iterations

for train_index, test_index in sss.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#To check wheher the test data set has the same proportion to the entire data set
#To ensure it is representative of the data
print(round((housing["income_cat"].value_counts() / len(housing)),2))
round((strat_test_set['income_cat'].value_counts()/len(strat_test_set)),2)


# In[12]:


#dropping the income_cat attribute to restore the original data
for set in (strat_train_set, strat_test_set):
    set.drop(['income_cat'], axis=1, inplace = True)


# Training Set Data Visualization

# In[13]:


#Exclusively using the train_set
housing = strat_train_set.copy()


# In[14]:


#mapping out the pop density 
get_ipython().run_line_magic('matplotlib', 'inline')
housing.plot(kind ='scatter', x = 'longitude', y = 'latitude',figsize =(10,15), color='#712a95', alpha = 0.25)
plt.show()


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
housing.plot(kind ='scatter', x = 'longitude', y = 'latitude', figsize=(10,15),
             alpha = 0.45, s=housing['population']/100, label='Population',
            c='median_house_value', cmap='jet', colorbar=True)
plt.legend()


# Feature Engineering & Data Cleaning

# In[16]:


#looking for correlations between the target variable and other attributes
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)


# In[17]:


##selecting the attributes with the highest correlation
attrs = ['median_house_value', 'median_income','total_rooms','housing_median_age']
scatter_matrix(housing[attrs], figsize =(15,10), color='#712a95')
plt.show()


# In[18]:


#combining attributes
housing['rooms/household']= housing['total_rooms']/housing['households']
housing['bedroom_to_room']=housing['total_bedrooms']/housing['total_rooms']
housing["population_per_household"]=housing["population"]/housing["households"]


# In[19]:


#checking correlation of new attributes
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)


# In[20]:


housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()


# In[ ]:





# In[21]:


#using sklearn Imputer to deal with missing values
imputer = SimpleImputer(strategy='median')

#dropping the ocean_proximity attr since median can only be computed on num values
housing_num = housing.drop('ocean_proximity', axis=1)
housing_num.isnull().sum()
#fitting the imputer instance to the data
imputer.fit(housing_num)
#transforming the training set by replacing the missing values w/ learned medians
X = imputer.transform(housing_num) #returns np array


# In[22]:


#imputer.statistics_ stores the results of the imputer
imputer.statistics_ == housing_num.median().values


# In[23]:


#converting the numpy array back to a df
housing_tr = pd.DataFrame(X, columns =housing_num.columns)
housing_tr.head()


# Handling of Text and categorical variables

# In[25]:


encoder = LabelBinarizer(sparse_output = True)
housing_cat = housing['ocean_proximity']
#sparse_output converts the output from ndarray to sparse matrix ##saving on memory
housing_cat_1hot = encoder.fit_transform(housing_cat)


# Custom Transformers for Feature Scaling & Transformation Pipelines

# In[26]:


#a custom transformer for combining the numerical attributes
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_br_per_room = True):  ##setting the hyperparameter add_br_per_room to True
        self.add_br_per_room = add_br_per_room
    def fit(self = X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:,3] / X[:,6]
        pop_per_household = X[:,5] / X[:,6]
        if self.add_br_per_room:
            br_per_room = X[:,4] / X[:, 3]
            return np.c_[X, rooms_per_household, pop_per_household, br_per_room]
        else:
            return np.c_[X, rooms_per_household, pop_per_household]
        
# attr_adder  = CombinedAttributesAdder(add_br_per_room = False)
# housing_extra_attr = attr_adder.transform(housing_num)


# In[27]:


##creating a custom transformer and a pipeline for numerical attributes

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names=attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    
num_attribs = list(housing_num)

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)), #fit_transform()
    ('imputer', SimpleImputer(strategy='median')),#fit_transform()
    ('attrib_adder', CombinedAttributesAdder()),  #fit_transform()
    ('std_scaler',StandardScaler())               #fit()
])

housing_num_tr = num_pipeline.fit_transform(housing_num)
print(type(housing_num_tr))


# In[28]:


cat_attribs = ['ocean_proximity']

#creating a custom transformer: MylabelBinarizer that takes 3 args instead of 2
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
         self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)

##categorical variables pipeline
cat_pipeline =Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', MyLabelBinarizer()),
])
    


# In[29]:


#joining transformers using FeatureUnion
full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared.shape
#housing


# Selecting and Trainig a Model

# In[30]:


#Training a linear regression model
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[31]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print('Predictions:\t',lin_reg.predict(some_data_prepared))
print('Labels:\t\t', list(some_labels))


# In[32]:


from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse) #root mean square error
lin_rmse

This implies that the model has a typical prediction error of $68,631
This is an underfitting model
# In[33]:


#using decision tree to find any complex non-linear rlshp
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# In[34]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

rmse is very unlikely to be 0; might be a case of overfitting data
# Performing K-fold Cross-Validation

# In[35]:


from sklearn.model_selection import cross_val_score

tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring='neg_mean_squared_error',cv=10)
tree_rmse_scores =np.sqrt(-tree_scores)  #cross-validation expects a utility function hence the -scores
tree_rmse_scores


# In[36]:


def display_scores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard deviation:', scores.std())
    
display_scores(tree_rmse_scores)


# In[37]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                         scoring='neg_mean_squared_error',cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# Ensemble Learning

# In[38]:


#using a random forest generator
from sklearn.ensemble import RandomForestClassifier

forest_reg = RandomForestClassifier()
forest_reg.fit(housing_prepared, housing_labels)
forest_prediction = forest_reg.predict(housing_prepared)
foreste_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
display_scores(forest_rmse)


# In[ ]:




