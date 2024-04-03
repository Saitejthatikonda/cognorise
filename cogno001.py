#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


data=pd.read_csv("C:/Users/dell/OneDrive/Desktop/Unemployment in India (1).csv")


# In[3]:


data


# In[4]:


## Display basic information about the dataset
data.info()


# In[5]:


## Display statistical summary
data.describe()


# In[7]:


## Drop rows with missing values
data.dropna(inplace=True)


# In[8]:


## Check for missing values
data.isnull().sum()


# In[10]:


data.columns


# In[11]:


pd.DataFrame(data.iloc[:,3])


# In[12]:


## Visualize the distribution of unemployment rate
sns.histplot(data.iloc[:,3], kde=True)
plt.title('Distribution of Unemployment Rate')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Frequency')
plt.show()


# In[13]:


## Drop irrelevant columns
data.drop(['Region', ' Frequency','Area'], axis=1, inplace=True)


# In[14]:


## Convert 'Date' column to datetime format
data[' Date'] = pd.to_datetime(data[' Date'])


# In[15]:


data.info()


# In[16]:


## Set 'Date' column as index
data.set_index(' Date', inplace=True)


# In[17]:


## Plotting time series of unemployment rate
sns.histplot(data[' Estimated Unemployment Rate (%)'], kde=True)
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.show()


# In[18]:


## Define features and target
X = data.drop(' Estimated Unemployment Rate (%)', axis=1)
y = data[' Estimated Unemployment Rate (%)']


# In[19]:


## Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[20]:


## Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)


# In[21]:


## Predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)


# In[22]:


## Evaluation metrics
print("Training MSE:", mean_squared_error(y_train, train_preds))
print("Testing MSE:", mean_squared_error(y_test, test_preds))
print("Training R^2:", r2_score(y_train, train_preds))
print("Testing R^2:", r2_score(y_test, test_preds))


# In[ ]:




