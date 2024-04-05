#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


df = pd.read_csv("C:\\Users\\dell\\OneDrive\\Desktop\\ds_salaries.csv")


# In[5]:


df.info()


# In[6]:


##PART 1: Data Cleaning

#Checking for null values, duplicate rows, and dropping unneeded columns.

df.isna().sum()


# In[7]:


df.describe()


# In[8]:


#Making a copy of the original data in case of errors and if we need it again later.
df_original = df.copy()
#Droping the first column Unnamed:0 and checking for duplicate rows.
df.drop(columns=['Unnamed: 0'], inplace=True)
duplicate_rows = df.duplicated()
df[duplicate_rows].head(10)


# In[9]:


#Droping the duplicates
df.drop_duplicates(inplace=True)
df.drop(columns=['salary','salary_currency'], inplace=True)
df.info()


# In[12]:


##PART 2: Visualization
#Plotting Salary and Number of Jobs based on various factors

import seaborn as sns
import matplotlib.pyplot as plt



# In[13]:


sns.set_style("whitegrid")

# Distribution of salaries in USD
plt.figure(figsize=(10, 6))
sns.histplot(df['salary_in_usd'], bins=30, kde=True)
plt.title('Distribution of Salaries (in USD)')
plt.xlabel('Salary (in USD)')
plt.ylabel('Frequency')
plt.show()


# In[14]:


##Distribution of Salaries:
#Most of the salaries seem to be concentrated in the lower range (~$100,000), and there are a few outliers in the higher salary range


#Plotting Salary vs Company Location (Country)
sorted_countries = df.groupby('company_location')['salary_in_usd'].mean().sort_values(ascending=False).index
plt.figure(figsize=(12, 6))
sns.barplot(x='company_location', y='salary_in_usd', data=df, order=sorted_countries)
plt.title('Salary Distribution Across Company Locations')
plt.xlabel('Country')
plt.ylabel('Salary (in USD)')
plt.xticks(rotation=90)
plt.show()


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.countplot(x='company_location', data=df, order=df['company_location'].value_counts().index)

plt.xlabel('Country')
plt.ylabel('Number of Jobs')
plt.title('Number of Jobs vs Company Location')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[16]:


##Number of Jobs by Company Location:¶
#We observe that a significant number of Jobs in our Data (>50%) are from the United States. However, companies located in Russia only account for 2 Jobs in our data. Because of fewer jobs, but much higher salaries, we are getting an incomplete picture, and this may skew our analysis or modeling.



sorted_roles = df.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False).index
plt.figure(figsize=(12, 6))
sns.barplot(x='job_title', y='salary_in_usd', data=df, order=sorted_roles)
plt.title('Salary Distribution Across Job Titles')
plt.xlabel('Job Title')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()


# In[17]:


##Number of Jobs based on Job Title

sorted_exp = ['EN', 'MI', 'SE', 'EX'] #Order by Seniority
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='experience_level', y='salary_in_usd', order=sorted_exp)
plt.title('Median Salaries vs. Experience Level')
plt.xlabel('Experience Level')
plt.ylabel('Salary (in USD)')
plt.show()


# In[18]:


##Median Salaries based on Experience Level¶
#Not surprisingly, Entry level jobs have the lowest salaries, followed by Mid Level jobs and Senior Level jobs, and Executive Level jobs have the highest salaries.


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.countplot(x='experience_level', data=df, order=sorted_exp)

plt.xlabel('Experience Level')
plt.ylabel('Number of Jobs')
plt.title('Number of Jobs vs Experience Level')
plt.xticks(rotation=0, ha="right")

plt.tight_layout()  
plt.show()


# In[19]:


#Number of Jobs based on Experience Level
#Executive level jobs are the least common, followed by Entry Level jobs. Mid-Level and Senior-Level Jobs account for about 80% of the Jobs in our data


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='employment_type', y='salary_in_usd', order=df.groupby('employment_type')['salary_in_usd'].median().sort_values().index)
plt.title('Median Salaries vs. Employment Type')
plt.xlabel('Employment Type')
plt.ylabel('Salary (in USD)')
plt.show()


# In[22]:


#Job Salaries vs Employment Type.


import seaborn as sns
import matplotlib.pyplot as plt

sorted_emp = df['employment_type'].value_counts().index

plt.figure(figsize=(8, 6))
sns.countplot(x='employment_type', data=df, order=sorted_emp)

ax = plt.gca()
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(1, 7), 
                textcoords='offset points',
                fontsize=16)

plt.xlabel('Employment Type')
plt.ylabel('Number of Jobs')
plt.xticks(rotation=0, ha="right")  
plt.title('Number of Jobs vs Employment Type')

plt.tight_layout() 
plt.show()



# In[23]:


#Number of Jobs vs Employment Type

sorted_size = ['S','M','L']
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='company_size', y='salary_in_usd', order=sorted_size)
plt.title('Median Salaries vs. Company Size')
plt.xlabel('Company Size')
plt.ylabel('Salary (in USD)')
plt.show()


# In[24]:


#Median Salaries vs Company Size



plt.figure(figsize=(8, 6))
sns.countplot(x='company_size', data=df, order=df['company_size'].value_counts().index)

ax = plt.gca()
for p in ax.patches:
    ax.annotate(f"{p.get_height():.0f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(1, 7), textcoords='offset points', fontsize=16)

plt.xlabel('Company Size')
plt.ylabel('Number of Jobs')
plt.xticks(rotation=0, ha="right")  
plt.title('Number of Jobs vs Company Size')

plt.tight_layout() 
plt.show()


# In[25]:


#PART 3: Model Building and Evaluation

threshold = 10

# Replace values with 'Other' if count is below threshold
replace_with_other = lambda col: col.replace(col.value_counts()[col.value_counts() < threshold].index, 'Other')
df['employee_residence'] = replace_with_other(df['employee_residence'])
df['company_location'] = replace_with_other(df['company_location'])
df['job_title'] = replace_with_other(df['job_title'])

# One-hot Encoding
df_encoded = pd.get_dummies(df, columns=['experience_level', 'employment_type', 'employee_residence', 'company_location', 'company_size', 'job_title'], drop_first=True)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_encoded[['salary_in_usd', 'remote_ratio']] = scaler.fit_transform(df_encoded[['salary_in_usd', 'remote_ratio']])

df_encoded.head()


# In[26]:


#Create correlation matrix

corr_matrix = df_encoded.corr()

plt.figure(figsize=(16, 13))
sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

