#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


import pandas as pd

# Using double backslashes to escape each backslash
df = pd.read_csv("C:\\Users\\dell\\Downloads\\archive\\cereal.csv")

# Alternatively, using raw string literal (prefixing with 'r')
# df = pd.read_csv(r"C:\Users\dell\Downloads\archive\cereal.csv")

# Continue with your code...


# In[6]:


# displaying a sample of the dataset to get an overview of the data structure
df.head(20)


# In[7]:


df.info()


# In[8]:


# Removing rows with negative values in 'sugars' and 'potass' columns
df = df[(df['sugars'] > 0) & (df['potass'] > 0)]


print(df)


# In[9]:


df.describe()


# In[10]:


# missing values in the dataset?
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)


# In[11]:


print(df.duplicated().sum())


# In[14]:


#  unique manufacturers in the dataset?
# creating a dictionary to map the manufacturer codes to their full names
manufacturer_mapping = {
    'A': 'American Home Food Products',
    'G': 'General Mills',
    'K': 'Kelloggs',
    'N': 'Nabisco',
    'P': 'Post',
    'Q': 'Quaker Oats',
    'R': 'Ralston Purina'
}

# replacing the manufacturer codes with their full names in the 'mfr' column
df['mfr'] = df['mfr'].map(manufacturer_mapping)


# In[15]:


unique_manufacturers = df['mfr'].unique()
print("Unique manufacturers:", unique_manufacturers)


# In[17]:


# exploring the distribution of cereal types (cold vs. hot)
cereal_type_counts = df['type'].value_counts()
print("Cereal Type Distribution:\n", cereal_type_counts)


# In[18]:


# investigating the summary statistics of numerical features
numerical_columns = ['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass', 'vitamins', 'weight', 'cups']
numerical_stats = df[numerical_columns].describe()
print("Summary Statistics of Numerical Features:\n", numerical_stats)


# In[19]:


# creating a bar chart showing the count of cereals by manufacturer (Deliverable)
# bar chart showing the count of cereals by manufacturer
plt.figure(figsize=(10, 6))
sns.countplot(x='mfr', data=df)
plt.xlabel('Manufacturer')
plt.ylabel('Count')
plt.title('Cereal Count by Manufacturer')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[20]:


plt.figure(figsize=(10, 6))
plt.hist(df['calories'], bins=10, edgecolor='black')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.title('Distribution of Calories')
plt.show()


# In[21]:


# summary statistics of vitamins
vitamins_stats = df['vitamins'].describe()

# visualization of the distribution of vitamins
plt.figure(figsize=(7, 5))
plt.hist(df['vitamins'], bins=3, edgecolor='black')
plt.xlabel('Vitamins')
plt.ylabel('Frequency')
plt.title('Distribution of Vitamins')
plt.xticks([0, 25, 100], ['0%', '25%', '100%'])
plt.show()


# In[22]:


# summary statistics of rating grouped by shelf placement
rating_stats = df.groupby('shelf')['rating'].describe()

# box plot of rating by shelf placement
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='shelf', y='rating')
plt.xlabel('Shelf Placement')
plt.ylabel('Rating')
plt.title('Relationship between Shelf Placement and Rating')
plt.show()

# shelf: display shelf (1, 2, or 3, counting from the floor)


# In[23]:


# Select the top 10 cereals with the highest ratings
top_10_cereals = df.nlargest(10, 'rating')

# Create a horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(top_10_cereals['name'], top_10_cereals['rating'])
plt.xlabel('Rating')
plt.ylabel('Cereal')
plt.title('Top 10 Cereals with the Highest Ratings')
plt.tight_layout()
plt.show()


# In[27]:


# Scatter plot: Calories vs Protein
plt.scatter(df['calories'], df['protein'])
plt.xlabel('Calories')
plt.ylabel('Protein')
plt.title('Calories vs Protein')
plt.show()


# In[28]:


# Created a pivot table with "mfr" and "type" as rows and columns, and "rating" as values
pivot_table = df.pivot_table(values='rating', index='mfr', columns='type')

# Create the heatmap using seaborn
sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt=".1f")

# Set the plot title and labels
plt.title('Cereal Rating by Manufacturer and Type')
plt.xlabel('Type')
plt.ylabel('Manufacturer')

# Display the heatmap
plt.show()


# In[29]:


nutritional_factors = ['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass', 'vitamins']
manufacturer_labels = ['American Home Food Products', 'General Mills', 'Kelloggs', 'Nabisco', 'Post', 'Quaker Oats', 'Ralston Purina']

mean_values = df.groupby('mfr')[nutritional_factors].mean()

plt.figure(figsize=(12, 6))
ax = mean_values.plot(kind='bar')
plt.xlabel('Manufacturer')
plt.ylabel('Mean Value')
plt.title('Nutritional Content by Manufacturer')
ax.legend(nutritional_factors, loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(range(len(manufacturer_labels)), manufacturer_labels, rotation='horizontal', ha='right')
plt.show()


# In[30]:


md = df[['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass']].groupby(df['mfr']).mean().reset_index()
sns.barplot(x='value', y='variable', hue='mfr', data=pd.melt(md, id_vars='mfr', var_name='variable'))
plt.title('Mean Nutrition Values by Manufacturer')
plt.xlabel('Mean Value')
plt.ylabel('Nutrition Features')
plt.legend(title='Manufacturers', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[31]:


import matplotlib.pyplot as plt

try:
    hr = df.groupby('mfr')['rating'].max().reset_index().sort_values('rating', ascending=False)
    plt.pie(hr['rating'], labels=hr['mfr'], autopct='%1.1f%%', startangle=90)
    plt.title('Highest-Rated Cereal Manufacturers')
    plt.axis('equal')
    plt.show()
except Exception as e:
    print("An error occurred:", e)


# In[32]:


# Calculate the average potassium for each manufacturer
avg_potassium = df.groupby('mfr')['potass'].mean().reset_index()

# Sort the data by potassium in descending order
avg_potassium = avg_potassium.sort_values('potass', ascending=False)

# Plot the average potassium values
plt.bar(avg_potassium['mfr'], avg_potassium['potass'])

# Add labels and title
plt.xlabel('Manufacturer')
plt.ylabel('Average Potassium')
plt.title('Average Potassium Content by Manufacturer')
plt.xticks(rotation=45, ha='right')

# Annotate the highest potassium value if available
nabisco_potassium = avg_potassium.loc[avg_potassium['mfr'] == 'N', 'potass']
if not nabisco_potassium.empty:
    nabisco_potassium = nabisco_potassium.values[0]
    plt.annotate('Nabisco', (2, nabisco_potassium), xytext=(2.5, nabisco_potassium + 50),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)

# Display the plot
plt.show()


# In[ ]:




