#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('mobile_dataset.csv')


# In[4]:


df


# In[5]:


df.columns


# In[6]:


df.isnull().sum()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


X = df.iloc[:,:-1]
Y = df.iloc[:,-1]


# In[10]:


X


# In[11]:


Y


# In[12]:


from sklearn.feature_selection import SelectKBest


# In[13]:


from sklearn.feature_selection import chi2


# In[14]:


ordered_rank_features = SelectKBest(score_func = chi2, k=20)


# In[15]:


ordered_features =ordered_rank_features.fit(X,Y)


# In[16]:


ordered_features


# In[17]:


dfscores = pd.DataFrame(ordered_features.scores_, columns = ['score'])


# In[18]:


dfcolumns = pd.DataFrame(X.columns)


# In[19]:


features_rank = pd.concat([dfcolumns,dfscores], axis =1)


# In[20]:


features_rank.columns = ['Features','Scores']


# In[21]:


features_rank


# In[22]:


features_rank.nlargest(10,'Scores')


# In[23]:


from sklearn.ensemble import ExtraTreesClassifier


# In[24]:


model = ExtraTreesClassifier()


# In[25]:


model.fit(X,Y)


# In[26]:


print(model.feature_importances_)


# In[31]:


ranked_features = pd.Series(model.feature_importances_, index = X.columns)
ranked_features.nlargest(10).plot(kind = 'bar')
plt.show()


# In[39]:


ranked_features


# In[40]:


df.corr()


# In[38]:


import seaborn as sns
corr = df.iloc[:,:-1].corr()
top_features = corr.index
plt.figure(figsize = (20,20))
sns.heatmap(df[top_features].corr(), annot = True)


# In[ ]:


threshold = 0.5


# In[41]:


df[top_features].corr()


# In[42]:


top_features


# Information Gain

# In[44]:


from sklearn.feature_selection import mutual_info_classif


# In[45]:


mutual_info = mutual_info_classif(X,Y)


# In[46]:


mutual_info


# In[47]:


mutual_data = pd.Series(mutual_info, index = X.columns)


# In[48]:


mutual_data


# In[ ]:




