#!/usr/bin/env python
# coding: utf-8

# In[285]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[286]:


df = pd.read_csv('IRIS.csv')


# In[287]:


df


# In[288]:


df.species.unique()


# In[289]:


df.info()


# In[290]:


df.describe()


# In[291]:


df['species']=df['species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})


# In[292]:


df['species'].unique()


# In[293]:


df.columns


# In[294]:


X = df.iloc[:,:-1]


# In[295]:


X


# In[296]:


Y = df['species']


# In[297]:


Y


# In[298]:


df['species'].dtype


# In[299]:


df.info()


# In[300]:


from sklearn.model_selection import train_test_split


# In[301]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.33, random_state=42)


# In[302]:


X_train


# In[303]:


X_test


# In[304]:


X_test.shape


# In[305]:


Y_train


# In[306]:


Y_test


# In[307]:


parameter = {'criterion': ['ginni', 'entropy', 'log_loss'],
             'splitter':['best','random'],
              'max_depth':[1,2,3,4,5],
              'max_features':['auto','sqrt','log2']
            }


# In[308]:


from sklearn.model_selection import GridSearchCV


# In[309]:


from sklearn.tree import DecisionTreeClassifier


# In[310]:


treemodel = DecisionTreeClassifier()
cv = GridSearchCV(treemodel,param_grid =parameter,cv=5,scoring='accuracy')


# In[311]:


cv.fit(X_train,Y_train)


# In[312]:


cv.best_params_


# In[313]:


y_pred = cv.predict(X_test)


# In[314]:


y_pred


# In[315]:


Y_test


# In[316]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[317]:


score = accuracy_score(y_pred,Y_test)


# In[318]:


score


# In[319]:


confusion_matrix(y_pred,Y_test)


# In[320]:


print(classification_report(y_pred,Y_test))


# In[ ]:





# In[ ]:





# In[ ]:




