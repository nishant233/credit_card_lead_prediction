#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


# In[2]:


train_data = pd.read_csv("train.csv", index_col = None)
test_data = pd.read_csv("test.csv", index_col = None)


# In[3]:


train_data.head()


# In[4]:


test_data.head()


# In[5]:


train_data.isnull().sum()


# In[6]:


test_data.isnull().sum()


# In[7]:


train_data = train_data.drop(columns = "ID", axis = 0)
test_data = test_data.drop(columns = "ID",axis = 0)


# In[8]:


train_data.head()


# In[9]:


test_data.head()


# In[10]:


#Deleted all rows which hae null value

train_data = train_data.dropna()
test_data = test_data.dropna()


# In[11]:


train_data.isnull().sum()


# In[12]:


test_data.isnull().sum()


# In[13]:


train_data.describe()


# In[14]:


test_data.describe()


# In[15]:


train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)


# In[16]:


train_data.head()


# In[17]:


test_data.head()


# In[42]:


features = train_data.drop(columns = ["Is_Lead"])
labels = np.array(train_data['Is_Lead'])


# In[43]:


labels


# In[44]:


from sklearn.model_selection import cross_val_score, train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42) 


# In[45]:


print(train_features.shape)
print(test_features.shape)
print(train_labels.shape)


print(test_labels.shape)


# In[46]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

model.fit(train_features, train_labels)


# In[47]:


prediction = model.predict(test_features)


# In[51]:


prediction


# In[64]:


test_labels


# In[76]:


predic_zerocount = 0
predic_onecount = 0
test_zerocount = 0
test_onecount = 0
for x in prediction:
    if x == 0:
        predic_zerocount = predic_zerocount+1
    else:
        predic_onecount = predic_onecount+1

for y in test_labels:
    if y == 0:
        test_zerocount = test_zerocount+1
    else:
        test_onecount = test_onecount+1
        
print(predic_zerocount)
print(predic_onecount)
print(test_zerocount)
print(test_onecount)

error = (test_zerocount-predic_zerocount)
print(error)

print("Error Percentage: ", (734/54100)*100)


# In[ ]:





# In[ ]:





# In[ ]:




