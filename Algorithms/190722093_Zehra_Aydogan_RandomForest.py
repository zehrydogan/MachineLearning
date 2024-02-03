#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[4]:


dataset = pd.read_csv('house.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[6]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[7]:


X_train


# In[8]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[13]:


classifier.estimators_


# In[16]:


from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(classifier.estimators_[5],filled=True)


# In[9]:


print(classifier.predict(sc.transform([[4.9,3,1.4,0.2]])))


# In[10]:


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[11]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[ ]:




