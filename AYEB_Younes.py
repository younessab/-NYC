#!/usr/bin/env python
# coding: utf-8

# # New York City Taxi Trip Duration

# ## Import

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


train=pd.read_csv("C:/nyctaxitripduration/train/train.csv")
test=pd.read_csv("C:/nyctaxitripduration/test/test.csv")


# ## Data explore 

# In[12]:


test.info()
train.info()


# In[16]:


train.describe()


# In[25]:


train.dtypes


# ## Outliers
# 

# In[28]:


plt.subplots(figsize=(14,5))
plt.title("Outliers visualization")
train.boxplot();


# ### Comme les deux colonnes ne sont pas de type date on va les convertir pour pouvoir les exploiter
# 
# pickup_datetime        object
# 
# dropoff_datetime       object

# In[31]:


train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')


# ### Un trajet est considéré comme etant un trajet il faut avoir un minimum d'un passager

# In[32]:


train = train[train['passenger_count']> 0]


# In[45]:


x_train = train[["vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude"]]


# In[36]:


y=train['trip_duration']
y.describe()


# ### On va importer les bibliothèques pour faire le random forest

# In[37]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score


# In[41]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[42]:


rand = RandomForestRegressor()


# In[43]:


rand.fit(x_train,y_train)


# ### Validation

# In[46]:


x_test = test[["vendor_id", "passenger_count","pickup_longitude", "pickup_latitude","dropoff_longitude","dropoff_latitude"]]
prediction = rand.predict(x_test)
prediction


# ### Predections

# In[47]:


sub = pd.read_csv("C:/nyctaxitripduration/sub/sample_submission.csv")
sub.head()


# In[48]:


submission = pd.DataFrame({'id': test.id, 'trip_duration': np.exp(prediction)})
submission.head()


# In[49]:


submission.to_csv('submission.csv', index=False)

