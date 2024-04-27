#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 


# In[4]:


df = pd.read_csv(r"D:\project\Bias_correction_ucl.csv")


# In[5]:


df


# In[6]:


df.drop("Date",axis=1,inplace=True)


# In[8]:


df


# In[7]:


df.head() 


# In[9]:


df.describe


# In[10]:


y = np.asarray(df.Next_Tmax) 
X = np.asarray(df.drop("Next_Tmax",axis=1)) 
X = np.nan_to_num(X) 
y = np.nan_to_num(y) 
print(np.isnan(X).sum()) 
print(np.isnan(X).sum())


# In[11]:


from sklearn.preprocessing import StandardScaler 
s = StandardScaler() 
X = s.fit_transform(X)  
X.shape


# In[12]:


import seaborn as sns 
import matplotlib.pyplot as plt 
plt.figure(figsize=(22,22))  
sns.heatmap(df.corr(), annot=True, annot_kws={"size": 10}) 
plt.show() 


# In[13]:


from sklearn.model_selection import train_test_split 
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2) 
from sklearn.linear_model import LinearRegression 
m = LinearRegression() 
m.fit(Xtrain,ytrain)  
y_pred = m.predict(Xtest) 
print("Absolute Error: %0.3f"%float(np.abs(ytest-y_pred).sum()/ len(y_pred))) 


# In[14]:


from sklearn.metrics import mean_squared_error 
print("Mean Squared Error: %0.3f"% mean_squared_error(ytest, y_pred)) 

