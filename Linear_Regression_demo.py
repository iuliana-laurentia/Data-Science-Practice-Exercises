#!/usr/bin/env python
# coding: utf-8

# In[20]:


#import necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


# In[21]:


#provide values for x, y
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])


# In[22]:


plt.scatter(x,y)
plt.show()


# In[23]:


#create the regression model
model = LinearRegression()


# In[24]:


#fit the regression model
model.fit(x, y)


# In[25]:


#find the coefficient of determination to see how well does the line fit the data
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)


# In[26]:


print('intercept:', model.intercept_)
print('slope:', model.coef_)


# In[27]:


#predict the valies of y with the slope and intercept from the fitted regression line
y_hat = model.predict(x)
print('predicted response:', y_hat)


# In[30]:


rg=sns.regplot(x,y)


# In[ ]:




