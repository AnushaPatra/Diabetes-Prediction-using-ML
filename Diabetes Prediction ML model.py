#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array


# In[12]:


daibetes_dataset = pd.read_csv(r"C:\Users\anush\OneDrive\Desktop\BALCO\Diabetes.csv")


# In[14]:


daibetes_dataset.head()


# In[16]:


daibetes_dataset.shape


# In[18]:


daibetes_dataset.describe()


# In[21]:


daibetes_dataset['Outcome'].value_counts()


# #### 0--> Non diabetic
# #### 1--> Diabetic

# In[23]:


daibetes_dataset.groupby('Outcome').mean()


# In[25]:


X = daibetes_dataset.drop(columns = 'Outcome', axis = 1)
Y = daibetes_dataset['Outcome']


# In[27]:


print(X)


# In[28]:


print(Y)


# In[68]:


scaler = StandardScaler()


# In[84]:


scaler.fit(X)


# In[70]:


standardized_data = scaler.transform(X)
print(standardized_data)


# In[72]:


X = standardized_data
Y = daibetes_dataset['Outcome']


# In[73]:


print(X)


# In[74]:


print(Y)


# #### Train Test Split

# In[75]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state=2)


# In[76]:


print(X.shape, X_train.shape, X_test.shape)


# #### Training the model

# In[77]:


classifier = svm.SVC(kernel = 'linear')


# In[78]:


#training the svc
classifier.fit(X_train, Y_train)


# #### Model evaluation

# In[79]:


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[80]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[81]:


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[82]:


print('Accuracy score of the test data : ', test_data_accuracy)


# #### Making a predictive system

# In[83]:


input_data = (4,111,72,47,207,37.1,1.39,56)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[ ]:





# In[ ]:




