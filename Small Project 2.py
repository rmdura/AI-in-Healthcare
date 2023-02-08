#!/usr/bin/env python
# coding: utf-8

# ### Import Packages

# In[65]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn import metrics
import pandas as pd
import numpy as np


# ### Import Data

# In[66]:


cancer = datasets.load_breast_cancer()
cancer_pandas = pd.DataFrame(data = cancer.data, columns=cancer.feature_names)


# ### Split Data

# In[82]:


x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.25, random_state = 1)


# ### Create Decision Tree Classifier

# In[83]:


dtree = tree.DecisionTreeClassifier(random_state = 10)
# grid search looks at gini and entropy, as well as max depths from 3-15
params = {'criterion':['gini','entropy'],'max_depth':[3,4,5,6,7,8,9,10,11,12,13,14,15]}
# 5 fold cross validation
grid = GridSearchCV(dtree, params, cv = 5)
grid.fit(x_train, y_train)


# ### Predict on Test Set

# In[84]:


y_prediction = grid.predict(x_test)


# ### Evaluate Model

# In[85]:


print("Model Accuracy:",metrics.accuracy_score(y_prediction, y_test))
print("Model Sensitivity:",metrics.recall_score(y_prediction, y_test))
print("Model Specificity:",metrics.precision_score(y_prediction, y_test, pos_label = 0))
print("Parameters used:", grid.best_params_)

