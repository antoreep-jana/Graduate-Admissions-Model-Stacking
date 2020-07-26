# Library imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


data = pd.read_csv('', index_col = '')
data1 = pd.read_csv('', index_col = '')

data = pd.concat([data, data1], axis = 0)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.drop([''], axis = 1), data[''], test_size = 0.2, random_state = 21)

# Feature Scaling




# Model Training







# Saving the model to disk 



# 