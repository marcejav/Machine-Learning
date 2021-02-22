# Data preprocessing

# import de libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import de dataset
dataset = pd.read_csv('2-data_preprocessing/data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Splitting the dataset into the training set and text set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
print("#" * 100)
print(x_train)
print("#" * 100)
print(x_test)
print("#" * 100)
print(y_train)
print("#" * 100)
print(y_test)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
print("#" * 100)
print(x_train)
print("#" * 100)
print(x_test)"""