# Data preprocessing

# import de libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import de dataset
dataset = pd.read_csv('2-data_preprocessing/data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
print(y)

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)

# Enconding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
print(x)
onehotencoder = OneHotEncoder(categories = 'auto')
x = onehotencoder.fit_transform(x).toarray()
print(x)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print(y)

# Splitting the dataset into the training set and text set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
print("#######################################")
print(x_train)
print(x_test)
print(y_train)
print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
print("#" * 100)
print(x_train)
print("#" * 100)
print(x_test)