# Multiple Linear Regression

# Multiple Linear Regression
# y = b0 + b1*x1 + b2*x2 ...

# import de libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import de dataset
dataset = pd.read_csv('5-multiple_linear_regression/50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Enconding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categories = 'auto')
x = onehotencoder.fit_transform(x).toarray()

# Avoiding the Dummy Variable Trap
x = x[:, 1:]

# Splitting the dataset into the training set and text set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
print("#" * 100)
print(x_train)
print("#" * 100)
print(x_test)"""

# Fitting Multiple Linear Regression in the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)

# Building the optimal model using Backward Elimination
import statsmodels.api as sm
x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1)
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()
x_opt = x[:, [0, 1, 3, 4, 5]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()
x_opt = x[:, [0, 3, 4, 5]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()
x_opt = x[:, [0, 3, 5]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()
x_opt = x[:, [0, 3]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()
print(regressor_ols.summary())