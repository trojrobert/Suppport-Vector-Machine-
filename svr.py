import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Feature scaling 
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
scale_y = StandardScaler()
X = scale_X.fit_transform(X)
y = scale_y.fit_transform(y)

# Fitting the Support Vector Machine model 
from sklearn.svm import SVR 
svr = SVR(kernel = 'rbf')
svr.fit(X,y)

# Predicting a new result
y_pred = scale_y.inverse_transform(svr.predict(scale_X.transform(np.array([[6.5]]))))

# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, svr.predict(X), color = 'blue')
plt.title('Support vector machine')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

