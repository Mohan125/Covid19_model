import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
dataset= pd.read_csv('csvfile.csv')
x= dataset.iloc[:,0:-1].values
y= dataset.iloc[:,-1].values
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)
from sklearn.preprocessing import PolynomialFeatures
pf=PolynomialFeatures(degree=4)
new_x=pf.fit_transform(x)
PolynomialRegression= LinearRegression()
PolynomialRegression.fit(new_x,y)
np.set_printoptions(precision=2)
plt.scatter(x,y,color='#FD542E' )
plt.plot(x,lr.predict(x),color='#00BFFF')
plt.grid(True)
plt.xlabel('Day')
plt.ylabel('No.of.Cases')
plt.title('Covid Cases in Tamil Nadu')
plt.show()
plt.scatter(x,y,color='#FD542E')
plt.plot(x,PolynomialRegression.predict(new_x),color='#00BFFF')
plt.grid(True)
plt.xlabel('Day')
plt.ylabel('No.of.Cases')
plt.title('Covid Cases in Tamil Nadu')
plt.show()
print(lr.predict([[95]]))
print(PolynomialRegression.predict(pf.fit_transform([[95]])))