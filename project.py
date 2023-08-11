#1,2,4,4,3,5,5   240=70%=168
#IMPORTS
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

#DATA HANDLING
data = pd.read_csv('project.csv')
print(data.head())
print(data.describe())
X = data[["water resource"]]
Y = data["stable"]
Y_ = data["selected"]

#DATA ANALYSIS
plt.scatter(X['water resource'], Y, color='b')
plt.xlabel('water resource')  
plt.ylabel('stable') 
plt.show()

#OBSERVATIONS
print("From the plot we can say that pobability of building a house is more if the water facilities are above 6")

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
mdl = LogisticRegression()
mdl.fit(X, Y_)
pred = mdl.predict([[7]])
print("Predicted value (LGR): ",pred[0])
print("Accuracy (LGR): ",mdl.score(X[:100], Y_[:100])*100)

plt.scatter(X['water resource'], Y_, color='b')
plt.plot(X['water resource'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('water resource')  
plt.ylabel('stable') 
plt.show()
print("-------------------------------------")

