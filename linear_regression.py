import matplotlib as plt
import numpy as np

from sklearn  import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
Diabetes=datasets.load_diabetes()
print(Diabetes)

# print(Diabetes)
# print(Diabetes.target_filename)
# print(Diabetes.data)
# print(Diabetes.target)
Diabetes_X_train=Diabetes.data[:-30]
Diabetes_Y_train=Diabetes.target[:-30]

Diabetes_X_test=Diabetes.data[-30:]
Diabetes_Y_test=Diabetes.target[-30:]

model=linear_model.LinearRegression()

model.fit(Diabetes_X_train,Diabetes_Y_train)

Diabetes_Y_Predict = model.predict(Diabetes_X_test)

print("Mean Squared Error :",mean_squared_error(Diabetes_Y_test,Diabetes_Y_Predict))
print("Weight ",model.coef_)
print("Intercept ",model.intercept_)
print("Acuuracy Score", r2_score(Diabetes_Y_test,Diabetes_Y_Predict))