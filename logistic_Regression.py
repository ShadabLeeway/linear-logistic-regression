# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score

# from sklearn import datasets ,linear_model
# iris=datasets.load_iris()
# # print(iris["DESCR"])
# X=iris['data'][:,3:]
# X_train=X[-30:]
# X_test=X[:-30]
# Y=iris['target']
# Y_train=Y[-30:]
# Y_test=Y[:-30]
# print(X)
# print(Y)

# # Train a logistic regression classifier
# clf=linear_model.LogisticRegression()
# clf.fit(X_train,Y_train)




# #visulisation
# # X_new= np.linspace(0,3,1000).reshape(-1,1)
# # Y_prob=clf.predict_proba(X_new)

# Y_pred=clf.redict_proba(X_test)


# # calculate the accuracy of the model
# accuracy = accuracy_score(Y_test, Y_pred)
# print('Accuracy:', accuracy)

# plt.plot(Y_test, Y_pred)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load the iris dataset
iris = load_iris()

# extract the features and target
X = iris.data[:, :2] # we'll use only the first two features for visualization
y = iris.target

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a logistic regression model and fit it to the training data
model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

# calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# visualize the data and decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(6, 4))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.xticks(())
plt.yticks(())
plt.show()