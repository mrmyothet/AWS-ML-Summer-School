import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from joblib import dump, load

df = pd.read_csv("./data/raw/iris.csv")

df = df.sample(frac=1, random_state=1).reset_index(drop=True)

# set features and target
X, y = df.iloc[:, :-1], df.iloc[:, -1]

print(X.shape)
print(y.shape)
print(y[:10])
print(X[:10])

x_train, y_train = X[:100], y[:100]
x_test, y_test = X[100:], y[100:]

print("X train : ", x_train.shape)
print("x test  : ", x_test.shape)

gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)
y_pred

accuracy = np.sum(y_test == y_pred) / len(y_test)
print("Model Accuracy: ", accuracy)

# Save the model
dump(gnb, "week2.joblib")
