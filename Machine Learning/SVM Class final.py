import pandas as pd
from sklearn import metrics
from sklearn.datasets import load_iris
iris = load_iris()

df = pd.DataFrame(iris.data)  # 4 cols
df['target'] = iris.target    # adding a col

print(df.head())

df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]

import matplotlib.pyplot as plt
'''
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0[0], df0[1],color="green",marker='+')
plt.scatter(df1[0], df1[1],color="blue",marker='*')
plt.show()
'''
# **Train Using Support Vector Machine (SVM)**
X = df.iloc[:, :4]
y = df.iloc[:,4]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Making the Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred)
print("Confusion MAtrix:\n",cm)

#Finding the accuracy of predicted value(Y_pred) by comparing with actual result(y_test).
print('accuracy',metrics.accuracy_score(y_test,y_pred))
#print("Prediction :",model.predict([[4.8,3.0,1.5,0.3]]))

# **Tune parameters**
# **1. Regularization (C)**

model_C = SVC(C=1)
model_C.fit(X_train, y_train)
print("Model using c=1 as reg ",model_C.score(X_test, y_test))

model_C = SVC(C=10)
model_C.fit(X_train, y_train)
print("Model using c=10 as reg ",model_C.score(X_test, y_test))

# **2. Gamma**
# controls the distance of influence of a single training point
# It does not do anything when the kernal is linear in nature
model_g = SVC(gamma=10)
model_g.fit(X_train, y_train)
print("Model score gamma",model_g.score(X_test, y_test))

# **3. Kernel**
model_linear_kernal = SVC(kernel='linear')
model_linear_kernal.fit(X_train, y_train)
print("Model score Kernal",model_linear_kernal.score(X_test, y_test))

