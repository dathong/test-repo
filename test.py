import pandas as pd
import numpy as np
from sklearn import datasets

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier



iris = datasets.load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],columns= iris['feature_names'] + ['target'])


d = {}
cv = 5
size = int(df.shape[0]/cv)
for i in range(0,cv):
    d[i+1] = df[size*i:size*(i+1)]

lg = LogisticRegression()


#----classify----

train_data = pd.concat([d[1],d[3],d[5],d[2]])
test_data = d[4]
lg.fit(train_data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']],
       train_data['target'])
yPred = lg.predict(test_data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']])
yTrue = test_data['target'].values.tolist()
print("yTrue = ",yTrue)
print("yPred = ",yPred)
print("acc = ",accuracy_score(yTrue, yPred))

#-----kNN----


knn = KNeighborsClassifier()
knn.fit(train_data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']],
       train_data['target'])
predictions = knn.predict(test_data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']])
print(accuracy_score(test_data['target'], predictions))
print(confusion_matrix(test_data['target'], predictions))
print(classification_report(test_data['target'], predictions))

#----split------
for i in range(1,6):
    for c in ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']:
        d[i][c] = (d[i][c] - d[i][c].mean())/d[i][c].std()


train_data = pd.concat([d[1],d[3],d[5],d[2]])
test_data = d[4]
lg.fit(train_data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']],
       train_data['target'])
yPred = lg.predict(test_data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']])
yTrue = test_data['target'].values.tolist()
print("yTrue = ",yTrue)
print("yPred = ",yPred)
print("acc = ",accuracy_score(yTrue, yPred))

knn = KNeighborsClassifier()
knn.fit(train_data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']],
       train_data['target'])
predictions = knn.predict(test_data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']])
print(accuracy_score(test_data['target'], predictions))
print(confusion_matrix(test_data['target'], predictions))
print(classification_report(test_data['target'], predictions))