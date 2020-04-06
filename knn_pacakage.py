from sklearn import neighbors
from sklearn import datasets
from sklearn.model_selection import train_test_split

knn=neighbors.KNeighborsClassifier()
iris_dataset=datasets.load_iris()
X, X_test, y, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)


knn.fit(X,y)
predictedLabel=knn.predict(X_test)

n=0
for i in range(38):
    if predictedLabel[i]==y_test[i]:
        n+=1

percent=n/38
print(percent)

# 调库居然有这么高的成功率
