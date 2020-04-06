
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
#  下面构建鹫尾花数据集的分类
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# 导入数据集
iris_dataset=load_iris()

X, X_test, y, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)


print(np.shape(X_test))
def knn_findone(x_jud,X,y,k):
    xx,yy=np.shape(X)
    tmp = X - x_jud
    tmp = tmp ** 2
    tmp=np.sqrt(tmp)
    for i in range(yy):
        ma = np.max(tmp[:, i])
        tmp[:, i] = tmp[:, i] / ma

    # 上述操作成功完成归一化操作
    distance = np.cumsum(tmp, axis=1)
    # distance=np.sqrt(distance)
    distance = distance[:, yy - 1]
    sortedDistance = distance.argsort()
    zero_type=one_type=two_type=0
    for j in range(k):
        flags = y[sortedDistance[j]]
        if flags == 0:
            zero_type += 1
        if flags == 1:
            one_type += 1
        if flags == 2:
            two_type += 1
    if zero_type>=one_type and zero_type>=two_type:
        return 0
    # 半天是这里的逻辑连接词  出了大问题
    elif two_type>=one_type and two_type>=zero_type:
        return 2
    else: return 1
    #return sortedDistance,distance,distance[sortedDistance]

k=2
qq=knn_findone(X_test[2],X,y,k)

type=0
for i in range(38):
    if y_test[i]==knn_findone(X_test[i],X,y,k):
        type+=1

print(type/38)
