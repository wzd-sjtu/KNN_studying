# 这是基础K-means算法的验证
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
k=4
def KNN_studying(x_jud,X,y,k):
    tmp=X-x_jud
    tmp = tmp ** 2
    tmp_x,tmp_y=np.shape(tmp)
    distance=np.cumsum(tmp,axis=1)
    #distance=np.sqrt(distance)
    xx,yy=np.shape(distance)
    distance_plus=np.zeros((xx,2))
    for i in range(xx):
        distance_plus[i,0]=distance[i,yy-1]
        distance_plus[i,1]=y[i]
    distance=distance[:,yy-1]
    # 返回对应的数组下标值
    sortedDistance = distance.argsort()
    zero_type=0
    one_type = 0
    two_type = 0
    for i in range(k):
        flags=y[(np.where(sortedDistance==i))[0]]
        flags=flags[0]
        if flags==0:
            zero_type+=1
        if flags==1:
            one_type+=1
        if flags==2:
            two_type+=1
    if zero_type>=one_type&zero_type>=two_type:
        return 0
    elif two_type>=one_type&two_type>=zero_type:
        return 2
    else: return 1


qq=KNN_studying(X_test[7],X,y,k)
print(y_test)
print(qq)
print(np.shape(X_test))
miss=0
for i in range(38):
    plt.scatter(i,KNN_studying(X_test[i],X,y,k),color='red')
    plt.scatter(i,y_test[i],color='blue')
    if y_test[i]!=KNN_studying(X_test[i],X,y,k):
        miss+=1
print(miss/38)
#plt.scatter(qq,y)
plt.show()
