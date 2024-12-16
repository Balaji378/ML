import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
iris=datasets.load_iris()
X=iris.data[:,:2]
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

svm_classifier=SVC(kernel='linear',C=1.0,random_state=42)
svm_classifier.fit(X_train,y_train)

x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,0.01),np.arange(y_min,y_max,0.01))

Z=svm_classifier.predict(np.c_[xx.ravel(),yy.ravel()])
Z=Z.reshape(xx.shape)

plt.contourf(xx,yy,Z,alpha=0.8,cmap='rainbow')
scatter=plt.scatter(X[:,0],X[:,1],c=y,edgecolors='k',cmap='rainbow')

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('SVM Decision Boundaries')
plt.legend(handles=scatter.legend_elements()[0],title='Classes')
plt.show()
