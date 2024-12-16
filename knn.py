q=pd.read_csv("fruits.csv")

xx=q.iloc[:, [0,1]].values

yy=q.iloc[:,-1].values

fruits={'orange': 1, 'apple':2}

yy=pd.Series (yy).map(fruits)

from sklearn.neighbors import KNeighborsClassifier

from mlxtend.plotting import plot_decision_regions

model=KNeighbors Classifier(n_neighbors=3)

model.fit(xx,yy)

yy=yy.to_numpy()

plot_decision_regions(xx,yy, clf=model)
