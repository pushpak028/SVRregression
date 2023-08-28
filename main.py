import numpy as np
import pandas pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandarScaler
from sklearn.svm import SVR

df = pd.read_csv("position_salary")

x = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values

scx = StandardScaler()
scy = StandardScaler()
y = y.reshape(len(y),1)
x = scx.fit_transform(x)
y = scy.fit_transform(y)

sv = SVR(kernel = 'rbf')
sv.fit(x,y)

x_grid = np.arrange(min(scx.inverse_transform(x)),max(scx.inverse_transform(x)),0.1)
x_grid = x_grid.reshape(len(x_grid),1)

plt.scatter(scx.inverse_transform(x),scy.inverse_transform(y))
plt.plot(x_grid , scy.inverse_transform(sv.predict(scx.transform(x_grid)).reshape(-1,1)))
