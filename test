from c_network import *

import matplotlib.pyplot as plt

dataMat=np.random.rand(100,2)*(-2)+1

print(dataMat)


assert (dataMat.shape==(100,2))
c=[]

cn=competitive_network(2,2,0.1)
cn.train(dataMat,1000)

for i in range(len(dataMat)):
    prediction=cn.prediction(dataMat[i])
    c.append(prediction*30)

dataMat=normalization_all(dataMat)
dataMat=np.array(dataMat)
plt.figure()
plt.scatter(dataMat[:,0],dataMat[:,1],c=c)
plt.xlim(-1.2,1.2)
plt.ylim(-1.2,1.2)
plt.show()
