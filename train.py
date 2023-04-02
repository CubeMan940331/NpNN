import network
import load_mnist as lm
import time
import numpy as np
npnn=network.net([28*28,128,128,128,10])
dataset=lm.load_mnist()
T=200
st=time.time()
eta=2
print(npnn.shape_list)
for t in range(T):
    ra=np.random.randint(60000,size=60000)
    for i in range(60):
        x_batch=dataset['x_train'][ra[i*1000:(i+1)*1000]]
        y_batch=dataset['y_train'][ra[i*1000:(i+1)*1000]]
        npnn.train(x_batch,y_batch,eta)
    cost=npnn.cost(dataset['x_test'],dataset['y_test'])
    acc=npnn.accuracy(y=dataset['y_test'])
    if acc>0.93: eta=1
    print(t+1,'cost %.4f' %(cost),end=' ')
    print('accruacy %.4f' %(acc))
print('time :',time.time()-st)
print(npnn.accuracy(dataset['x_test'],dataset['y_test']))
print(npnn.shape_list)
npnn.save('test',protect=True)
