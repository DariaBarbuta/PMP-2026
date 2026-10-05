import numpy as np
import random
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

X=0
l=[]
for i in range(10000):
    frizer=random.choices([1,2,3], weights=[3/13,6/13,4/13], k=1)[0] #3+6+4=13
    if frizer==1:
        X=np.random.exponential(scale=1/3) #media=1/lambda
    elif frizer==2:
        X=np.random.exponential(scale=1/6) 
    else:
        X=np.random.exponential(scale=1/4) 
    l.append(X)

m=np.mean(l)
deviation=np.std(l)

print(m)
print(deviation)
densitate = gaussian_kde(l)
x_valori = np.linspace(min(l), max(l), 1000)
y_valori = densitate(x_valori)

plt.plot(x_valori, y_valori)
plt.show()