import numpy as np

a=np.array([[0,0,0],[0,0,0]])
b=np.array([0,1,1])
c=a/b
print(c)
c[np.isnan(c)]=0
print(c)