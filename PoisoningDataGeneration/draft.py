import cupy as cp
import numpy as np

a=cp.array([[2,2,3,4],[3,4,5,6]])
b=cp.array([[1,2,3,4],[4,5,6,7]])
c=np.array([[2,2,3,4],[3,4,5,6]])
print(c)
print(np.append(np.zeros(2).transpose(),c))
print(cp.matmul(a[:,3:4],a[0:1,:]))
# print(cp.matmul(a,b))
print(a[:,:,None])
print(b[:,None])
print((cp.matmul(a[:,:,None],b[:,None])))