from numpy import *
from numpy import linalg

a=array([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]])
b=array([[[10,20,30],[40,50,60]],[[10,20,30],[40,50,60]]])
print(subtract(a,b))
print(linalg.norm(subtract(a,b)))