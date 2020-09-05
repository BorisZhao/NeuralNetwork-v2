import os
import os.path
rootdir = "C:/Users/zhaobo/Desktop/0_expanded/"
files = os.listdir(rootdir)
b=1
for name in files:
    a=os.path.splitext(name)
    print(a[0])
    newname = str(b)+'.jpg'
    b = b + 1
    os.rename(rootdir+name,rootdir+newname)