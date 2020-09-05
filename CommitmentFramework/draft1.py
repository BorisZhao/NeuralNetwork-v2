import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


# b=255-a
# im=Image.fromarray(a.astype('uint8'))
# im.save('../dataset/1_gray.jpg')

fig = plt.figure()
ax = Axes3D(fig)

pt_list=[]

list_score=[]

n=100
precision=5

target='1'
# size=279


# for i in range(0,1000):
#     pt_list.append(np.random.randn(1,3))
# pt_list[-1]=np.array([[10,20,30]])
# print(pt_list)
for i in range(329,1096):
    # print(i)
    pt_list.append(np.array(Image.open(f'C:/Users/zhaobo/Desktop/train/{target}/{i}.jpg').convert('L')).reshape(1,35*35))
    # im = Image.fromarray(pt_list[-1].reshape(50,50).astype('uint8'))
    # im.save(f'../dataset/{target}/{i}_gray.jpg')


for i in range(0,len(pt_list)):
    score=0
    for j in range(0,len(pt_list)):
        score+=np.linalg.norm(pt_list[i][0]-pt_list[j][0])
        # print(np.linalg.norm(pt_list[i][0]-pt_list[-2][0]))
    list_score.append(score)

temp=list_score.copy()
temp.sort()
print(temp)
list_index_edge=[list_score.index(i) for i in temp[-n:]]
list_index_center=[list_score.index(i) for i in temp[:n]]
list_index_edge.reverse()
print(list_index_edge)

list_commitment_edge=[]
list_commitment_center=[]

index=list_index_center.copy()

for i in index:
    distance=[]
    for j in index:
        distance.append(np.linalg.norm(pt_list[i]-pt_list[j]))
    # print(distance)
    temp=distance.copy()
    temp.sort()
    cm=[]
    for cnt in range(0, precision):
        cm.append(pt_list[index[distance.index(temp[cnt])]])
    list_commitment_center.append(np.array(cm).mean(axis=0))


index=list_index_edge.copy()

for i in index:
    distance=[]
    for j in index:
        distance.append(np.linalg.norm(pt_list[i]-pt_list[j]))
    temp=distance.copy()
    temp.sort()
    cm=[]
    for cnt in range(0, precision):
        cm.append(pt_list[index[distance.index(temp[cnt])]])
    list_commitment_edge.append(np.array(cm).mean(axis=0))
# distance=[]
# for i in index:
#     distance.append(np.linalg.norm(pt_list[index[0]][0]-pt_list[i][0]))
# temp=distance.copy()
# temp.sort()
# index=[]
# for i in temp:
#     index.append(list_index_center[distance.index(i)])
# # index.pop(0)
# while True:
#     temp=[]
#     if len(index)>=precise:
#         for i in range(0,precise):
#             temp.append(pt_list[index[i]])
#             index.pop(0)
#         list_commitment_center.append(np.array(temp).mean(axis=0))
#         print(list_commitment_center)
#     else:
#         for i in index:
#             temp.append(pt_list[i])
#             index.pop(0)
#         list_commitment_center.append(np.array(temp).mean(axis=0))
#         break
#
#
# index = list_index_edge.copy()
# random.shuffle(index)
# distance = []
# for i in index:
#     distance.append(np.linalg.norm(pt_list[index[0]][0] - pt_list[i][0]))
# temp = distance.copy()
# temp.sort()
# index = []
# for i in temp:
#     index.append(list_index_edge[distance.index(i)])
# index.pop(0)
# while True:
#     temp = []
#     if len(index) >= precise:
#         for i in range(0, precise):
#             temp.append(pt_list[index[i]])
#             index.pop(0)
#         list_commitment_edge.append(np.array(temp).mean(axis=0))
#     else:
#         for i in index:
#             temp.append(pt_list[i])
#             index.pop(0)
#         list_commitment_edge.append(np.array(temp).mean(axis=0))
#         break

# while True:
#     distance=[]
#     for j in index:
#         distance.append(np.linalg.norm(pt_list[index[0]][0]-pt_list[j][0]))
#     temp=distance.copy()
#     temp.sort()
#     temp.pop(0)
#     if len(temp)>=precise:
#         idx=[distance.index(d) for d in temp[:precise]]
#         temp=[]
#         for j in [index[i] for i in idx]:
#             temp.append(pt_list[j][0])
#         list_commitment_center.append(np.array(temp).mean(axis=0))
#         temp=[]
#         for i, cnt in zip(index,range(0,len(index))):
#             if not (cnt in idx):
#                 temp.append(i)
#         index=temp.copy()
#     else:
#         idx = [distance.index(d) for d in temp[:]]
#         temp = []
#         for j in [index[i] for i in idx]:
#             temp.append(pt_list[j][0])
#         list_commitment_center.append(np.array(temp).mean(axis=0))
#         temp = []
#         for i, cnt in zip(index, range(0, len(index))):
#             if not (cnt in idx):
#                 temp.append(i)
#         index = temp.copy()
#         break
# #
# index=list_index_edge.copy()
# random.shuffle(index)
# while True:
#     distance=[]
#     for j in index:
#         distance.append(np.linalg.norm(pt_list[index[0]][0]-pt_list[j][0]))
#     temp=distance.copy()
#     temp.sort()
#     temp.pop(0)
#     if len(temp)>=precise:
#         idx=[distance.index(d) for d in temp[:precise]]
#         temp=[]
#         for j in [index[i] for i in idx]:
#             temp.append(pt_list[j][0])
#         list_commitment_edge.append(np.array(temp).mean(axis=0))
#         temp=[]
#         for i, cnt in zip(index,range(0,len(index))):
#             if not (cnt in idx):
#                 temp.append(i)
#         index=temp.copy()
#     else:
#         idx = [distance.index(d) for d in temp[:]]
#         temp = []
#         for j in [index[i] for i in idx]:
#             temp.append(pt_list[j][0])
#         list_commitment_edge.append(np.array(temp).mean(axis=0))
#         temp = []
#         for i, cnt in zip(index, range(0, len(index))):
#             if not (cnt in idx):
#                 temp.append(i)
#         index = temp.copy()
#         break

# for i in list_index_edge:
#     cm=pt_list[i][0]
#     a = cm.reshape(100, 100)
#     im = Image.fromarray(a.astype('uint8'))
#     im.save(f'../dataset/{target}/commitment/edge_{i}.jpg')
#
# for i in list_index_center:
#     cm=pt_list[i][0]
#     a = cm.reshape(100, 100)
#     im = Image.fromarray(a.astype('uint8'))
#     im.save(f'../dataset/{target}/commitment/center_{i}.jpg')

for cm,cnt in zip(list_commitment_edge,range(0,len(list_commitment_edge))):
    a=cm.reshape(35,35)
    im = Image.fromarray(a.astype('uint8'))
    im.save(f'C:/Users/zhaobo/Desktop/cm-edge/{target}/cm{cnt}_edge.jpg')

for cm,cnt in zip(list_commitment_center,range(0,len(list_commitment_center))):
    a=cm.reshape(35,35)
    im = Image.fromarray(a.astype('uint8'))
    im.save(f'C:/Users/zhaobo/Desktop/cm-center/{target}/cm{cnt}_center.jpg')



# x=[]
# y=[]
# z=[]
# for pt in pt_list:
#     x.append(pt[0][0])
#     y.append(pt[0][1])
#     z.append(pt[0][2])
# ax.scatter(x,y,z,color='b', s=5.5, label='samples')

x=[]
y=[]
z=[]
for pt in list_index_edge:
    x.append(pt_list[pt][0][0])
    y.append(pt_list[pt][0][1])
    z.append(pt_list[pt][0][2])
ax.scatter(x,y,z,color='r', s=5.5, label='informative samples')

# x=[]
# y=[]
# z=[]
# for pt in list_index_center:
#     x.append(pt_list[pt][0][0])
#     y.append(pt_list[pt][0][1])
#     z.append(pt_list[pt][0][2])
# ax.scatter(x,y,z,color='y', s=5.5,label='representative samples')

# x=[]
# y=[]
# z=[]
# for cm in list_commitment_center:
#     x.append(cm[0][0])
#     y.append(cm[0][1])
#     z.append(cm[0][2])
# ax.scatter(x,y,z, color='g', marker='>', s=20.5, label='commitment')
#
x=[]
y=[]
z=[]
for cm in list_commitment_edge:
    x.append(cm[0][0])
    y.append(cm[0][1])
    z.append(cm[0][2])
ax.scatter(x,y,z, color='g', marker='>', s=20.5)

plt.legend()
plt.show()