from __future__ import division
import heapq
import multiprocessing

from Network import *
from time import *
from flask import Flask, request
import requests
import json
import threading
import matplotlib.pyplot as plt
from numpy import *
import socket

app = Flask(__name__)

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# initialization
time_start = time()

# neural network parameters
rate_learning = 0.001
rate_regularization = 0
network = Network([561, 1122, 6], rate_learning, rate_regularization)

# training parameters
path_training = '../train9.csv'
size_batch = 10
round_epoch = 5
sample_training, label_training = LoadDataset(path_training, size_batch, True)

# testing parameters
path_testing = '../test9.csv'
sample_test, label_test=LoadDataset(path_testing,10,True)

# annealing parameters
rate_annealing = 10
threshold_differential = 0.000
threshold_loss = 1.0

# gradient clipping parameters
rate_gradient_clipping = 1
range_gradient_clipping = 1

# federated parameters
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('8.8.8.8', 80))
addr = s.getsockname()[0]
ip=range(9000,9010)
n=10
f=3
S=7
list_w=[]
list_score=zeros(n).tolist()
cnt=0

def Broadcast(temp:dict, port):
    result = requests.post('http://{}:{}/Krum'.format(addr, port), data=temp)

def Collusion(temp:dict, port):
    result = requests.post('http://{}:{}/Collusion'.format(addr, port), data=temp)

def PoisoningBroadcast(temp:dict, port):
    result = requests.post('http://{}:{}/Poisoning'.format(addr, port), data=temp)

def Update():
    global cnt
    global network
    if cnt==350:
        return
    print('Done.')
    print('Global Epoch {}...'.format(cnt), end='')
    if (not cnt==0) and (cnt % range_gradient_clipping==0):
        network.rate_learning=network.rate_learning*1
    Test(network, sample_test, label_test)
    Train(network, sample_training, label_training, round_epoch, range_gradient_clipping, rate_gradient_clipping, threshold_differential, threshold_loss, rate_annealing)
    w = []
    for m in network.list_weight:
        w.append(m.tolist())
    temp = {}
    temp['w'] = json.dumps(w)
    # if compromised:
    #     for i in range(9007,9010):
    #         try:
    #             t = threading.Thread(target=Collusion, args=(temp, i))
    #             t.start()
    #         except:
    #             print('Failed to broadcast parameters to 192.168.0.{}'.format(i))
    # else:
    #     for i in ip:
    #         try:
    #             t = threading.Thread(target=Broadcast, args=(temp, i))
    #             t.start()
    #         except:
    #             print('Failed to broadcast parameters to 192.168.0.{}'.format(i))
    try:
        t = threading.Thread(target=Collusion, args=(temp, 9009))
        t.start()
    except:
        print('Failed to broadcast parameters to 192.168.0.{}'.format(9009))

@app.route('/Krum', methods=['POST'])
def Krum():
    global list_w
    global list_score
    global network
    global cnt
    global list_model_benign
    with lock:
        temp = []
        for l in json.loads(request.form['w']):
            temp.append(array(l))
        list_w.append(temp.copy())
        if compromised:
            if len(list_w)==n-f:
                list_model_benign=list_w.copy()
            if len(list_model_benign)+len(list_model_compromised) == n:
                threading.Thread(target=Attack).start()
        if len(list_w)==n:
            for i in range(0,n):
                temp=[]
                for j in range(0,n):
                    norm=0
                    for sub in subtract(list_w[i], list_w[j]):
                        norm+=linalg.norm(sub)
                    temp.append(norm)
                temp.sort()
                list_score[i]=sum(temp[0:(n-f-1)])
            # index_w_final=list_score.index(np.min(list_score))
            list_tmp=list_score.copy()
            list_tmp.sort()
            print(list_tmp)
            index_w_final = [list_score.index(i) for i in list_tmp[:S]]
            # network.list_weight=list_w[index_w_final]
            temp=[]
            for i in index_w_final:
                temp.append(list_w[i])
            # temp=list_w[i in index_w_final]
            temp=sum(temp, axis=0)
            network.list_weight = divide(temp, S).tolist()
            cnt+=1
            list_w = []
            list_score = zeros(n).tolist()
            update = threading.Thread(target=Update)
            update.start()
        return ''

@app.route('/activate',methods=['GET'])
def activate():
    update = threading.Thread(target=Update)
    update.start()
    return 'Activated.', 200

@app.route('/plot_train',methods=['GET'])
def get_training_plot():
    return json.dumps(network.history_train_loss)

@app.route('/plot_test',methods=['GET'])
def get_testing_plot():
    return json.dumps(network.history_test_acc)

@app.route('/plot_test_loss',methods=['GET'])
def get_testing_loss_plot():
    return json.dumps(network.history_test_loss)

# Poisoning attack in 'Local Model Poisoning Attacks to Byzantine-Robust Federated Learning', Usenix'20

# compromising switch
compromised=True

# benign local models from benign nodes
list_model_benign=[]

# benign local models from compromised nodes
list_model_compromised=[]

# number of parameters
d=(561+1)*1122+(1122+1)*6

def Set_Lambda(m:int, c:int, d:int, benign_model:list):
    w_re = []
    for i in network.list_weight:
        w_re.append(cp.asnumpy(i))
    if not len(benign_model) == m-c:
        raise Exception('The size of benign group doesn\'t match to security threshold. Please check implementation.')
    list_score=np.zeros(m-c)
    list_distance=[]
    for i in range(0, m-c):
        temp = []
        for j in range(0, m-c):
            norm = 0
            for sub in subtract(benign_model[i], benign_model[j]):
                norm += linalg.norm(sub)
            temp.append(norm)
        temp.sort()
        list_score[i] = sum(temp[0:(m - c - 1)])
        norm=0
        for sub in subtract(benign_model[i], w_re):
            norm += linalg.norm(sub)
        list_distance.append(norm)
    min_score=np.min(list_score)
    max_distance=max(list_distance)
    return (min_score/((m-2*c-1)*pow(d,0.5)))+(max_distance/pow(d,0.5))

@app.route('/Collusion', methods=['POST'])
def CompromisedCommunication():
    global list_model_compromised
    global list_model_benign
    with lock:
        temp = []
        for l in json.loads(request.form['w']):
            temp.append(array(l))
        list_model_compromised.append(temp.copy())
        if len(list_model_benign) + len(list_model_compromised) == n:
            threading.Thread(target=Attack).start()
        return ''

def Attack():
    # TODO: implement attack
    # pass
    global network
    global list_model_benign
    global list_model_compromised
    # w_re=network.list_weight.copy()
    w_re=[]
    for i in network.list_weight:
        w_re.append(cp.asnumpy(i))
    list_w_temp=(list_model_benign + list_model_compromised).copy()
    list_score = zeros(n).tolist()
    for i in range(0, n):
        temp = []
        for j in range(0, n):
            norm = 0
            for sub in subtract(list_w_temp[i], list_w_temp[j]):
                norm += linalg.norm(sub)
            temp.append(norm)
        temp.sort()
        list_score[i] = sum(temp[0:(n - f - 1)])
    list_tmp = list_score.copy()
    list_tmp.sort()
    # print(list_tmp)
    index_w_final = [list_score.index(i) for i in list_tmp[:1]]
    temp = []
    for i in index_w_final:
        temp.append(list_w_temp[i])
    temp = sum(temp, axis=0)
    w_before_attack = divide(temp, 1).tolist()
    s=[]
    # print(w_before_attack[0].__class__,w_re[0].__class__,list_w_temp[2][0].__class__)
    # print(w_before_attack[0]-w_re[0])
    for i,j in zip(w_before_attack,w_re):
        s.append(np.sign(np.subtract(i,j)))
    # s=np.subtract(w_before_attack,w_re)
    # s=np.sign(s)
    # print(s)
    s=asarray(s)
    Lambda=Set_Lambda(n,f,d,list_model_benign)
    w_poisoned=np.subtract(w_re, s * Lambda)
    # start selection
    while True:
        # print(w_poisoned)
        list_w_temp=list_model_benign.copy()
        for i in range(0,f):
            list_w_temp.append(w_poisoned)
        list_score = zeros(n).tolist()
        for i in range(0, n):
            temp = []
            for j in range(0, n):
                norm = 0
                for sub in subtract(list_w_temp[i], list_w_temp[j]):
                    norm += linalg.norm(sub)
                temp.append(norm)
            temp.sort()
            list_score[i] = sum(temp[0:(n - f - 1)])
        list_tmp = list_score.copy()
        list_tmp.sort()
        # print(list_tmp)
        index_w_final = [list_score.index(i) for i in list_tmp[:1]]
        temp = []
        for i in index_w_final:
            temp.append(list_w_temp[i])
        temp = sum(temp, axis=0)
        w_after_attack = divide(temp, 1).tolist()
        # print(np.array(w_after_attack)[0]==np.array(w_poisoned)[0])
        # print(linalg.norm(w_after_attack[0] - w_poisoned[0]))
        # print(w_after_attack.__class__)
        # print(w_poisoned.__class__)
        # print(w_after_attack)
        # print(w_poisoned)
        norm = 0
        for sub in w_after_attack - w_poisoned:
            norm += linalg.norm(sub)
        print(norm)
        if norm==0 or (Lambda < 0.00001):
            list_model_benign=[]
            list_model_compromised=[]
            if Lambda<0.00001:
                print('Poisoning Failed')
            else:
                print('Poisoning Success!')
            w = []
            for m in w_poisoned:
                w.append(m.tolist())
            temp = {}
            temp['w'] = json.dumps(w)
            for i in ip:
                try:
                    t = threading.Thread(target=Broadcast, args=(temp, i))
                    t.start()
                except:
                    print('Failed to broadcast parameters to 192.168.0.{}'.format(i))
            for i in range(9007,9009):
                try:
                    t = threading.Thread(target=PoisoningBroadcast, args=(temp, i))
                    t.start()
                except:
                    print('Failed to broadcast parameters to 192.168.0.{}'.format(i))
            break
        else:
            Lambda=Lambda/2
            w_poisoned = np.subtract(w_re, s * Lambda)

if __name__ == '__main__':
    lock= threading.Lock()
    app.run(host='{}'.format(addr), port=9009)