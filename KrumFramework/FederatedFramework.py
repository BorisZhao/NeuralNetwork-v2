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

# initialization
time_start = time()

# neural network parameters
rate_learning = 0.01
rate_regularization = 0
network = Network([561, 1122, 6], rate_learning, rate_regularization)

# training parameters
path_training = 'subject1.csv'
size_batch = 3
round_epoch = 10
sample_training, label_training = LoadDataset(path_training, size_batch)

# testing parameters
path_testing = 'subject17.csv'
sample_test, label_test=LoadDataset(path_testing)

# annealing parameters
rate_annealing = 10
threshold_differential = 0.0000
threshold_loss = 1.0

# gradient clipping parameters
rate_gradient_clipping = 1
range_gradient_clipping = 50

# federated parameters
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('8.8.8.8', 80))
addr = s.getsockname()[0]
ip=range(9000,9010)
n=10
f=3
list_w=[]
list_score=zeros(n).tolist()
activated=False
cnt=0

def Broadcast(temp:dict, port):
    result = requests.post('http://{}:{}/FederatedAverage'.format(addr, port), data=temp)

def Update():
    global cnt
    global network
    print('Global Epoch {}...'.format(cnt), end='')
    Test(network, sample_test, label_test)
    Train(network, sample_training, label_training, round_epoch, range_gradient_clipping, rate_gradient_clipping, threshold_differential, threshold_loss, rate_annealing)
    w = []
    for m in network.list_weight:
        w.append(m.tolist())
    temp = {}
    temp['w'] = json.dumps(w)
    for i in ip:
        try:
            t = threading.Thread(target=Broadcast, args=(temp, i))
            t.start()
        except:
            print('Failed to broadcast parameters to 192.168.0.{}'.format(i))

@app.route('/Krum', methods=['POST'])
def Krum():
    global list_w
    global list_score
    global network
    global cnt
    temp = []
    for l in json.loads(request.form['w']):
        temp.append(array(l))
    list_w.append(temp.copy())
    if len(list_w)==n:
        for i in range(0,n):
            temp=[]
            for j in range(0,n):
                norm=0
                for sub in subtract(list_w[i], list_w[j]):
                    norm+=linalg.norm(sub)
                temp.append(norm)
            temp.sort()
            list_score[i]=sum(temp[0:(n-f+1)])
        index_w_final=list_score.index(min(list_score))
        network.list_weight=list_w[index_w_final]
        cnt+=1
        list_w = []
        list_score = zeros(n).tolist()
        update = threading.Thread(target=Update)
        update.start()
    return ''

@app.route('/activate',methods=['GET'])
def activate():
    global activated
    if not activated:
        update = threading.Thread(target=Update)
        update.start()
        activate=True
    return 'Activated.', 200

@app.route('/plot_train',methods=['GET'])
def get_training_plot():
    return json.dumps(network.history_train_loss)

@app.route('/plot_test',methods=['GET'])
def get_testing_plot():
    return json.dumps(network.history_test_loss)

def app_run():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    ip = s.getsockname()[0]
    app.run(host='{}'.format(ip), port=9000)

if __name__ == '__main__':
    main_app = multiprocessing.Process(target=app_run)
    main_app.start()