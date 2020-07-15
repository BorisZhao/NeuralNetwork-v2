import threading

from Network import *
from time import *
from flask import Flask, request
import requests
import json
from numpy import *
import socket
import multiprocessing
from multiprocessing import Pipe, Queue, Manager
from multiprocessing.managers import BaseManager
import pickle

app = Flask(__name__)

# initialization
time_start = time()

# neural network parameters
rate_learning = 0.01
rate_regularization = 0
network = Network([561, 1122, 6], rate_learning, rate_regularization)

# training parameters
path_training = '../subject1.csv'
size_batch = 3
round_epoch = 10
# sample_training=[]
# label_training=[]

# testing parameters
path_testing = '../subject17.csv'
# sample_test=[]
# label_test=[]

# annealing parameters
rate_annealing = 10
threshold_differential = 0.0000
threshold_loss = 1.0

# gradient clipping parameters
rate_gradient_clipping = 1
range_gradient_clipping = 50

# federation parameters
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('8.8.8.8', 80))
addr = s.getsockname()[0]
ip=range(9000,9010)
list_w=[]
list_updated=[]
list_n=[]
C=1
K=10
cnt=0
activated=False

#process communication


def Broadcast(temp:dict, port):
    result = requests.post('http://{}:{}/FederatedAverage'.format(addr, port), data=temp)

def Update(weight, train_loss, test_loss, sample_training, label_training, sample_test, label_test, queue):
    global network
    network.list_weight=weight
    network.history_train_loss=train_loss
    network.history_test_loss=test_loss
    Test(network, sample_test, label_test)
    Train(network, sample_training, label_training, round_epoch, range_gradient_clipping, rate_gradient_clipping, threshold_differential, threshold_loss, rate_annealing)
    # input.send((network.history_test_loss, network.history_train_loss))
    queue.put((network.history_test_loss, network.history_train_loss))
    w=[]
    for m in network.list_weight:
        w.append(m.tolist())
    temp = {}
    temp['w'] = json.dumps(w)
    temp['n'] = len(sample_training)
    temp['port'] = 9000
    try:
        for i in ip:
            threading.Thread(target=Broadcast, args=(temp,i)).start()
    except:
        pass

@app.route('/FederatedAverage', methods=['POST'])
def FederatedAverage():
    global network
    global list_w
    global list_updated
    global list_n
    global cnt
    global sample_training, label_training, sample_test, label_test
    global queue
    if request.form['port'] in list_updated:
        # nothing
        print("denied")
        return ''
    if len(list_w)<C*K:
        # add received weight to pool
        list_updated.append(request.form['port'])
        temp=[]
        for l in json.loads(request.form['w']):
            temp.append(array(l))
        list_w.append(temp.copy())
        list_n.append(int(request.form['n']))
        print(list_updated)
        if len(list_w)==C*K:
            # weighted average
            n=sum(list_n)
            list_n=divide(list_n, n)
            for i in range(0,len(list_w)):
                list_w[i]=multiply(list_w[i],list_n[i])
            list_w=array(list_w).sum(axis=0)
            network.list_weight=list_w.tolist().copy()
            network.history_test_loss, network.history_train_loss=queue.get()
            cnt += 1
            list_updated = []
            list_w = []
            list_n = []
            print('Global Epoch {}...'.format(cnt))
            weight, train_loss, test_loss = (network.list_weight, network.history_train_loss, network.history_test_loss)
            update = multiprocessing.Process(target=Update, args=(weight, train_loss, test_loss, sample_training, label_training, sample_test, label_test, queue))
            update.start()
    return ''

@app.route('/activate',methods=['GET'])
def activate():
    global activated
    global sample_training, label_training, sample_test, label_test
    global queue
    global network
    if not activated:
        print('Global Epoch {}...'.format(cnt))
        weight, train_loss, test_loss = (network.list_weight, network.history_train_loss, network.history_test_loss)
        update = multiprocessing.Process(target=Update, args=(weight, train_loss, test_loss, sample_training, label_training, sample_test, label_test, queue))
        update.start()
        activated=True
    return 'Activated.', 200

@app.route('/plot_train',methods=['GET'])
def get_training_plot():
    return json.dumps(network.history_train_loss)

@app.route('/plot_test',methods=['GET'])
def get_testing_plot():
    return json.dumps(network.history_test_loss)

def app_run():
    app.run(host='{}'.format(addr), port=9000)

if __name__ == '__main__':
    # multiprocessing.Process(target=app_run).start()
    queue=Queue()
    m=Manager()
    sample_training, label_training = LoadDataset(path_training, size_batch)
    sample_test, label_test = LoadDataset(path_testing)
    app.run(host='{}'.format(addr), port=9000)