import threading

from Network import *
from time import *
from flask import Flask, request
import requests
import json
from numpy import *
import socket
import multiprocessing
from multiprocessing import Pipe, Queue
from multiprocessing.managers import BaseManager
import pickle

app = Flask(__name__)

# initialization
time_start = time()

# neural network parameters
rate_learning = 0.001
rate_regularization = 0
network = Network([561, 1122, 6], rate_learning, rate_regularization)

# training parameters
path_training = '../train1.csv'
size_batch = 10
round_epoch = 5
# sample_training=[]
# label_training=[]

# testing parameters
path_testing = '../test1.csv'
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
list_finished=[]
list_n=[]
C=1
K=10
cnt=0
activated=False

#process communication


def Broadcast(temp:dict, port):
    result = requests.post('http://{}:{}/FederatedAverage'.format(addr, port), data=temp)

@app.route('/Signal', methods=['POST'])
def Signal():
    global list_finished
    if request.form['port'] in list_finished:
        print('Singal denied')
        return ''
    list_finished.append(request.form['port'])
    if len(list_finished) ==C*K:
        print('Global Epoch {}...'.format(cnt))
        update = threading.Thread(target=Update)
        update.start()
        list_finished=[]
    return ''

def Update():
    global network
    Test(network, sample_test, label_test)
    Train(network, sample_training, label_training, round_epoch, range_gradient_clipping, rate_gradient_clipping, threshold_differential, threshold_loss, rate_annealing)
    w=[]
    for m in network.list_weight:
        w.append(m.tolist())
    temp = {}
    temp['w'] = json.dumps(w)
    temp['n'] = len(sample_training)
    temp['port'] = 9001
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
    # global queue
    with lock:
        if request.form['port'] in list_updated:
            # nothing
            print("denied")
            return ''
        # if len(list_updated)<C*K:
        # add received weight to pool
        list_updated.append(request.form['port'])
        temp = []
        for l in json.loads(request.form['w']):
            temp.append(array(l))
        list_w.append(temp.copy())
        list_n.append(int(request.form['n']))
        print(list_updated)
        if len(list_updated) == C * K:
            list_updated = []
            # weighted average
            n = sum(list_n)
            list_n = divide(list_n, n)
            for i in range(0, C * K):
                list_w[i] = multiply(list_w[i], list_n[i])
            list_n = []
            list_w = array(list_w).sum(axis=0)
            network.list_weight = list_w.tolist().copy()
            list_w = []
            # network.history_test_loss, network.history_train_loss=queue.get()
            cnt += 1
            # weight, train_loss, test_loss = (network.list_weight, network.history_train_loss, network.history_test_loss)
            # for i in ip:
            #     result=requests.post('http://{}:{}/Signal'.format(addr, i), data=({'port':9000}))
            print('Global Epoch {}...'.format(cnt))
            update = threading.Thread(target=Update)
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
        # weight, train_loss, test_loss = (network.list_weight, network.history_train_loss, network.history_test_loss)
        update = threading.Thread(target=Update)
        update.start()
        activated=True
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

if __name__ == '__main__':
    # multiprocessing.Process(target=app_run).start()
    lock=threading.Lock()
    sample_training, label_training = LoadDataset(path_training, size_batch)
    sample_test, label_test = LoadDataset(path_testing)
    app.run(host='{}'.format(addr), port=9001)