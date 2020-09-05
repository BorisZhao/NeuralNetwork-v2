import threading

from CommitmentFramework.Network_BP_CM import *
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
import datetime
import cupy.cuda

app = Flask(__name__)
device=0

# initialization
time_start = time()

# neural network parameters
rate_learning = 0.001
rate_regularization = 0
# network = Network([50*50, 25*25, 725, 512,200,100,100,100,100,100,50, 2], rate_learning, rate_regularization)
# network = Network([50*50, 50*25, 25*25, 12*25, 6*25, 3*25, 40, 20, 10, 4, 2], rate_learning, rate_regularization)
# network = Network([35*35, 35*35,35*25,35*25,35*15,35*15,35*5,30*5,25*5,20*5,10*5, 2], rate_learning, rate_regularization)
with cupy.cuda.Device(device):
    network = Network([35*35, 1000,2], rate_learning, rate_regularization)

# training parameters
path_training = '../dataset-commit/subject1/train.csv'
size_batch = 1
round_epoch = 1
# sample_training=[]
# label_training=[]

# testing parameters
path_testing = '../dataset-commit/subject1/test.csv'
# sample_test=[]
# label_test=[]

# cm parameters
# path_cmc = '../dataset-commit/subject1/cm-center.csv'
# path_cme = '../dataset-commit/subject1/cm-edge.csv'

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
ip=range(9000,9001)
list_w=[]
list_updated=[]
list_finished=[]
list_n=[]
C=1
K=1
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
    global cnt
    while True:
        print('Global Epoch {} at {}...'.format(cnt, datetime.datetime.now()))
        cnt+=1
        # if cnt%100 ==0:
        #     network.rate_learning=network.rate_learning*0.1
        if cnt==400:
            break
        with cupy.cuda.Device(device):
            Test_train(network, sample_training, label_training)
            Test_test(network, sample_test, label_test)
            Test_cmc(network, sample_cm, label_cm)
            # Test_cme(network, sample_cme, label_cme)
            Train(network, sample_training, label_training, round_epoch, range_gradient_clipping, rate_gradient_clipping, threshold_differential, threshold_loss, rate_annealing)
        # sleep(10)
    # w=[]
    # for m in network.list_weight:
    #     w.append(m.tolist())
    # temp = {}
    # temp['w'] = json.dumps(w)
    # temp['n'] = len(sample_training)
    # temp['port'] = 9000
    # try:
    #     for i in ip:
    #         threading.Thread(target=Broadcast, args=(temp,i)).start()
    # except:
    #     pass

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
        temp=[]
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
            for i in range(0, C*K):
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
    # if not activated:

        # weight, train_loss, test_loss = (network.list_weight, network.history_train_loss, network.history_test_loss)
    update = threading.Thread(target=Update)
    update.start()
    activated=True
    return 'Activated.', 200

@app.route('/plot_train_loss',methods=['GET'])
def get_training_loss():
    return json.dumps(network.history_train_loss)

@app.route('/plot_train_acc',methods=['GET'])
def get_training_acc():
    return json.dumps(network.history_train_acc)

@app.route('/plot_test_acc',methods=['GET'])
def get_testing_acc():
    return json.dumps(network.history_test_acc)

@app.route('/plot_test_loss',methods=['GET'])
def get_testing_loss():
    return json.dumps(network.history_test_loss)

@app.route('/plot_cm_acc',methods=['GET'])
def get_cm_acc():
    return json.dumps(network.history_CMC_acc)

@app.route('/plot_cm_loss',methods=['GET'])
def get_cm_loss():
    return json.dumps(network.history_CMC_loss)

# @app.route('/plot_cme_acc',methods=['GET'])
# def get_cme_acc():
#     return json.dumps(network.history_CME_acc)
#
# @app.route('/plot_cme_loss',methods=['GET'])
# def get_cme_loss():
#     return json.dumps(network.history_CME_loss)

if __name__ == '__main__':
    # multiprocessing.Process(target=app_run).start()
    lock=threading.Lock()
    average, std = GetAverageAndStd(path_training)
    sample_training, label_training = LoadDataset(path_training, size_batch, normalization=True, average=average, std=std)
    sample_test, label_test = LoadDataset(path_testing, 1, normalization=True, average=average, std=std)
    # sample_cmc, label_cmc = LoadDataset(path_cmc, 1, normalization=True, average=average, std=std)
    # print(sample_cmc[0])
    sample_cm, label_cm = Commitment(sample_training,label_training,5)
    sample_training, label_training = LoadDatasetLabelFlipping(path_training, size_batch, normalization=True, average=average, std=std, threshold=1700)
    # print(sample_cmc[0])
    # sample_cme, label_cme = LoadDataset(path_cme, 1, normalization=True, average=average, std=std)
    # print(len(sample_training[0][0]))
    app.run(host='{}'.format(addr), port=9000)