from Network import *
from time import *
from flask import Flask, request
import requests
import json
import threading
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
path_training = '../train0.csv'
size_batch = 3
round_epoch = 10
sample_training, label_training = LoadDataset(path_training, size_batch)

# testing parameters
path_testing = '../train1.csv'
sample_test, label_test=LoadDataset(path_testing)

# annealing parameters
rate_annealing = 10
threshold_differential = 0.0000
threshold_loss = 1.0

# gradient clipping parameters
rate_gradient_clipping = 1
range_gradient_clipping = 50

# federation parameters
list_w=[]
list_updated=[]
list_n=[]
C=1
K=1
cnt=0
activated=False

def Update():
    global cnt
    global network
    print('Global Epoch {}...'.format(cnt), end='')
    Test(network, sample_test, label_test)
    Train(network, sample_training, label_training, round_epoch, range_gradient_clipping, rate_gradient_clipping, threshold_differential, threshold_loss, rate_annealing)
    # if cnt % 30 == 0:
    #     plt.plot(network.history_loss, color='blue')
    #     plt.pause(0.01)
    w=[]
    for m in network.list_weight:
        w.append(m.tolist())
    # for i in ip:
    temp = {}
    temp['w'] = json.dumps(w)
    temp['n'] = len(sample_training)
    try:
        # for i in ip:
            # requests.post('http://192.168.0.{}/FederatedAverage'.format(i), data=temp)
            # result=requests.post('http://192.168.0.{}:9000/FederatedAverage'.format(i), data=temp)
        result = requests.post('http://172.26.152.178:9000/FederatedAverage', data=temp)
    except:
        print("Failed to broadcast parameters to 172.26.152.178")

@app.route('/FederatedAverage', methods=['POST'])
def FederatedAverage():
    global network
    global list_w
    global list_updated
    global list_n
    global cnt
    if request.remote_addr in list_updated:
        print("denied")
        return ''
    if len(list_w)<C*K:
        # add received weight to pool
        list_updated.append(request.remote_addr)
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
            update = threading.Thread(target=Update)
            update.start()
            cnt += 1
            list_updated = []
            list_w = []
            list_n = []
    return ''

@app.route('/activate',methods=['GET'])
def activate():
    global activated
    if not activated:
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

def app_run():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    ip = s.getsockname()[0]
    app.run(host='{}'.format(ip), port=9000)

if __name__ == '__main__':
    main_app = threading.Thread(target=app_run)
    # main_app.daemon = True
    main_app.start()
    # sleep(5)
    # plot=threading.Thread(target=plot)
    # plot.start()
# while True:
#     pass

        # print('Done')

# w=[]
# for m in network.list_weight:
#     # print(m)
#     w.append(m.tolist())
# # print(len(w))
# ww=json.dumps(w)
# print(array(json.loads(ww)[0]))
# print(array(json.loads(ww)[1]))


