import requests
import json
import threading
import matplotlib.pyplot as plt
import ast
import numpy as np
import socket

loss_0=[]
loss_1=[]
loss_2=[]
loss_3=[]
loss_4=[]
loss_5=[]
loss_6=[]
loss_7=[]
loss_8=[]
loss_9=[]

loss_train=[[], [], [], [], [], [], [], [], [], []]
acc_train=[[], [], [], [], [], [], [], [], [], []]
acc_test=[[], [], [], [], [], [], [], [], [], []]
loss_test=[[], [], [], [], [], [], [], [], [], []]
acc_cm=[[], [], [], [], [], [], [], [], [], []]
loss_cm=[[], [], [], [], [], [], [], [], [], []]
# acc_cme=[[], [], [], [], [], [], [], [], [], []]
# loss_cme=[[], [], [], [], [], [], [], [], [], []]

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('8.8.8.8', 80))
addr = s.getsockname()[0]

pool_thread=[]

activated=False



def activate(port):
    try:
        result = requests.get(f'http://{addr}:{port}/activate')
        print(result.text)
    except:
        print(f"Failed when activating {addr}:{port}")

print("This is the console of FL-test. Copyright© Bo Zhao, all rights reserved.")

def plot(i,port):
    try:
        result_train_loss = requests.get(f'http://{addr}:{port}/plot_train_loss')
        result_train_acc = requests.get(f'http://{addr}:{port}/plot_train_acc')
        result_test_acc = requests.get(f'http://{addr}:{port}/plot_test_acc')
        result_test_loss = requests.get(f'http://{addr}:{port}/plot_test_loss')
        result_cm_acc = requests.get(f'http://{addr}:{port}/plot_cm_acc')
        result_cm_loss = requests.get(f'http://{addr}:{port}/plot_cm_loss')
        # result_cme_acc = requests.get(f'http://{addr}:{port}/plot_cme_acc')
        # result_cme_loss = requests.get(f'http://{addr}:{port}/plot_cme_loss')
        loss_train[i] = np.array(ast.literal_eval(result_train_loss.text))
        acc_train[i] = np.array(ast.literal_eval(result_train_acc.text))
        acc_test[i] = np.array(ast.literal_eval(result_test_acc.text))
        loss_test[i] = np.array(ast.literal_eval(result_test_loss.text))
        acc_cm[i] = np.array(ast.literal_eval(result_cm_acc.text))
        loss_cm[i] = np.array(ast.literal_eval(result_cm_loss.text))
        # acc_cme[i] = np.array(ast.literal_eval(result_cme_acc.text))
        # loss_cme[i] = np.array(ast.literal_eval(result_cme_loss.text))
    except:
        print(f"Failed when getting plot from 139.198.0.182:{port}")

while True:
    cmd=input(">>>")
    if cmd=='plot' or cmd=='p':
        pool_thread = []
        for i,port in zip(range(0,10),range(9000,9010)):
            t=threading.Thread(target=plot, args=(i,port))
            pool_thread.append(t)
            # try:
            #     result_train = requests.get(f'http://{addr}:{port}/plot_train')
            #     result_test = requests.get(f'http://{addr}:{port}/plot_test')
            #     loss_train[i]=np.array(ast.literal_eval(result_train.text))
            #     rate_err[i] = np.array(ast.literal_eval(result_test.text))
            #     plt.plot(loss_train[i], label=f'loss_{i}')
            #     # plt.plot(rate_err[i], label=f'err_{i}', linestyle='-.')
            # except:
            #     print(f"Failed when getting plot from 139.198.0.182:{port}")
            #     continue
        for i in range(0,10):
            pool_thread[i].start()
        for i in range(0,1):
            pool_thread[i].join()
        # for i in range(0, 10):
            plt.plot(loss_train[i], label=f'loss_train_{i}', linestyle='-')
            plt.plot(acc_train[i], label=f'acc_train_{i}', linestyle='-.')
            plt.plot(acc_test[i], label=f'acc_test_{i}', linestyle='-.')
            plt.plot(loss_test[i], label=f'loss_test_{i}', linestyle='-')
            plt.plot(acc_cm[i], label=f'acc_cm_{i}', linestyle='-.')
            plt.plot(loss_cm[i], label=f'loss_cm_{i}', linestyle='-')
            # plt.plot(acc_cme[i], label=f'acc_cme_{i}', linestyle='-.')
            # plt.plot(loss_cme[i], label=f'loss_cme_{i}', linestyle='-')
        # try:
        #     plt.plot(np.mean(np.array(rate_err), axis=0),label=f'acc_global', linestyle='-.')
        # except:
        #     print('Please wait for vector alignment...')
        #     continue
        plt.grid(True)
        plt.title("Training loss and ERR")
        plt.xlabel('Iteration')
        plt.legend()
        plt.show()
        plt.close()
    elif cmd=='activate' or cmd=='a':
        if not activated:
            for port in range(9000,9010):
                t=threading.Thread(target=activate, args=(port,))
                t.start()
                activated=True
        else:
            print('Please type "confirm" to confirm restart:')
            confirm=input()
            if confirm == 'confirm':
                for port in range(9000, 9010):
                    t = threading.Thread(target=activate, args=(port,))
                    t.start()
            else:
                print('Misoperation. Ignored.')
    elif cmd=='\r':
        continue
    elif cmd=='quit' or cmd=='q':
        print("Leaving...")
        break
    else:
        print("Invalid command.")