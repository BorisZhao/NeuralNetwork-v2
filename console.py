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
rate_err=[[], [], [], [], [], [], [], [], [], []]

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('8.8.8.8', 80))
addr = s.getsockname()[0]

pool_thread=[]

def activate(port):
    try:
        result = requests.get(f'http://{addr}:{port}/activate')
        print(result.text)
    except:
        print(f"Failed when activating {addr}:{port}")

print("This is the console of FL-test. Copyright© Bo Zhao, all rights reserved.")

def plot(i,port):
    try:
        result_train = requests.get(f'http://{addr}:{port}/plot_train')
        result_test = requests.get(f'http://{addr}:{port}/plot_test')
        loss_train[i] = np.array(ast.literal_eval(result_train.text))
        rate_err[i] = np.array(ast.literal_eval(result_test.text))
    except:
        print(f"Failed when getting plot from 139.198.0.182:{port}")

while True:
    cmd=input(">>>")
    if cmd=='plot' or cmd=='p':
        for i,port in zip(range(0,10),range(9000,9010)):
            t=threading.Thread(target=plot, args=(i,port))
            t.start()
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
            pool_thread[i].join()
        plt.grid(True)
        plt.title("Training loss and ERR")
        plt.xlabel('Iteration')
        for i in range(0, 10):
            plt.plot(loss_train[i], label=f'loss_{i}')
            plt.plot(rate_err[i], label=f'err_{i}', linestyle='-.')
            # plt.plot(np.mean(np.array(rate_err), axis=0),label=f'err_global', linestyle='-.')
        plt.legend()
        plt.show()
    elif cmd=='activate' or cmd=='a':
        for port in range(9000,9010):
            t=threading.Thread(target=activate, args=(port,))
            t.start()
    elif cmd=='\r':
        continue
    elif cmd=='quit' or cmd=='q':
        print("Leaving...")
        break
    else:
        print("Invalid command.")