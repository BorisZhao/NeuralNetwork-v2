import requests
import json
import threading
import matplotlib.pyplot as plt
import ast
import numpy as np

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


def activate(port):
    try:
        result = requests.get(f'http://139.198.0.182:{port}/activate')
        print(result.text)
    except:
        print(f"Failed when activating 139.198.0.182:{port}")

print("This is the console of FL-test. Copyright© Bo Zhao, all rights reserved.")

while True:
    cmd=input(">>>")
    if cmd=='plot' or cmd=='p':
        plt.grid(True)
        plt.title("Training loss and ERR")
        plt.xlabel('Iteration')
        for i,port in zip(range(0,10),range(9000,9010)):
            try:
                result_train = requests.get(f'http://139.198.0.182:{port}/plot_train')
                result_test = requests.get(f'http://139.198.0.182:{port}/plot_test')
                loss_train[i]=np.array(ast.literal_eval(result_train.text))
                rate_err[i] = np.array(ast.literal_eval(result_test.text))
                plt.plot(loss_train[i], label=f'loss_{i}')
                # plt.plot(rate_err[i], label=f'err_{i}', linestyle='-.')
            except:
                print(f"Failed when getting plot from 139.198.0.182:{port}")
                continue
        plt.plot(np.mean(np.array(rate_err), axis=0),label=f'err_global', linestyle='-.')
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


