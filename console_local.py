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

# plot
# def plot():


#     plt.plot(network.history_loss, color='blue')
#     plt.pause(5)
#     while True:
#         plt.plot(network.history_loss)
#         plt.pause(0.01)

def activate(port):
    try:
        result = requests.get(f'http://26.26.26.1:{port}/activate')
        print(result.text)
    except:
        print(f"Failed when activating 26.26.26.1:{port}")

print("This is the console of FL-test. Copyright© Bo Zhao, all rights reserved.")

while True:
    cmd=input(">>>")
    if cmd=='plot' or cmd=='p':
        plt.grid(True)
        plt.title("Training loss and ERR")
        plt.xlabel('Iteration')
        # for i,port in zip(range(0,10),range(9000,9010)):
            # if i==2:
        try:
            result_train = requests.get(f'http://26.26.26.1:9000/plot_train')
            result_test = requests.get(f'http://26.26.26.1:9000/plot_test')
            # print(result.text.__class__)
            loss_train[0]=np.array(ast.literal_eval(result_train.text))
            rate_err[0] = np.array(ast.literal_eval(result_test.text))
            plt.plot(loss_train[0], label=f'loss_{0}')
            plt.plot(rate_err[0], label=f'err_{0}', linestyle='-.')
        except:
            print(f"Failed when getting plot from 172.26.152.178:9000")
            continue
        # print(np.array(rate_err))
        # plt.plot(np.mean(np.array(rate_err), axis=0),label=f'err_global', linestyle='-.')
        # try:
        #     result_train = requests.get(f'http://172.26.154.224:9000/plot_train')
        #     result_test = requests.get(f'http://172.26.154.224:9000/plot_test')
        #     # print(result.text.__class__)
        #     loss_train[-1]=ast.literal_eval(result_train.text)
        #     rate_err[-1]=ast.literal_eval(result_test.text)
        #     plt.plot(loss_train[-1], label=f'client 0 (local)', linestyle='-.')
        #     plt.plot(rate_err[-1], label=f'error rate client 0 (local)', linestyle='-.')
        # except:
        #     print(f"Failed when getting plot from 172.26.154.224:9000")

        # plt.show()
        # plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)
        plt.legend()
        plt.show()
    elif cmd=='activate' or cmd=='a':
        # for port in range(9000,9010):
        t=threading.Thread(target=activate, args=(9000,))
        t.start()
    elif cmd=='\r':
        continue
    elif cmd=='quit' or cmd=='q':
        print("Leaving...")
        break
    else:
        print("Invalid command.")


