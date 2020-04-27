import requests
import json
import threading
import matplotlib.pyplot as plt
import ast

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

loss=[[],[],[],[],[],[],[],[],[],[],[]]

# plot
# def plot():


#     plt.plot(network.history_loss, color='blue')
#     plt.pause(5)
#     while True:
#         plt.plot(network.history_loss)
#         plt.pause(0.01)

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
        plt.title("Training loss of clients")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        for i,port in zip(range(0,10),range(9000,9010)):
            # if i==2:
            try:
                result = requests.get(f'http://139.198.0.182:{port}/plot')
                # print(result.text.__class__)
                loss[i]=ast.literal_eval(result.text)
                plt.plot(loss[i], label=f'client {i}')
            except:
                print(f"Failed when getting plot from 139.198.0.182:{port}")
                continue
        # try:
        #     result = requests.get(f'http://192.168.88.227:9000/plot')
        #     # print(result.text.__class__)
        #     loss[-1]=ast.literal_eval(result.text)
        #     plt.plot(loss[-1], label=f'client 2 (local)', linestyle='-.')
        # except:
        #     print(f"Failed when getting plot from 192.168.88.227:9000")
        # plt.show()
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


