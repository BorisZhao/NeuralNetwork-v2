# copyright© Bo Zhao, all rights reserved. E-MAIL: bozhao@nuaa.edu.cn
# This work is supported by Liming Fang @ Nanjing University of Aeronautics and Astronautics
# from numpy import *
import cupy.random as rd
import numpy as np
import cupy as cp
# from numpy.random import *
import matplotlib.pyplot as plt
import csv
# import cupy as cp
# import minpy.numpy as mnp

# neural network
class Network:
    list_weight = []
    list_loss_history = []
    rate_learning = 0
    rate_regularization = 0
    term_regularization = 0
    # Adam parameters
    beta_1=0.9
    beta_2=0.999
    epsron=0.00000001
    list_mt=[]
    list_vt=[]
    iteration=0
    # plot
    history_train_loss = []
    history_test_loss = []
    history_test_acc = []

    # initialization
    def __init__(self, list_description: list, rate_learning: float, rate_regularization: float):
        for i in range(0, len(list_description) - 1):
            # self.list_weight.append(cp.zeros((list_description[i + 1], list_description[i] + 1)))
            self.list_mt.append(cp.zeros((list_description[i + 1], list_description[i] + 1)))
            self.list_vt.append(cp.zeros((list_description[i + 1], list_description[i] + 1)))
            # rd.seed(0)
            self.list_weight.append(rd.normal(loc=0.0, scale=1, size=(list_description[i + 1], list_description[i] + 1)))
            # print(self.list_weight[0])
            self.rate_learning = rate_learning
            self.rate_regularization = rate_regularization
            for m in self.list_weight:
                self.term_regularization += cp.sum(cp.asarray(np.fabs(cp.asnumpy(m))))
            self.term_regularization *= 2

    # sigmoid neuron function
    def Neuron(self, inX):
        # Sigmoid
        return 1.0 / (1 + cp.exp(-inX))

        #Softplus
        # return cp.log(1+cp.exp(-cp.abs(inX)))+cp.maximum(inX,0)

    def Neuron_D(self, inX):
        # Sigmoid
        return inX*(1-inX)

        # Softplus
        # return 1.0 / (1 + cp.exp(-inX))

    # forward propagation
    def Predict(self, array_input: list):
        layer_current = cp.array(array_input.copy())
        if len(layer_current[0]) != len(self.list_weight[0][0]) - 1:
            raise Exception("The size of input vector should be {}".format(len(self.list_weight[0][0]) - 1))
        for m in self.list_weight:
            layer_current = cp.asarray(np.append(cp.asnumpy(cp.ones((len(array_input), 1))), cp.asnumpy(layer_current), axis=1))
            layer_current = cp.matmul(layer_current, cp.asarray(m.transpose(), dtype='float32'))
            layer_current = self.Neuron(layer_current)
        # print(layer_current)
        return layer_current

    def GradientDescent(self, array_input: list, array_label: list):
        a_i = []
        loss_sum = 0

        # forward propagation
        layer_current = cp.array(array_input.copy())
        label_current = cp.array(array_label.copy())
        if len(layer_current[0]) != len(self.list_weight[0][0]) - 1:
            raise Exception("The size of input vector should be {}".format(len(self.list_weight[0][0]) - 1))
        elif len(label_current[0]) != len(self.list_weight[-1]):
            raise Exception("The size of label vector should be {}".format(len(self.list_weight[-1])))
        for m in self.list_weight:
            layer_current = cp.asarray(np.append(cp.asnumpy(cp.ones((len(array_input), 1))), cp.asnumpy(layer_current), axis=1))
            a_i.append(layer_current)
            layer_current = cp.matmul(layer_current, cp.asarray(m.transpose(), dtype='float32'))
            layer_current = self.Neuron(layer_current)
        # print(layer_current)
        a_i.append(layer_current)
        # print(cp.sum(cp.abs(cp.subtract(a_i[-1], label_current))))

        # backward propagation
        self.iteration+=1
        loss = cp.subtract(a_i[-1], label_current) + self.term_regularization * self.rate_regularization
        loss_sum = cp.sum(cp.abs(loss))
        # derivative_ReLu = multiply(a_i[-1], subtract(ones((len(a_i[-1]), len(a_i[-1][0]))), a_i[-1]))
        derivative_ReLu = self.Neuron_D(a_i[-1])
        rate_error = cp.multiply(loss, derivative_ReLu)
        gradient_current_average = cp.divide(cp.matmul(rate_error[:, :, None], a_i[-2][:, None]).sum(axis=0),
                                          len(array_input))
        self.list_mt[-1]=cp.add(cp.multiply(self.list_mt[-1],self.beta_1),cp.multiply(gradient_current_average, (1-self.beta_1)))
        self.list_vt[-1]=cp.add(cp.multiply(self.list_vt[-1], self.beta_2),cp.multiply(cp.power(gradient_current_average,2),(1-self.beta_2)))
        mt_=cp.divide(self.list_mt[-1], 1-self.beta_1**self.iteration)
        vt_=cp.divide(self.list_vt[-1], 1-self.beta_2**self.iteration)
        self.list_weight[-1]=cp.subtract(cp.asarray(self.list_weight[-1], dtype='float32'), cp.multiply(cp.divide(mt_, cp.add(cp.power(vt_, 0.5), self.epsron)), self.rate_learning))
        # self.list_weight[-1] = cp.subtract(cp.asarray(self.list_weight[-1], dtype='float32'), cp.multiply(gradient_current_average, self.rate_learning))
        for i in range(-2, -len(self.list_weight) - 1, -1):
            # derivative_ReLu = multiply(a_i[i], subtract(ones((len(a_i[i]), len(a_i[i][0]))), a_i[i]))
            derivative_ReLu = self.Neuron_D(a_i[i])
            rate_error = cp.asarray(np.delete(cp.asnumpy(cp.multiply(cp.matmul(rate_error, self.list_weight[i + 1]), derivative_ReLu)), 0, axis=1))
            gradient_current_average = cp.divide(cp.matmul(rate_error[:, :, None], a_i[i - 1][:, None]).sum(axis=0),
                                              len(array_input))
            self.list_mt[i] = cp.add(cp.multiply(self.list_mt[i], self.beta_1),
                                   cp.multiply(gradient_current_average, (1 - self.beta_1)))
            self.list_vt[i] = cp.add(cp.multiply(self.list_vt[i], self.beta_2),
                                   cp.multiply(cp.power(gradient_current_average, 2), (1 - self.beta_2)))
            mt_ = cp.divide(self.list_mt[i], 1 - self.beta_1**self.iteration)
            vt_ = cp.divide(self.list_vt[i], 1 - self.beta_2**self.iteration)
            self.list_weight[i] = cp.subtract(cp.asarray(self.list_weight[i], dtype='float32'),
                                            cp.multiply(cp.divide(mt_, cp.add(cp.power(vt_, 0.5), self.epsron)),
                                                     self.rate_learning))
            # self.list_weight[i] = cp.subtract(cp.asarray(self.list_weight[i], dtype='float32'), cp.multiply(gradient_current_average, self.rate_learning))
        self.list_loss_history.append(loss_sum / len(array_input))


# load training set
def LoadDataset(abs_path: str, size_batch: int = 1, normalization=False):
    print("Loading training set...", end='')
    sample = []
    label = []
    f = open(abs_path)
    f_csv = csv.reader(f)
    sample_temp_2 = []
    label_temp_2 = []
    for i, row in enumerate(f_csv):
        if i == 0:
            continue
        sample_temp_1 = []
        # user-defined rules of label and pre-process required
        # label_temp_1=[0,0,0]
        # label_temp_1[int(row[-1])-1]=1
        label_temp_1 = [1, 0, 0, 0, 0, 0] if row[-1] == 'WALKING' else [0, 1, 0, 0, 0, 0] if row[-1] == 'SITTING' else [
            0, 0, 1, 0, 0, 0] if row[-1] == 'STANDING' else [0, 0, 0, 1, 0, 0] if row[-1] == 'LAYING' else [0, 0, 0, 0,
                                                                                                            1, 0] if \
            row[-1] == 'WALKING_UPSTAIRS' else [0, 0, 0, 0, 0, 1]
        row.pop(-1)
        # row.pop(0)

        # row.pop(0)
        for num in row:
            if num == '':
                sample_temp_1.append(0.00)
            else:
                sample_temp_1.append(float(num))
        sample_temp_2.append(sample_temp_1.copy())
        label_temp_2.append(label_temp_1.copy())
        if i % size_batch == 0:
            # print(sample_temp_2)
            if normalization:
                # print(sample_temp_2)
                sample_temp_2=cp.asarray(sample_temp_2)
                sample_temp_2=cp.add(sample_temp_2,0.001)
                mu=cp.mean(sample_temp_2, axis=0)+0.001
                sigma=cp.std(sample_temp_2, axis=0)+0.001
                sample_temp_2=((sample_temp_2-mu)/sigma).tolist()
                # print(sample_temp_2)
            sample.append(sample_temp_2.copy())
            label.append(label_temp_2.copy())
            sample_temp_2.clear()
            label_temp_2.clear()
    f.close()
    print("Done")
    return sample, label

def LoadDatasetPoisoned(abs_path: str, size_batch: int = 1, normalization=False):
    print("Loading training set...", end='')
    sample = []
    label = []
    f = open(abs_path)
    f_csv = csv.reader(f)
    sample_temp_2 = []
    label_temp_2 = []
    import random
    for i, row in enumerate(f_csv):
        if i == 0:
            continue
        sample_temp_1 = []
        # user-defined rules of label and pre-process required
        # label_temp_1=[0,0,0]
        # label_temp_1[int(row[-1])-1]=1
        label_temp_1 = [0, 1, 0, 0, 0, 0] if row[-1] == 'WALKING' else [0, 0, 1, 0, 0, 0] if row[-1] == 'SITTING' else [
            0, 0, 0, 1, 0, 0] if row[-1] == 'STANDING' else [0, 0, 0, 0, 1, 0] if row[-1] == 'LAYING' else [0, 0, 0, 0,
                                                                                                            0, 1] if \
            row[-1] == 'WALKING_UPSTAIRS' else [1, 0, 0, 0, 0, 0]
        row.pop(-1)
        # row.pop(0)

        # row.pop(0)
        for num in row:
            if num == '':
                sample_temp_1.append(0.00)
            else:
                sample_temp_1.append(float(num))
        sample_temp_2.append(sample_temp_1.copy())
        label_temp_2.append(label_temp_1.copy())
        if i % size_batch == 0:
            # print(sample_temp_2)
            if normalization:
                # print(sample_temp_2)
                sample_temp_2 = cp.asarray(sample_temp_2)
                sample_temp_2 = cp.add(sample_temp_2, 0.001)
                mu = cp.mean(sample_temp_2, axis=0) + 0.001
                sigma = cp.std(sample_temp_2, axis=0) + 0.001
                sample_temp_2 = ((sample_temp_2 - mu) / sigma).tolist()
                # print(sample_temp_2)
            sample.append(sample_temp_2.copy())
            label.append(label_temp_2.copy())
            sample_temp_2.clear()
            label_temp_2.clear()
    f.close()
    print("Done")
    return sample, label

# Training
def Train(network: Network, sample_train: list, label_train: list, round_epoch: int, range_gradient_clipping: float,
          rate_gradient_clipping: float, threshold_differential: float, threshold_loss: float, rate_annealing: float):
    # start training
    # print("Start training...")
    for cnt in range(0, round_epoch):
        # network.beta_1 = pow(network.beta_1, i * round_epoch+cnt+1)
        # network.beta_2 = pow(network.beta_2, i * round_epoch+cnt+1)
        # print('Epoch {}...'.format(cnt), end='')
        # gradient clipping
        if cnt % range_gradient_clipping == 0:
            network.rate_learning *= rate_gradient_clipping
        # simulated annealing
        if cnt > 2:
            # print(network.list_loss_history[-2])
            if abs(network.list_loss_history[-2] - network.list_loss_history[-1]) < threshold_differential and network.list_loss_history[-1] > threshold_loss:
                network.rate_learning *= rate_annealing
        # batch training
        for batch_sample, batch_label in zip(sample_train, label_train):
            network.GradientDescent(batch_sample, batch_label)
    # loss calculation
    network.history_train_loss.append((np.sum(network.list_loss_history) / len(network.list_loss_history)).tolist())
    # print(network.history_train_loss[-1])
    network.list_loss_history=[]


def Test(network: Network, sample_test: list, label_test: list):
    # predict
    err = 0
    loss=[]
    for batch_sample, batch_label in zip(sample_test,label_test):
        loss.append(cp.sum(cp.abs(cp.subtract(cp.asarray(network.Predict(batch_sample)[0]), cp.asarray(batch_label[0])))))
    loss=cp.mean(cp.asarray(loss)).tolist()
    network.history_test_loss.append(loss)
    for batch_sample, batch_label in zip(sample_test, label_test):
        # print(network.Predict(batch_sample))
        # print(batch_sample)
        # print('===============')
        # print("Predict:{}".format(network.Predict(batch_sample)[0]))
        # print('label:{}'.format(cp.asnumpy(batch_label[0])))
        if np.argmax(cp.asnumpy(network.Predict(batch_sample)[0])) == np.argmax(cp.asnumpy(batch_label[0])):
            continue
        else:
            err += 1
    network.history_test_acc.append(1-(err / len(sample_test)))
    # print(network.history_test_err[-1])

# # statistic
# time_end=time()
# print("Done. Time in total:")
# print(str(time_end-time_start)+"s")
# print("=========parameters=========")
# print('Learning rate:{}'.format(rate_learning))
# print('Normalization rate:{}'.format(rate_regularization))
# print('batch size:{}'.format(size_batch))
# print('epoch round:{}'.format(round_epoch))
# print("=========output=========")
#
# # load test set
# f=open('E:\Project\Python\dataset\human-activity-recognition-with-smartphones\train0.csv')
# f_csv=csv.reader(f)
# for i, row in enumerate(f_csv):
#     sample_test = []
#     if i == 0:
#         continue
#     if i>300 and i<321:
#         print(row[-1])
#         row.pop(-1)
#         # row.pop(0)
#         temp=[]
#         for num in row:
#             if num == '':
#                 temp.append(0.00)
#             else:
#                 temp.append(float(num))
#         sample_test.append(temp)
#         print(sample_test)
#         print(network.Predict(sample_test))
# f.close()
# plt.show()
