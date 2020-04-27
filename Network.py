# copyright© Bo Zhao, all rights reserved. E-MAIL: bozhao@nuaa.edu.cn
# This work is supported by Liming Fang @ Nanjing University of Aeronautics and Astronautics
from numpy import *
from numpy.random import *
import matplotlib.pyplot as plt
import csv


# neural network
class Network:
    list_weight = []
    list_loss_history = []
    rate_learning = 0
    rate_regularization = 0
    term_regularization = 0
    # plot
    history_loss = []

    # initialization
    def __init__(self, list_description: list, rate_learning: float, rate_regularization: float):
        for i in range(0, len(list_description) - 1):
            self.list_weight.append(zeros((list_description[i + 1], list_description[i] + 1)))
            # self.list_weight.append(rand(list_description[i + 1], list_description[i] + 1))
            self.rate_learning = rate_learning
            self.rate_regularization = rate_regularization
            for m in self.list_weight:
                self.term_regularization += sum(fabs(m))
            self.term_regularization *= 2

    # sigmoid neuron function
    def Sigmoid(self, inX):
        return 1.0 / (1 + exp(-inX))

    # forward propagation
    def Predict(self, array_input: list):
        layer_current = array(array_input.copy())
        if len(layer_current[0]) != len(self.list_weight[0][0]) - 1:
            raise Exception("The size of input vector should be {}".format(len(self.list_weight[0][0]) - 1))
        for m in self.list_weight:
            layer_current = append(ones([len(array_input), 1]), layer_current, axis=1)
            layer_current = matmul(layer_current, m.transpose())
            layer_current = self.Sigmoid(layer_current)
        return layer_current

    def GradientDescent(self, array_input: list, array_label: list):
        a_i = []
        loss_sum = 0

        # forward propagation
        layer_current = array(array_input.copy())
        label_current = array(array_label.copy())
        if len(layer_current[0]) != len(self.list_weight[0][0]) - 1:
            raise Exception("The size of input vector should be {}".format(len(self.list_weight[0][0]) - 1))
        elif len(label_current[0]) != len(self.list_weight[-1]):
            raise Exception("The size of label vector should be {}".format(len(self.list_weight[-1])))
        for m in self.list_weight:
            layer_current = append(ones([len(array_input), 1]), layer_current, axis=1)
            a_i.append(layer_current)
            layer_current = matmul(layer_current, m.transpose())
            layer_current = self.Sigmoid(layer_current)
        a_i.append(layer_current)

        # backward propagation
        loss = add(subtract(a_i[-1], label_current), self.term_regularization * self.rate_regularization)
        loss_sum = sum(abs(loss))
        derivative_sigmoid = multiply(a_i[-1], subtract(ones([len(a_i[-1]), len(a_i[-1][0])]), a_i[-1]))
        rate_error = multiply(loss, derivative_sigmoid)
        gradient_current_average = divide(matmul(rate_error[:, :, None], a_i[-2][:, None]).sum(axis=0),
                                          len(array_input))
        self.list_weight[-1] = subtract(self.list_weight[-1], multiply(gradient_current_average, self.rate_learning))
        for i in range(-2, -len(self.list_weight) - 1, -1):
            derivative_sigmoid = multiply(a_i[i], subtract(ones([len(a_i[i]), len(a_i[i][0])]), a_i[i]))
            rate_error = delete(multiply(matmul(rate_error, self.list_weight[i + 1]), derivative_sigmoid), 0, axis=1)
            gradient_current_average = divide(matmul(rate_error[:, :, None], a_i[i - 1][:, None]).sum(axis=0),
                                              len(array_input))
            self.list_weight[i] = subtract(self.list_weight[i], multiply(gradient_current_average, self.rate_learning))
        self.list_loss_history.append(loss_sum / len(array_input))




# load training set
def LoadDataset(abs_path: str, size_batch: int = 1):
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
        label_temp_1 = [1, 0, 0, 0, 0, 0] if row[-1] == 'WALKING' else [0, 1, 0, 0, 0, 0] if row[-1] == 'SITTING' else [
            0, 0, 1, 0, 0, 0] if row[-1] == 'STANDING' else [0, 0, 0, 1, 0, 0] if row[-1] == 'LAYING' else [0, 0, 0, 0,
                                                                                                            1, 0] if \
            row[-1] == 'WALKING_UPSTAIRS' else [0, 0, 0, 0, 0, 1]
        row.pop(-1)
        # row.pop(0)
        for num in row:
            if num == '':
                sample_temp_1.append(0.00)
            else:
                sample_temp_1.append(float(num))
        sample_temp_2.append(sample_temp_1.copy())
        label_temp_2.append(label_temp_1.copy())
        if i % size_batch == 0:
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
        # print('Epoch {}...'.format(cnt), end='')
        # gradient clipping
        if cnt % range_gradient_clipping == 0:
            network.rate_learning *= rate_gradient_clipping
        # simulated annealing
        if cnt > 2:
            if abs(network.history_loss[-2] - network.history_loss[-1]) < threshold_differential and network.history_loss[-1] > threshold_loss:
                network.rate_learning *= rate_annealing
        # batch training
        for batch_sample, batch_label in zip(sample_train, label_train):
            network.GradientDescent(batch_sample, batch_label)
        # loss calculation
        network.history_loss.append(sum(network.list_loss_history) / len(network.list_loss_history))
        network.list_loss_history.clear()
        # print('Done')

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
# f=open('E:\Project\Python\dataset\human-activity-recognition-with-smartphones\subject1.csv')
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
