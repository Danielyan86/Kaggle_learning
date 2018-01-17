#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import csv
import random
from functools import reduce

from numpy import *
import pandas as pd


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


class Node(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        self.output = output

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        self.upstream.append(conn)

    def calc_output(self):
        # 每个节点的输出算法，N元一次方程求和
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        # 结果放入激活函数
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str


class ConstNode(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str


class Layer(object):
    def __init__(self, layer_index, node_count):
        self.layer_index = layer_index
        self.nodes = []
        # 初始化节点对象
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        for node in self.nodes:
            print(node)


class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0

    def calc_gradient(self):
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def update_weight(self, rate):
        self.calc_gradient()
        self.weight += rate * self.gradient

    def get_gradient(self):
        return self.gradient

    def __str__(self):
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)


class Connections(object):
    def __init__(self):
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

    def dump(self):
        for conn in self.connections:
            print(conn)


class Network(object):
    def __init__(self, layers):
        self.connections = Connections()
        self.layers = []
        # 计算网络层数
        layer_count = len(layers)
        node_count = 0
        # 初始化网络层，网错层对象append在self.layers 里面，而节点对象又在layer里面被初始化
        # Connections 仅仅作为Connection的集合对象，提供一些集合操作, 而layer有是节点对象合集
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    def train(self, labels, data_set, rate, epoch):
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)
                # print 'sample %d training finished' % d

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def calc_delta(self, label):
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self, label, sample):
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def predict(self, sample):
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return list(map(lambda node: node.output, self.layers[-1].nodes[:-1]))

    def dump(self):
        for layer in self.layers:
            layer.dump()


class Normalizer(object):
    def __init__(self):
        self.mask = [
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
        ]

    def norm(self, number):
        return list(map(lambda m: 0.9 if number & m else 0.1, self.mask))

    def denorm(self, vec):
        binary = list(map(lambda i: 1 if i > 0.5 else 0, vec))
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x, y: x + y, binary)


def mean_square_error(vec1, vec2):
    return 0.5 * reduce(lambda a, b: a + b,
                        list(map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                                 zip(vec1, vec2)
                                 ))
                        )


def gradient_check(network, sample_feature, sample_label):
    '''
    梯度检查
    network: 神经网络对象
    sample_feature: 样本的特征
    sample_label: 样本的标签
    '''
    # 计算网络误差
    network_error = lambda vec1, vec2: \
        0.5 * reduce(lambda a, b: a + b,
                     list(map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                              zip(vec1, vec2))))

    # 获取网络在当前样本下每个连接的梯度
    network.get_gradient(sample_feature, sample_label)

    # 对每个权重做梯度检查    
    for conn in network.connections.connections:
        # 获取指定连接的梯度
        actual_gradient = conn.get_gradient()

        # 增加一个很小的值，计算网络的误差
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)

        # 减去一个很小的值，计算网络的误差
        conn.weight -= 2 * epsilon  # 刚才加过了一次，因此这里需要减去2倍
        error2 = network_error(network.predict(sample_feature), sample_label)

        # 根据式6计算期望的梯度值
        expected_gradient = (error2 - error1) / (2 * epsilon)

        # 打印
        print('expected gradient: \t%f\nactual gradient: \t%f' % (
            expected_gradient, actual_gradient))


def train_data_set():
    # with open("./data/train_new.csv", "r", newline='') as csvfile:
    #     csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     content = [line for line in csv_reader]
    # data_set = []
    # for line in content:
    #     # for number in line[1:]:
    #     #     new_line = int(float(number))
    #     new_line = [float(number) for number in line[1:]]
    #     data_set.append(new_line)
    #
    # origin_lable = [int(line[0]) for line in content]
    # labels = _convert_label(origin_lable)
    # return labels, data_set
    pd_reader = pd.read_csv("./data/train_new.csv")
    origin_lable = pd_reader.Survived.values.tolist()
    lables = _convert_label(origin_lable)
    pd_reader = pd_reader.drop(['Survived'], 1)
    data_set = pd_reader.values.tolist()
    return lables, data_set


def _convert_label(lables):
    new_list = []
    for item in lables:
        if item == 0:
            new_list.append([0.9, 0.1])
        elif item == 1:
            new_list.append([0.1, 0.9])
    return new_list


def train(network):
    assert isinstance(network, object)
    labels, data_set = train_data_set()
    network.train(labels, data_set, 0.01, 1000)


def test(network, data):
    normalizer = Normalizer()
    norm_data = normalizer.norm(data)
    predict_data = network.predict(norm_data)
    print('\ttestdata(%u)\tpredict(%u)' % (
        data, normalizer.denorm(predict_data)))


def correct_ratio(network):
    normalizer = Normalizer()
    correct = 0.0
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print('correct_ratio: %.2f%%' % (correct / 256 * 100))


def gradient_check_test():
    net = Network([2, 2, 2])
    sample_feature = [0.9, 0.1]
    sample_label = [0.9, 0.1]
    gradient_check(net, sample_feature, sample_label)


if __name__ == '__main__':
    # gradient_check_test()
    # 设置神经网络初始化参数，初始化神经网络,列表长度表示网络层数，每个数字表示每一层节点个数
    #
    # with open("./data/train_new.csv", "r", newline='') as csvfile:
    #     csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     test_data = []
    #     expected_data = []
    #     for line in csv_reader:
    #         test_data.append([float(number) for number in line])
    #         # expected_data.append(int(line[0]))
    #
    # net = Network([6, 4, 2])
    # train(net)
    # right_number = 0
    # for data in test_data:
    #     result_two_dimension = net.predict(data[1:])
    #     result = 0 if result_two_dimension[0] > result_two_dimension[1] else 1
    #     if result == data[0]:
    #         right_number = right_number + 1
    # print(right_number)
    #
    # with open("./data/test_new.csv", "r", newline='') as csvfile:
    #     csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     test_data = []
    #     for line in csv_reader:
    #         test_data.append([float(num) for num in line])
    #
    # result_list_final = []
    # for data in test_data:
    #     result_two_dimension = net.predict(data[1:])
    #     result = 0 if result_two_dimension[0] > result_two_dimension[1] else 1
    #     result_list_final.append(result)
    #
    # with open("./data/my_submission.csv", "w", newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     print(result_list_final)
    #     result_combine = list(zip(range(892, 1310), result_list_final))
    #     # print(list(result_combine))
    #     for item in result_combine:
    #         csv_writer.writerow(item)

    net = Network([7, 5, 5, 2])
    train(net)

    pd_reader = pd.read_csv("./data/test_new.csv")
    test_data_list = pd_reader.values.tolist()
    print(test_data_list)
    for item in test_data_list:
        print(net.predict(item))
    res_list = [net.predict(n) for n in test_data_list]
    final_res = [
    ]
    for item in res_list:
        if item[0] > item[1]:
            final_res.append(0)
        else:
            final_res.append(1)
    final_res = list(zip(range(892, 1310), final_res))
    pd.DataFrame(final_res).to_csv("./data/my_submission.csv", index=False)
