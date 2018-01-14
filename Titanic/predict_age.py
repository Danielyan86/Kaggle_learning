# !/usr/bin/env python
# -*- coding: UTF-8 -*-

from Titanic.perceptron import Perceptron


# 定义激活函数f
def active_function(x):
    return x
    # return x if x > 0 else 0


class LinearUnit(Perceptron):
    def __init__(self, input_num):
        '''初始化线性单元，设置输入参数的个数'''
        Perceptron.__init__(self, input_num, activator_fun=active_function)


def get_training_dataset():
    # 构建训练数据
    import csv
    with open("./data/train_age.csv", "r", newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        content = [line for line in csv_reader]
        print(content)
        y = [int(float(line[3])) for line in content]
        for line in content:
            print(line)
        x = []
        for line in content:
            line.pop(3)
            try:
                line[-1] = float(line[-1])
            finally:
                print(line, line[-1])
            x.append([int(n) for n in line])
    input_vecs, labels = x, y
    return input_vecs, labels


def train_linear_unit():
    '''
    使用数据训练线性单元
    '''
    # 创建感知器，输入参数的特征数为6（不算年龄）
    lu = LinearUnit(6)
    # 训练，迭代10轮, 学习速率为0.01
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 100, 0.01)
    # 返回训练好的线性单元
    return lu


if __name__ == '__main__':
    '''训练线性单元'''

    # data = pd.read_csv('iris.data', header=None)
    # take [0, 1] columns of data to be x, take 4 column and do categorical codes as y.
    # y now is an array composed by 0 (stands for Iris-setosa), 1 (stands for Iris-versicolor), and
    # 2 (stands for Iris-virginica)
    # x, y = data[[0, 1, 2, 3]], pd.Categorical(data[4]).codes
    # pd.to_list

    # print(x, y)
    linear_unit = train_linear_unit()
    print("predict")
    print(linear_unit.predict([0, 1, 1, 1, 0, 82.1708]))

    # 打印训练获得的权重
