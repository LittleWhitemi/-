# -*- encoding: utf-8 -*-
"""
Filename         :handwrite.py
Description      :利用2层全连接网络对mnist数据集进行分类
Time             :2022/03/27
Author           :杨映光
Version          :2.0
"""
import numpy
import scipy.special
from tqdm import tqdm


# 类定义
class Neuralnetwork:
    # 初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # 初始权重值：正态随机：正态概率分布采样权重，其中平均值为0，标准方差为节点传入链接数目的开方
        self.wih = numpy.random.normal(0.0, pow(self.hnodes,
                                                -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes,
                                                -0.5), (self.onodes, self.hnodes))

        # 学习率
        self.lr = learningrate

        # 激活函数S函数
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # 训练神经网络
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # 输出层的敏感度，即输出层输出的误差*输出层输出的导数： (target - actual)*final_outputs * (1.0 - final_outputs)
        output_errors = (targets - final_outputs) * final_outputs * (1.0 - final_outputs)
        # 隐含层层的敏感度，即输出层输出的误差*输出层输出的导数： (target - actual)*final_outputs * (1.0 - final_outputs)
        hidden_errors = numpy.dot(self.who.T, output_errors) * hidden_outputs * (1.0 - hidden_outputs)
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot(output_errors, numpy.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot(hidden_errors, numpy.transpose(inputs))

        pass

    # 查询神经网络
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        # 计算隐含层输入信号: I_hidden = W_in_hidden .*input
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 利用激活函数计算隐含层输出信号: O_hidden = f(I_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层的输入信号: I_out = W_hidden_out .*O_hidden
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 利用激活函数计算隐含层输出信号: O_output = f(I_out)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        pass


# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
# 学习率 0.1
learning_rate = 0.2
# create instance of neural network
n = Neuralnetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 添加训练文件到列表list
training_data_file = open(r"C:\Users\Veteran\Desktop\ipython\作业：手写数字识别\Handwriting_dates/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# 开始训练神经网络
# 设置训练轮数：5
epochs = 5
for e in range(epochs):
    print('Epoch %d/%d' % (e, epochs))
    # 在训练数据集中遍历所有的recode
    for recode in tqdm(training_data_list):
        # split(','):将数据根据逗号‘，’拆分
        all_values = recode.split(',')
        # 设置输入数据
        inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        # 创建期望输出
        targets = numpy.ones(output_nodes) * 0.01
        # 设置第一个标签输出 all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# # 保存参数
parameters = {"w_ih": n.wih,
              "w_ho": n.who}

numpy.save('parameters.npy', parameters)

# 加载参数
parameters = numpy.load('parameters.npy', allow_pickle='TRUE')
n.wih = parameters.item()['w_ih']
n.who = parameters.item()['w_ho']

# 测试神经网络
# 添加测试文件到list
test_data_file = open(r"C:\Users\Veteran\Desktop\ipython\作业：手写数字识别\Handwriting_dates/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# 对训练结束的神经网络经行测试，并用scorecard经行记分，即正确的次数
scorecard = []

# 遍历整个测试样本
for recode in test_data_list:
    # split(','):将数据根据逗号‘，’拆分
    all_values = recode.split(',')
    # 设置正确的标签并经行显示
    correct_label = int(all_values[0])
    print(correct_label, "期望值")
    # 设置输入数据
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    # 查询神经网络输出
    outputs = n.query(inputs)
    # 确定输出标签
    label = numpy.argmax(outputs)
    print(label, "实际网络输出值")
    # 进行计分； append()函数：在列表ls最后(末尾)添加一个元素object
    if correct_label == label:
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

# 计算当前正确率
scorecard_array = numpy.asfarray(scorecard)
print("准确率为", scorecard_array.sum() / scorecard_array.size * 100, '%')
