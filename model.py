import numpy as np
import matplotlib.pyplot as plt
import os

# sigmoid激活函数
def sigmoid(input_x):
    return 1 / (1 + np.exp(-input_x))
# 计算sigmoid函数的gradient值
def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))
# softmax 函数
def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)
# 定义cross_entropy loss函数
def cross_entropy(x, y):
    return np.sum(np.nan_to_num(-y*np.log(x)-(1-y)*np.log(1-x)))

class NN():

    def __init__(self, lr=0.001, num_in=100, num_out=10, hidden=4, weight_scale=1e-3, L2=0.5, epoch=200, lr_schedule='decay'):
        self.lr = lr                         #学习率
        self.num_in = num_in                 #输入层数量
        self.num_out = num_out               #输出层数量
        self.params = {}                     #储存参数字典
        self.hidden = hidden                 #隐藏层节点数
        self.weight_scale = weight_scale     #标准化初始权重
        self.L2 = L2                         #正则化因子
        self.epoch = epoch                   #epoch
        self.lr_schedule = lr_schedule       #学习率策略，包括<'multi_step'> <'decay'>

        self.loss = []
        self.acc = []

        self.flag_init_weight = False
        if self.flag_init_weight == False:
            self.init_weights()

    def init_weights(self):
        """初始化权重"""
        assert self.flag_init_weight == False
        self.params['W1'] = np.random.randn(self.num_in, self.hidden) * self.weight_scale
        self.params['W2'] = np.random.randn(self.hidden, self.num_out) * self.weight_scale
        self.params['b1'] = np.zeros(self.hidden, )
        self.params['b2'] = np.zeros(self.num_out, )
        self.flag_init_weight = True

    def loss_softmax(self, x, y):

        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        N = x.shape[0]
        loss = -np.sum(log_probs[np.arange(N), y]) / N
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        return loss, dx

    def train(self, input, y):
        # 训练模型
        for i in range(self.epoch):
            # 前向传播
            hidden1_ = np.dot(input, self.params['W1']) + self.params['b1']
            hidden1 = sigmoid(hidden1_)

            output = np.dot(hidden1, self.params['W2']) + self.params['b2']

            # 计算L2正则化损失
            loss, dout = self.loss_softmax(output, y)
            loss += 0.5 * self.L2 * (
                np.sum(np.square(self.params['W2'])) + np.sum(
                np.square(self.params['W1'])))

            self.loss.append(loss)

            # 计算准确率
            y_pred = np.argmax(output, axis=1).reshape(1, -1)
            y_true = y.reshape(1, -1)
            sum_ = 0.0
            for c in range(y_pred.shape[1]):
                if y_pred[0, c] == y_true[0, c]:
                    sum_ = sum_ + 1
                    acc = 100.0 * sum_ / y_pred.shape[1]
                    self.acc.append(acc)

            if i % 10 == 0:
                print('Epochs {} -- Acc: [{:.3f}%], Loss: [{:.5f}]'.format(i, 100.0 * sum_ / y_pred.shape[1], loss))

            # 计算梯度
            #dout = dout * (1 - output) * output

            dW2 = np.dot(hidden1.T, dout)
            db2 = np.sum(dout, axis=0)
            dhidden1 = np.dot(dout, self.params['W2'].T) * (1 - hidden1) * hidden1

            dW1 = np.dot(input.T, dhidden1)  # 2*4 and 4*2 => 2*2
            db1 = np.sum(dhidden1, axis=0)  # 1 * 2

            # L2正则化
            dW2 += self.params['W2'] * self.L2
            dW1 += self.params['W1'] * self.L2

            # 反向传播
            self.params['W2'] -= self.lr * dW2
            self.params['b2'] -= self.lr * db2
            self.params['W1'] -= self.lr * dW1
            self.params['b1'] -= self.lr * db1
            #学习率策略
            if self.lr_schedule == 'decay':
                self.lr = self.lr * 0.999
            elif self.lr_schedule == 'multi_step':
                if i < 500:
                    self.lr = self.lr
                elif i < 1000 and i >= 500:
                    self.lr -= self.lr * 0.6
                else:
                    self.lr = self.lr * 0.1

                if self.lr < 0.1:
                    self.lr = 0.1

            if i == self.epoch - 1:
                y_pred = np.argmax(output, axis=1).reshape(1, -1)
                y_true = y.reshape(1, -1)
                sum_ = 0.0
                for c in range(y_pred.shape[1]):
                    if y_pred[0, c] == y_true[0, c]:
                        sum_ = sum_ + 1
                print('Epochs {} -- Acc: [{:.3f}%], Loss: [{:.5f}]'.format(i, 100.0 * sum_ / y_pred.shape[1], loss))

    def test(self, input, y):
        # 测试模型
        hidden1_ = np.dot(input, self.params['W1']) + self.params['b1']
        hidden1 = sigmoid(hidden1_)

        hidden2_ = np.dot(hidden1, self.params['W2']) + self.params['b2']
        output = sigmoid(hidden2_)


        # 计算准确率
        y_pred = np.argmax(output, axis=1).reshape(1, -1)
        y_true = y.reshape(1, -1)
        sum_ = 0.0
        for c in range(y_pred.shape[1]):
            if y_pred[0, c] == y_true[0, c]:
                sum_ = sum_ + 1
        print('Test acc is {:.5f}'.format(sum_ / y_pred.shape[1]))
        return sum_ / y_pred.shape[1]

    def get_loss_history(self):
        return self.loss

    def get_acc_history(self):
        return self.acc

    def save(self, filename):
        np.savez_compressed(
            file=os.path.join(os.curdir, 'models', filename),
            params=self.params
        )
