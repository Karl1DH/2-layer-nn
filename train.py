import numpy as np
import csv
from model import NN
import matplotlib.pyplot as plt
#读取数据集
train = csv.reader(open('mnist_train.csv', 'r'))
train_content = []
for line in train:
    train_content.append(line)
test = csv.reader(open('mnist_test.csv', 'r'))
test_content = []
for line in test:
    test_content.append(line)
train_content = np.array(train_content, dtype=np.float32)
test_content = np.array(test_content, dtype=np.float32)

train_label = np.array(train_content[:, 0], dtype=np.int)
train_x = train_content[:,1 :]
test_label = np.array(test_content[:, 0], dtype=np.int)
test_x = test_content[:, 1:]

assert train_x.shape[1] == test_x.shape[1]
print('Number of input is %d' % train_x.shape[1])
num_input = train_x.shape[1]
train_x = (train_x - 255/2) / 255
test_x = (test_x - 255/2) / 255


hidden = [20,30,40]
learning_rates = [0.5,0.3,0.1]
L2 = [0, 1e-2, 1e-5]
best_ = {'accuracy': 0}

for h in hidden:
    for lr in learning_rates:
        for L in L2:
            print(f"Current layer: {h}, Current learning rate: {lr}, Current L2 penalty: {L}")
            model = NN(lr=lr, num_in=784, num_out=10, hidden=h, weight_scale=0.1, L2=L, epoch=200,
                     lr_schedule='decay')
            model.train(train_x, train_label)  # [0:1000]
            accuracy=model.test(test_x, test_label)
            if accuracy > best_['accuracy']:
                best_['accuracy'] = accuracy
                best_['layer'] = h
                best_['learning_rate'] = lr
                best_['L2'] = L
print(best_)

#可视化
net = NN(lr=0.5, num_in=784, num_out=10, hidden=40, weight_scale=0.1, L2=0, epoch=200,
                     lr_schedule='decay')
net.train(train_x, train_label)  # [0:1000]
net.test(test_x, test_label)
net.save("model")
#损失
loss = net.get_loss_history()
plt.plot(loss)
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig("figs/Loss Curve.png")
plt.close()

#准确率
acc = net.get_acc_history()
plt.plot(acc)
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig("figs/Accuracy Curve.png")
plt.close()

# 可视化每层的网络参数
layer1_weights = net.params["W1"].flatten().tolist()
plt.hist(layer1_weights, bins=100)
plt.title("layer1 weights")
plt.xlabel("value")
plt.ylabel("frequency")
plt.savefig("figs/W1.png")
plt.close()

layer2_weights = net.params["W2"].flatten().tolist()
plt.hist(layer2_weights, bins=30)
plt.title("layer2 weights")
plt.xlabel("value")
plt.ylabel("frequency")
plt.savefig("figs/W2.png")
plt.close()

layer1_biases = net.params["b1"].flatten().tolist()
plt.hist(layer1_biases, bins=10)
plt.title("layer1 biases")
plt.xlabel("value")
plt.ylabel("frequency")
plt.savefig("figs/b1.png")
plt.close()

layer2_biases = net.params["b2"].flatten().tolist()
plt.hist(layer2_biases, bins=10)
plt.title("layer2 biases")
plt.xlabel("value")
plt.ylabel("frequency")
plt.savefig("figs/b2.png")
plt.close()