import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


# def initialize_parameters(n_x, n_h, n_y):
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


def initialize_parameters(n_x, n_h, n_y):

    np.random.seed(2)   # 产生的随机数既不会改变了，就是每次重新运行都会产生同一个随机数
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    # 用断言确保数据格式的正确
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}


    return parameters

# 测试initialize_parameters
# print("=========================测试initialize_parameters=========================")
# n_x, n_h, n_y = initialize_parameters_test_case()
# parameters = initialize_parameters(n_x, n_h, n_y)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    # 前向传播法计算
    # 第一层到第二层
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    # 第二层到第五层
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    # 使用断言确保数据格式正确
    assert(A2.shape == (1,X.shape[1]))
    cache = {"Z1": Z1,
              "A1": A1,
              "Z2": Z2,
              "A2": A2}

    return (A2, cache)

# 测试forward_propagation
# print("=========================测试forward_propagation=========================")
# X_assess, parameters = forward_propagation_test_case()
# A2, cache = forward_propagation(X_assess, parameters)
# print(np.mean(cache["Z1"]), np.mean(cache["A1"]), np.mean(cache["Z2"]), np.mean(cache["A2"]))


def cost_function(A2, Y, parameters):
    logprobs = np.multiply(np.log(A2), Y)
    cost = - np.sum(logprobs)

    return cost


# 测试compute_cost
# print("=========================测试compute_cost=========================")
# A2 , Y_assess , parameters = compute_cost_test_case()
# print("cost = " + str(cost_function(A2, Y_assess, parameters)))

# 反向传播算法
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[0]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2-Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads
# #测试backward_propagation
# print("=========================测试backward_propagation=========================")
# parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
#
# grads = backward_propagation(parameters, cache, X_assess, Y_assess)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("db2 = "+ str(grads["db2"]))


def update_parameters(parameters, grads):
    learning_rate = 0.01
    W1 = parameters["W1"]
    dW1 = grads["dW1"]
    W1 = W1 - learning_rate*dW1
    b1 = parameters["b1"]
    db1 = grads["db1"]
    b1 = b1 - learning_rate*db1
    W2 = parameters["W2"]
    dW2 = grads["dW2"]
    W2 = W2 - learning_rate * dW2
    b2 = parameters["b2"]
    db2 = grads["db2"]
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


# # 测试update_parameters
# print("=========================测试update_parameters=========================")
# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


def mystudy(X, Y, number, n_h, print_cost=False):

    n_x = X.shape[0]  # 只输出列数就写1，输出行数就写0
    n_y = Y.shape[0]

    # 初始化
    # n_x, n_h, n_y = initialize_parameters_test_case()
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    for i in range(number):
        # 前向传播
        A2 , cache = forward_propagation(X, parameters)
        # 计算误差
        cost = cost_function(A2, Y, parameters)
        # 反向传播
        grads = backward_propagation(parameters, cache, X, Y)
        # 更新参数
        parameters = update_parameters(parameters, grads)

        if print_cost:
            if i % 1000 == 0:
                print('第', i, '循环, 成本为：', str(cost))
    return parameters


# #测试nn_model
# print("=========================测试my_study=========================")
# X_assess, Y_assess = nn_model_test_case()
#
# parameters = mystudy(X_assess, Y_assess,  number=10000, print_cost=True)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
def predict(parameters, X):

    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)

    return predictions


X, Y = load_planar_dataset()
parameters = mystudy(X, Y, n_h=8, number=10000, print_cost=True) # 更换节点数量效果明显
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
predictions = predict(parameters, X)
print ('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
plt.show()