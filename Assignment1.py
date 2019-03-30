import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import matplotlib.pyplot as plt
import math

# function 1: load the batch
# return X: contains  the  image  pixel  data
#        Y: contains the one-hot representation of the label for each image
#        y: contains the label for each image
def load_batch(file): 
    with open(file, 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
    
    X = batch['data'].transpose() / 255
    
    y = batch['labels']
    
    Y = np.zeros((len(y), 10))
    for idx, val in enumerate(y):
        Y[idx][val] = 1
    
    return X, Y.transpose(), y

# function 2: initialize model W and b with random gaussian values
# with zero mean and standard deviation of 0.01
def init_model_W_b(W_first, W_second, b_first, b_second):
    zero_mean = 0
    std_dev = 0.01
    np.random.seed(0)
    W = np.random.normal(zero_mean, std_dev, size = (W_first, W_second))
    b = np.random.normal(zero_mean, std_dev, size = (b_first, b_second))
    return W, b

# evaluate the network
def evaluate_classifier(X, W, b):
    # formula s = WX + b
    s = np.matmul(W, X)
    s = np.add(s, b)
    
    # p = softmax(s), softmax function exp(s) / 1^T exp(s)
    exp = [np.exp(i) for i in s]
    one_trans_exp = np.matmul(np.ones((1, len(exp))), exp)
    softmax = np.divide(exp, one_trans_exp)

    return softmax

#Visualizing CIFAR 10
def view(X):
    X = X.transpose().reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    fig, axes1 = plt.subplots(5,5,figsize=(3,3))
    for j in range(5):
        for k in range(5):
            i = np.random.choice(range(len(X)))
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(X[i:i+1][0])

# computes the cost function for a set of images
def compute_cost(X, Y, W, b, lmbda):
    one_over_data_magnitude = 1 / len(X[0])
    p = evaluate_classifier(X, W, b)
    Y_T_p = np.matmul(Y.transpose(), p)
    l_cross = np.log(Y_T_p) * -1
    l_cross_sum = l_cross.sum() 
    
    W_squared = W ** 2
    W_squared_sum = W_squared.sum()
    
    cost = (one_over_data_magnitude * l_cross_sum) + (lmbda * W_squared_sum)

    return cost

# computes the accuracy of the network’s
# predictions on a set of data
def compute_accuracy(X, y, W, b):
    p = evaluate_classifier(X, W, b)
    p_trans = p.transpose()
    
    arg_max = []
    for i in p_trans:
        arg_max.append(np.argmax(i))
        
    pred_corr = 0
    for i in range(len(arg_max)):
        if arg_max[i] == y[i]:
            pred_corr += 1
    
    accuracy = pred_corr / len(arg_max)
    
    return accuracy

# computes gradient descent based on 
# last slide of lecture 3
def compute_gradients(X, Y, P, W, lmbda):
    G_batch = np.subtract(Y, P) * -1
    L_w_r_t_W = 1 / len(Y[0]) * G_batch.dot(X.transpose())
    L_w_r_t_b = 1 / len(Y[0]) * G_batch.dot(np.ones((len(Y[0]), 1)))
    grad_W = L_w_r_t_W + (2 * lmbda * W)
    grad_b = L_w_r_t_b
    
    return grad_W, grad_b

def main():
    batch_1_X, batch_1_Y, batch_1_y = load_batch('Datasets/cifar-10-batches-py/data_batch_1')
    batch_2_X, batch_2_Y, batch_2_y = load_batch('Datasets/cifar-10-batches-py/data_batch_2')
    test_batch_X, test_batch_Y, test_batch_y = load_batch('Datasets/cifar-10-batches-py/test_batch')
    print(batch_1_X)

    k = 10 # number of lables
    d = 3072 # dimentionality of each image 32x32x3 = 3072
    model_W, model_b = init_model_W_b(k, d, k, 1)

    p_batch_1 = evaluate_classifier(batch_1_X, model_W, model_b)
    p_batch_2 = evaluate_classifier(batch_2_X, model_W, model_b)
    p_test_batch = evaluate_classifier(test_batch_X, model_W, model_b)
    print("Probability for batch 1: ")
    print(p_batch_1)
    print(p_batch_1.shape)
    print((p_batch_1.transpose())[0].sum())

    J = compute_cost(batch_1_X, batch_1_Y, model_W, model_b, 1)

    acc = compute_accuracy(batch_1_X, batch_1_y, model_W, model_b)

    grad_W, grad_b = compute_gradients(batch_1_X, batch_1_Y, p_batch_1, model_W, 1)
    print(grad_W.shape)
    print(grad_b.shape)

    #print(ComputeGradsNum(batch_1_X, batch_1_Y, model_W, model_b, 0))


if __name__ == '__main__':
    main()