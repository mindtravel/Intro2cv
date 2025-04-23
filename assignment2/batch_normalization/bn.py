import numpy as np
import cv2
import os
import sys

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BP import sigmoid

# eps may help you to deal with numerical problem
eps = 1e-5
def bn_forward_test(x, gamma, beta, mean, var):

    #----------------TODO------------------
    # Implement forward 
    #----------------TODO------------------
    x_hat = (x - mean) / (np.sqrt(var) + eps)
    out = gamma * x_hat + beta
    return out

def bn_forward_train(x, gamma, beta):

    #----------------TODO------------------
    # Implement forward
    #----------------TODO------------------
    # print(x.shape)
    sample_mean = np.mean(x, axis=0)
    sample_var = np.var(x, axis=0)
    x_hat = (x - sample_mean) / (np.sqrt(sample_var) + eps)
    out = gamma * x_hat + beta
        
    # save intermidiate variables for computing the gradient when backward
    cache = (gamma, x, sample_mean, sample_var, x_hat)
    return out, cache
    
def bn_backward(dout, cache):
    # print(dout.shape)

    #----------------TODO------------------
    # Implement backward 
    #----------------TODO------------------
    gamma, x, mu, var, x_hat = cache
    batchsize = x.shape[0]

    dgamma = (dout * x_hat).sum()
    dbeta = dout.sum()
    dx_hat = dout * gamma
    # d_mean = d_x_hat
    # d_var
    # dx = d_x_hat * (eps + np.sqrt(sample_var))
    dvar = np.sum(dx_hat * (x - mu) * -0.5 * (var + eps)**(-1.5), axis=0)
    
    # 计算关于均值的梯度
    dmu = np.sum(dx_hat * -1/np.sqrt(var + eps), axis=0) + dvar * np.mean(-2 * (x - mu), axis=0)
    
    # 计算关于输入x的梯度
    dx = dx_hat / np.sqrt(var + eps) + dvar * 2 * (x - mu) / batchsize + dmu / batchsize
    
    
    return dx, dgamma, dbeta

# This function may help you to check your code
def print_info(x):
    print('mean:', np.mean(x,axis=0))
    print('var:',np.var(x,axis=0))
    print('------------------')
    return 

if __name__ == "__main__":
    HW_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # input data
    train_data = np.zeros((9,784)) 
    for i in range(9):
        train_data[i,:] = cv2.imread(os.path.join(HW_dir, "mnist_subset", f"{i}.png"), cv2.IMREAD_GRAYSCALE).reshape(-1)/255.
    gt_y = np.zeros((9,1)) 
    gt_y[0] =1  

    val_data = np.zeros((1,784)) 
    val_data[0,:] = cv2.imread(os.path.join(HW_dir, "mnist_subset", "9.png"), cv2.IMREAD_GRAYSCALE).reshape(-1)/255.
    val_gt = np.zeros((1,1)) 

    np.random.seed(14)

    # Intialize MLP  (784 -> 16 -> 1)
    MLP_layer_1 = np.random.randn(784,16)
    MLP_layer_2 = np.random.randn(16,1)

    # Initialize gamma and beta
    gamma = np.random.randn(16)
    beta = np.random.randn(16)

    lr=1e-1
    loss_list=[]

    # ---------------- TODO -------------------
    # compute mean and var for testing
    # add codes anywhere as you need
    # ---------------- TODO -------------------
    mean = 0.
    var = 0.

    # training 
    for i in range(50):
        # Forward
        output_layer_1 = train_data.dot(MLP_layer_1)
        output_layer_1_bn, cache = bn_forward_train(output_layer_1, gamma, beta)
        # update runtime mean and var from cache 
        mean = cache[2]
        var = cache[3]
        
        output_layer_1_act = sigmoid(output_layer_1_bn)  #sigmoid activation function
        # debug: activate this to test without batchnorm
        # output_layer_1_act = sigmoid(output_layer_1)  #sigmoid activation function
        output_layer_2 = output_layer_1_act.dot(MLP_layer_2)
        pred_y = sigmoid(output_layer_2)  #sigmoid activation function

        # compute loss 
        loss = -( gt_y * np.log(pred_y) + (1-gt_y) * np.log(1-pred_y)).sum()
        print("iteration: %d, loss: %f" % (i+1 ,loss))
        loss_list.append(loss)

        # Backward : compute the gradient of paratmerters of layer1 (grad_layer_1) and layer2 (grad_layer_2)
        grad_pred_y = -(gt_y/pred_y) + (1-gt_y)/(1-pred_y)
        grad_activation_func = grad_pred_y * pred_y * (1-pred_y) 
        grad_layer_2 = output_layer_1_act.T.dot(grad_activation_func)
        grad_output_layer_1_act = grad_activation_func.dot(MLP_layer_2.T)
        
        
        # debug: activate this to test without batchnorm
        # grad_output_layer_1  = grad_output_layer_1_act * (1-output_layer_1_act) * output_layer_1_act
        grad_output_layer_1_bn  = grad_output_layer_1_act * (1-output_layer_1_act) * output_layer_1_act
        grad_output_layer_1, grad_gamma, grad_beta = bn_backward(grad_output_layer_1_bn, cache)
        grad_layer_1 = train_data.T.dot(grad_output_layer_1)

        # update parameters
        gamma -= lr * grad_gamma
        beta -= lr * grad_beta
        MLP_layer_1 -= lr * grad_layer_1
        MLP_layer_2 -= lr * grad_layer_2
    
    # validate
    output_layer_1 = val_data.dot(MLP_layer_1)
    output_layer_1_bn = bn_forward_test(output_layer_1, gamma, beta, mean, var)
    output_layer_1_act = sigmoid(output_layer_1_bn) #sigmoid activation function
    output_layer_2 = output_layer_1_act.dot(MLP_layer_2)
    pred_y = 1 / (1+np.exp(-output_layer_2))  #sigmoid activation function
    loss = -( val_gt * np.log(pred_y) + (1-val_gt) * np.log(1-pred_y)).sum()
    print("validation loss: %f" % (loss))
    loss_list.append(loss)

    os.makedirs(os.path.join(os.path.join(HW_dir), "results"), exist_ok=True)
    np.savetxt(os.path.join(os.path.join(HW_dir), "results", "bn_loss.txt"), loss_list)