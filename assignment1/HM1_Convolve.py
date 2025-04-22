import numpy as np
from utils import read_img, write_img

def padding(img, padding_size, type):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """
    padding_img = np.zeros((img.shape[0]+2*padding_size, img.shape[1]+2*padding_size))
    if type=="zeroPadding":
        padding_img[padding_size:padding_size+img.shape[0], padding_size:padding_size+img.shape[1]] = img
        return padding_img
    
    elif type=="replicatePadding":
        padding_img[padding_size:padding_size+img.shape[0], padding_size:padding_size+img.shape[1]] = img
        padding_img[:padding_size,:padding_size] = img[0,0]
        padding_img[img.shape[0]+padding_size:,:padding_size] = img[-1,0]
        padding_img[:padding_size,img.shape[1]+padding_size:] = img[0,-1]
        padding_img[img.shape[0]+padding_size:,img.shape[1]+padding_size:] = img[-1,-1]
        
        padding_img[:padding_size,padding_size:img.shape[1]+padding_size] = np.tile(img[0,:],(padding_size,1))
        padding_img[img.shape[0]+padding_size:,padding_size:img.shape[1]+padding_size] = np.tile(img[-1,:],(padding_size,1))
        padding_img[padding_size:img.shape[0]+padding_size,:padding_size] = np.tile(img[:,0],(padding_size,1)).T
        padding_img[padding_size:img.shape[0]+padding_size,img.shape[1]+padding_size:] = np.tile(img[:,-1],(padding_size,1)).T
        return padding_img


def convol_with_Toeplitz_matrix(img, kernel):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """
    #zero padding
    # img = np.ones([2,2])
    # kernel = np.arange(3*3).reshape([3,3])+1
    
    org_shape = img.shape
    # img = padding(img, kernel.shape[0]//2,"zeroPadding")
    shift_x = img.shape[0] - kernel.shape[0] + 1
    shift_y = img.shape[1] - kernel.shape[1] + 1
    
    pos1 = np.zeros(img.shape)
    pos1[:kernel.shape[0],:kernel.shape[1]] = kernel

    x = np.arange(img.size)
    y = np.arange(img.size)
    y = np.where(np.logical_and(y // np.shape(img)[1] < shift_y, y % np.shape(img)[1] < shift_y))
    idx_x, idx_y = np.meshgrid(x, y)
    idx = (idx_x - idx_y) % img.size
    output = pos1.reshape(-1)[idx] @ img.reshape(-1)
    return output.reshape(org_shape)


def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float)
        Outputs:
            output: array(float)
    """
    # img = padding(img, kernel.shape[0]//2,"zeroPadding")
    shift_x = img.shape[0] - kernel.shape[0] + 1
    shift_y = img.shape[1] - kernel.shape[1] + 1
    #build the sliding-window convolution here
    pos1 = np.arange(img.size).reshape(img.shape)[:kernel.shape[0],:kernel.shape[1]].reshape(-1)-1
    pos2 = np.arange(img.size).reshape(img.shape)[:shift_x,:shift_y].reshape(-1)
    pos_x, pos_y = np.meshgrid(pos1, pos2)
    idx = (pos_x + pos_y + 1) % img.size
    output = (img.reshape(-1)[idx]) @ kernel.reshape(-1)
    return output.reshape([shift_x,shift_y])


def Gaussian_filter(img):
    padding_img = padding(img, 1, "replicatePadding")
    gaussian_kernel = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
    output = convolve(padding_img, gaussian_kernel)
    return output

def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output

def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output



if __name__=="__main__":

    np.random.seed(111)
    input_array=np.random.rand(6,6)
    input_kernel=np.random.rand(3,3)

    # task1: padding
    zero_pad =  padding(input_array,1,"zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt",zero_pad)

    replicate_pad = padding(input_array,1,"replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt",replicate_pad)


    #task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    #task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)
    print("finish task3")
    #task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("lenna.png")/255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x*255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y*255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur*255)




    