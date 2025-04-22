import numpy as np
from utils import  read_img, draw_corner, write_img
from HM1_Convolve import convolve, Sobel_filter_x,Sobel_filter_y,padding



def corner_response_function(input_img, window_size, alpha, threshold):
    """
        The function you need to implement for Q3.
        Inputs:
            input_img: array(float)
            window_size: int
            alpha: float
            threshold: float
        Outputs:
            corner_list: array
    """
    # padding_image = padding(input_img, window_size//2, "zeroPadding")
    I_x = Sobel_filter_x(input_img)
    I_y = Sobel_filter_y(input_img)

    
    I_xx = I_x**2
    I_yy = I_y**2
    I_xy = I_x*I_y
    
    window = np.ones((window_size,window_size))/(window_size**2)
    I_xx_conv = convolve(I_xx, window)
    I_yy_conv = convolve(I_yy, window)
    I_xy_conv = convolve(I_xy, window)
    det_M = I_xx_conv*I_yy_conv - I_xy_conv**2
    
    trace_M = I_xx_conv + I_yy_conv
    
    r = det_M - alpha*trace_M**2
    r = r > threshold

    
    corner_id = np.argwhere(r)
    corner = r.reshape(-1, 1)
    corner_list = np.concatenate((corner_id, corner), axis = 1)
    # please solve the corner_response_function of each window,
    # and keep windows with theta > threshold.
    # you can use several functions from HM1_Convolve to get 
    # I_xx, I_yy, I_xy as well as the convolution result.
    # for detials of corner_response_function, please refer to the slides.


    return corner_list # array, each row contains information about one corner, namely (index of row, index of col, theta)



if __name__=="__main__":

    #Load the input images
    input_img = read_img("hand_writting.png")/255.

    #you can adjust the parameters to fit your own implementation 
    window_size = 5
    alpha = 0.04
    threshold =0.04

    corner_list = corner_response_function(input_img,window_size,alpha,threshold)

    # NMS
    corner_list_sorted = sorted(corner_list, key = lambda x: x[2], reverse = True)
    NML_selected = [] 
    NML_selected.append(corner_list_sorted[0][:-1])
    dis = 10
    for i in corner_list_sorted :
        for j in NML_selected :
            if(abs(i[0] - j[0]) <= dis and abs(i[1] - j[1]) <= dis) :
                break
        else :
            NML_selected.append(i[:-1])


    #save results
    draw_corner("hand_writting.png", "result/HM1_HarrisCorner.png", NML_selected)
