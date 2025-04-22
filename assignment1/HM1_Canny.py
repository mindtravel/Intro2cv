import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y, convolve, padding
from utils import read_img, write_img

def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float) 
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """
    eps = 1e-10
    magnitude_grad = np.sqrt(x_grad**2 + y_grad**2)
    direction_grad = np.arctan2(y_grad + eps, x_grad + eps)
    return magnitude_grad, direction_grad 

def non_maximal_suppressor(grad_mag, grad_dir):
    """
        The function you need to implement for Q2 b).
        Inputs:
            grad_mag: array(float) 
            grad_dir: array(float)
        Outputs:
            output: array(float)
    """   
    
    def bilinear_interpolation(y, x):
        """
            Given (y, x), return the gradient of the pixel
            using bilinear interpolation.
        """
        y1 = np.floor(y).astype(int)
        y2 = y1 + 1
        x1 = np.floor(x).astype(int)
        x2 = x1 + 1

        y1 = np.clip(y1, 0, grad_mag.shape[0]-1)
        y2 = np.clip(y2, 0, grad_mag.shape[0]-1)
        x1 = np.clip(x1, 0, grad_mag.shape[1]-1)
        x2 = np.clip(x2, 0, grad_mag.shape[1]-1)
        
        w11 = (y2 - y) * (x2 - x) 
        w12 = (y2 - y) * (x - x1)
        w21 = (y - y1) * (x2 - x)
        w22 = (y - y1) * (x - x1)

        return w11 * grad_mag[y1, x1] + w12 * grad_mag[y1, x2] + w21 * grad_mag[y2, x1] + w22 * grad_mag[y2, x2]


    grad_y, grad_x = grad_mag * np.sin(grad_dir), grad_mag * np.cos(grad_dir) 
    x, y = np.meshgrid(np.arange(grad_mag.shape[0]), np.arange(grad_mag.shape[1]))
    neighbor1_x, neighbor1_y = x + grad_x, y + grad_y
    neighbor2_x, neighbor2_y = x - grad_x, y - grad_y
    neighbor1 = bilinear_interpolation(neighbor1_y, neighbor1_x)
    neighbor2 = bilinear_interpolation(neighbor2_y, neighbor2_x)
    NMS_output = grad_mag * (grad_mag >= neighbor1) * (grad_mag >= neighbor2)
    
    return NMS_output            


def hysteresis_thresholding(img) :
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float) 
        Outputs:
            output: array(float)
    """


    #you can adjust the parameters to fit your own implementation 
    low_ratio = 0.10
    high_ratio = 0.30
    
    high_mask = img >= high_ratio
    low_mask = img >= low_ratio
    mask = high_mask
    i = 0
    while i < 20:
        i+=1
        mask = padding(mask, 1, "zeroPadding")
        mask = (convolve(mask, np.ones((3,3))) > 0) & low_mask
    return mask
    return low_mask * img
    return high_mask * img



if __name__=="__main__":

    #Load the input images
    input_img = read_img("Lenna.png")/255

    #Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)

    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)

    #Compute the magnitude and the direction of gradient
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(x_grad, y_grad)
    # write_img("result/Mymagnitude_grad.png", magnitude_grad*255)
    # write_img("result/Mydirection_grad.png", direction_grad*255)

    #NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)
    # write_img("result/MyNMS.png", NMS_output*255)

    #Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output)
    
    write_img("result/HM1_Canny_result.png", output_img*255)
