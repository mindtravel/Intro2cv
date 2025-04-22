import numpy as np
from utils import draw_save_plane_with_points, normalize, test_save_plane_with_points


if __name__ == "__main__":


    np.random.seed(0)
    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    # to simplify this problem, we provide the number of inliers and outliers here

    noise_points = np.loadtxt("HM1_ransac_points.txt")


    #RANSAC
    # Please formulate the palnace function as:  A*x+B*y+C*z+D=0     

    sample_time = 12 # the minimal time that can guarantee the probability of at least one hypothesis does not contain any outliers is larger than 99.9%
    distance_threshold = 0.1


    # sample points group
    sample_id = np.random.choice(130, sample_time * 3, replace=False)
    sampled_points = noise_points[sample_id].reshape(sample_time, 3, 3)
    # print(sampled_points)
    line = (np.linalg.inv(sampled_points) @ np.array([-1, -1, -1])).reshape(-1, 1, 3)
    points = noise_points.reshape(1, -1, 3)
    dist = (np.sum(line * points, axis=-1) + 1)/np.sqrt(np.sum(line[0]**2, axis=-1))
#     print(dist.shape)
    dist = np.abs(np.sum(line * points, axis=-1) + 1)/np.sqrt(np.sum(line[0]**2))
    n_inliners = np.sum(dist < distance_threshold, axis=1)
    best_line_id = np.argmax(n_inliners)
    # noise_points = noise_points.reshape(1, -1, 3)
#     print(best_line_id)
    # print(noise_points.shape)
    
    
    # estimate the plane with sampled points group
    best_inliners = noise_points[dist[best_line_id] < distance_threshold]
    least_square_plane = np.linalg.inv(best_inliners.T @ best_inliners) @ best_inliners.T @ -np.ones((best_inliners.shape[0]))



    # minimize the sum of squared perpendicular distances of all inliers with least-squared method 
#     print(least_square_plane.shape)
    pf = [least_square_plane[0],
          least_square_plane[1],
          least_square_plane[2],1]
    # draw the estimated plane with points and save the results 
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0  
    pf = normalize(pf)
    draw_save_plane_with_points(pf, noise_points,"result/HM1_RANSAC_fig.png") 
    # test_save_plane_with_points(pf, noise_points, noise_points[0:1],"result/HM1_RANSAC_fig.png") 
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)
    np.savetxt('result/HM1_RANSAC_sample_time.txt', np.array([sample_time]))
 