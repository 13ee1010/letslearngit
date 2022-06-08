#
import pickle
import numpy as np
import matplotlib.pyplot as plt


def read_pickle(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        # --
    return data

'''
def get_neighbour_value(index,k,arr_b):
    
    if(index[k] >= len(arr_b) or index[k]<0):
        return 0 
    else:
        return arr_b[index[k]]
'''

def get_value(arr,i):
    if(i >=len(arr) or i< 0):
        return 0
    else:
        return arr[i]

def normalised(y_data, z_data, n_data):

    """ Normalise the y_data"""
    norm_y = np.linalg.norm(y_data)
    y_norm = y_data / norm_y
    
    """ Normalise the z_data"""
    norm_z = np.linalg.norm(z_data)
    z_norm = z_data / norm_z

    """ Normalise the n_data"""
    norm_n = np.linalg.norm(n_data)
    n_norm = n_data / norm_n

    return y_norm, z_norm, n_norm

def distance_fun(lens, est):
    y_lense, z_lense, n_lense = lens[:,0], lens[:,1], lens[:,2]
    y_est, z_est, n_est = est[:,0], est[:,1], est[:,2]
    for i in range(len(n_est)):
        if( n_est[i] > 175.0):
            n_est[i] = np.sin(np.deg2rad(n_est[i]))
    y_lense_norm, z_lense_norm, n_lense_norm = normalised(y_lense,z_lense,n_lense)
    y_est_norm, z_est_norm, n_est_norm = normalised(y_est, z_est, n_est)
    bound = len(y_lense)
    bound = int(np.sqrt(bound))

    #index = [0,0,0,0,0,0,0,0,0]
    manhatten_distance = [[0]*len(est) for i in range(len(est))]
    euclidian_distance = [[0]*len(est) for i in range(len(est))]

    for i in range(len(est)):
        upper_right = get_value(n_lense_norm,i+(bound+1))
        upper_centre = get_value(n_lense_norm,i+bound)
        upper_left = get_value(n_lense_norm,i+(bound-1))
        e_left = get_value(n_lense_norm,i-1)
        centre = get_value(n_lense_norm,i)
        e_right = get_value(n_lense_norm,i+1)
        bottom_left = get_value(n_lense_norm,i-(bound+1))
        bottom_centre = get_value(n_lense_norm,i-(bound))
        bottom_right = get_value(n_lense_norm,i-(bound-1))
        if(i%(bound)==0):
            upper_left = 0
            e_left = 0
            bottom_left = 0
        if(i%(bound) == (bound-1)):
            upper_right = 0
            e_right = 0
            bottom_right = 0
        '''
        index[0] = upper_right
        index[1] = upper_centre
        index[2] = upper_left
        index[3] = e_left
        index[4] = i
        index[5] = e_right
        index[6] = bottom_left
        index[7] = bottom_centre
        index[8] = bottom_right
        '''

        k = 0 
        for j in range(len(est)):
             upper_right_est = get_value(n_est_norm,j+(bound+1))
             upper_centre_est = get_value(n_est_norm,j+(bound))
             upper_left_est = get_value(n_est_norm,j+(bound-1))
             e_left_est = get_value(n_est_norm,j-1)
             centre_est = get_value(n_est_norm,j)
             e_right_est = get_value(n_est_norm,j+1)
             bottom_left_est = get_value(n_est_norm,j-(bound+1))
             bottom_centre_est = get_value(n_est_norm,j-(bound))
             bottom_right_est = get_value(n_est_norm,j-(bound-1))

             if(j%(bound)==0):
                 upper_left_est = 0
                 e_left_est = 0
                 bottom_left_est = 0

             if(j%(bound) == (bound-1)):
                 upper_right_est = 0
                 e_right_est = 0
                 bottom_right_est = 0

             manhatten_distance[i][j] = abs(y_lense_norm[i]-y_est_norm[i]) + abs(z_lense_norm[i] - z_est_norm[i]) + abs(upper_right - upper_right_est) + abs(upper_centre - upper_centre_est) + abs(upper_left-upper_left_est) + abs(e_left-e_left_est) + 1*abs(centre-centre_est) + abs(e_right-e_right_est) + abs(bottom_left-bottom_left_est) + abs(bottom_centre-bottom_centre_est) + abs(bottom_right-bottom_right_est)
             euclidian_distance[i][j] = np.sqrt(np.square(y_lense_norm[i]-y_est_norm[i]) + np.square(z_lense_norm[i] - z_est_norm[i]) + np.square(upper_right-upper_right_est) + np.square(upper_centre-upper_centre_est) + np.square(upper_left-upper_left_est) + np.square(e_left-e_left_est) + (np.square(centre-centre_est)) + np.square(e_right-e_left_est) + np.square(bottom_left-bottom_left_est) + np.square(bottom_centre-bottom_centre_est) + np.square(bottom_right-bottom_right_est))


    return manhatten_distance, euclidian_distance   

'''      
def euclidian_distance_fun(lens,est):
    y_lense, z_lense, n_lense = lens[:,0], lens[:,1], lens[:,2]
    y_est, z_est, n_est = est[:,0], est[:,1], est[:,2]
    y_lense_norm, z_lense_norm, n_lense_norm = normalised(y_lense,z_lense,n_lense)
    y_est_norm, z_est_norm, n_est_norm = normalised(y_est, z_est, n_est)
    index = [0,0,0,0,0,0,0,0,0]
    euclidian_distance = [[0]*len(est) for i in range(len(est))]
    
    for i in range(len(est)):
        upper_right = get_value(n_lense_norm,i-4)
        upper_centre = get_value(n_lense_norm,i-3)
        upper_left = get_value(n_lense_norm,i-2)
        e_left = get_value(n_lense_norm,i-1)
        centre = get_value(n_lense_norm,i)
        e_right = get_value(n_lense_norm,i+1)
        bottom_left = get_value(n_lense_norm,i+2)
        bottom_centre = get_value(n_lense_norm,i+3)
        bottom_right = get_value(n_lense_norm,i+4)

        
        index[0] = upper_right
        index[1] = upper_centre
        index[2] = upper_left
        index[3] = e_left
        index[4] = i
        index[5] = e_right
        index[6] = bottom_left
        index[7] = bottom_centre
        index[8] = bottom_right
        

        k = 0 
        for j in range(len(est)):
            upper_right_est = get_value(n_est_norm,j-4)
            upper_centre_est = get_value(n_est_norm,j-3)
            upper_left_est = get_value(n_est_norm,j-2)
            e_left_est = get_value(n_est_norm,j-1)
            centre_est = get_value(n_est_norm,j)
            e_right_est = get_value(n_est_norm,j+1)
            bottom_left_est = get_value(n_est_norm,j+2)
            bottom_centre_est = get_value(n_est_norm,j+3)
            bottom_right_est = get_value(n_est_norm,j+4)
            euclidian_distance[i][j] = abs(y_lense_norm[i]-y_est_norm[i]) + abs(z_lense_norm[i] - z_est_norm[i]) + abs(upper_right - upper_right_est) + abs(upper_centre - upper_centre_est) + abs(upper_left-upper_left_est) + abs(e_left-e_left_est) + np.square(centre-centre_est) + abs(e_right-e_right_est) + abs(bottom_left-bottom_left_est) + abs(bottom_centre-bottom_centre_est) + abs(bottom_right-bottom_right_est)
            euclidian_distance[i][j] = np.sqrt(np.square(y_lense_norm[i]-y_est_norm[i]) + np.square(z_lense_norm[i] - z_est_norm[i]) + np.square(upper_right-upper_right_est) + np.square(upper_centre-upper_centre_est) + np.square(upper_left-upper_left_est) + np.square(e_left-e_left_est) + np.square(centre-centre_est) + np.square(e_right-e_left_est) + np.square(bottom_left-bottom_left_est) + np.square(bottom_centre-bottom_centre_est) + np.square(bottom_right-bottom_right_est))
    return euclidian_distance

'''

def main():
    #fin = "./data/cm_1024_51x51_1.5.pkl"
    fin = "./data/cm_256_16x16_1.5.pkl"
    #fin = "./data/cm_128_8x8_1.5.pkl"
    #fin = "./data/cm_102_5x5_1.5.pkl"
    data = read_pickle(fin)
    lens, est, img_size = data['lens'], data['est'], data['img_size']
    count_euclidian = 0
    count_manhatten = 0
    fig, axs = plt.subplots(2)
    
    y_lense, z_lense, n_lense = lens[:,0], lens[:,1], lens[:,2]
    y_lense_norm, z_lense_norm, n_lense_norm = normalised(y_lense,z_lense,n_lense)
    
    y, z, n = est[:,0], est[:,1], est[:,2]

    #print("len est",np.sqrt(len(est)))
    length = int(np.sqrt(len(est)))
    #print("printing length")
    #print(length)
    for i in range(length):
        a_trim = est[i*length:i*length+length]
        #print(a_trim)
        for i in range(length):
            for k in range(i+1,length):
                if a_trim[i][1] > a_trim[k][1]:
                    a_trim[i][1],a_trim[k][1] = a_trim[k][1],a_trim[i][1]
                    a_trim[i][2],a_trim[k][2] = a_trim[k][2],a_trim[i][2]
                    a_trim[i][0],a_trim[k][0] = a_trim[k][0],a_trim[i][0]
    
    
    y_est, z_est, n_est = est[:,0], est[:,1], est[:,2]
    euclidian_distance = [[0]*(len(est)) for i in range(len(est))]
    manhatten_distance = [[0]*(len(est)) for i in range(len(est))]


    manhatten_distance, euclidian_distance = distance_fun(lens,est)
    #print(euclidian_distance) 
    
    minInRows_euclidian = np.argmin(euclidian_distance, axis=1)
    minInRows_manhatten = np.argmin(manhatten_distance, axis=1)
    
    for k in range(len(minInRows_euclidian)):
        
        if(k == minInRows_euclidian[k]):
            count_euclidian = count_euclidian + 1
    euclidian_distance = np.array(euclidian_distance)
    print("Number of matches_euclidian",(count_euclidian))

    for k in range(len(minInRows_manhatten)):
        
        if(k == minInRows_manhatten[k]):
            count_manhatten = count_manhatten + 1
    manhatten_distance = np.array(manhatten_distance)
    print("Number of matches_manhatten",(count_manhatten))

    


   
if __name__ == "__main__":
    main()
