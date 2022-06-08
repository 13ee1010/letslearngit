#
import pickle
import numpy as np
import matplotlib.pyplot as plt


def read_pickle(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        # --
    return data

times = 0 

def get_neighbour_value(index,k,arr_b):
    
    if(index[k] >= len(arr_b) or index[k] < 0):
        return 0 
    else:
        return arr_b[index[k]]


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
    index = [0,0,0,0,0,0,0,0,0]
    value = [0,0,0,0,0,0,0,0,0]
    value2 = [0,0,0,0,0,0,0,0,0]
    value3 = [0,0,0,0,0,0,0,0,0]

    manhatten_distance = [[0]*9 for i in range(len(est))]
    euclidian_distance = [[0]*9 for i in range(len(est))]
    bound = len(y_lense)
    bound = int(np.sqrt(bound))
    for i in range(len(est)):

        upper_right = i+(bound+1)
        upper_centre = i+bound
        upper_left = i+(bound-1)
        e_left = i-1
        e_right = i+1
        bottom_left = i-(bound+1)
        bottom_centre = i-(bound)
        bottom_right = i-(bound-1)
    
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
        for j in range(len(index)):
            value[k]  = get_neighbour_value(index,k,y_est_norm)
            value2[k] = get_neighbour_value(index,k,z_est_norm)
            value3[k] = get_neighbour_value(index,k,n_est_norm)
            #manhatten_distance[i][j] = abs(y_lense_norm[i] - value) + abs(z_lense_norm[i]-value2) + 0.5*abs(n_lense_norm[i]- value3)
            k = k+1

        if(i%(bound)==0):
            value[0] = 0
            value[3] = 0
            value[6] = 0

            value2[0] = 0
            value2[3] = 0
            value2[6] = 0 

            value3[0] = 0
            value3[3] = 0 
            value3[6] = 0

        if(i%(bound) == (bound-1)):
            value[5] = 0
            value[8] = 0

            value2[5] = 0
            value2[8] = 0

            value3[5] = 0
            value3[8] = 0
        
        if(i==(bound-1)):
            value[2] = 0 
            value2[2] = 0 
            value3[2] = 0
        k = 0 
        for j in range(len(index)):
            manhatten_distance[i][j] = abs(y_lense_norm[i] - value[j]) + abs(z_lense_norm[i]-value2[j]) + abs(n_lense_norm[i]- value3[j])

            euclidian_distance[i][j] = np.sqrt(np.square(y_lense_norm[i]- value[j]) + np.square(z_lense_norm[i] - value2[j])) + np.square(n_lense_norm[i] - value3[j])
            

    return manhatten_distance,euclidian_distance   




def main():
    #fin = "./data/cm_1024_51x51_1.5.pkl"
    #fin = "./data/cm_256_16x16_1.5.pkl"
    #fin = "./data/cm_128_8x8_1.5.pkl"
    #fin = "./data/cm_102_5x5_1.5.pkl"
    fin = "./data/cm_256_8x8_2.0_8_2.0.pkl"
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
    euclidian_distance = [[0]*9 for i in range(len(est))]
    manhatten_distance = [[0]*9 for i in range(len(est))]


    manhatten_distance,euclidian_distance = distance_fun(lens,est)
    
    
    #print(euclidian_distance) 
    print(manhatten_distance)
    minInRows_euclidian = np.argmin(euclidian_distance, axis=1)
    minInRows_manhatten = np.argmin(manhatten_distance, axis=1)
    #print("HELOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    print(minInRows_euclidian)
    print(minInRows_manhatten)

    #print('min value of every Row: ', len(minInRows))
    for k in range(len(minInRows_euclidian)):
        #print("lense image %d and ellipse %d " % (k, minInRows[k]))
        if(4 == minInRows_euclidian[k]):
            count_euclidian = count_euclidian + 1
    euclidian_distance = np.array(euclidian_distance)
    print("Number of matches_euclidian",(count_euclidian))
    


    #print('min value of every Row: ', len(minInRows))
    for k in range(len(minInRows_manhatten)):
        #print("lense image %d and ellipse %d " % (k, minInRows[k]))
        if(4 == minInRows_manhatten[k]):
            count_manhatten = count_manhatten + 1
    manhatten_distance = np.array(manhatten_distance)
    print("Number of matches_manhatten",(count_manhatten))
    


   
if __name__ == "__main__":
    main()
