#
from operator import ne
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
range_1 = range_2 = range_3 = range_4 = range_5 = range_6 = range_7 = range_8 = range_9 = 0
range_est_1 = range_est_2 = range_est_3 = range_est_4 = range_est_5 = range_est_6 = range_est_7 = range_est_8 = range_est_9 = 0


def norm_data(data):
    mean_data = np.mean(data)
    std_data = np.std(data,ddof=1)
    return (data-mean_data)/(std_data)

def ncc(data0,data1):
    return(1.0/(len(data0)-1))*np.sum(norm_data(data0)*norm_data(data1))

def distance_fun(lens, est):
    global range_1, range_2, range_3, range_4, range_5, range_6, range_7, range_8, range_9
    global range_est_1, range_est_2, range_est_3, range_est_4, range_est_5, range_est_6, range_est_7, range_est_8, range_est_9

    range_1 = range_2 = range_3 = range_4 = range_5 = range_6 = 0
    range_est_1 = range_est_2 = range_est_3 = range_est_4 = range_est_5 = range_est_6 = 0

    y_lense, z_lense, n_lense = lens[:,0], lens[:,1], lens[:,2]
    y_est, z_est, n_est = est[:,0], est[:,1], est[:,2]
    
    
    for i in range(len(n_est)):
        if( n_est[i] > 175.0):
            n_est[i] = np.sin(np.deg2rad(n_est[i]))
    y_lense_norm, z_lense_norm, n_lense_norm = normalised(y_lense,z_lense,n_lense)
    y_est_norm, z_est_norm, n_est_norm = normalised(y_est, z_est, n_est)
    
    '''
    print(n_lense)
    print(n_est)
    
    print(n_lense_norm)
    '''
    #n_est_norm = "{:.3f}".format(n_est_norm)
    #print("%.3f" %(n_est_norm))
    

    bound = len(y_lense)
    bound = int(np.sqrt(bound))
    angle_diff = 0

    neighbourhood = [0,0,0,0,0,0,0,0,0]
    range_lens = [0,0,0,0,0,0,0,0,0]
    range_est = [0,0,0,0,0,0,0,0,0]
    manhatten_distance = [[0]*len(est) for i in range(len(est))]
    euclidian_distance = [[0]*len(est) for i in range(len(est))]
    n_cross_correlation = [[0]*len(est) for i in range(len(est))]

    for i in range(len(est)):
        
        range_1 = range_2 = range_3 = range_4 = range_5 = range_6 = range_7 = range_8 = range_9 = 0
        range_lens = [0,0,0,0,0,0,0,0,0]
        neighbourhood[0] = get_value(n_lense_norm,i+(bound+1))
        neighbourhood[1] = get_value(n_lense_norm,i+bound)
        neighbourhood[2] = get_value(n_lense_norm,i+(bound-1))
        neighbourhood[3] = get_value(n_lense_norm,i-1)
        neighbourhood[4] = get_value(n_lense_norm,i)
        neighbourhood[5] = get_value(n_lense_norm,i+1)
        neighbourhood[6] = get_value(n_lense_norm,i-(bound+1))
        neighbourhood[7] = get_value(n_lense_norm,i-(bound))
        neighbourhood[8] = get_value(n_lense_norm,i-(bound-1))

        if(i%(bound)==0):
            neighbourhood[2] = 0
            neighbourhood[3] = 0
            neighbourhood[6] = 0
        if(i%(bound) == (bound-1)):
            neighbourhood[0] = 0
            neighbourhood[5] = 0
            neighbourhood[8] = 0

        for m in range(len(neighbourhood)):
            if (neighbourhood[m] >=0 and neighbourhood[m] <= 0.1) :
                range_lens[0] = range_lens[0] + 1
        

        for m in range(len(neighbourhood)):
            if(neighbourhood[m] > 0.1 and neighbourhood[m] <= 0.2):
                range_lens[1] = range_lens[1] + 1

        for m in range(len(neighbourhood)):
            if(neighbourhood[m] > 0.2 and neighbourhood[m] <= 0.3):
                range_lens[2] = range_lens[2] + 1

        for m in range(len(neighbourhood)):
            if(neighbourhood[m] > 0.3 and neighbourhood[m] <= 0.4):
                range_lens[3] = range_lens[3] + 1

        for m in range(len(neighbourhood)):
            if(neighbourhood[m] > 0.4 and neighbourhood[m] <= 0.5):
                range_lens[4] = range_lens[4] + 1

        for m in range(len(neighbourhood)):
            if(neighbourhood[m] > 0.5 and neighbourhood[m] <= 0.6):
                range_lens[5] = range_lens[5] + 1
        
        for m in range(len(neighbourhood)):
            if(neighbourhood[m] > 0.6 and neighbourhood[m] <= 0.7):
                range_lens[6] = range_lens[6] + 1
        
        for m in range(len(neighbourhood)):
            if(neighbourhood[m] > 0.7 and neighbourhood[m] <= 0.8):
                range_lens[7] = range_lens[7] + 1

        for m in range(len(neighbourhood)):
            if(neighbourhood[m] > 0.8 and neighbourhood[m] <= 0.9):
                range_lens[8] = range_lens[8] + 1
        
        #print("range_lenssssss")
        #print(range_lens)
        #print("neighbourhood count % d %d %d %d %d %d %d %d %d" %(range_1,range_2,range_3,range_4,range_5,range_6,range_7, range_8, range_9)) 
        for j in range(len(est)):
            
            range_est_1 = range_est_2 = range_est_3 = range_est_4 = range_est_5 = range_est_6 = range_est_7 = range_est_8 = range_est_9 = 0
            range_est = [0,0,0,0,0,0,0,0,0]
            neighbourhood[0] = get_value(n_est_norm,j+(bound+1))
            neighbourhood[1] = get_value(n_est_norm,j+(bound))
            neighbourhood[2] = get_value(n_est_norm,j+(bound-1))
            neighbourhood[3] = get_value(n_est_norm,j-1)
            neighbourhood[4] = get_value(n_est_norm,j)
            neighbourhood[5] = get_value(n_est_norm,j+1)
            neighbourhood[6] = get_value(n_est_norm,j-(bound+1))
            neighbourhood[7] = get_value(n_est_norm,j-(bound))
            neighbourhood[8] = get_value(n_est_norm,j-(bound-1))
             
             
            if(j%(bound)==0):
                neighbourhood[2] = 0
                neighbourhood[3] = 0
                neighbourhood[6] = 0

            if(j%(bound) == (bound-1)):
                neighbourhood[0] = 0
                neighbourhood[5] = 0
                neighbourhood[8] = 0

            for m in range(len(neighbourhood)):
                if(neighbourhood[m] >=0 and neighbourhood[m] <= 0.1):
                    range_est[0] = range_est[0] + 1

            for m in range(len(neighbourhood)):
                if(neighbourhood[m] > 0.1 and neighbourhood[m] <= 0.2):
                    range_est[1] = range_est[1] + 1

            for m in range(len(neighbourhood)):
                if(neighbourhood[m] > 0.2 and neighbourhood[m] <= 0.3):
                    range_est[2] = range_est[2] + 1

            for m in range(len(neighbourhood)):
                if(neighbourhood[m] > 0.3 and neighbourhood[m] <= 0.4):
                    range_est[3] = range_est[3] + 1

            for m in range(len(neighbourhood)):
                if(neighbourhood[m] > 0.4 and neighbourhood[m] <= 0.5):
                    range_est[4] = range_est[4] + 1

            for m in range(len(neighbourhood)):
                if(neighbourhood[m] > 0.5 and neighbourhood[m] <= 0.6):
                    range_est[5] = range_est[5] + 1
            
            for m in range(len(neighbourhood)):
                if(neighbourhood[m] > 0.6 and neighbourhood[m] <= 0.7):
                    range_est[6] = range_est[6] + 1
            
            for m in range(len(neighbourhood)):
                if(neighbourhood[m] > 0.7 and neighbourhood[m] <= 0.8):
                    range_est[7] = range_est[7] + 1
            
            for m in range(len(neighbourhood)):
                if(neighbourhood[m] > 0.8 and neighbourhood[m] <= 0.9):
                    range_est[8] = range_est[8] + 1

            #print("rangeeeee eadstttttttt")
            #print(range_est)
            n_cross_correlation[i][j] = ncc(range_lens,range_est)
            angle_diff = np.sin(np.deg2rad(n_est[i])) - np.sin(np.deg2rad(n_lense[j]))
            #print("neighbourhood count2 % d %d %d %d %d %d %d %d %d" %(range_est_1,range_est_2,range_est_3,range_est_4,range_est_5,range_est_6,range_est_7, range_est_8, range_est_9))
            euclidian_distance[i][j] = np.sqrt(np.square(y_lense_norm[i] - y_est_norm[j]) + np.square(z_lense_norm[i] - z_est_norm[j]) + np.square(angle_diff)  +  np.square(range_lens[0]-range_est[0]) + np.square(range_lens[1]-range_est[1]) + np.square(range_lens[2]-range_est[2]) + np.square(range_lens[3] - range_est[3]) + np.square(range_lens[4] - range_est[4]) + np.square(range_lens[5] - range_est[5]) + np.square(range_lens[6] - range_est[6]) + np.square(range_lens[7] - range_est[7]) + np.square(range_lens[8] - range_est[8]))
            manhatten_distance[i][j] = abs(y_lense_norm[i]-y_est_norm[j]) + abs(z_lense_norm[i] - z_est_norm[j]) + abs(angle_diff) + abs(range_lens[0] - range_est[0]) + abs(range_lens[1]-range_est[1]) + abs(range_lens[2]-range_est[2]) + abs(range_lens[3] - range_est[3]) + abs(range_lens[4] - range_est[4]) + abs(range_lens[5] - range_est[5]) + abs(range_lens[6] - range_est[6]) + abs(range_lens[7] - range_est[7]) + abs(range_lens[8] - range_est[8])
            #manhatten_distance[i][j] = abs(y_lense_norm[i]-y_est_norm[i]) + abs(z_lense_norm[i] - z_est_norm[i]) + abs(upper_right - upper_right_est) + abs(upper_centre - upper_centre_est) + abs(upper_left-upper_left_est) + abs(e_left-e_left_est) + 1*abs(centre-centre_est) + abs(e_right-e_right_est) + abs(bottom_left-bottom_left_est) + abs(bottom_centre-bottom_centre_est) + abs(bottom_right-bottom_right_est)
            #euclidian_distance[i][j] = np.sqrt(np.square(y_lense_norm[i]-y_est_norm[i]) + np.square(z_lense_norm[i] - z_est_norm[i]) + np.square(upper_right-upper_right_est) + np.square(upper_centre-upper_centre_est) + np.square(upper_left-upper_left_est) + np.square(e_left-e_left_est) + (np.square(centre-centre_est)) + np.square(e_right-e_left_est) + np.square(bottom_left-bottom_left_est) + np.square(bottom_centre-bottom_centre_est) + np.square(bottom_right-bottom_right_est))
                                            #np.sin(np.deg2rad(n_est[i])) - np.sin(np.deg2rad(n_lense[j]))

    return manhatten_distance, euclidian_distance, n_cross_correlation   



def main():
    #fin = "./data/cm_1024_51x51_1.5.pkl"
    #fin = "./data/cm_256_16x16_1.5.pkl"
    #fin = "./data/cm_128_8x8_1.5.pkl"
    #fin = "./data/cm_102_5x5_1.5.pkl"
    #fin = "./data/cm_256_8x8_2.0_8_2.0.pkl"
    #fin = "./data/cm_512_8x8_2.0_8_1.5.pkl"
    #fin = "./data/cm_256_8x8_2.0_8_5.0.pkl"
    #fin = "./data/cm_256_8x8_2.0_8_1.2.pkl"
    #fin = "./data/cm_512_16x16_2.0_8_1.3.pkl"
    #fin = "./data/cm_256_16x16_3.0_8_1.5.pkl"
    #fin = "./data/cm_256_16x16_2.0_8_1.5.pkl"
    #fin = "./data/cm_512_16x16_2.0_8_0.9.pkl"
    #fin = "./data/cm_102_5x5_2.0_8_2.5.pkl"
    #fin ="./data/cm_1024_51x51_2.0_8_5.1.pkl"
    fin = "./data/cm_128_4x4_2.0_8_1.9.pkl"
    #fin = "./data/cm_512_16x16_2.0_8_1.9.pkl"
    #fin = "./data/cm_1024_32x32_2.0_8_1.1.pkl"
    #fin = "./data/cm_512_16x16_2.0_8_1.3.pkl"

    data = read_pickle(fin)
    lens, est, img_size = data['lens'], data['est'], data['img_size']
    count_euclidian = 0
    count_manhatten = 0
    cc_count = 0
    fig, axs = plt.subplots(2)
    range_1 = range_2 = range_3 = range_4 = range_5 = range_6 = 0
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
    n_cross_correlation = [[0]*(len(est)) for i in range(len(est))]

    manhatten_distance, euclidian_distance, n_cross_correlation = distance_fun(lens,est)
    #n_cross_correlation = np.linalg.inv(n_cross_correlation)

    print("euclidian distance")
    print(euclidian_distance) 
    #print("manhatten distance")
    #print(manhatten_distance)
    #print("cross_corelatiron")
    #print(n_cross_correlation)

    minInRows_euclidian = np.argmin(euclidian_distance, axis=1)
    minInRows_manhatten = np.argmin(manhatten_distance, axis=1)
    maxInRows_cc = np.argmax(n_cross_correlation, axis=1)

    #print("mAX IN ROWS")
    #print(maxInRows_cc)
    print("minimum rows")
    print(minInRows_manhatten)



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
     
    '''
    for i in range(len(est)):
        for j in range(len(est)):
            if( i == j):
                if(manhatten_distance[i][j] == 0):
                    count_manhatten = count_manhatten + 1

    for i in range(len(est)):
        for j in range(len(est)):
            if (i == j):
                if(euclidian_distance[i][j] == 0):
                    count_euclidian = count_euclidian + 1 
    
    
    for i in range(len(est)):
        for j in range(len(est)):
            if (i == j):
                if(n_cross_correlation[i][j] >= 0.999999999 or n_cross_correlation[i][j]<= 1.0000000):
                    cc_count = cc_count + 1 
    
    euclidian_distance = np.array(euclidian_distance)
    print("Number of matches_euclidian",(count_euclidian))
    
    
    manhatten_distance = np.array(manhatten_distance)
    print("Number of matches_manhatten",(count_manhatten))
    print("Number of matches_cross_correlation",(cc_count))
    '''
    


   
if __name__ == "__main__":
    main()
