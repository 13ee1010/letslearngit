from cgi import print_directory
from time import process_time_ns
from turtle import right
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

def read_pickle(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        # --
    return data

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



def distance_fun(lens,est):
    global range_1, range_2, range_3, range_4, range_5, range_6, range_7, range_8, range_9
    global range_est_1, range_est_2, range_est_3, range_est_4, range_est_5, range_est_6, range_est_7, range_est_8, range_est_9
    range_1 = range_2 = range_3 = range_4 = range_5 = range_6 = range_7 = range_8 = range_9 = 0
    range_est_1 = range_est_2 = range_est_3 = range_est_4 = range_est_5 = range_est_6 = range_est_7 = range_est_8 = range_est_9 = 0

    y_lense, z_lense, n_lense = lens[:,0], lens[:,1], lens[:,2]
    y_est, z_est, n_est = est[:,0], est[:,1], est[:,2]
     
    '''
    for i in range(len(n_est)):
        if( n_est[i] > 175.0):
            n_est[i] = np.sin(np.deg2rad(n_est[i]))
    '''
    y_lense_norm, z_lense_norm, n_lense_norm = normalised(y_lense,z_lense,n_lense)
    y_est_norm, z_est_norm, n_est_norm = normalised(y_est, z_est, n_est)

   
    bound = len(y_lense)
    bound = int(np.sqrt(bound))
    
    check = 0.0
    check2= 0.0
    centreX = 0
    centreY = 0
    centreX1 = 0
    centreY1 = 0

    neighbour1 = [0] *10
    neighbour2 = [0] *10
    index = [0]*9
    angle_diff = 0 
    count1 = 0
    count2 = 0
    count3 = 0

    manhatten_distance = [[0]*len(est) for i in range(len(est))]
    euclidian_distance = [[0]*len(est) for i in range(len(est))]
    
    for k in range (len(est)):
        
        count1 = 0
        centreX = z_lense[k]
        centreY = y_lense[k] 
        neighbour1 =[0 for i in range(len(neighbour1))] 
        neighbour2 = [0 for i in range(len(neighbour1))] 
        
        for i in range (len(est)):
            for j in range(len(est)):
                if(i==j):
                    check = ((z_lense[i]) - (centreX))*((z_lense[i])-(centreX)) + ((y_lense[j])-(centreY))*((y_lense[j])-(centreY))
                    #check = abs(z_lense[i] - centreX)  +  abs(y_lense[j] - centreY)
                    #check2 = (((z_est_norm[i] - centreX1)*(z_est_norm[i] - centreX1))  + ((y_est_norm[j] - centreY1)*(y_est_norm[j] - centreY1))) 
                    #print(check)                                                                   
                    if(check <= 2.56):
                        neighbour1[count1] = n_est_norm[i]
                        count1 = count1 + 1
                    
        #print("neighbour1")
        #print(neighbour1)
        #print(count1)           
        range_1 = range_2 = range_3 = range_4 = range_5 = range_6 = range_7= range_8 = range_9 = 0
        
        
        for m in range(len(neighbour1)):    
            if (neighbour1[m] > 0 and neighbour1[m] <= 0.2) :
                range_1 = range_1 + 1
        

        for m in range(len(neighbour1)):
            if(neighbour1[m] > 0.2 and neighbour1[m] <= 0.4):
                range_2 = range_2 + 1

        for m in range(len(neighbour1)):
            if(neighbour1[m] > 0.4 and neighbour1[m] <= 0.6):
                range_3 = range_3 + 1

        for m in range(len(neighbour1)):
            if(neighbour1[m] > 0.6 and neighbour1[m] <= 0.8):
                range_4 = range_4 + 1

        for m in range(len(neighbour1)):
            if(neighbour1[m] > 0.8 and neighbour1[m] <= 1.0):
                range_5 = range_5 + 1
        



















        
        '''
        for m in range(len(neighbour1)):    
            if (neighbour1[m] > 0 and neighbour1[m] <= 0.1) :
                range_1 = range_1 + 1
        

        for m in range(len(neighbour1)):
            if(neighbour1[m] > 0.1 and neighbour1[m] <= 0.2):
                range_2 = range_2 + 1

        for m in range(len(neighbour1)):
            if(neighbour1[m] > 0.2 and neighbour1[m] <= 0.3):
                range_3 = range_3 + 1

        for m in range(len(neighbour1)):
            if(neighbour1[m] > 0.3 and neighbour1[m] <= 0.4):
                range_4 = range_4 + 1

        for m in range(len(neighbour1)):
            if(neighbour1[m] > 0.4 and neighbour1[m] <= 0.5):
                range_5 = range_5 + 1

        for m in range(len(neighbour1)):
            if(neighbour1[m] > 0.5 and neighbour1[m] <= 0.6):
                range_6 = range_6 + 1
        
        for m in range(len(neighbour1)):
            if(neighbour1[m] > 0.6 and neighbour1[m] <= 0.7):
                range_7 = range_7 + 1
        
        for m in range(len(neighbour1)):
            if(neighbour1[m] > 0.7 and neighbour1[m] <= 0.8):
                range_8 = range_8 + 1

        for m in range(len(neighbour1)):
            if(neighbour1[m] > 0.8 and neighbour1[m] <= 0.9):
                range_9 = range_9 + 1
        
        '''
        
        #print("Range1")
        #print("range1 = %d range 2=%d range3=%d range4=%d range5=%d range6=%d range7=%d range8=%d range9=%d"%(range_1,range_2,range_3,range_4,range_5,range_6,range_7,range_8,range_9))            
        
        for x in range(len(est)):

            count2 = 0
            centreX1 = z_est[x]
            centreY1 = y_est[x]
            #print(centreX1)
            #print(centreY1)
            neighbour2 = [0 for i in range(len(neighbour2))] 
            range_est_1 = range_est_2 = range_est_3 = range_est_4 = range_est_5 = range_est_6 = range_est_7 = range_est_8 = range_est_9 = 0
            
            for l in range(len(est)):
                for m in range(len(est)):
                    if(l==m):
                        check2 = (((z_est[l] - centreX1)*(z_est[l] - centreX1))  + ((y_est[m] - centreY1)*(y_est[m] - centreY1))) 
                        
                        if(check2 <= 1256):
                            neighbour2[count2] = n_est_norm[l] 
                            count2 = count2 + 1 

            #print("Count is ---------------------")
            #print(neighbour2)       
            #print(count2)
            
            
            for m in range(len(neighbour2)):
                if(neighbour2[m] > 0 and neighbour2[m] <= 0.2):
                    range_est_1 = range_est_1 + 1

            for m in range(len(neighbour2)):
                if(neighbour2[m] > 0.2 and neighbour2[m] <= 0.4):
                    range_est_2 = range_est_2 + 1

            for m in range(len(neighbour2)):
                if(neighbour2[m] > 0.4 and neighbour2[m] <= 0.6):
                    range_est_3 = range_est_3 + 1

            for m in range(len(neighbour2)):
                if(neighbour2[m] > 0.6 and neighbour2[m] <= 0.8):
                    range_est_4 = range_est_4 + 1

            for m in range(len(neighbour2)):
                if(neighbour2[m] > 0.8 and neighbour2[m] <= 1.0):
                    range_est_5 = range_est_5 + 1

            
            '''
            for m in range(len(neighbour2)):
                if(neighbour2[m] >0 and neighbour2[m] <= 0.1):
                    range_est_1 = range_est_1 + 1

            for m in range(len(neighbour2)):
                if(neighbour2[m] > 0.1 and neighbour2[m] <= 0.2):
                    range_est_2 = range_est_2 + 1

            for m in range(len(neighbour2)):
                if(neighbour2[m] > 0.2 and neighbour2[m] <= 0.3):
                    range_est_3 = range_est_3 + 1

            for m in range(len(neighbour2)):
                if(neighbour2[m] > 0.3 and neighbour2[m] <= 0.4):
                    range_est_4 = range_est_4 + 1

            for m in range(len(neighbour2)):
                if(neighbour2[m] > 0.4 and neighbour2[m] <= 0.5):
                    range_est_5 = range_est_5 + 1

            for m in range(len(neighbour2)):
                if(neighbour2[m] > 0.5 and neighbour2[m] <= 0.6):
                    range_est_6 = range_est_6 + 1
            
            for m in range(len(neighbour2)):
                if(neighbour2[m] > 0.6 and neighbour2[m] <= 0.7):
                    range_est_7 = range_est_7 + 1
            
            for m in range(len(neighbour2)):
                if(neighbour2[m] > 0.7 and neighbour2[m] <= 0.8):
                    range_est_8 = range_est_8 + 1
            
            for m in range(len(neighbour2)):
                if(neighbour2[m] > 0.8 and neighbour2[m] <= 0.9):
                    range_est_9 = range_est_9 + 1
            
            '''
            #print("Range2")
            
            angle_diff = np.sin(np.deg2rad(n_est[k])) - np.sin(np.deg2rad(n_lense[x]))
            
            #print("range_est1 = %d range_est2=%d range_est3=%d range_est4=%d range_est5=%d range_est6=%d range_est7=%d range_est8=%d range_est9=%d"%(range_est_1,range_est_2,range_est_3,range_est_4,range_est_5,range_est_6,range_est_7,range_est_8,range_est_9))
            manhatten_distance[k][x] = 1*abs(z_lense_norm[k] - z_est_norm[x]) + 1*abs(y_lense_norm[k] - y_est_norm[x]) + 1*abs(angle_diff) + 0.3*(abs(range_1 - range_est_1) + abs(range_2-range_est_2) + abs(range_3-range_est_3) + abs(range_4 - range_est_4) + abs(range_5 - range_est_5))
            #euclidian_distance[k][x] = np.sqrt(np.square(z_lense_norm[k]- z_est_norm[x]) + np.square(y_lense_norm[k] - y_est_norm[x]) + np.square(angle_diff) + 0.3*(np.square(range_1-range_est_1) + np.square(range_2-range_est_2) + np.square(range_3-range_est_3) + np.square(range_4 - range_est_4) + np.square(range_5 - range_est_5) + np.square(range_6 - range_est_6) + np.square(range_7 - range_est_7) + np.square(range_8 - range_est_8) + np.square(range_9 - range_est_9)))
            #manhatten_distance[k][x] = 1*abs(z_lense_norm[k] - z_est_norm[x]) + 1*abs(y_lense_norm[k] - y_est_norm[x]) + 1*abs(n_lense_norm[k] - n_est_norm[x]) 
    return manhatten_distance,euclidian_distance

def main():
 


    #fin = "./data/cm_1024_51x51_1.5.pkl"
    #fin = "./data/cm_256_16x16_1.5.pkl"
    #fin = "./data/cm_128_8x8_1.5.pkl"
    #fin = "./data/cm_102_5x5_1.5.pkl"
    #fin = "./data/cm_256_8x8_2.0_8_2.0.pkl"
    #fin = "./data/cm_512_8x8_2.0_8_1.5.pkl"
    #fin = "./data/cm_256_8x8_2.0_8_9.1.pkl"
    #fin = "./data/cm_256_8x8_2.0_8_3.3.pkl"
    fin = "./data/cm_512_16x16_2.0_8_1.1.pkl"
    #fin = "./data/cm_256_16x16_3.0_8_1.5.pkl"
    #fin = "./data/cm_256_16x16_2.0_8_1.5.pkl"
    #fin = "./data/cm_512_16x16_2.0_8_2.9.pkl"
    #fin = "./data/cm_102_5x5_2.0_8_2.5.pkl"
    #fin ="./data/cm_1024_51x51_2.0_8_5.1.pkl"
    #fin = "./data/cm_128_4x4_2.0_8_2.1.pkl"
    #fin = "./data/cm_512_16x16_2.0_8_1.9.pkl"
    #fin = "./data/cm_1024_32x32_2.0_8_1.1.pkl"
    #fin = "./data/cm_512_16x16_2.0_8_1.9.pkl"
    #fin ="./data/cm_1024_51x51_2.0_8_5.1.pkl"
    


    data = read_pickle(fin)
    lens, est, img_size = data['lens'], data['est'], data['img_size']
    count_euclidian = 0
    count_manhatten = 0
    fig, axs = plt.subplots(2)
    range_1 = range_2 = range_3 = range_4 = range_5 = range_6 = 0


    # extract lense and laser spot information here 
    y_lense, z_lense, n_lense = lens[:,0], lens[:,1], lens[:,2]
    y_lense_norm, z_lense_norm, n_lense_norm = normalised(y_lense,z_lense,n_lense)
    
    y, z, n = est[:,0], est[:,1], est[:,2]
    
   
    #print("esttttttttttttttttt")
    #print(est)
    
    length = int(np.sqrt(len(est)))
    
    '''
    for i in range(length):
        a_trim = est[i*length:i*length+length]
        for i in range(length):
            for k in range(i+1,length):
                if a_trim[i][1] > a_trim[k][1]:
                    a_trim[i][1],a_trim[k][1] = a_trim[k][1],a_trim[i][1]
                    a_trim[i][2],a_trim[k][2] = a_trim[k][2],a_trim[i][2]
                    a_trim[i][0],a_trim[k][0] = a_trim[k][0],a_trim[i][0]
    
    print("print esttt after sorting")
    print(est)
    '''

    y_est, z_est, n_est = est[:,0], est[:,1], est[:,2]
    euclidian_distance = [[0]*(len(est)) for i in range(len(est))]
    manhatten_distance = [[0]*(len(est)) for i in range(len(est))]
    manhatten_distance,euclidian_distance = distance_fun(lens,est)


    #print("manhatten distance")
    #print(manhatten_distance)

    minInRows_euclidian = np.argmin(euclidian_distance, axis=1)
    minInRows_manhatten = np.argmin(manhatten_distance, axis=1)
    
    #print(minInRows_manhatten)
    #print(minInRows_euclidian)

    '''
    for i in range(len(est)):
        for j in range(len(est)):
            if(i == j):
                if(manhatten_distance[i][j] == 0):
                    count_manhatten = count_manhatten + 1

    for i in range(len(est)):
        for j in range(len(est)):
            if (i == j):
                if(euclidian_distance[i][j] == 0):
                    count_euclidian = count_euclidian + 1 
    '''
    
    for k in range(len(minInRows_euclidian)):
        #print("{}       |       {} ".format(k, minInRows_euclidian[k]))
        if(k == minInRows_euclidian[k]):
            count_euclidian = count_euclidian + 1
    euclidian_distance = np.array(euclidian_distance)
    #print("Number of matches_euclidian",(count_euclidian))
    
    #print("lense index  | laser spot index")
    for k in range(len(minInRows_manhatten)):
        #print("{}       |       {} ".format(k, minInRows_manhatten[k]))
        if(k == minInRows_manhatten[k]):
            count_manhatten = count_manhatten + 1
    manhatten_distance = np.array(manhatten_distance)
    #print("Number of matches_manhatten",(count_manhatten))

   
if __name__ == "__main__":
    t0= time.time()
    main()
    t1 = time.time() - t0
    print("Time elapsed: ", t1)
   