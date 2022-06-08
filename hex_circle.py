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

def get_value(arr_2d,i,j):
    if((i >=len(arr_2d) or i< 0) or (j >= len(arr_2d) or j < 0)):
        return 0
    else:
        return arr_2d[i][j]

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
    count_1 = 0 
    count_2 = 0
    neighbour1 = [0] *10
    neighbour2 = [0] *10
    range_lense = [0]*5
    range_est = [0]*5 

    bound = len(y_lense)
    bound = int(np.sqrt(bound))

    print("bound is ", bound)


    # After normalisation, convert angle matrix into 2d matrix
    z_lense_2d = np.reshape(z_lense, (bound,bound))
    z_est_2d = np.reshape(z_est, (bound, bound))

    y_lense_2d = np.reshape(y_lense, (bound,bound))
    y_est_2d = np.reshape(y_est, (bound,bound))

    n_lense_norm_2d = np.reshape(n_lense_norm, (bound,bound))
    n_lense_2d = np.reshape(n_lense, (bound,bound))

    '''
    print(n_lense)
    print(n_est)
    '''
    #print("z_lense norm")
    #print(z_lense_norm)

    #print("Z_est norm")
    #print(z_est_norm)


    #print("y_lense_norm")
    #print(y_lense_norm)
    #print("y_est_norm")
    #print(y_est_norm)

    #print("")
    #n_est_norm = "{:.3f}".format(n_est_norm)
    #print("%.3f" %(n_est_norm))
    neighboursX = [0]*6
    neighboursY = [0]*6
    manhatten_distance = [[0]*len(est) for i in range(len(est))]
    euclidian_distance = [[0]*len(est) for i in range(len(est))]

    for i in range(len(z_lense_2d)):
        for j in range(len(z_est_2d)):

            #i = 0 
            #j = 0 
            # centre of lense image
            centreX = z_lense_2d[i][j]  
            centreY = y_lense_2d[i][j] 
            
            print("center X = %d centerY = %d"%(centreX,centreY))
            count1 = 0
            

            check = (z_lense_2d[i][j] - centreX)*(z_lense_2d[i][j] - centreX) + (y_lense_2d[i][j] - centreY)*(y_lense_2d[i][j] - centreY)

            if(check <= 4.2):
                neighbour1[count1] = n_lense_2d[i][j]
                count1 = count1 + 1

            #print(neighbour1)
            for m in range(len(neighbour1)):    
                if (neighbour1[m] > 0 and neighbour1[m] <= 0.2) :
                    range_lense[0] = range_lense[0] + 1
        

            for m in range(len(neighbour1)):
                if(neighbour1[m] > 0.2 and neighbour1[m] <= 0.4):
                    range_lense[1] = range_lense[1] + 1

            for m in range(len(neighbour1)):
                if(neighbour1[m] > 0.4 and neighbour1[m] <= 0.6):
                    range_lense[2] = range_lense[2] + 1

            for m in range(len(neighbour1)):
                if(neighbour1[m] > 0.6 and neighbour1[m] <= 0.8):
                    range_lense[3] = range_lense[3] + 1

            for m in range(len(neighbour1)):
                if(neighbour1[m] > 0.8 and neighbour1[m] <= 1.0):
                    range_lense[4] = range_lense[4] + 1        

            for i in range(len(z_lense_2d)):
                for j in range(len(z_est_2d)):
                    #neighbours's centre in hexagon region
                    if(i%2 == 0):
                        neighboursX[0] = get_value(z_est_2d, i-1, j-1)
                        neighboursX[1] = get_value(z_est_2d, i-1, j)
                        neighboursX[2] = get_value(z_est_2d, i, j-1)
                        neighboursX[3] = get_value(z_est_2d, i,j)
                        neighboursX[4] = get_value(z_est_2d, i, j+1)
                        neighboursX[5] = get_value(z_est_2d, i+1, j-1)
                        neighboursX[6] = get_value(z_est_2d, i+1, j)

                        neighboursY[0] = get_value(y_est_2d, i-1, j-1)
                        neighboursY[1] = get_value(y_est_2d, i-1, j)
                        neighboursY[2] = get_value(y_est_2d, i, j-1)
                        neighboursY[3] = get_value(y_est_2d, i,j)
                        neighboursY[4] = get_value(y_est_2d, i, j+1)
                        neighboursY[5] = get_value(y_est_2d, i+1, j-1)
                        neighboursY[6] = get_value(y_est_2d, i+1, j)


                    else:
                        neighboursX[0] = get_value(z_est_2d, i-1, j)
                        neighboursX[1] = get_value(z_est_2d, i-1, j+1)
                        neighboursX[2] = get_value(z_est_2d, i, j-1)
                        neighboursX[3] = get_value(z_est_2d, i,j)
                        neighboursX[4] = get_value(z_est_2d, i, j+1)
                        neighboursX[5] = get_value(z_est_2d, i+1, j)
                        neighboursX[6] = get_value(z_est_2d, i+1, j+1)

                        neighboursY[0] = get_value(y_est_2d, i-1, j)
                        neighboursY[1] = get_value(y_est_2d, i-1, j+1)
                        neighboursY[2] = get_value(y_est_2d, i, j-1)
                        neighboursY[3] = get_value(y_est_2d, i,j)
                        neighboursY[4] = get_value(y_est_2d, i, j+1)
                        neighboursY[5] = get_value(y_est_2d, i+1, j)
                        neighboursY[6] = get_value(y_est_2d, i+1, j+1)
            '''

            '''
            print("range1 = %d range 2=%d range3=%d range4=%d range5=%d range6=%d range7=%d range8=%d range9=%d"%(range_1,range_2,range_3,range_4,range_5,range_6,range_7,range_8,range_9))            
            '''

            for k in range(len(neighboursX)):
                count2 = 0 
                centreX1 = neighboursX[k]
                centreY1 = neighboursY[k]
                
                neighbour2 = [0 for h in range(len(neighbour2))]
                range_est = [0 for g in range(len(range_est))]

                for l in range(len(est)):
                    for m in range(len(est)):
                        if(l==m):
                            check2 = (((z_est[l] - centreX1)*(z_est[l] - centreX1))  + ((y_est[l] - centreY1)*(y_est[l] - centreY1))) 
                            if(check2 <= 900):
                                #print("value of l is ",l)
                                neighbour2[count2] = n_est_norm[l] 
                                count2 = count2 + 1 

                #print(neighbour2)

                for m in range(len(neighbour2)):
                    if(neighbour2[m] > 0 and neighbour2[m] <= 0.2):
                        range_est[0] = range_est[0] + 1

                for m in range(len(neighbour2)):
                    if(neighbour2[m] > 0.2 and neighbour2[m] <= 0.4):
                        range_est[1] = range_est[1] + 1

                for m in range(len(neighbour2)):
                    if(neighbour2[m] > 0.4 and neighbour2[m] <= 0.6):
                        range_est[2] = range_est[2] + 1

                for m in range(len(neighbour2)):
                    if(neighbour2[m] > 0.6 and neighbour2[m] <= 0.8):
                        range_est[3] = range_est[3] + 1

                for m in range(len(neighbour2)):
                    if(neighbour2[m] > 0.8 and neighbour2[m] <= 1.0):
                        range_est[4] = range_est[4] + 1


                '''
                print("range_est1 = %d range_est2=%d range_est3=%d range_est4=%d range_est5=%d range_est6=%d range_est7=%d range_est8=%d range_est9=%d"%(range_est_1,range_est_2,range_est_3,range_est_4,range_est_5,range_est_6,range_est_7,range_est_8,range_est_9))
                '''
                #manhatten_distance[count_1][count_2] = abs(y_lense_norm[count_1] - y_est_norm[count_2]) + abs(z_lense_norm[count_1] - z_est_norm[count_2]) + abs(n_lense_norm[count_1] - n_est_norm[count_2]) + 0.8*(abs(range_1 - range_est_1) + abs(range_2-range_est_2) + abs(range_3-range_est_3) + abs(range_4 - range_est_4) + abs(range_5 - range_est_5) + abs(range_6 - range_est_6) + abs(range_7 - range_est_7) + abs(range_8 - range_est_8) + abs(range_9 - range_est_9))
                #euclidian_distance[count_1][count_2] = np.sqrt(np.square(y_lense_norm[count_1]- y_est_norm[count_2]) + np.square(z_lense_norm[count_1] - z_est_norm[count_2]) + np.square(n_lense_norm[count_1] - n_est_norm[count_2]) + 0.8*(np.square(range_1-range_est_1) + np.square(range_2-range_est_2) + np.square(range_3-range_est_3) + np.square(range_4 - range_est_4) + np.square(range_5 - range_est_5) + np.square(range_6 - range_est_6) + np.square(range_7 - range_est_7) + np.square(range_8 - range_est_8) + np.square(range_9 - range_est_9)))

                    
            

    return manhatten_distance, euclidian_distance   

def main():
    #fin = "./data/cm_1024_51x51_1.5.pkl"
    #fin = "./data/cm_256_16x16_1.5.pkl"
    #fin = "./data/cm_128_8x8_10.1.pkl"
    #fin = "./data/cm_102_5x5_1.5.pkl"
    #fin = "./data/cm_256_8x8_2.0_8_2.0.pkl"
    #fin = "./data/cm_512_8x8_2.0_8_1.5.pkl"
    fin = "./data/cm_128_4x4_2.0_8_9.1.pkl"
    #fin = "./data/cm_256_8x8_2.0_8_1.2.pkl"
    #fin = "./data/cm_512_16x16_2.0_8_1.9.pkl"
    #fin = "./data/cm_1024_32x32_2.0_8_1.1.pkl"
    #fin = "./data/cm_256_8x8_2.0_8_7.5.pkl"
    #fin = "./data/cm_256_8x8_2.0_8_1.2.pkl"
    #fin = "./data/cm_512_16x16_2.0_8_0.9.pkl"
    #fin = "./data/cm_256_8x8_2.0_8_1.2.pkl"

    data = read_pickle(fin)
    lens, est, img_size = data['lens'], data['est'], data['img_size']
    count_euclidian = 0
    count_manhatten = 0
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


    manhatten_distance, euclidian_distance = distance_fun(lens,est)

    #print("euclidian distance")
    #print(euclidian_distance) 
    #print("manhatten distance")
    #print(manhatten_distance)

    minInRows_euclidian = np.argmin(euclidian_distance, axis=1)
    minInRows_manhatten = np.argmin(manhatten_distance, axis=1)
    #print("minimum rows")
    #print(minInRows_manhatten)

    #print("Minimum rows in euclidian")
    #print(minInRows_euclidian)
    '''
    for i in range(len(est)):
        for j in range(len(est)):
            if( i == j):
                if(manhatten_distance[i][j] == 0):l
                    count_manhatten = count_manhatten + 1

    for i in range(len(est)):
        for j in range(len(est)):
            if (i == j):
                if(euclidian_distance[i][j] == 0):
                    count_euclidian = count_euclidian + 1 
    '''
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
