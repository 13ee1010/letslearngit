#
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import time

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


def normalised_single_data(y_data):
    norm_y = np.linalg.norm(y_data)
    y_norm = y_data / norm_y

    return y_norm

def get_value(arr_b,i):
    if((i >=len(arr_b) or i< 0)):
        return 0
    else:
        return arr_b[i]

def get_neighbour_value(index,k,arr_b):
    
    if(index[k] >= len(arr_b) or index[k] < 0):
        return 0 
    else:
        return arr_b[index[k]]



def distance_fun(lens, est):
    global range_1, range_2, range_3, range_4, range_5, range_6, range_7, range_8, range_9
    global range_est_1, range_est_2, range_est_3, range_est_4, range_est_5, range_est_6, range_est_7, range_est_8, range_est_9
    range_1 = range_2 = range_3 = range_4 = range_5 = range_6 = range_7 = range_8 = range_9 = 0
    range_est_1 = range_est_2 = range_est_3 = range_est_4 = range_est_5 = range_est_6 = range_est_7 = range_est_8 = range_est_9 = 0
    global k,n
    k = 0 
    n = 0

    y_lense, z_lense, n_lense = lens[:,0], lens[:,1], lens[:,2]
    y_est, z_est, n_est = est[:,0], est[:,1], est[:,2]
    bound = len(y_lense)
    bound = int(np.sqrt(bound))
    neighboursX = [0]*7
    neighboursY = [0]*7
    neighboursA = [0]*7
    value =  [0]*7
    value1 = [0]*7
    value2 = [0]*7

    # After normalisation, convert angle matrix into 2d matrix
    z_lense_2d = np.reshape(z_lense, (bound,bound))
    z_est_2d = np.reshape(z_est, (bound, bound))

    y_lense_2d = np.reshape(y_lense, (bound,bound))
    y_est_2d = np.reshape(y_est, (bound,bound))

   
    y_lense_norm, z_lense_norm, n_lense_norm = normalised(y_lense,z_lense,n_lense)
    y_est_norm, z_est_norm, n_est_norm = normalised(y_est, z_est, n_est)
    index = [0]*7
    value = [0]*7
    value2 = [0]*7
    value3 = [0]*7
    centreX_value = [0,0,0,0,0,0,0,0,0]
    centreY_value = [0,0,0,0,0,0,0,0,0]
    manhatten_distance = [[0]*7 for i in range(len(est))]
    euclidian_distance = [[0]*7 for i in range(len(est))]
    range_lense = [0]*5
    range_est = [0]*5 
   
    neighbour1 = [0] *10
    neighbour2 = [0] *10
    n_lense_norm_2d = np.reshape(n_lense_norm, (bound,bound))
    n_lense_2d = np.reshape(n_lense, (bound,bound))
    
    print(" Bound is %d",bound)
    
    for i in range(len(est)):
        
        #i = 5
        count1 = 0
        centreX = z_lense[i]
        centreY = y_lense[i] 

        #print("centreX is %d and centreY is %d"%(centreX,centreY))
        neighbour1 =[0]*(len(neighbour1))
        neighbour2 = [0]*(len(neighbour2))
        range_lense = [0 for k in range(len(range_lense))]
        range_est = [0 for g in range(len(range_est))]

        
        for x in range (len(est)):
            for y in range(len(est)):
                if(x==y):
                    check = ((z_lense[x]) - (centreX))*((z_lense[x])-(centreX)) + ((y_lense[x])-(centreY))*((y_lense[x])-(centreY))
                    if(check <= 4.2):
                        neighbour1[count1] = n_lense_norm[x]
                        count1 = count1 + 1
        #print("neighbours inside the circle")
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

        #print(range_lense)
        #get neighbour value 

        
       
        if(k == bound):
            k = 0 
            n = n + 1
        k = k + 1
        #print("value of k is %d and n is %d"%(k,n))   


        

        if(n%2 == 0):
            #print("VAUE OF even row i",i)
            upper_left =  i+(bound-1)
            upper_right = i+(bound)
            e_left = i-1
            e_right = i+1
            bottom_left = i-(bound+1)
            bottom_right = i-(bound)   

            index[0] = upper_left
            index[1] = upper_right
            index[2] = e_left
            index[3] = i
            index[4] = e_right
            index[5] = bottom_left
            index[6] = bottom_right

            for k in range(len(index)):
                value[k]  = get_neighbour_value(index,k,y_est_norm)
                value2[k] = get_neighbour_value(index,k,z_est_norm)
                value3[k] = get_neighbour_value(index,k,n_est_norm)
                neighboursX[k] = get_neighbour_value(index,k,z_est)
                neighboursY[k] = get_neighbour_value(index,k,y_est)
            #k = k+1

            '''
            neighboursX[0] = get_value(z_est, i+(bound-1)) # upper left
            neighboursX[1] = get_value(z_est, i+(bound))   # upper right
            neighboursX[2] = get_value(z_est, i-1)         # extreme left
            neighboursX[3] = get_value(z_est, i)           # centre
            neighboursX[4] = get_value(z_est, i+1)         # extreme right
            neighboursX[5] = get_value(z_est, i-(bound+1))  #lower left
            neighboursX[6] = get_value(z_est, i-(bound))    # lower right
            
            neighboursY[0] = get_value(y_est, i+(bound-1))
            neighboursY[1] = get_value(y_est, i+(bound))
            neighboursY[2] = get_value(y_est, i-1)
            neighboursY[3] = get_value(y_est, i)
            neighboursY[4] = get_value(y_est, i+1)
            neighboursY[5] = get_value(y_est, i-(bound+1))
            neighboursY[6] = get_value(y_est, i-(bound))
            '''   
        else:

            upper_left =  i+(bound)
            upper_right = i+(bound+1)
            e_left = i-1
            e_right = i+1
            bottom_left = i-(bound)
            bottom_right = i-(bound-1)   

            index[0] = upper_left
            index[1] = upper_right
            index[2] = e_left
            index[3] = i
            index[4] = e_right
            index[5] = bottom_left
            index[6] = bottom_right

            for k in range(len(index)):
                value[k]  = get_neighbour_value(index,k,y_est_norm)
                value2[k] = get_neighbour_value(index,k,z_est_norm)
                value3[k] = get_neighbour_value(index,k,n_est_norm)
                neighboursX[k] = get_neighbour_value(index,k,z_est)
                neighboursY[k] = get_neighbour_value(index,k,y_est)

            '''
            #print("VAUE OF odd row i",i)
            neighboursX[0] = get_value(z_est, i+(bound))       # upper left
            neighboursX[1] = get_value(z_est, i+(bound+1))     # upper right
            neighboursX[2] = get_value(z_est, i-1)             # extreme left
            neighboursX[3] = get_value(z_est, i)               # centre
            neighboursX[4] = get_value(z_est, i+1)             # extreme right
            neighboursX[5] = get_value(z_est, i-(bound))       #lower left
            neighboursX[6] = get_value(z_est, i-(bound-1))     # lower right
            
            neighboursY[0] = get_value(y_est,  i+(bound))
            neighboursY[1] = get_value(y_est, i+(bound+1))
            neighboursY[2] = get_value(y_est,  i-1 )
            neighboursY[3] = get_value(y_est, i)
            neighboursY[4] = get_value(y_est, i+1)
            neighboursY[5] = get_value(y_est, i-(bound) )
            neighboursY[6] = get_value(y_est, i-(bound-1))
            '''
            
            
        #print(neighboursA)
        #print(neighboursY)

        #print("X co-ordinates in hexagon neighbours",neighboursX)
        #print("X co-ordinates in hexagon neighbours",neighboursY)
        #print(neighboursX)
        #print(neighboursY)
       

        '''
        print(centreX_value)
        print(centreY_value)
        
        print("Range lenseeeeeeeeeeeeee is")
        print(range_lense)
        '''
        range_norm = normalised_single_data(range_lense)
        

        for j in range(7):
            
            #print(n_est_norm)
            count2 = 0 
            centreX1 = neighboursX[j]
            centreY1 = neighboursY[j]
            Angle1 = neighboursA[j]            

            #print(centreX1)
            #print(centreY1)
            #print(Angle1)
            neighbour2 = [0 for h in range(len(neighbour2))] 

            range_est = [0 for g in range(len(range_est))]

            

            for l in range(len(est)):
                for m in range(len(est)):
                    if(l==m):
                        check2 = (((z_est[l] - centreX1)*(z_est[l] - centreX1))  + ((y_est[l] - centreY1)*(y_est[l] - centreY1))) 
                       
                        
                        if(check2 <= 1100):
                            
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
            if(j==3):
                print("range_est_isssssssssssssssss")
                print(range_est)
                print(neighbour2)
            '''
            range_est_norm = normalised_single_data(range_est)

            '''
            print("Range2")
            print("range_est1 = %d range_est2=%d range_est3=%d range_est4=%d range_est5=%d" %(range_est_1,range_est_2,range_est_3,range_est_4,range_est_5))
            '''
            manhatten_distance[i][j] = 1*(abs(y_lense_norm[i] - value[j])) + 1*(abs(z_lense_norm[i]-value2[j])) + 1*(abs(n_lense_norm[i]- value3[j])) + 1*(abs(range_norm[0] - range_est_norm[0]) + abs(range_norm[1]-range_est_norm[1]) + abs(range_norm[2]-range_est_norm[2]) + abs(range_norm[3] - range_est_norm[3]) + abs(range_norm[4] - range_est_norm[4]))                                      
            #manhatten_distance[i][j] = abs(y_lense_norm[i] - value[j]) + abs(z_lense_norm[i]-value2[j]) + abs(n_lense_norm[i]- value3[j])                                    
            #manhatten_distance[i][j] = 0.3*(abs(range_norm[0] - range_est_norm[0]) + abs(range_norm[1]-range_est_norm[1]) + abs(range_norm[2]-range_est_norm[2]) + abs(range_norm[3] - range_est_norm[3]) + abs(range_norm[4] - range_est_norm[4]))                                      

            #euclidian_distance[i][j] = np.sqrt(np.square(y_lense_norm[i]- value[j]) + np.square(z_lense_norm[i] - value2[j])) + np.square(n_lense_norm[i] - value3[j])
            

    return manhatten_distance,euclidian_distance   




def main():
    
    #fin = "./data/cm_1024_51x51_1.5.pkl"
    #fin = "./data/cm_256_16x16_1.5.pkl"
    #fin = "./data/cm_128_8x8_1.5.pkl"
    #fin = "./data/cm_102_5x5_1.5.pkl"
    #fin = "./data/cm_256_8x8_2.0_8_1.1.pkl"
    #fin = "./data/cm_512_8x8_2.0_8_1.5.pkl"
    #fin = "./data/cm_256_8x8_2.0_8_1.1.pkl"
    #fin = "./data/cm_256_8x8_2.0_8_1.1.pkl"
    #fin = "./data/cm_512_16x16_2.0_8_1.1.pkl"
    #fin = "./data/cm_256_16x16_3.0_8_1.5.pkl"
    #fin = "./data/cm_256_16x16_2.0_8_1.5.pkl"
    #fin = "./data/cm_512_16x16_2.0_8_1.3.pkl"
    #fin = "./data/cm_102_5x5_2.0_8_2.5.pkl"
    #fin ="./data/cm_1024_51x51_2.0_8_5.1.pkl"
    #fin = "./data/cm_128_4x4_2.0_8_9.1.pkl"
    #fin = "./data/cm_512_16x16_2.0_8_1.9.pkl"
    fin = "./data/cm_1024_32x32_2.0_8_1.5.pkl"
    #fin = "./data/cm_512_16x16_2.0_8_1.9.pkl"
    #fin ="./data/cm_1024_51x51_2.0_8_5.1.pkl"
    #fin = "./data/cm_512_16x16_2.0_8_1.1.pkl"
    
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
    euclidian_distance = [[0]*7 for i in range(len(est))]
    manhatten_distance = [[0]*7 for i in range(len(est))]


    manhatten_distance,euclidian_distance = distance_fun(lens,est)
    
    
    #print(euclidian_distance) 
    print(manhatten_distance)
    minInRows_euclidian = np.argmin(euclidian_distance, axis=1)
    minInRows_manhatten = np.argmin(manhatten_distance, axis=1)
    #print("HELOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    #print(minInRows_euclidian)
    print(minInRows_manhatten)

    #print('min value of every Row: ', len(minInRows))
    for k in range(len(minInRows_euclidian)):
        #print("lense image %d and ellipse %d " % (k, minInRows[k]))
        if(4 == minInRows_euclidian[k]):
            count_euclidian = count_euclidian + 1
    euclidian_distance = np.array(euclidian_distance)
    #print("Number of matches_euclidian",(count_euclidian))
    


    #print('min value of every Row: ', len(minInRows))
    for k in range(len(minInRows_manhatten)):
        #print("lense image %d and ellipse %d " % (k, minInRows[k]))
        if(4 == minInRows_manhatten[k]):
            count_manhatten = count_manhatten + 1
    manhatten_distance = np.array(manhatten_distance)
    print("Number of matches_manhatten",(count_manhatten))
    


   
if __name__ == "__main__":
    t0= time.time()
    main()
    t1 = time.time() - t0
    print("Time elapsed: ", t1)
