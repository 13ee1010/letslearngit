from cgi import print_directory
from turtle import right
import numpy as np

def normalised(arr):

    norm_y = np.linalg.norm(arr)
    y_norm = arr / norm_y
     
    return y_norm

arr = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
arr_b = np.array([1.01,2.02,3.03,4.04,5.05,6.06,7.03,8.04,9.06,10.03,11.04,12.05,13.06,14.07,15.08,16.09,17.01,18.06,19.02,20.07,21.09,22.02,23.06,24.08,25.08])
arr_norm = normalised(arr)

arr_b_norm = normalised(arr_b)

neighbours = [0,0,0,0,0,0,0]
arr_2d = np.reshape(arr_norm, (5, 5))
arr_b_2d = np.reshape(arr_b_norm, (5,5))

#print(arr_2d)
#print(arr_b_2d)

manhatten_distance = [[0]*len(arr) for i in range(len(arr))]

count_1 = 0 
count_2 = 0
#print(arr_2d)

rows = len(arr_2d)
column = len(arr_2d[0])

def get_value(arr_2d,i,j):
    if((i >=5 or i < 0) or (j >= 5 or j < 0)):
        return 0
    else:
        return arr_2d[i][j]


for i in range(5):
    
    for j in range(5):
        
        #print(count_1)
        count_2 = 0
        range_1 = range_2 = range_3 = range_4 = range_5 = range_6 = range_7= range_8 = range_9 =  0
        if(i%2 == 0):
            neighbours[0] = get_value(arr_2d, i-1, j-1)
            neighbours[1] = get_value(arr_2d, i-1, j)
            neighbours[2] = get_value(arr_2d, i, j-1)
            neighbours[3] = get_value(arr_2d, i,j)
            neighbours[4] = get_value(arr_2d, i, j+1)
            neighbours[5] = get_value(arr_2d, i+1, j-1)
            neighbours[6] = get_value(arr_2d, i+1, j)
        else:
            neighbours[0] = get_value(arr_2d, i-1, j)
            neighbours[1] = get_value(arr_2d, i-1, j+1)
            neighbours[2] = get_value(arr_2d, i, j-1)
            neighbours[3] = get_value(arr_2d, i,j)
            neighbours[4] = get_value(arr_2d, i, j+1)
            neighbours[5] = get_value(arr_2d, i+1, j)
            neighbours[6] = get_value(arr_2d, i+1, j+1)
        
    

        for m in range(len(neighbours)):
            if (neighbours[m] >=0 and neighbours[m] <= 0.1) :
                range_1 = range_1 + 1
        

        for m in range(len(neighbours)):
            if(neighbours[m] > 0.1 and neighbours[m] <= 0.2):
                range_2 = range_2 + 1

        for m in range(len(neighbours)):
            if(neighbours[m] > 0.2 and neighbours[m] <= 0.3):
                range_3 = range_3 + 1

        for m in range(len(neighbours)):
            if(neighbours[m] > 0.3 and neighbours[m] <= 0.4):
                range_4 = range_4 + 1

        for m in range(len(neighbours)):
            if(neighbours[m] > 0.4 and neighbours[m] <= 0.5):
                range_5 = range_5 + 1

        for m in range(len(neighbours)):
            if(neighbours[m] > 0.5 and neighbours[m] <= 0.6):
                range_6 = range_6 + 1
        
        for m in range(len(neighbours)):
            if(neighbours[m] > 0.6 and neighbours[m] <= 0.7):
                range_7 = range_7 + 1
        
        for m in range(len(neighbours)):
            if(neighbours[m] > 0.7 and neighbours[m] <= 0.8):
                range_8 = range_8 + 1

        for m in range(len(neighbours)):
            if(neighbours[m] > 0.8 and neighbours[m] <= 0.9):
                range_9 = range_9 + 1
    

        #print(neighbours)
        #print("neighbourhood count % d %d %d %d %d %d %d %d %d" %(range_1,range_2,range_3,range_4,range_5,range_6,range_7, range_8, range_9))

        
        for k in range(5):
            for l in range(5):
                
                
                range_est_1 = range_est_2 = range_est_3 = range_est_4 = range_est_5 = range_est_6 = range_est_7 = range_est_8 = range_est_9 = 0
                if(k%2 == 0):
                    neighbours[0] = get_value(arr_b_2d, k-1, l-1)
                    neighbours[1] = get_value(arr_b_2d, k-1, l)
                    neighbours[2] = get_value(arr_b_2d, k, l-1)
                    neighbours[3] = get_value(arr_b_2d, k,l)
                    neighbours[4] = get_value(arr_b_2d, k, l+1)
                    neighbours[5] = get_value(arr_b_2d, k+1, l-1)
                    neighbours[6] = get_value(arr_b_2d, k+1, l)
                else:
                    neighbours[0] = get_value(arr_b_2d, k-1, l)
                    neighbours[1] = get_value(arr_b_2d, k-1, l+1)
                    neighbours[2] = get_value(arr_b_2d, k, l-1)
                    neighbours[3] = get_value(arr_b_2d, k,l)
                    neighbours[4] = get_value(arr_b_2d, k, l+1)
                    neighbours[5] = get_value(arr_b_2d, k+1, l)
                    neighbours[6] = get_value(arr_b_2d, k+1, l+1)
                    
                #print(neighbours)
                for m in range(len(neighbours)):
                    if(neighbours[m] >=0 and neighbours[m] <= 0.1):
                        range_est_1 = range_est_1 + 1

                for m in range(len(neighbours)):
                    if(neighbours[m] > 0.1 and neighbours[m] <= 0.2):
                        range_est_2 = range_est_2 + 1

                for m in range(len(neighbours)):
                    if(neighbours[m] > 0.2 and neighbours[m] <= 0.3):
                        range_est_3 = range_est_3 + 1

                for m in range(len(neighbours)):
                    if(neighbours[m] > 0.3 and neighbours[m] <= 0.4):
                        range_est_4 = range_est_4 + 1

                for m in range(len(neighbours)):
                    if(neighbours[m] > 0.4 and neighbours[m] <= 0.5):
                        range_est_5 = range_est_5 + 1

                for m in range(len(neighbours)):
                    if(neighbours[m] > 0.5 and neighbours[m] <= 0.6):
                        range_est_6 = range_est_6 + 1
            
                for m in range(len(neighbours)):
                    if(neighbours[m] > 0.6 and neighbours[m] <= 0.7):
                        range_est_7 = range_est_7 + 1
            
                for m in range(len(neighbours)):
                    if(neighbours[m] > 0.7 and neighbours[m] <= 0.8):
                        range_est_8 = range_est_8 + 1
            
                for m in range(len(neighbours)):
                    if(neighbours[m] > 0.8 and neighbours[m] <= 0.9):
                        range_est_9 = range_est_9 + 1
                
                
        
                #print(neighbours)
                #print("neighbourhood count2 % d %d %d %d %d %d %d %d %d" %(range_est_1,range_est_2,range_est_3,range_est_4,range_est_5,range_est_6,range_est_7, range_est_8, range_est_9))
                manhatten_distance[count_1][count_2] = abs(range_1 - range_est_1) + abs(range_2-range_est_2) + abs(range_3-range_est_3) + abs(range_4 - range_est_4) + abs(range_5 - range_est_5) + abs(range_6 - range_est_6) + abs(range_7 - range_est_7) + abs(range_8 - range_est_8) + abs(range_9 - range_est_9)
                count_2 = count_2 + 1

                #print("count 1 %d count2 %d" %(count_1, count_2))
        count_1 = count_1 + 1
        #print("count 1 %d count2 %d" %(count_1, count_2))
print(manhatten_distance) 



