from cgi import print_directory
from time import process_time_ns
from turtle import right
import numpy as np
import matplotlib.pyplot as plt

arr = np.array  ([1,1,1,1,1,  2,2,2,2,2,  3,3,3,3,3, 4,4,4,4,4, 5,5,5,5,5])
arr_b = np.array([0,1,2,3,4,  0.5,1.5,2.5,3.5,4.5,  0,1,2,3,4,  0.5,1.5,2.5,3.5,4.5, 0,1,2,3,4])
arr_n = np.array ([10,20,30,40,50,   60,70,80,90,100,  90,100,110,120,130,   140,150,160,170,20,  20,40,50,60,70])



arr_c = np.array([2,2,2,2,2,    3,3,3,3,3, 4,4,4,4,4, 5,5,5,5,5, 6,6,6,6,6])
arr_d = np.array([0,1,2,3,4,  0.5,1.5,2.5,3.5,4.5,  0,1,2,3,4,  0.5,1.5,2.5,3.5,4.5, 0,1,2,3,4])
arr_n1 = np.array ([10,20,30,40,50,   60,70,80,90,100,  90,100,110,120,130,   140,150,160,170,20,  20,40,50,60,70])



arr_2 = np.reshape(arr,(5,5))
arr_b2 = np.reshape(arr_b,(5,5))

check = 0.0
check2= 0.0
centreX = arr[2]
centreY = arr_b[2]

centreX1 = 0
centreY1 = 0
neighbour1 = [0,0,0,0,0,0,0,0,0,0,0]
neighbour2 = [0,0,0,0,0,0,0,0,0,0,0]

count1 = 0
count2 = 0
count3 = 0

count_manhatten = 0 

manhatten_distance = [[0]*len(arr) for i in range(len(arr))]
euclidian_distance = [[0]*len(arr) for i in range(len(arr))]

for k in range (len(arr)):
    count1 = 0
    centreX = arr[k]
    centreY = arr_b[k] 
    neighbour1 = [0,0,0,0,0,0,0,0,0,0,0]
    neighbour2 = [0,0,0,0,0,0,0,0,0,0,0]

    for i in range (len(arr)):
        for j in range(len(arr_b)):
            if(i==j):
                check = ((arr[i]) - (centreX))*((arr[i])-(centreX)) + ((arr_b[j])-(centreY))*((arr_b[j])-(centreY))
                #check2 = (((arr_c[i] - centreX1)*(arr_c[i] - centreX1))  + ((arr_d[j] - centreY1)*(arr_d[j] - centreY1))) 
                                                                                         
                if(check <= 1):
                    neighbour1[count1] = arr_n[i]
                    count1 = count1 + 1
                    
                    #print("neighbour1")
                    #print(neighbour1)
                    
    range_1 = range_2 = range_3 = range_4 = range_5 = range_6 = range_7= range_8 = range_9 = 0

    for m in range(len(neighbour1)):    
        if (neighbour1[m] >0 and neighbour1[m] <= 20) :
            range_1 = range_1 + 1
        

    for m in range(len(neighbour1)):
        if(neighbour1[m] > 20 and neighbour1[m] <= 40):
            range_2 = range_2 + 1

    for m in range(len(neighbour1)):
        if(neighbour1[m] > 40 and neighbour1[m] <= 60):
            range_3 = range_3 + 1

    for m in range(len(neighbour1)):
        if(neighbour1[m] > 60 and neighbour1[m] <= 80):
            range_4 = range_4 + 1

    for m in range(len(neighbour1)):
        if(neighbour1[m] > 80 and neighbour1[m] <= 100):
            range_5 = range_5 + 1

    for m in range(len(neighbour1)):
        if(neighbour1[m] > 100 and neighbour1[m] <= 120):
            range_6 = range_6 + 1
        
    for m in range(len(neighbour1)):
        if(neighbour1[m] > 120 and neighbour1[m] <= 140):
            range_7 = range_7 + 1
        
    for m in range(len(neighbour1)):
        if(neighbour1[m] > 140 and neighbour1[m] <= 160):
            range_8 = range_8 + 1

    for m in range(len(neighbour1)):
        if(neighbour1[m] > 160 and neighbour1[m] <= 180):
            range_9 = range_9 + 1

    #print("range1 = %d range 2=%d range3=%d range4=%d range5=%d range6=%d range7=%d range8=%d range9=%d"%(range_1,range_2,range_3,range_4,range_5,range_6,range_7,range_8,range_9))            

    for x in range(len(arr)):
        count2 = 0
        centreX1 = arr_c[x]
        centreY1 = arr_d[x]
        neighbour2 = [0,0,0,0,0,0,0,0,0,0,0]
        range_est_1 = range_est_2 = range_est_3 = range_est_4 = range_est_5 = range_est_6 = range_est_7 = range_est_8 = range_est_9 = 0
        for l in range(len(arr_c)):
            for m in range(len(arr_d)):
                if(l==m):
                    check2 = (((arr_c[l] - centreX1)*(arr_c[l] - centreX1))  + ((arr_d[m] - centreY1)*(arr_d[m] - centreY1)))    
                    if(check2 <= 1):
                        neighbour2[count2] = arr_n1[l] 
                        count2 = count2 + 1 
                        #print("neighbour12")
                        #print(neighbour2)       
            
        
        for m in range(len(neighbour2)):
            if(neighbour2[m] >0 and neighbour2[m] <= 20):
                range_est_1 = range_est_1 + 1

        for m in range(len(neighbour2)):
            if(neighbour2[m] > 20 and neighbour2[m] <= 40):
                range_est_2 = range_est_2 + 1

        for m in range(len(neighbour2)):
            if(neighbour2[m] > 40 and neighbour2[m] <= 60):
                range_est_3 = range_est_3 + 1

        for m in range(len(neighbour2)):
            if(neighbour2[m] > 60 and neighbour2[m] <= 80):
                range_est_4 = range_est_4 + 1

        for m in range(len(neighbour2)):
            if(neighbour2[m] > 80 and neighbour2[m] <= 100):
                range_est_5 = range_est_5 + 1

        for m in range(len(neighbour2)):
            if(neighbour2[m] > 100 and neighbour2[m] <= 120):
                range_est_6 = range_est_6 + 1
            
        for m in range(len(neighbour2)):
            if(neighbour2[m] > 120 and neighbour2[m] <= 140):
                range_est_7 = range_est_7 + 1
            
        for m in range(len(neighbour2)):
            if(neighbour2[m] > 140 and neighbour2[m] <= 160):
                range_est_8 = range_est_8 + 1
            
        for m in range(len(neighbour2)):
            if(neighbour2[m] > 160 and neighbour2[m] <= 180):
                range_est_9 = range_est_9 + 1
        
        #print("range_est1 = %d range_est2=%d range_est3=%d range_est4=%d range_est5=%d range_est6=%d range_est7=%d range_est8=%d range_est9=%d"%(range_est_1,range_est_2,range_est_3,range_est_4,range_est_5,range_est_6,range_est_7,range_est_8,range_est_9))
        manhatten_distance[k][x] = abs(range_1 - range_est_1) + abs(range_2-range_est_2) + abs(range_3-range_est_3) + abs(range_4 - range_est_4) + abs(range_5 - range_est_5) + abs(range_6 - range_est_6) + abs(range_7 - range_est_7) + abs(range_8 - range_est_8) + abs(range_9 - range_est_9)
        
    '''
    print("neighbour1")
    print(neighbour1)
    '''
        
    '''
    print("number of elememnts found in a circle")
    print(count1)

    print("number of elemnets found in circle 2")
    print(count2)

    '''

print(manhatten_distance)

minInRows_manhatten = np.argmin(manhatten_distance, axis=1)
print(minInRows_manhatten)

for k in range(len(minInRows_manhatten)):
    if(k == minInRows_manhatten[k]):
            count_manhatten = count_manhatten + 1
    manhatten_distance = np.array(manhatten_distance)
print("Number of matches_manhatten",(count_manhatten))

'''
figure, axes = plt.subplots()

draw_circle = plt.Circle((centreX, centreY), 1,fill=False)
#draw_circle = plt.Circle((centreX1, centreY1), 1,fill=False)
axes.set_aspect(1)
axes.add_artist(draw_circle)

#plt.plot(arr_c, arr_d, 'ro')
plt.plot(arr, arr_b, 'ro')
plt.show()
'''