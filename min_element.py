from cgi import print_directory
from turtle import right
import numpy as np

arr = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
arr_b = np.array([1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1,10.1,11.1,12.1,13.1,14.1,15.1,16.1,17.1,18.1,19.1,20.1,21.1,22.1,23.2,24.1,25.4])
index = [0,0,0,0,0,0,0,0,0]
value = [0,0,0,0,0,0,0,0,0]
#arr_2d = np.reshape(arr,(3,3))
manhatten_distance = [[0]*9 for i in range(len(arr))]
count = 0

bound = len(arr)
bound = int(np.sqrt(bound))
print("bound %d", bound)

def get_value(index,k,i):
    if (index[k] >= len(arr) or index[k] < 0):
        return 0
    else:
       return arr[index[k]]

for i in range(len(arr)):
    upper_right = i+(bound+1)
    upper_centre = i+(bound)
    upper_left = i+(bound-1)
    e_left = i-1
    e_right = i+1
    bottom_left = i-(bound+1)
    bottom_centre = i-(bound)
    bottom_right = i-(bound-1)
    
    index[0] = upper_left
    index[1] = upper_centre
    index[2] = upper_right
    index[3] = e_left
    index[4] = i
    index[5] = e_right
    index[6] = bottom_left
    index[7] = bottom_centre
    index[8] = bottom_right
    
  
    
    k = 0
    for j in range(9):
        value[k] = get_value(index,k,i)
        k= k+1
    
    if(i%(bound) == 0):
        value[0] = 0
        value[3] = 0
        value[6] = 0
    
    if(i%(bound) == (bound-1)):
        value[5] = 0
        value[8] = 0

    if(i==(bound-1)):
        value[2] = 0
    print(value)
    k = 0
    for j in range(9):
        manhatten_distance[i][j] = abs(arr[i] - value[j])
        
            
            
#print(manhatten_distance)


minInRows = np.argmin(manhatten_distance, axis=1)
#print(minInRows)
#print(minInRows[1])
for x in range(len(minInRows)):
        #print("lense image %d and ellipse %d " % (k, minInRows[k]))
        if(x == minInRows[x]):
            count = count+1
print("Number of matches",(count))
