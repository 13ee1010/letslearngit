from cgi import print_directory
import numpy as np

numbers = np.array([[1,2,3],[3,2,4],[1,4,6]])
manhatten_distance = [[0]*9 for i in range(9)]
print(numbers)
A = np.arange(9)
A = A.reshape(3,3)
count = 0
trace = 0
print(A)
b = 0
#index = [5,0]
num_neighbor = 1
for i in range (3):
    for j in range(3):
        left = max(0,i-num_neighbor)
        right = max(0,i+num_neighbor+1)
        bottom = max(0,j-num_neighbor)
        top = max(0,j+num_neighbor+1)
        
        #print(count)
        if(left > 3):
            left = 3
        if(right > 3):
            right = 3
        if(bottom > 3):
            bottom = 3
        if(top > 3):
            top = 3
        trace = 0  # trace = total number of neighbours 
        for k in range(left,right):
            for l in range(bottom,top):
                manhatten_distance[count][trace] = abs(numbers[i][j] - A[k][l])
                #print(trace)
                trace = trace + 1
        count = count + 1         
print(manhatten_distance)