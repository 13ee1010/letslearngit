from cgi import print_directory
from turtle import right
import numpy as np

arr = np.array([1,2,3,4,5,6,7,8,9])
arr_b = np.array([1.1,2.2,3.3,4.4,5.5,6.6,7.3,8.4,9.6])
index = [0,0,0,0,0,0,0,0,0]
value_lense = [0,0,0,0,0,0,0,0,0]
value_est = [0,0,0,0,0,0,0,0,0]
arr_2d = np.reshape(arr,(3,3))
manhatten_distance = [[0]*9 for i in range(9)]
euclidian_distance = [[0]*9 for i in range(9)]
bound = len(arr)
bound = int(np.sqrt(bound))
print(bound)
count_euclidian = 0
count_manhatten = 0

'''
def get_value(index,k):
    if (index[k] >=9 or index[k] < 0):
        return 0
    else:
       return arr_b[index[k]]
'''
def get_value(arr,i):
    if(i >=9 or i < 0):
        return 0
    else:
        return arr[i]

for i in range(9):
    upper_right = get_value(arr,i+(bound+1))
    upper_centre = get_value(arr,i+bound)
    upper_left = get_value(arr,i+(bound-1))
    e_left = get_value(arr,i-1)
    centre = get_value(arr,i)
    e_right = get_value(arr,i+1)
    bottom_left = get_value(arr,i-(bound+1))
    bottom_centre = get_value(arr, i-(bound))
    bottom_right = get_value(arr,i-(bound-1))
    
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
    for j in range(9):

        upper_right_est = get_value(arr_b,j+(bound+1))
        upper_centre_est = get_value(arr_b,j+bound)
        upper_left_est = get_value(arr_b,j+(bound-1))
        e_left_est = get_value(arr_b,j-1)
        centre_est = get_value(arr_b,j)
        e_right_est = get_value(arr_b,j+1)
        bottom_left_est = get_value(arr_b,j-(bound+1))
        bottom_centre_est = get_value(arr_b,j-(bound))
        bottom_right_est = get_value(arr_b,j-(bound-1))

        if(j%(bound)==0):
            upper_left_est = 0
            e_left_est = 0
            bottom_left_est = 0

        if(j%(bound) == (bound-1)):
            upper_right_est = 0
            e_right_est = 0
            bottom_right_est = 0
        '''
        print("first matrix")
        print(upper_left)
        print(upper_centre)
        print(upper_right)
        print(e_left)
        print(centre)
        print(e_right)
        print(bottom_left)
        print(bottom_centre)
        print(bottom_right)

        print("SECOND MATRIX")
        print(upper_left_est)
        print(upper_centre_est)
        print(upper_right_est)
        print(e_left_est)
        print(centre_est)
        print(e_right_est)
        print(bottom_left_est)
        print(bottom_centre_est)
        print(bottom_right_est)
        '''

        euclidian_distance[i][j] = np.sqrt(np.square(upper_right-upper_right_est) + np.square(upper_centre-upper_centre_est) + np.square(upper_left-upper_left_est) + np.square(e_left-e_left_est) + np.square(centre-centre_est) + np.square(e_right-e_right_est)+ np.square(bottom_left-bottom_left_est) + np.square(bottom_centre-bottom_centre_est) + np.square(bottom_right-bottom_right_est))
        manhatten_distance[i][j] = abs(upper_right - upper_right_est) + abs(upper_centre - upper_centre_est) + abs(upper_left-upper_left_est) + abs(e_left-e_left_est) + abs(centre-centre_est) + abs(e_right-e_right_est) + abs(bottom_left-bottom_left_est) + abs(bottom_centre-bottom_centre_est) + abs(bottom_right-bottom_right_est)
        #print("distance is")
        #print(distance)
        
            
            
print(manhatten_distance)
print(euclidian_distance)


minInRows_manhatten = np.argmin(manhatten_distance, axis=1)
#print(minInRows_manhatten)
#print(minInRows[1])
for x in range(len(minInRows_manhatten)):
        #print("lense image %d and ellipse %d " % (k, minInRows[k]))
        if(x == minInRows_manhatten[x]):
            count_manhatten = count_manhatten+1
#print("Number of matches",(count_manhatten))


minInRows_euclidian = np.argmin(euclidian_distance, axis=1)
#print(minInRows_euclidian)
#print(minInRows[1])
for x in range(len(minInRows_euclidian)):
        #print("lense image %d and ellipse %d " % (k, minInRows[k]))
        if(x == minInRows_euclidian[x]):
            count_euclidian = count_euclidian + 1
#print("Number of matches",(count_manhatten))
