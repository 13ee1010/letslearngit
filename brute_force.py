import numpy as np

 # first set 
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([[2,4,6],[8,10,12],[14,16,18]])

y_lense, z_lense, n_lense = a[:,0], a[:,1], a[:,2]

y_est, z_est, n_est = b[:,0], b[:,1], b[:,2] 

print("printing y_lense")
print(y_lense)

print("printing z_lense")
print(z_lense)

print("printing n_lense")
print(n_lense)

print("printing y_est")
print(y_est)

print("printing z_est")
print(z_est)

print("printing n_est")
print(n_est)


"""
print("Printing A ---------")
print(a)
print("Printing B --------")
print(b)
print(a[0][1])
"""

euclidian_distance = [[0]*len(a) for i in range(len(a))]
print(euclidian_distance)

""""
for i in range (len(a)):
    for j in range (len(a)):
        for k in range (len(a)):
            euclidian_distance[i][j] = euclidian_distance[i][j] +  abs(a[i][k] - b[j][k])
print("printing euclidian distance--------------")                                
print(euclidian_distance)        
"""

for i in range(len(a)):
    for j in range(len(a)):
        euclidian_distance[i][j] = abs(y_lense[i]-y_est[j]) + abs(z_lense[i] - z_est[j]) + abs(n_lense[i] - n_est[j])

print("printing euclidian matrix")
print(euclidian_distance)
# find the minimum value in a row 
minInRows = np.argmin(euclidian_distance, axis=1)
print('min value of every Row: ', minInRows)



