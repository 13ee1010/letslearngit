import numpy as np

A = np.arange(9)
A = A.reshape(3,3)

print(A)

index = [1,1]
print(A[1,1])

num_neighbor = 1

left = max(0,index[0]-num_neighbor)
right = max(0,index[0]+num_neighbor+1)

bottom = max(0,index[1]-num_neighbor)
top = max(0,index[1]+num_neighbor+1)

sample = A[left:right,bottom:top]
print(sample)