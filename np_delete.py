import numpy as np 

a = np.array([[12,6,15],[16,2,12],[8,1,9],[4,5,3],[2,3,2],[1,2,1]])
print(a)
#a_trim = a[0:3,]
print("---------------------------------------")
#print(a_trim)
print("---------------------------------------")
#b_trim = a[3:6,]
#print(b_trim)
for i in range(2):
    a_trim = a[i*3:i*3+3]
    print(a_trim)
    for i in range(3):
        for k in range(i+1,3):
            if a_trim[i][1] > a_trim[k][1]:
                a_trim[i][1],a_trim[k][1] = a_trim[k][1],a_trim[i][1]

print("----------------------------------------")
print(a)