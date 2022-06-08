from cgi import print_directory
from time import process_time_ns
from turtle import right
import numpy as np
import matplotlib.pyplot as plt
import pickle


def read_pickle(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        # --
    return data

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

#fin = "./data/cm_1024_51x51_1.5.pkl"
#fin = "./data/cm_256_16x16_1.5.pkl"
fin = "./data/cm_128_4x4_2.0_8_1.3.pkl"
#fin = "./data/cm_102_5x5_1.5.pkl"
#fin = "./data/cm_256_8x8_2.0_8_2.0.pkl"
#fin = "./data/cm_512_8x8_2.0_8_1.5.pkl"
data = read_pickle(fin)

length = 0 
lens, est, img_size = data['lens'], data['est'], data['img_size']

y, z, n = lens[:,0], lens[:,1], lens[:,2]
y_lense, z_lense, n_lense = normalised(y,z,n)
 
print(z_lense)

print(y_lense)
y, z, n = est[:,0], est[:,1], est[:,2]
length = int(np.sqrt(len(est)))
   
for i in range(length):
    a_trim = est[i*length:i*length+length]
    #print(a_trim)
    for i in range(length):
        for k in range(i+1,length):
            if a_trim[i][1] > a_trim[k][1]:
                a_trim[i][1],a_trim[k][1] = a_trim[k][1],a_trim[i][1]
                a_trim[i][2],a_trim[k][2] = a_trim[k][2],a_trim[i][2]
                a_trim[i][0],a_trim[k][0] = a_trim[k][0],a_trim[i][0]

y_est,z_est,n_est = est[:,0], est[:,1], est[:,2] 
y_est,z_est,n_est = normalised(y,z,n)


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

neighbour1 = [0] * 100
neighbour2 = [0] * 100

count1 = 0
count2 = 0

for k in range(1):
    count1 = 0
    count2 = 0
    centreX = z_lense[k]
    centreY = y_lense[k]
    
    
    centreX1 = z_est[k]
    centreY1 = y_est[k]
    
    neighbour1 = [0 for i in range(len(neighbour1))] 
    neighbour2 = [0 for i in range(len(neighbour2))] 

    print("centreX",centreX)
    print("centerY",centreY)
    print("----------------")
    print("centreX1",centreX1)
    print("centreY1",centreY1)
    print("-----------------")
    
    for i in range (1):
        for j in range(len(est)):
            if(i==j):
                check = ((z_lense[i]) - (centreX))*((z_lense[i])-(centreX)) + ((y_lense[j])-(centreY))*((y_lense[j])-(centreY))
                check2 = (((z_est[i] - centreX1)*(z_est[i] - centreX1))  + ((y_est[j] - centreY1)*(y_est[j] - centreY1))) 
                                                                                         
                if(check <= 0.000625):
                    #neighbour1[count1] = arr_n[i]
                    count1 = count1 + 1
                    
                
                if(check2 <= 0.000625):
                    #neighbour2[count2] = arr_n1[i]
                    count2 = count2 + 1


    print("neighbour1")
    print(neighbour1)

    print("neighbours2")
    print(neighbour2)

    print("number of elememnts found in a circle")
    print(count1)

    print("number of elemnets found in circle 2")
    print(count2)

fig, axs = plt.subplots()

#y, z, n = est[:,0], est[:,1], est[:,2]
draw_circle = plt.Circle((centreX1, centreY1), 0.025,fill=False)
axs.set_aspect(1)
axs.add_artist(draw_circle)

plt.plot(z_est, y_est, 'ro')

'''
axs[1].scatter(z, y)
axs[1].title.set_text("Lens positions")
for i, txt in enumerate(n):
    axs[0].annotate(txt, (z[i], y[i]))

y, z, n = est[:,0], est[:,1], est[:,2]
    
for i in range(len(n)):
    if( n[i] > 175.0):
        n[i] = np.sin(np.deg2rad(n[i]))

 
axs[2].scatter(z, y)
axs[2].title.set_text("Laserspot positions")
for i, txt in enumerate(n):
    axs[2].annotate("%.1f"%txt, (z[i], y[i]))

'''
plt.show()

    




#plt.plot(arr_c, arr_d, 'ro')
#plt.plot(arr, arr_b, 'ro')

