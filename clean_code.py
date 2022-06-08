#
import pickle
import numpy as np
import matplotlib.pyplot as plt


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

def distance_fun(lens, est):
    y_lense, z_lense, n_lense = lens[:,0], lens[:,1], lens[:,2]
    y_est, z_est, n_est = est[:,0], est[:,1], est[:,2]
    for i in range(len(n_est)):
        if( n_est[i] > 175.0):
            n_est[i] = np.sin(np.deg2rad(n_est[i]))

    y_lense_norm, z_lense_norm, n_lense_norm = normalised(y_lense,z_lense,n_lense)
    y_est_norm, z_est_norm, n_est_norm = normalised(y_est, z_est, n_est)

    manhatten_distance = [[0]*len(est) for i in range(len(est))]
    euclidian_distance = [[0]*len(est) for i in range(len(est))]
    for i in range(len(est)):
        for j in range(len(est)):
            manhatten_distance[i][j] = abs(y_lense_norm[i]-y_est_norm[j]) + abs(z_lense_norm[i] - z_est_norm[j]) + (abs(n_lense_norm[i] - n_est_norm[j]))
            euclidian_distance[i][j] = np.sqrt(((np.square((y_lense_norm[i]-y_est_norm[j]))) + ((np.square(z_lense_norm[i] - z_est_norm[j]  )))  + 1*(np.square(n_lense_norm[i] - n_est_norm[j]))))
    return manhatten_distance, euclidian_distance     
            
def euclidian_distance_fun(lens,est):
    y_lense, z_lense, n_lense = lens[:,0], lens[:,1], lens[:,2]
    y_est, z_est, n_est = est[:,0], est[:,1], est[:,2]

    y_lense_norm, z_lense_norm, n_lense_norm = normalised(y_lense,z_lense,n_lense)
    y_est_norm, z_est_norm, n_est_norm = normalised(y_est, z_est, n_est)
    
    euclidian_distance = [[0]*len(est) for i in range(len(est))]

    for i in range(len(est)):
        for j in range(len(est)):
            euclidian_distance[i][j] = np.sqrt(((np.square((y_lense_norm[i]-y_est_norm[j]))) + ((np.square(z_lense_norm[i] - z_est_norm[j]  )))  + 1*(np.square(n_lense_norm[i] - n_est_norm[j]))))
    
    return euclidian_distance



def main():
    #fin = "./data/cm_1024_51x51_1.5.pkl"
    fin = "./data/cm_256_16x16_1.5.pkl"
    #fin = "./data/cm_128_8x8_1.5.pkl"
    #fin = "./data/cm_102_5x5_1.5.pkl"
    data = read_pickle(fin)
    lens, est, img_size = data['lens'], data['est'], data['img_size']
    count_euclidian = 0
    count_manhatten = 0
    fig, axs = plt.subplots(2)
    
    y_lense, z_lense, n_lense = lens[:,0], lens[:,1], lens[:,2]
    y_lense_norm, z_lense_norm, n_lense_norm = normalised(y_lense,z_lense,n_lense)
    
    y, z, n = est[:,0], est[:,1], est[:,2]

    print("len est",np.sqrt(len(est)))
    length = int(np.sqrt(len(est)))
    print("printing length")
    print(length)
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
    euclidian_distance = [[0]*len(est) for i in range(len(est))]
    manhatten_distance = [[0]*len(est) for i in range(len(est))]


    manhatten_distance,euclidian_distance = distance_fun(lens,est)
    #euclidian_distance = euclidian_distance_fun(lens,est)

   
    minInRows_euclidian = np.argmin(euclidian_distance, axis=1)
    minInRows_manhatten = np.argmin(manhatten_distance, axis=1)
    #print(minInRows_euclidian)
    #print(minInRows_manhatten)

    #print('min value of every Row: ', len(minInRows))
    for k in range(len(minInRows_euclidian)):
        #print("lense image %d and ellipse %d " % (k, minInRows[k]))
        if(k == minInRows_euclidian[k]):
            count_euclidian = count_euclidian + 1
    euclidian_distance = np.array(euclidian_distance)
    print("Number of matches_euclidian",(count_euclidian))
    


    #print('min value of every Row: ', len(minInRows))
    for k in range(len(minInRows_manhatten)):
        #print("lense image %d and ellipse %d " % (k, minInRows[k]))
        if(k == minInRows_manhatten[k]):
            count_manhatten = count_manhatten + 1
    manhatten_distance = np.array(manhatten_distance)
    print("Number of matches_manhatten",(count_manhatten))
   


   
    #plt.show()

    #y, z, n = lens[:,0], lens[:,1], lens[:,2]
    """
    axs[2].scatter(z, y)
    axs[2].title.set_text("Lens positions")
    for i, txt in enumerate(n):
        axs[0].annotate(txt, (z[i], y[i]))

    axs[0].scatter(z_lense, y_lense)
    axs[0].title.set_text("Lens positions")
    for i, txt in enumerate(n):
        axs[0].annotate(txt, (z_lense[i], y_lense[i]))

    y, z, n = est[:,0], est[:,1], est[:,2]
    axs[1].scatter(z_est, y_est)
    axs[1].title.set_text("Laserspot positions")
    for i, txt in enumerate(n):
        axs[1].annotate("%.1f"%txt, (z_est[i], y_est[i]))

    plt.show()
    """



if __name__ == "__main__":
    main()
