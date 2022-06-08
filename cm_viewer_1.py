#
import pickle
import numpy as np
import matplotlib.pyplot as plt


def read_pickle(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        # --
    return data


def main():
    #fin = "./data/cm_1024_51x51_1.5.pkl"
    #fin = "./data/cm_256_16x16_1.5.pkl"
    #fin = "./data/cm_128_8x8_1.5.pkl"
    #fin = "./data/cm_102_5x5_1.5.pkl"
    fin = "./data/cm_256_8x8_2.0_8_2.0.pkl"
    data = read_pickle(fin)
    lens, est, img_size = data['lens'], data['est'], data['img_size']

    #print("="*80)
    #print(" ... microlens position in physical unit (e.g., um), array size %dx%d :"%(lens.shape[0],lens.shape[1]))
    #print(lens)

    #print("-"*80)
    #print(" ... estimated ellipse positions in pixel unit, img size %dx%d, array size %dx%d :"%(img_size[0],img_size[1],est.shape[0],est.shape[1]))
    #print(est)

    #print("-"*80)
    #print(" ... visualize it")

    fig, axs = plt.subplots(2)
    
    y_lense, z_lense, n_lense = lens[:,0], lens[:,1], lens[:,2]

    print(z_lense) 
    #print("standardised y of lense--------")
    y_standard = (y_lense - np.average(y_lense)) / (np.std(y_lense))
    #print(y_standard)

    #print("standardised z of lense--------")
    z_standard = (z_lense - np.average(z_lense)) / (np.std(z_lense))
    #print(z_standard)

    #print("standardised n of lense--------")
    n_standard = (n_lense - np.average(n_lense)) / (np.std(n_lense))
    #print(n_standard)
    

    #print("Norm printing lens -----------------------")
    #print(n)
    norm_y_lense = np.linalg.norm(y_lense)
    y_lense = y_lense / norm_y_lense
    #print("normalised y of lense--------")
    #print(y_lense)
   

    norm_z_lense = np.linalg.norm(z_lense)
    z_lense = z_lense / norm_z_lense
    print("normalised z of lense--------")
    print(z_lense)

    norm_n_lense = np.linalg.norm(n_lense)
    n_lense = n_lense / norm_n_lense
    #print("normalised n of lense--------")
    #print(n_lense)
    
    print("printing lense ----------")
    #print(len(lens))
    

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
    #print("Modified est printing------------------------------------")
    
    y_est, z_est, n_est = est[:,0], est[:,1], est[:,2]
    
    #print(y_est)
    #print("LETS  Shifted y ellipse")
    #print(z_est)

    #print("standardised y of lense--------")
    y_est_standard = (y_est - np.average(y_est)) / (np.std(y_est))
    #print(y_est_standard)

    #print("standardised z of lense--------")
    z_est_standard = (z_est - np.average(z_est)) / (np.std(z_est))
    #print(z_est_standard)

    #print("standardised n of lense--------")
    n_est_standard = (n_est - np.average(n_est)) / (np.std(n_est))
    #print(n_est_standard)



    #print(est)
    norm_y_est = np.linalg.norm(y_est)
    y_est = y_est / norm_y_est
    #print("normalised y of laser--------")
    #print(y_est)

    norm_z_est = np.linalg.norm(z_est)
    z_est = z_est / norm_z_est
    print("normalised z of lense--------")
    print(z_est)

    norm_n_est = np.linalg.norm(n_est)
    n_est = n_est / norm_n_est
    #print("normalised n of lense--------")
    #print(n_est)

    #print("printing ellipse")
    #print(est)
    #print("printing length of est")
    #print(len(n))
    
    euclidian_distance = [[0]*len(est) for i in range(len(est))]
    manhatten_distance = [[0]*len(est) for i in range(len(est))]
    euclidian_distance_std = [[0]*len(est) for i in range(len(est))]
    manhatten_distance_std = [[0]*len(est) for i in range(len(est))]

    """"
    for i in range(len(est)):
        euclidian_distance[i] =  np.sqrt((np.square(y_lense[0]-y_est[i])  + np.square(z_lense[0] - z_est[i])  + np.square(n_lense[0] - n_est[i])))
        #manhatten_distance[i] =  abs(y_lense[i]-y_est[i]) + abs(z_lense[i] - z_est[i]) + abs(n_lense[i] - n_est[i])
    """
    for i in range(len(est)):
        for j in range(len(est)):
            manhatten_distance[i][j] = abs(y_lense[i]-y_est[j]) + abs(z_lense[i] - z_est[j]) + (abs(n_lense[i] - n_est[j]))
            euclidian_distance[i][j] = np.sqrt(( (np.square(   (y_lense[i]-y_est[j] + 0.00  ))) + ((np.square(z_lense[i] - z_est[j] + 0.00 )))  + 0.1*(np.square(n_lense[i] - n_est[j]))))
            #euclidian_distance[i][j] = np.sqrt((((np.square(y_lense[i]-y_est[j])))))
    
    for i in range(len(est)):
        for j in range(len(est)):
            manhatten_distance_std[i][j] = abs(y_standard[i]-y_est_standard[j]) + abs(z_standard[i] - z_est_standard[j]) + abs(n_standard[i] - n_est_standard[j])
            euclidian_distance_std[i][j] = np.sqrt((np.square(y_standard[i]-y_est_standard[j])  + np.square(z_standard[i] - z_est_standard[j])  + np.square(n_standard[i] - n_est_standard[j])))

    #print("PRINTING EUCLIDIAN DISTANCE HERE")
    #print(euclidian_distance)
    count=0
    minInRows = np.argmin(euclidian_distance, axis=1)
    #print(minInRows)
    #print('min value of every Row: ', len(minInRows))
    for k in range(len(minInRows)):
        #print("lense image %d and ellipse %d " % (k, minInRows[k]))
        if(k == minInRows[k]):
            count = count+1
    euclidian_distance = np.array(euclidian_distance)
    #print("Number of matches",(count))
   
    #print("Manhatten distance")
    #print(euclidian_distance)

    minInRows_std = np.argmin(manhatten_distance_std, axis=1)
    # print('min value of every Row: ', minInRows_std)
    manhatten_distance_std = np.array(manhatten_distance_std)
    #sizes = (np.random.sample(size=z.size) * 10) ** 2
    #axs[1].scatter(z, y)
    #axs[1].title.set_text("Laserspot positions")
    #for i, txt in enumerate(n):
    #    axs[1].annotate("%.1f"%txt, (z[i], y[i]))

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
