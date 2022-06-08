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

    # fin = "./data/cm_1024_51x51_1.5.pkl"
    #fin = "./data/cm_256_16x16_1.5.pkl"
    #fin = "./data/cm_128_8x8_1.5.pkl"
    fin = "./data/cm_102_5x5_1.5.pkl"
    #fin = "./data/cm_128_4x4_2.0_8_1.9.pkl"
    #fin = "./data/cm_256_8x8_2.0_8_2.0.pkl"
    #fin = "./data/cm_512_8x8_2.0_8_1.5.pkl"
    #fin = "./data/cm_256_8x8_1.0_8_2.5.pkl"
    #fin = "./data/cm_256_8x8_2.0_8_5.0.pkl"
    #fin = "./data/cm_256_8x8_2.0_8_7.5.pkl"
    #fin = "./data/cm_256_8x8_2.0_8_1.2.pkl"
    #fin = "./data/cm_512_16x16_2.0_8_0.9.pkl"
    #fin = "./data/cm_512_16x16_2.0_16_10.1.pkl"
    #fin = "./data/cm_512_16x16_2.0_8_1.9.pkl"
    #fin = "./data/cm_102_5x5_2.0_8_2.5.pkl"
    #fin = "./data/cm_256_16x16_2.0_8_100.0.pkl"
    #fin ="./data/cm_1024_51x51_2.0_8_5.1.pkl"
    #fin = "./data/cm_256_16x16_2.0_8_5.0.pkl"
    #fin = "./data/cm_1024_32x32_2.0_8_1.1.pkl"
    fin = "./data/cm_128_4x4_2.0_8_9.1.pkl"

    data = read_pickle(fin)
    lens, est, img_size = data['lens'], data['est'], data['img_size']

    #print("="*80)
    #print(" ... microlens position in physical unit (e.g., um), array size %dx%d :"%(lens.shape[0],lens.shape[1]))
    print(lens)

    #print("-"*80)
    #print(" ... estimated ellipse positions in pixel unit, img size %dx%d, array size %dx%d :"%(img_size[0],img_size[1],est.shape[0],est.shape[1]))
    print(est)

    #print("-"*80)
    #print(" ... visualize it")

    fig, axs = plt.subplots(2)

    
    y, z, n = lens[:,0], lens[:,1], lens[:,2]
    print(len(y))
    axs[0].scatter(z, y)
    axs[0].title.set_text("Lens positions")
    for i, txt in enumerate(n):
        axs[0].annotate(txt, (z[i], y[i]))
    
    
    y, z, n = est[:,0], est[:,1], est[:,2]
    print(len(y))
    for i in range(len(n)):
        if( n[i] > 175.0):
            n[i] = np.sin(np.deg2rad(n[i]))

    
    axs[1].scatter(z, y)
    axs[1].title.set_text("Laserspot positions")
    for i, txt in enumerate(n):
        axs[1].annotate("%.1f"%txt, (z[i], y[i]))
    
    '''
    y, z, n = est[:,0], est[:,1], est[:,2]
    y_shifted = y+10
    z_shifted = z-10
    axs[1].scatter(z_shifted, y_shifted)
    axs[1].title.set_text("Laserspot positions shifted")
    for i, txt in enumerate(n):
        axs[1].annotate("%.1f"%txt, (z_shifted[i], y_shifted[i]))
    '''
    plt.show()



if __name__ == "__main__":
    main()
