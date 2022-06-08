# Trung-Hieu Tran @IPVS
# 210916

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
# import imutils
import math
import scipy.misc
import imageio
import cv2
import pickle
import h5py
import os
import utils
import cca_computation_no_fp as alg_nofp

def generate_ellipse_rand_shift(ny, nx, array, conf):
    '''
    Generate elipses which fall within the circle in hexagon arrangement.
    input:
      + ny, nx :  num of lenses in y, x directions
      + array: (ny*nx)x3, each row [ iy, ix, angle]
      + width: width of output image in pixel
    '''
    # estimate radius in pixel
    # W = r+2 r nx
    # H = 2r + (ny-1) \sqrt(3) r
    width = conf['width']
    r = width/(2*nx+1)
    H = math.ceil(2*r + (ny-1) * math.sqrt(3)*r)
    W = width
    print(" Image size %dx%d, lens array size %dx%d, radius of a lens %d"%(H,W,ny,nx,r))
    # elipse size
    a = int(conf['a'])
    b = int(a*1./conf['eratio'])
    print(' ... ellipse size: %dx%d'%(2*a,2*b))
    img = np.zeros([H,W])
    params = []
    num = len(array)
    rshift = np.random.rand(num,2) * r/conf['local_shift_factor']
    print(" num: ", num, " r: ", r)
    for ii, val in enumerate(array):
        dy,dx = rshift[ii,0], rshift[ii,1]
        cy,cx = int(val[0]*r + dy), int(val[1]*r + dx)
        angle = val[2]
        img = cv2.ellipse(img,
                          (cx,cy),
                          (a,b),
                          angle,
                          0,360, color=1, thickness=-1)
        params.append([cx,cy,angle,a,b])
    return img, params


def build_lenses_list():
    '''
    build a list of lenses'cogs
    read in the lens file 'vcsel.mat'
    lenses are organized in hexagon form.
    rotation angle is (val-1)*10
    assuming radius is 1
    return:
       + ny, nx: number of lenses by y and x direciont
       + an array Nx3, N is number of lenses, each has [y, x, angle]
    '''
    fmat = "./vcsel.mat"
    mat = h5py.File(fmat,'r')
    array = np.asarray(mat['bg'])
    array = array.T
    H,W = array.shape[:2]
    ny, nx = H, int(W/2) # number of lense by y and by x
    llenses = []
    for iy in range(ny):
        start_y = 1 + iy*math.sqrt(3)
        row = array[iy,:]
        if iy%2==0:
            row = row[::2]
            start_x = 1
        else:
            row = row[1::2]
            start_x = 2
        for ix,val in enumerate(row):
            item = [start_y, start_x+ix*2, (val-1)*10]
            llenses.append(item)
    alenses = np.asarray(llenses)
    return ny, nx, alenses


def get_lenses_array(conf):
    '''
    config: 
        conf = {}
        conf['width'] = 64
        conf['eratio'] = 2 # ratio of major and minor axis.
        conf['max_ny'] = 4 # <1 mean no change, >1 indicate number of lens in y dir
        conf['max_nx'] = 4 # <1 mean no change, >1 indicate number of lens in x dir
    '''
    # get an array of lenses
    ny, nx, alenses = build_lenses_list()
    # set number of lenese
    nny = ny if conf['max_ny']<1 else conf['max_ny']
    nnx = nx if conf['max_nx']<1 else conf['max_nx']
    array = alenses.reshape([ny,nx,3])
    array = array[:nny,:nnx,:]
    array = array.reshape([-1,3])
    return nny, nnx, array


def main():
    # load image + ground
    #512_16x16_2.0_8_1.3.pkl"
    conf = {}
    conf['output'] = "./data"
    conf['width'] = 1024
    conf['eratio'] = 2.0 # ratio of major and minor axis.
    conf['max_ny'] = 32# <1 mean no change, >1 indicate number of lens in y dir
    conf['max_nx'] = 32 # <1 mean no change, >1 indicate number of lens in x dir
    conf['local_shift_factor'] = 1.5 # small -> more overlapping, large less
    # overlapping
    conf['a'] = 8 # number of pixel of major axis
    conf['seed'] = 111 # random seed

    np.random.seed(conf['seed'])

    if not os.path.exists(conf['output']):
        os.makedirs(conf['output'])

    nny, nnx, arr_lens = get_lenses_array(conf)
    img, params = generate_ellipse_rand_shift(nny,nnx, arr_lens, conf)
    print("Paramassssssssssssssss")
    params = np.asarray(params,dtype=np.float)
    print(params)


    tmp_params = [list(params[ii])+[ii] for ii in range(len(params))]
    # print(tmp_params)
    utils.cca_show(img, img, img, tmp_params, color='r')

    acc_mem, connected, LABEL, lbl_img = alg_nofp.cca_lbl_1st_step(img, max_lbl_width=20)
    acc_mem, connected = alg_nofp.cca_lbl_2nd_step(acc_mem, connected, LABEL)
    est_params, lbl_img_final = alg_nofp.cca_extract_info(acc_mem, LABEL, lbl_img)
    est_params = np.asarray(est_params,dtype=np.float)
    print("estimated parameterssss")
    print(est_params)


    utils.cca_show(img, lbl_img, lbl_img_final,est_params, color='r')

    arr_lens = np.asarray(arr_lens,dtype=float)
    print("Arrayyyy lenssssssssss")
    print(arr_lens)
    
    est_params = np.asarray(est_params,dtype=float)

    print("Printing length of estimated parameters")
    print(len(est_params))
    length = int(np.ceil((np.sqrt(len(est_params)))))
    print(length)

    for i in range(length):
        a_trim = est_params[i*length:i*length+length]
        for i in range(length):
            for k in range(i+1,length):
                if a_trim[i][0] > a_trim[k][0]:
                    a_trim[i][1],a_trim[k][1] = a_trim[k][1],a_trim[i][1]
                    a_trim[i][2],a_trim[k][2] = a_trim[k][2],a_trim[i][2]
                    a_trim[i][0],a_trim[k][0] = a_trim[k][0],a_trim[i][0]
                    a_trim[i][3],a_trim[k][3] = a_trim[k][3],a_trim[i][3]
                    a_trim[i][4],a_trim[k][4] = a_trim[k][4],a_trim[i][4]
    

    print("'after sorting")
    print(est_params)


    arr_est = est_params[:,:3]
    # swich value of x,y
    arr_est[:, 0], arr_est[:, 1] = arr_est[:, 1], arr_est[:, 0].copy()
    print(arr_est)

    fstr = "cm_%d_%dx%d_%.1f_%d_%.1f.pkl"%(conf['width'],
                                           conf['max_ny'],
                                           conf['max_nx'],
                                           conf['eratio'],
                                           conf['a'],
                                           conf['local_shift_factor'])
    fout = os.path.join(conf['output'],fstr)
    data={'lens': arr_lens,
          'est': arr_est,
          'img_size': img.shape}

    with open(fout, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
