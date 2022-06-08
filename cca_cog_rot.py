#!/usr/bin/python3
'''
Bit accurate model for cog rot
'''

from fxpmath import Fxp
import os
import numpy as np
import math
import pickle
import imageio
import matplotlib.pyplot as plt
import cca_computation_no_fp as alg_nofp
import generate_ellipse as gen_ellipse
import utils


def gen_img_test_software():
    # load image + ground
    conf = {}
    conf['width'] = 256
    conf['eratio'] = 2.0 # ratio of major and minor axis.
    conf['max_ny'] = 8 # <1 mean no change, >1 indicate number of lens in y dir
    conf['max_nx'] = 8 # <1 mean no change, >1 indicate number of lens in x dir
    conf['local_shift_factor'] = 2.0 # small -> more overlapping, large less
    # overlapping
    conf['a'] = 8 # number of pixel of major axis

    nny, nnx, arr_lens = gen_ellipse.get_lenses_array(conf)
    # img, params =  gen_ellipse.gen_ellipse_with_conf(conf)
    img, params = gen_ellipse.generate_ellipse_rand_shift(nny,nnx, arr_lens, conf)
    params = np.asarray(params,dtype=np.float)

    tmp_params = [list(params[ii])+[ii] for ii in range(len(params))]
    # print(tmp_params)
    utils.cca_show(img, img, img, tmp_params, color='r')

    acc_mem, connected, LABEL, lbl_img = alg_nofp.cca_lbl_1st_step(img, max_lbl_width=20)
    acc_mem, connected = alg_nofp.cca_lbl_2nd_step(acc_mem, connected, LABEL)
    est_params, lbl_img_final = alg_nofp.cca_extract_info(acc_mem, LABEL, lbl_img)
    # print(est_params)
    utils.cca_show(img, lbl_img, lbl_img_final,est_params, color='r')

    arr_lens = np.asarray(arr_lens,dtype=float)
    # print(arr_lens)
    est_params = np.asarray(est_params,dtype=float)
    arr_est = est_params[:,:3]
    # swich value of x,y
    arr_est[:, 0], arr_est[:, 1] = arr_est[:, 1], arr_est[:, 0].copy()
    # print(arr_est)

    fout = "../data/sw/cm_%d_%dx%d_%.1f.pkl"%(conf['width'],
                                              conf['max_ny'],
                                              conf['max_nx'],
                                              conf['eratio'])
    data={'lens': arr_lens,
          'est': arr_est,
          'img_size': img.shape}

    with open(fout, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    gen_img_test_software()
