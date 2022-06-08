#!/bin/bash
'''
Computation without considering fixed point math
'''
import numpy as np
from matplotlib import pyplot as plt
import math


def eig_calc(A):
    ''' 
    A should have the structure [[a, b], [b, c]]
    return einval, einvec
    '''
    A = np.array(A)
    a,b,c = A[0,0], A[0,1], A[1,1]
    delta = (a-c)**2 + 4*b**2
    lam1 = (a+c+math.sqrt(delta))/2
    lam2 = (a+c-math.sqrt(delta))/2
    v1 = [lam1-b-c, a+b-lam1]
    v2 = [lam2-b-c, a+b-lam2]
    # handle special case
    if b == 0 and a ==c:
        v1 = [1,0]
        v2 = [0,1]
    elif b!=0 and a ==c:
        if b>0:
            v1 = [1,1]
            v2 = [-1,1]
        else:
            v1 = [-1,1]
            v2 = [1,1]
    # normalize
    # print(v1, v2)
    v1 = v1/ np.linalg.norm(v1)
    v2 = v2/ np.linalg.norm(v2)
    einval = np.array([lam1,lam2]).T
    einvec = np.array([v1,v2]).T
    return einval, einvec

# put above codes into functions
def cca_lbl_1st_step(img, max_lbl_width=20, debug=None):
    '''
    1st step : pass through image, label pixels and constructe connected list.
    input:
      + max_lbl_width = 20 # keeping 2^10-1 label
    output:
      + acc_mem: accumulator memory
      + connected: group of connected label
      + LABEL: number of label existed in image
      + lbl_img: first round labeled image
    '''
    # running cca
    # 6 accumulators S, X, Y, XY, X2, Y2, link
    acc_mem = np.zeros([2**max_lbl_width,7],dtype=np.int) # keeping accumulator
    connected = [] # keeping connected pair.
    H,W = img.shape[:2]
    INVALID = int(2**max_lbl_width-1)
    # pre_row = np.zeros([1,W]) + INVALID # set to invalid value
    cur_row = np.zeros([W,],dtype=np.int) + INVALID
    lbl_img = np.zeros_like(img,dtype=np.int)+INVALID

    LABEL = 0
    list_lbls = [] # keeping list of assigned label per non-background pixel
    list_conns = [] # keeping list of connected label per assigned label
    for iy in range(H):
        pre_lbl = INVALID # previous lbl
        pre_row = cur_row
        for ix in range(W):
            pix = img[iy,ix]
            nw = pre_row[ix-1] if ix>0 else INVALID
            nn = pre_row[ix]
            ne = pre_row[ix+1] if ix<(W-1) else INVALID
            ww = pre_lbl
            # find min label
            nachbar_min_lbl = min(nw,nn,ne,ww)        
            conn_lbl = [0, INVALID] # for testbench
            if pix==0: # is background
                lbl = INVALID
            else:
                if nachbar_min_lbl==INVALID:
                    lbl = LABEL
                    LABEL +=1
                else:
                    lbl = nachbar_min_lbl
                    # update connected groups
                    # can only 1 from these two cases happen
                    if lbl != ne and ne !=INVALID and nn==INVALID:
                        connected.append([lbl,ne])
                        conn_lbl = [1, ne] # flag, lbl
                    if lbl != ww and ww !=INVALID  and nn ==INVALID:
                        connected.append([lbl,ww])
                        conn_lbl = [1, ww]
                # update accumulators
                accs = acc_mem[lbl,:]
                accs[0]+= 1 # S
                accs[1]+= ix # X
                accs[2]+= iy # Y
                accs[3]+= ix*iy # XY
                accs[4]+= ix*ix # X^2
                accs[5]+= iy*iy # Y^2
                accs[6] = lbl
                acc_mem[lbl,:] = accs[:]
            # for debug + testbench
            list_lbls.append([iy,ix,lbl])
            list_conns.append(conn_lbl)
            # update current row
            cur_row[ix] = lbl
            # update lbl img
            lbl_img[iy,ix] = lbl
            # don't forget to update pre_pix
            pre_lbl = lbl
    # ...
    if debug is not None:
       debug['lbls'] = list_lbls
       debug['cons'] = list_conns

    return acc_mem, connected, LABEL, lbl_img

def cca_lbl_2nd_step(acc_mem, connected, LABEL):
    '''
    process conected label list and update labeling
    input:
      + acc_mem: accumlator memory
      + connected: list of connected label
      + LABEL: number of LABEL was used
    output:
      + acc_mem: updated acc_mem
      + connected: updated list of connected component (debug only)
    '''
    # process connected group
    for couple in connected:
        ia,ib = couple # b => a
        acca, accb = acc_mem[ia], acc_mem[ib]
        if accb[6] != ib: # b is moved to accb[6] => add new couple for latter processing
            if ia != accb[6]: # not a duplication
                connected.append([min(ia,accb[6]),max(ia,accb[6])])
            continue
        elif acca[6] != ia: # a is move to acca[6] => add new couple for later processing
            connected.append([acca[6],ib])
            continue
        else: # process as normal
            acca[:6] = acca[:6] + accb[:6]
            accb[6] = ia # mark it as moved to ia
    return acc_mem, connected

def cca_check_duplication(connected):
    #import sys
    #np.set_printoptions(threshold=sys.maxsize)
    #lbl_img[lbl_img==INVALID]=LABEL
    #print(lbl_img[:,:40])
    # lbl_img = INVALID-lbl_img
    #lbl_img[lbl_img==INVALID] = LABEL
    #lbl_img = LABEL - lbl_img
    #print(len(connected), LABEL)
    #print(connected)
    myset = set()
    for x in connected:
        a = '_'.join([str(lbl) for lbl in x])
        myset.add(a)
    print(" Duplication check: origin ", len(connected), " after reduction ", len(myset))
    #print(myset)

def cca_extract_info(acc_mem, LABEL,lbl_img):
    '''
    Extract COG, Rot
    Input:
      + acc_mem: acculator mem
      + LABEL: number of assigned label
      + lbl_img: pre labeled image
    Output:
      + cca_list: list of params [ xc, yc, rot, a, b, lbl]
      + lbl_img_final: relabeled image
    '''
    # update lbl_imge
    lbl_img_final = np.copy(lbl_img)
    cca_list = []
    for ii in range(LABEL):
        acc = acc_mem[ii]
        if acc[6]!=ii: # moved - ignore
            # relabling the image (not fully completed, since we don't trace until the last of linked list)
            # for visualization only.
            lbl_img_final[lbl_img_final==ii] = acc[6]
        else: # valid label - compute COG, ROT
            S, X, Y, XY, X2, Y2, link = acc[:7]
            xc = X*1.0/S
            yc = Y*1.0/S
            m02 = Y2 - Y**2*1.0/S # should div by S
            m20 = X2 - X**2*1.0/S # should div by S
            m11 = XY - X*Y*1.0/S # shoudl div by S
            cov = [[m20, m11],[m11,m02]]
            eigval, eigvec = eig_calc(cov)
            vec = eigvec[:,0]
            if vec[1]==0 or math.isnan(vec[0]) or math.isnan(vec[1]):
                rot = 0
            else:
                rot = np.arctan(vec[1]/vec[0])
            rot = rot/np.pi*180.0 # -1 since y-axis up side down.
            rot = 180+rot if rot<0 else rot
            a, b = list(0.5*np.sqrt(eigval))
            cca_list.append([xc,yc,rot, a,b,ii])

    return cca_list, lbl_img_final
