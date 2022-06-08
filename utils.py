#!/usr/bin/python3

import numpy as np
import math
from matplotlib import pyplot as plt

def cca_show(img, lbl_img, lbl_img_final, cca_list, color='black'):
    '''
    input:
      + img: binary image
      + lbl_img: pre labeled image
      + lbl_img_final: 2nd labeled image 
      + cca_list: for annotation
    '''
    INVALID = np.amax(lbl_img)+1
    lbl_img += 1
    lbl_img[lbl_img==INVALID] = 0
    lbl_img_final +=1
    lbl_img_final[lbl_img_final==INVALID]=0

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(lbl_img, cmap='nipy_spectral')
    plt.axis('off')
    ax = plt.subplot(133)
    plt.imshow(lbl_img_final, cmap='nipy_spectral')
    plt.axis('off')
    # annotation
    font = {'family': 'serif',
            'color':  'white',
            'weight': 'bold',
            'size': 16,
            }
    for grp in cca_list:
        xc,yc,rot, a,b, lbl = grp
        cog = plt.Circle((xc,yc), 0.5, color=color)
        ax.add_patch(cog)
        # plt.text(xc, yc, r'$%d$'%(lbl), fontdict=font)
        # draw arrow
        fc = 'r'
        ec = fc
        rad = rot*math.pi/180.0
        y_scale, x_scale = math.sin(rad)*b, math.cos(rad)*b
        arr = plt.arrow(xc,yc,x_scale,y_scale,
                        fc=fc, ec=ec, width = 1)
        ax.add_patch(arr)
        # plt.text(xc+x_scale, yc+y_scale, r'$%.1f$'%(rot), fontdict=font)
        plt.text(xc-2, yc-4, r'$%.1f$'%(rot), fontdict=font)

    plt.tight_layout()
    plt.show()


def show_img(img, cmap='gray'):
    iscolor = False if len(img.shape)== 2 else True
    if iscolor:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)
    plt.show()
