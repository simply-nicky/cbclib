'''
CCB_pred module consists of some functions that does some diffraciton prediction
based on forward modeling.

'''
import sys,os
sys.path.append(os.path.realpath(__file__))
import h5py
import numpy as np
import scipy
import matplotlib
#matplotlib.use('pdf')
import matplotlib.pyplot as plot

import CCB_pat_sim


def kout_pred(OR_mat,k_in_cen,HKL_int):
    #NA. is the numeriacal aperture, now a single value salar, in radians.
    OR_mat=OR_mat.reshape(3,3)
    k_in_cen=np.array(k_in_cen).reshape(3,1)
    num_q=HKL_int.shape[0]
    K_in_pred=np.zeros((num_q,3))
    K_out_pred=np.zeros((num_q,3))

    for num in range(num_q):
        hkl=HKL_int[num,:].reshape(3,1)
        q_int=OR_mat@hkl
        n=np.cross(q_int,k_in_cen,axis=0)
        n_u=n/np.linalg.norm(n,axis=0)
        p_u=np.cross(n_u,q_int,axis=0)
        p_u=p_u/np.linalg.norm(p_u,axis=0)
        p=np.sqrt(np.linalg.norm(k_in_cen,axis=0)**2-np.linalg.norm(q_int/2,axis=0)**2)*p_u
        k_in_s=p-q_int/2
        k_out_s=k_in_s+q_int
        ################## change to the mid-points of the streaks
        K_in_SL, K_out_SL = CCB_pat_sim.source_line_scan(k_in_cen,OR_mat,hkl,rot_ang_step=0.05,rot_ang_range=3.0)
        k_in = np.nanmean(K_in_SL,axis=0)
        k_out = k_in + q_int
        ##################
        K_in_pred[num,:]=k_in.reshape(-1)
        K_out_pred[num,:]=k_out.reshape(-1)
    return K_in_pred,K_out_pred

def kout_pred8(OR_mat,k_in_cen,HKL_int):
    #NA. is the numeriacal aperture, now a single value salar, in radians.
    OR_mat=OR_mat.reshape(3,3)
    k_in_cen=np.array(k_in_cen).reshape(3,1)
    num_q=HKL_int.shape[0]
    K_in_pred=np.zeros((num_q,3,8))
    K_out_pred=np.zeros((num_q,3,8))

    for num in range(num_q):
        for num_c in range(8):
            hkl=HKL_int[num,:,num_c].reshape(3,1)
            q_int=OR_mat@hkl
            n=np.cross(q_int,k_in_cen,axis=0)
            n_u=n/np.linalg.norm(n,axis=0)
            p_u=np.cross(n_u,q_int,axis=0)
            p_u=p_u/np.linalg.norm(p_u,axis=0)
            p=np.sqrt(np.linalg.norm(k_in_cen,axis=0)**2-np.linalg.norm(q_int/2,axis=0)**2)*p_u
            k_in=p-q_int/2
            k_out=k_in+q_int
            K_in_pred[num,:,num_c]=k_in.reshape(-1)
            K_out_pred[num,:,num_c]=k_out.reshape(-1)
    return K_in_pred,K_out_pred
