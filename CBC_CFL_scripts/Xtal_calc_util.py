#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'Xtal_calc_util' generates the lattice constant matrix in real and reciprocal space for
a give lattice constant set.

usage:

    A_gen <a>  <b>  <c> <alpha> <beta> <gamma>


Created on Wed Dec 12 13:59:04 2018
@author: chufeng
"""

import numpy as np
import sys,os
#import matplotlib.pyplot as plt

#

def deg2rad(deg):

	rad=deg/180*np.pi
	return rad

def rad2deg(rad):

	deg=rad/np.pi*180
	return deg


def A_gen(lp):
	'''
	'A_gen' generates the lattice constant matrix in real and reciprocal space for
	a give lattice constant set.
	Use:
    	(A,Astar) = A_gen(lp)
	where lp=[a,b,c,alpha,beta,gamma], a ndarray type.
	'''
	#check lp, 6 element array
	lp=np.array(lp)
	if lp.shape[0]!=6:
		sys.exit('lattice constant must be a 6-element array!')
	a_rl=np.array([lp[0],0,0])
	b_rl=np.array([lp[1]*np.cos(deg2rad(lp[5])),lp[1]*np.sin(deg2rad(lp[5])),0])
	c_rl=np.zeros((3,))
	c_rl[0]=lp[2]*np.cos(deg2rad(lp[4]))
	c_rl[1]=lp[2]*(np.cos(deg2rad(lp[3]))-np.cos(deg2rad(lp[4]))*np.cos(deg2rad(lp[5])))/np.sin(deg2rad(lp[5]))
	c_rl[2]=np.sqrt(lp[2]**2-c_rl[0]**2-c_rl[1]**2)

	if np.isnan(c_rl[2]):
		sys.exit('the lattice constants are non-physical')

	A=np.stack((a_rl,b_rl,c_rl),axis=1)

	Vol=np.dot(a_rl,np.cross(b_rl,c_rl))
	a_star=np.cross(b_rl,c_rl)/Vol
	b_star=np.cross(c_rl,a_rl)/Vol
	c_star=np.cross(a_rl,b_rl)/Vol
	A_star=np.stack((a_star,b_star,c_star),axis=1)

	return A, A_star


def Get_relp_q(A_star,HKL):

	HKL=np.array(HKL)
	if HKL.shape[0]!=3:
		sys.exit('check HKL')
	HKL=HKL.reshape((3,1))
	if A_star.shape!=(3,3):
		sys.exit('check A_star')
	q=A_star@HKL

	return q

def Get_HKL(A_star,q):

	q=np.array(q)
	if q.shape[0]!=3:
		sys.exit('check q')
	q=q.reshape((3,1))
	if A_star.shape!=(3,3):
		sys.exit('check A_star')
	HKL_frac=np.linalg.solve(A_star,q)
	HKL_int=np.round(HKL_frac)
	HKL_int=HKL_int.astype(int)
	return HKL_frac,HKL_int

def delta_calc(q_1,q_2):

	delta_q=q_1-q_2
	delta_q_abs=np.linalg.norm(np.abs(delta_q))

	return delta_q_abs


def est_res(cam_len,photon_energy,Geom_dict):
    '''
    It estimates the resolution limit given the scattering geometry.
    '''
    pixel_size=np.float64(110e-6)
    Geom_file_name=Geom_dict['Geom_file_name']
    x_pix=Geom_dict['x_pix']
    y_pix=Geom_dict['y_pix']
    center_ind=Geom_dict['center_ind']
    Assemble_size=np.maximum((np.abs(x_pix)).max(),(np.abs(y_pix)).max())

    sct_angl_max=np.arctan(1.4*Assemble_size*pixel_size/cam_len)
    wave_len=12398/photon_energy #units: Angstrom and eV
    res_limit=wave_len/(2*np.sin(sct_angl_max/2)) #angstrom
    q_limit=1/res_limit # Angstrom^-1
    return res_limit,q_limit

def HKL_extend(HKL):
    '''
    HKL_ext extends the positive HKL index to 8 quadrants.
    '''
    HKL=np.array(HKL).astype(np.int16).reshape(3,)
    H=HKL[0]
    K=HKL[1]
    L=HKL[2]
    HKL_ext=np.zeros((8,3))
    row_counter=0
    if (H<0 or K<0 or L<0):
        sys.exit('The HKL_ext should not operate on negative HKL')
    else:
        pass
    if H==0:
        if K==0:
            if L==0:
                HKL_ext[row_counter,:]=HKL
                row_counter=row_counter+1
            else:
                HKL_ext[row_counter,:]=HKL
                HKL_ext[row_counter+1,:]=np.array([0,0,-L]).astype(np.int16)
                row_counter=row_counter+2
        else:
            if L==0:
                HKL_ext[row_counter,:]=HKL
                HKL_ext[row_counter+1,:]=np.array([0,-K,0]).astype(np.int16)
                row_counter=row_counter+2
            else:
                HKL_ext[row_counter,:]=HKL
                HKL_ext[row_counter+1,:]=np.array([0,K,-L]).astype(np.int16)
                HKL_ext[row_counter+2,:]=np.array([0,-K,L]).astype(np.int16)
                HKL_ext[row_counter+3,:]=np.array([0,-K,-L]).astype(np.int16)
                row_counter=row_counter+4
    else:
        if K==0:
            if L==0:
                HKL_ext[row_counter,:]=HKL
                HKL_ext[row_counter+1,:]=np.array([-H,0,0]).astype(np.int16)
                row_counter=row_counter+2
            else:
                HKL_ext[row_counter,:]=HKL
                HKL_ext[row_counter+1,:]=np.array([H,0,-L]).astype(np.int16)
                HKL_ext[row_counter+2,:]=np.array([-H,0,L]).astype(np.int16)
                HKL_ext[row_counter+3,:]=np.array([-H,0,-L]).astype(np.int16)
                row_counter=row_counter+4

        else:
            if L==0:
                HKL_ext[row_counter,:]=HKL
                HKL_ext[row_counter+1,:]=np.array([H,-K,0]).astype(np.int16)
                HKL_ext[row_counter+2,:]=np.array([-H,K,0]).astype(np.int16)
                HKL_ext[row_counter+3,:]=np.array([-H,-K,0]).astype(np.int16)
                row_counter=row_counter+4
            else:
                HKL_ext[row_counter,:]=HKL
                HKL_ext[row_counter+1,:]=np.array([H,K,-L]).astype(np.int16)
                HKL_ext[row_counter+2,:]=np.array([H,-K,L]).astype(np.int16)
                HKL_ext[row_counter+3,:]=np.array([-H,K,L]).astype(np.int16)
                HKL_ext[row_counter+4,:]=np.array([H,-K,-L]).astype(np.int16)
                HKL_ext[row_counter+5,:]=np.array([-H,K,-L]).astype(np.int16)
                HKL_ext[row_counter+6,:]=np.array([-H,-K,L]).astype(np.int16)
                HKL_ext[row_counter+7,:]=np.array([-H,-K,-L]).astype(np.int16)
                row_counter=row_counter+8

    return HKL_ext[:row_counter,:].reshape(-1,3)

def gen_sf_list(lp,res_cut_hi,res_cut_low):
    lp=np.array(lp)
    ind_max=np.ceil(lp[0:3].max()/res_cut_hi).astype(np.int16)#mas possible HKL indice
    ######################################

        ##############################################
    sf_list_ext=np.array([]).reshape(-1,4)
    _,A_star=A_gen(lp)
    row_counter=0
    for H in range(ind_max+1):
        for K in range(ind_max+1):
            for L in range(ind_max+1):
                    HKL=np.array([H,K,L]).astype(np.int16)

                    q=Get_relp_q(A_star,HKL)
                    resol_q=np.linalg.norm(q)
                    if resol_q==0:
                        resol=10000
                    else:
                        resol=1/resol_q
                    HKL_ext=HKL_extend(HKL)
                    HKL_ext=np.concatenate((HKL_ext,resol*np.ones((HKL_ext.shape[0],1))),axis=-1)
                    sf_list_ext=np.append(sf_list_ext,HKL_ext,axis=0)
                    row_counter=row_counter+HKL_ext.shape[0]
    sf_list_ext=sf_list_ext[:row_counter,:]
    ind=np.argsort(sf_list_ext[:,3])
    ind=ind[::-1]
    sf_list_ext=sf_list_ext[ind,:]

    ind_low=np.where(sf_list_ext[:,3]<=res_cut_low)
    ind_hi=np.where(sf_list_ext[:,3]>=res_cut_hi)
    ind_low=np.minimum(ind_low[0][0],sf_list_ext.shape[0]-1)
    if ind_hi[0].shape[0]==0:
        ind_hi=sf_list_ext[0]-1
    else:
        ind_hi=ind_hi[0][-1]
    sf_list_ext=sf_list_ext[ind_low:ind_hi+1,:]
    return sf_list_ext


#def Euler2Q():


'''
'Euler2Q' calculates the roation matrix from a given set of 3 Euler angles
according to 'CFL convention'
Use:
    Q=Euler2Q(Euler_angles)
where the Euler_angles=[phi,theta,alpha], a ndarray type.
the (phi,theta) angle pair defines the rotation axis, and the
alpha defines the rotation angle around the axis.
phi:   the angle from the x-axis to the projection of the rotation axis onto the
       x-y plane, in z-direction.
theta: the angle from the z-axis to the rotation axis.
alpha: the rotation angle around the rotaion axis defined by (phi,theta) pair.

'''
#    return Q

#def Q2Euler():

'''
'Q2Euler' calculates the Euler angles according from the given rotation matrix
Q.
Q must be a orthogonal matrix:
    det(Q) = 1 right hand rotation
    det(Q) =-1 left hand rotation

'''

 #   return Euler_angle
