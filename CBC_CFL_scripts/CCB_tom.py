'''
CCB_tom.py contains functions that serve for the tomographic 
reconstruction of the Crystal diffraction efficiency map for 
Convergent Beam X-ray Crystallography.
'''

import sys,os
sys.path.append(os.path.realpath(__file__))
import numpy as np
from skimage import measure, morphology, feature
import scipy
import glob
import h5py
import re
import CCB_ref
import CCB_pred
import CCB_pat_sim
import CCB_read
#import gen_match_figs as gm
import CCB_streak_det
import matplotlib
#matplotlib.use('TkAgg') # To be adjusted for the batch job mode.
import matplotlib.pyplot as plt
import h5py
import scipy.ndimage as ndi
import CCB_int_proc
import DE_gather
import DE_output
import CCB_ref
import pickle as pk

def fft_arry(diff_dict):
	bins_arry_x = diff_dict['bins_arry_x']
	bins_arry_y = diff_dict['bins_arry_y']
	bins_ind_cx = np.digitize(np.array([-4.170e8]),bins_arry_x)
	bins_ind_cy = np.digitize(np.array([-3.210e8]),bins_arry_y)
	step = (bins_arry_x[-1]-bins_arry_x[0])/(bins_arry_x.shape[0]-1)	
	n = int(np.round(4.5e8/step,decimals=0))

	p_arry = np.zeros((4*n+1,4*n+1))	
	p_arry[2*n-n:2*n+n+1,2*n-n:2*n+n+1] = diff_dict['Int_arry'][bins_ind_cx[0]-1-n:bins_ind_cx[0]+n,bins_ind_cy[0]-1-n:bins_ind_cy[0]+n]
	ft_arry = np.fft.fftshift(np.fft.fft2(p_arry))	
	return ft_arry

def get_rot_mat(frame_no,res_file):
	res_file = os.path.abspath(res_file)
	res_arry = DE_output.gm.read_res(res_file)
	ind = (res_arry[:,0]==frame_no).nonzero()[0][0]
		
	OR_angs=tuple(res_arry[ind,1:4])
	theta,phi,alpha = OR_angs
	rot_mat = CCB_ref.Rot_mat_gen(theta,phi,alpha)@CCB_ref.rot_mat_yaxis(-frame_no)
	cam_len = res_arry[ind,4]
	k_out_osx = res_arry[ind,5]
	k_out_osy = res_arry[ind,6]
	geometry_dict = {'frame_no':frame_no,'rot_mat':rot_mat,'cam_len':cam_len,'k_out_osx':k_out_osx,\
					'k_out_osy':k_out_osy}

	return geometry_dict

def compute_3d_coord(ft_arry,frame_no):
	arry_c = np.array([(ft_arry.shape[0]-1)/2-1,(ft_arry.shape[1]-1)/2-1])
	y_coord0,x_coord0 = np.meshgrid(np.arange(ft_arry.shape[0])-arry_c[0],np.arange(ft_arry.shape[1])-arry_c[1])
	z_coord0 = np.zeros_like(x_coord0)
	x_coord = np.zeros_like(x_coord0)
	y_coord = np.zeros_like(x_coord0)
	z_coord = np.zeros_like(x_coord0)
	x_coord_inv = np.zeros_like(x_coord0)
	y_coord_inv = np.zeros_like(x_coord0)
	z_coord_inv = np.zeros_like(x_coord0)


	geometry_dict = get_rot_mat(frame_no,'/home/lichufen/CCB_ind/Best_GA_res.txt') #default the res_file
	rot_mat = geometry_dict['rot_mat']
	for ind, value in np.ndenumerate(x_coord0):
		#print(ind,value)
		xyz0 = np.array([x_coord0[ind],y_coord0[ind],z_coord0[ind]]).reshape(3,1)
		xyz = rot_mat@xyz0
		xyz_inv = np.linalg.inv(rot_mat)@xyz0
		#print(xyz0,xyz)
		x_coord[ind] = xyz[0]
		y_coord[ind] = xyz[1]
		z_coord[ind] = xyz[2]
		x_coord_inv[ind] = xyz_inv[0]
		y_coord_inv[ind] = xyz_inv[1]
		z_coord_inv[ind] = xyz_inv[2]
		
	#x_3d_ind = x_coord + arry_c[0]
	#y_3d_ind = y_coord + arry_c[1]
	#z_3d_ind = z_coord + arry_c.max()
	
	x_coord_round = (np.round(x_coord,decimals=0)).astype(np.int)
	y_coord_round = (np.round(y_coord,decimals=0)).astype(np.int)
	z_coord_round = (np.round(z_coord,decimals=0)).astype(np.int)
	
	x_coord_inv_round = (np.round(x_coord_inv,decimals=0)).astype(np.int)
	y_coord_inv_round = (np.round(y_coord_inv,decimals=0)).astype(np.int)
	z_coord_inv_round = (np.round(z_coord_inv,decimals=0)).astype(np.int)

	xyz_dict = {'x_coord':x_coord,'y_coord':y_coord,'z_coord':z_coord,\
					'x_coord_round':x_coord_round,'y_coord_round':y_coord_round,'z_coord_round':z_coord_round,\
					'x_coord_inv':x_coord_inv,'y_coord_inv':y_coord_inv,'z_coord_inv':z_coord_inv,\
					'x_coord_inv_round':x_coord_inv_round,'y_coord_inv_round':y_coord_inv_round,\
					'z_coord_inv_round':z_coord_inv_round,'ft_arry':ft_arry}
	return xyz_dict


def assemble_3d_ft(bins=100,save=True):
	dset = CCB_int_proc.Dataset('/home/lichufen/CCB_ind/K_map_fr101.txt')
	frame_no_list = []
	diff_dict_list = []
	xyz_dict_list = []
	for f in dset.frame_obj_list:
		print('frame %d'%f.frame_no)
		frame_no_list.append(f.frame_no)
		diff_dict = f.get_diff_arry(bins=bins)
		xyz_dict = compute_3d_coord(fft_arry(diff_dict),f.frame_no)
		diff_dict_list.append(diff_dict)
		xyz_dict_list.append(xyz_dict)
	
	ft_arry_size = (xyz_dict_list[0]['ft_arry'].shape[0]-1)/2
	ft_3d_size = int(np.round(1.5*ft_arry_size,decimals=0))
	ft_3d_arry = np.zeros((2*ft_3d_size+1,2*ft_3d_size+1,2*ft_3d_size+1))
	
	counter = np.zeros_like(ft_3d_arry)
	sum_ft_3d_arry = np.zeros_like(ft_3d_arry,dtype=np.complex)
	ave_ft_3d_arry = np.zeros_like(ft_3d_arry,dtype=np.complex)
	#sum_ft_3d_arry = np.zeros_like(ft_3d_arry,dtype=np.float)
	#ave_ft_3d_arry = np.zeros_like(ft_3d_arry,dtype=np.float)
	for k in range(len(frame_no_list)):
		frame_no = frame_no_list[k]
		print('%d'%frame_no)
		xyz_dict = xyz_dict_list[k]
		ft_arry = xyz_dict['ft_arry']
		x_coord_round = xyz_dict['x_coord_round']
		y_coord_round = xyz_dict['y_coord_round']
		z_coord_round = xyz_dict['z_coord_round']
		x_ind_round = x_coord_round + ft_3d_size
		y_ind_round = y_coord_round + ft_3d_size
		z_ind_round = z_coord_round + ft_3d_size

		x_coord_inv_round = xyz_dict['x_coord_inv_round']
		y_coord_inv_round = xyz_dict['y_coord_inv_round']
		z_coord_inv_round = xyz_dict['z_coord_inv_round']
		x_ind_inv_round = x_coord_inv_round + ft_3d_size
		y_ind_inv_round = y_coord_inv_round + ft_3d_size
		z_ind_inv_round = z_coord_inv_round + ft_3d_size

		#ft_arry = np.abs(ft_arry) #modulus only
		for ind, value in np.ndenumerate(ft_arry):
			counter[x_ind_round[ind],y_ind_round[ind],z_ind_round[ind]]+= 1
			sum_ft_3d_arry[x_ind_round[ind],y_ind_round[ind],z_ind_round[ind]]+= ft_arry[ind] #modulus only?
			counter[x_ind_inv_round[ind],y_ind_inv_round[ind],z_ind_inv_round[ind]]+= 1
			sum_ft_3d_arry[x_ind_inv_round[ind],y_ind_inv_round[ind],z_ind_inv_round[ind]]+= ft_arry[ind] #
	ave_ft_3d_arry = sum_ft_3d_arry/(counter+sys.float_info.epsilon)
		
	ft_3d_dict = {'counter':counter,'sum_ft_3d_arry':sum_ft_3d_arry,'ave_ft_3d_arry':ave_ft_3d_arry,\
						'frame_no_list':frame_no_list,'xyz_dict_list':xyz_dict_list,'diff_dict_list':diff_dict_list}
	if save:
		with open('ft_3d_dict.pkl','wb') as pf:
			pk.dump(ft_3d_dict,pf)
	
	return ft_3d_dict
